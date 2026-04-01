/**
 * gesture_kernel.cpp — Hand gesture recognition kernel implementation
 *
 * Pipeline: events → activity map → binary mask → blob detection →
 *           convex hull → defect counting → gesture classification
 */

#include "gesture_kernel.h"

#include <cstring>
#include <cstdio>
#include <algorithm>
#include <queue>

/* ────────────────────────────────────────────────────────────────── */
/*  Construction / Destruction                                       */
/* ────────────────────────────────────────────────────────────────── */

GestureKernel::GestureKernel(int width, int height, const char* tflite_model_path)
    : width_(width), height_(height)
{
    int n = width * height;
    activity_  = new float[n]();
    binary_    = new uint8_t[n]();
    label_map_ = new int[n]();
    resize_buf_ = new uint8_t[128 * 128 * 3]();
    if (tflite_model_path)
        load_tflite_model(tflite_model_path);
}

GestureKernel::~GestureKernel()
{
    delete[] activity_;
    delete[] binary_;
    delete[] label_map_;
    delete[] resize_buf_;
}

/* ────────────────────────────────────────────────────────────────── */
/*  Event Ingestion                                                  */
/* ────────────────────────────────────────────────────────────────── */

void GestureKernel::ingest_events(const dvs_event_t* events, uint32_t count)
{
    /* Apply temporal decay first */
    decay_activity();

    /* Accumulate new events */
    for (uint32_t i = 0; i < count; i++) {
        int x = events[i].x;
        int y = events[i].y;
        if (x < 0 || x >= width_ || y < 0 || y >= height_)
            continue;

        int idx = y * width_ + x;
        if (events[i].polarity == 1) {
            activity_[idx] += ACTIVITY_INCREMENT;
        } else {
            activity_[idx] += ACTIVITY_DECREMENT; /* OFF events also show motion */
        }

        /* Clamp */
        if (activity_[idx] > 255.0f) activity_[idx] = 255.0f;
    }
}

/* ────────────────────────────────────────────────────────────────── */
/*  Pipeline Stage: Temporal Decay                                   */
/* ────────────────────────────────────────────────────────────────── */

void GestureKernel::decay_activity()
{
    int n = width_ * height_;
    for (int i = 0; i < n; i++) {
        activity_[i] *= ACTIVITY_DECAY_FACTOR;
        if (activity_[i] < 1.0f) activity_[i] = 0.0f;
    }
}

/* ────────────────────────────────────────────────────────────────── */
/*  Pipeline Stage: Binarization                                     */
/* ────────────────────────────────────────────────────────────────── */

void GestureKernel::apply_threshold()
{
    int n = width_ * height_;
    for (int i = 0; i < n; i++) {
        binary_[i] = (activity_[i] >= ACTIVITY_THRESHOLD) ? 1 : 0;
    }

    /* Simple 3×3 morphological close (dilate then erode) to fill gaps */
    auto morph_op = [&](uint8_t* src, uint8_t* dst, bool dilate) {
        memset(dst, 0, n);
        for (int y = 1; y < height_ - 1; y++) {
            for (int x = 1; x < width_ - 1; x++) {
                if (dilate) {
                    /* Dilate: output=1 if any neighbor=1 */
                    uint8_t val = 0;
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++)
                            val |= src[(y + dy) * width_ + (x + dx)];
                    dst[y * width_ + x] = val;
                } else {
                    /* Erode: output=1 if all neighbors=1 */
                    uint8_t val = 1;
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++)
                            val &= src[(y + dy) * width_ + (x + dx)];
                    dst[y * width_ + x] = val;
                }
            }
        }
    };

    uint8_t* temp = new uint8_t[n];

    /* Dilate */
    morph_op(binary_, temp, true);
    /* Erode */
    morph_op(temp, binary_, false);

    delete[] temp;
}

/* ────────────────────────────────────────────────────────────────── */
/*  Pipeline Stage: Connected Component Labeling (BFS flood-fill)    */
/* ────────────────────────────────────────────────────────────────── */

int GestureKernel::connected_components()
{
    int n = width_ * height_;
    memset(label_map_, 0, n * sizeof(int));

    int label = 0;
    std::queue<int> q;

    for (int i = 0; i < n; i++) {
        if (binary_[i] == 0 || label_map_[i] != 0)
            continue;

        label++;
        label_map_[i] = label;
        q.push(i);

        while (!q.empty()) {
            int idx = q.front();
            q.pop();

            int y = idx / width_;
            int x = idx % width_;

            /* 4-connected neighbors */
            static const int dx[] = {-1, 1, 0, 0};
            static const int dy[] = {0, 0, -1, 1};
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d];
                int ny = y + dy[d];
                if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_)
                    continue;
                int ni = ny * width_ + nx;
                if (binary_[ni] == 1 && label_map_[ni] == 0) {
                    label_map_[ni] = label;
                    q.push(ni);
                }
            }
        }
    }

    return label;
}

/* ────────────────────────────────────────────────────────────────── */
/*  Pipeline Stage: Find Largest Blob                                */
/* ────────────────────────────────────────────────────────────────── */

int GestureKernel::find_largest_blob(int num_labels)
{
    if (num_labels == 0) return 0;

    std::vector<int> counts(num_labels + 1, 0);
    int n = width_ * height_;
    for (int i = 0; i < n; i++) {
        if (label_map_[i] > 0)
            counts[label_map_[i]]++;
    }

    int best_label = 0;
    int best_count = 0;
    for (int l = 1; l <= num_labels; l++) {
        if (counts[l] > best_count) {
            best_count = counts[l];
            best_label = l;
        }
    }

    return (best_count >= MIN_BLOB_AREA) ? best_label : 0;
}

/* ────────────────────────────────────────────────────────────────── */
/*  Pipeline Stage: Blob Feature Extraction                          */
/* ────────────────────────────────────────────────────────────────── */

void GestureKernel::extract_blob_features(int label, BlobFeatures& feat)
{
    memset(&feat, 0, sizeof(feat));
    if (label == 0) return;

    int min_x = width_, max_x = 0;
    int min_y = height_, max_y = 0;
    long sum_x = 0, sum_y = 0;
    int area = 0;

    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (label_map_[y * width_ + x] == label) {
                area++;
                sum_x += x;
                sum_y += y;
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }

    feat.area = area;
    feat.bbox_x = min_x;
    feat.bbox_y = min_y;
    feat.bbox_w = max_x - min_x + 1;
    feat.bbox_h = max_y - min_y + 1;
    feat.centroid.x = (int)(sum_x / area);
    feat.centroid.y = (int)(sum_y / area);

    int bbox_area = feat.bbox_w * feat.bbox_h;
    feat.fill_ratio = (bbox_area > 0) ? (float)area / bbox_area : 0;
    feat.aspect_ratio = (feat.bbox_h > 0) ? (float)feat.bbox_w / feat.bbox_h : 1.0f;

    /* Compute perimeter (count boundary pixels) */
    int perimeter = 0;
    for (int y = 0; y < height_; y++) {
        for (int x = 0; x < width_; x++) {
            if (label_map_[y * width_ + x] != label) continue;
            /* Check if any 4-neighbor is background */
            bool is_boundary = false;
            static const int dx[] = {-1, 1, 0, 0};
            static const int dy[] = {0, 0, -1, 1};
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d];
                int ny = y + dy[d];
                if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_ ||
                    label_map_[ny * width_ + nx] != label) {
                    is_boundary = true;
                    break;
                }
            }
            if (is_boundary) perimeter++;
        }
    }
    feat.compactness = (area > 0) ? (float)(perimeter * perimeter) / area : 0;
}

/* ────────────────────────────────────────────────────────────────── */
/*  TFLite Inference                                                 */
/* ────────────────────────────────────────────────────────────────── */


void GestureKernel::resize_activity_to_model_input()
{
    /* Bilinear resize: activity_ (width_×height_ float, 0–255)
     * → resize_buf_ (128×128×3 uint8, grayscale replicated to 3 channels) */
    const int IN_W = width_, IN_H = height_;
    const int OUT  = 128;

    for (int oy = 0; oy < OUT; oy++) {
        float sy = oy * (float)(IN_H - 1) / (OUT - 1);
        int   y0 = (int)sy;
        int   y1 = (y0 + 1 < IN_H) ? y0 + 1 : y0;
        float fy = sy - y0;

        for (int ox = 0; ox < OUT; ox++) {
            float sx = ox * (float)(IN_W - 1) / (OUT - 1);
            int   x0 = (int)sx;
            int   x1 = (x0 + 1 < IN_W) ? x0 + 1 : x0;
            float fx = sx - x0;

            float v = activity_[y0 * IN_W + x0] * (1 - fx) * (1 - fy)
                    + activity_[y0 * IN_W + x1] *      fx  * (1 - fy)
                    + activity_[y1 * IN_W + x0] * (1 - fx) *      fy
                    + activity_[y1 * IN_W + x1] *      fx  *      fy;

            uint8_t pix = (v < 0.0f) ? 0u : (v > 255.0f) ? 255u : (uint8_t)v;
            int base = (oy * OUT + ox) * 3;
            resize_buf_[base + 0] = pix;
            resize_buf_[base + 1] = pix;
            resize_buf_[base + 2] = pix;
        }
    }
}

bool GestureKernel::load_tflite_model(const char* path)
{
    tflite_model_ = tflite::FlatBufferModel::BuildFromFile(path);
    if (!tflite_model_) {
        fprintf(stderr, "[tflite] Failed to load model: %s\n", path);
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*tflite_model_, resolver);
    builder(&tflite_interp_);
    if (!tflite_interp_) {
        fprintf(stderr, "[tflite] Failed to build interpreter\n");
        return false;
    }

    tflite_interp_->SetNumThreads(1);
    if (tflite_interp_->AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "[tflite] AllocateTensors() failed\n");
        return false;
    }

    const TfLiteTensor* inp = tflite_interp_->input_tensor(0);
    fprintf(stderr, "[tflite] Loaded %s — input [%d,%d,%d,%d] type=%d\n",
            path,
            inp->dims->data[0], inp->dims->data[1],
            inp->dims->data[2], inp->dims->data[3],
            (int)inp->type);
    return true;
}

gesture_class_t GestureKernel::run_tflite_inference(float& confidence)
{
    /* Label mapping from gesture_labels.txt:
       keras idx 0 → GESTURE_FIST      (proto 2)
       keras idx 1 → GESTURE_POINTING  (proto 3)
       keras idx 2 → GESTURE_OPEN_HAND (proto 1)
       keras idx 3 → GESTURE_PEACE     (proto 4)
       keras idx 4 → GESTURE_THUMBS_UP (proto 5) */
    static const gesture_class_t kKerasToProto[5] = {
        GESTURE_FIST,
        GESTURE_POINTING,
        GESTURE_OPEN_HAND,
        GESTURE_PEACE,
        GESTURE_THUMBS_UP,
    };

    uint8_t* input_data = tflite_interp_->typed_input_tensor<uint8_t>(0);
    memcpy(input_data, resize_buf_, 128 * 128 * 3);

    if (tflite_interp_->Invoke() != kTfLiteOk) {
        confidence = 0.0f;
        return GESTURE_NONE;
    }

    const uint8_t* out = tflite_interp_->typed_output_tensor<uint8_t>(0);
    int     best_idx = 0;
    uint8_t best_val = out[0];
    for (int i = 1; i < 5; i++) {
        if (out[i] > best_val) { best_val = out[i]; best_idx = i; }
    }

    confidence = best_val * (1.0f / 256.0f);
    return kKerasToProto[best_idx];
}

/* ────────────────────────────────────────────────────────────────── */
/*  Top-Level: Run Full Pipeline                                     */
/* ────────────────────────────────────────────────────────────────── */

GestureResult GestureKernel::classify()
{
    GestureResult result;
    memset(&result, 0, sizeof(result));
    result.gesture = GESTURE_NONE;

    /* Stage 1: Threshold activity map to binary */
    apply_threshold();

    /* Stage 2: Connected component labeling */
    int num_labels = connected_components();

    /* Stage 3: Find largest blob */
    int dominant = find_largest_blob(num_labels);
    if (dominant == 0) return result;

    /* Stage 4: Extract shape features */
    extract_blob_features(dominant, result.features);

    /* Stage 5: TFLite inference — skip sparse frames to avoid false positives */
    if (result.features.area >= MIN_BLOB_AREA * 4) {
        resize_activity_to_model_input();
        result.gesture = run_tflite_inference(result.confidence);
        if (result.confidence < 0.55f) {
            result.gesture    = GESTURE_NONE;
            result.confidence = 0.0f;
        }
    }

    return result;
}
