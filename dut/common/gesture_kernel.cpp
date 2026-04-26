// Gesture recognition kernel — shared across all DUT targets.
//
// Platform-specific inference back-end is resolved at link time via:
//   extern "C" int32_t gesture_run_inference(void* input, void* output);
//
// E1x EVK: provided by eff-import (model compiled to native RISC-V at build time).
// RPi 4:   provided by dut/rpi4/tflite_backend.cpp (TFLite v2.18.0 interpreter).
//
// Input tensor:  int8_t[TFLM_INPUT_W * TFLM_INPUT_H * TFLM_INPUT_CH]
//   (uint8 pixel value − 128; model uses inference_input_type=uint8, zp=−128)
// Output tensor: int8_t[TFLM_NUM_CLASSES]
//   (uint8 probabilities stored as int8 with zp=−128;
//    float_prob = ((int)output[i] + 128) / 255.0f)
// Returns 0 on success.

#include "gesture_kernel.h"

#include <cstring>

// __efficient__ marks hot loops for CGRA acceleration on the E1x EVK.
// It is a no-op on all other targets.
#ifndef __efficient__
#define __efficient__
#endif

extern "C" int32_t gesture_run_inference(void* input, void* output);

/* ── Static buffers ───────────────────────────────────────────────── */

static const int W = DVS_FRAME_WIDTH;
static const int H = DVS_FRAME_HEIGHT;
static const int N = DVS_FRAME_WIDTH * DVS_FRAME_HEIGHT;

static float    s_activity[N];
static uint8_t  s_binary[N];
static int      s_label_map[N];
static uint8_t  s_resize_buf[TFLM_INPUT_W * TFLM_INPUT_H * TFLM_INPUT_CH];

/* BFS queue — worst case all pixels */
static int      s_bfs_queue[N];

/* ── Label mapping (matches train_gesture_model.py CLASS_NAMES sort order) ── */
/* Keras alphabetical order: fist, one, palm, peace, thumb_up */
static const gesture_class_t kKerasToProto[TFLM_NUM_CLASSES] = {
    GESTURE_FIST,       /* keras idx 0 */
    GESTURE_POINTING,   /* keras idx 1 */
    GESTURE_OPEN_HAND,  /* keras idx 2 */
    GESTURE_PEACE,      /* keras idx 3 */
    GESTURE_THUMBS_UP,  /* keras idx 4 */
};

/* ── Internal helpers ─────────────────────────────────────────────── */

__efficient__ static void decay_activity(void)
{
    for (int i = 0; i < N; i++) {
        float v = s_activity[i] * ACTIVITY_DECAY_FACTOR;
        s_activity[i] = (v < 1.0f) ? 0.0f : v;
    }
}

__efficient__ static void apply_threshold(void)
{
    for (int i = 0; i < N; i++)
        s_binary[i] = (s_activity[i] >= ACTIVITY_THRESHOLD) ? 1 : 0;

    /* 3×3 morphological close (dilate then erode) */
    static uint8_t tmp[N];

    /* Dilate */
    memset(tmp, 0, N);
    for (int y = 1; y < H - 1; y++) {
        for (int x = 1; x < W - 1; x++) {
            uint8_t v = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    v |= s_binary[(y + dy) * W + (x + dx)];
            tmp[y * W + x] = v;
        }
    }

    /* Erode */
    memset(s_binary, 0, N);
    for (int y = 1; y < H - 1; y++) {
        for (int x = 1; x < W - 1; x++) {
            uint8_t v = 1;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    v &= tmp[(y + dy) * W + (x + dx)];
            s_binary[y * W + x] = v;
        }
    }
}

static int connected_components(void)
{
    memset(s_label_map, 0, N * sizeof(int));

    static const int dx4[] = {-1, 1, 0, 0};
    static const int dy4[] = {0, 0, -1, 1};

    int label = 0;

    for (int start = 0; start < N; start++) {
        if (s_binary[start] == 0 || s_label_map[start] != 0)
            continue;

        label++;
        s_label_map[start] = label;

        int qh = 0, qt = 0;
        s_bfs_queue[qt++] = start;

        while (qh < qt) {
            int idx = s_bfs_queue[qh++];
            int y = idx / W, x = idx % W;

            for (int d = 0; d < 4; d++) {
                int nx = x + dx4[d], ny = y + dy4[d];
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                int ni = ny * W + nx;
                if (s_binary[ni] == 1 && s_label_map[ni] == 0) {
                    s_label_map[ni] = label;
                    s_bfs_queue[qt++] = ni;
                }
            }
        }
    }

    return label;
}

static int find_largest_blob(int num_labels)
{
    if (num_labels == 0) return 0;

    static int counts[N];
    memset(counts, 0, (num_labels + 1) * sizeof(int));

    for (int i = 0; i < N; i++)
        if (s_label_map[i] > 0)
            counts[s_label_map[i]]++;

    int best = 0, best_cnt = 0;
    for (int l = 1; l <= num_labels; l++) {
        if (counts[l] > best_cnt) { best_cnt = counts[l]; best = l; }
    }

    return (best_cnt >= MIN_BLOB_AREA) ? best : 0;
}

static void extract_blob_features(int label, blob_features_t* feat)
{
    memset(feat, 0, sizeof(*feat));
    if (label == 0) return;

    static const int dx4[] = {-1, 1, 0, 0};
    static const int dy4[] = {0, 0, -1, 1};

    int min_x = W, max_x = 0, min_y = H, max_y = 0;
    long sum_x = 0, sum_y = 0;
    int area = 0, perimeter = 0;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (s_label_map[y * W + x] != label) continue;
            area++;
            sum_x += x; sum_y += y;
            if (x < min_x) min_x = x;
            if (x > max_x) max_x = x;
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;

            for (int d = 0; d < 4; d++) {
                int nx = x + dx4[d], ny = y + dy4[d];
                if (nx < 0 || nx >= W || ny < 0 || ny >= H ||
                    s_label_map[ny * W + nx] != label) {
                    perimeter++;
                    break;
                }
            }
        }
    }

    feat->area = area;
    feat->bbox_x = min_x; feat->bbox_y = min_y;
    feat->bbox_w = max_x - min_x + 1;
    feat->bbox_h = max_y - min_y + 1;
    feat->centroid_x = (int)(sum_x / area);
    feat->centroid_y = (int)(sum_y / area);

    int bbox_area = feat->bbox_w * feat->bbox_h;
    feat->fill_ratio   = (bbox_area > 0) ? (float)area / bbox_area : 0.0f;
    feat->aspect_ratio = (feat->bbox_h > 0) ? (float)feat->bbox_w / feat->bbox_h : 1.0f;
    feat->compactness  = (area > 0) ? (float)(perimeter * perimeter) / area : 0.0f;
}

__efficient__ static void resize_activity_to_model_input(void)
{
    /* Bilinear resize: activity (W×H float [0-255])
     * → resize_buf (128×128×3 uint8, grayscale→3ch) */
    const int OUT = TFLM_INPUT_W;

    for (int oy = 0; oy < OUT; oy++) {
        float sy = oy * (float)(H - 1) / (OUT - 1);
        int   y0 = (int)sy;
        int   y1 = (y0 + 1 < H) ? y0 + 1 : y0;
        float fy = sy - y0;

        for (int ox = 0; ox < OUT; ox++) {
            float sx = ox * (float)(W - 1) / (OUT - 1);
            int   x0 = (int)sx;
            int   x1 = (x0 + 1 < W) ? x0 + 1 : x0;
            float fx = sx - x0;

            float v = s_activity[y0 * W + x0] * (1.0f - fx) * (1.0f - fy)
                    + s_activity[y0 * W + x1] *         fx  * (1.0f - fy)
                    + s_activity[y1 * W + x0] * (1.0f - fx) *         fy
                    + s_activity[y1 * W + x1] *         fx  *         fy;

            uint8_t pix = (v < 0.0f) ? 0u : (v > 255.0f) ? 255u : (uint8_t)v;
            int base = (oy * OUT + ox) * TFLM_INPUT_CH;
            s_resize_buf[base + 0] = pix;
            s_resize_buf[base + 1] = pix;
            s_resize_buf[base + 2] = pix;
        }
    }
}

static gesture_class_t run_inference(float* confidence)
{
    *confidence = 0.0f;

    /* Shift uint8 → int8 (zero-point = −128) to match model's input quant */
    static int8_t input_buf[TFLM_INPUT_W * TFLM_INPUT_H * TFLM_INPUT_CH];
    for (int i = 0; i < TFLM_INPUT_W * TFLM_INPUT_H * TFLM_INPUT_CH; i++)
        input_buf[i] = (int8_t)((int)s_resize_buf[i] - 128);

    int8_t output_buf[TFLM_NUM_CLASSES] = {};
    if (gesture_run_inference(input_buf, output_buf) != 0) return GESTURE_NONE;

    /* Dequantize int8 output (zp=−128) to [0,1] probability and find argmax */
    int best_idx = 0;
    int best_val = (int)output_buf[0] + 128;
    for (int i = 1; i < TFLM_NUM_CLASSES; i++) {
        int val = (int)output_buf[i] + 128;
        if (val > best_val) { best_val = val; best_idx = i; }
    }

    *confidence = best_val / 255.0f;
    return kKerasToProto[best_idx];
}

/* ── Public API ───────────────────────────────────────────────────── */

int gesture_kernel_init(void)
{
    memset(s_activity,  0, sizeof(s_activity));
    memset(s_binary,    0, sizeof(s_binary));
    memset(s_label_map, 0, sizeof(s_label_map));
    return 0;
}

void gesture_kernel_ingest(const dvs_event_t* events, uint32_t count)
{
    decay_activity();

    for (uint32_t i = 0; i < count; i++) {
        int x = events[i].x, y = events[i].y;
        if (x < 0 || x >= W || y < 0 || y >= H) continue;

        int idx = y * W + x;
        if (events[i].polarity == 1)
            s_activity[idx] += ACTIVITY_INCREMENT;
        else
            s_activity[idx] += ACTIVITY_DECREMENT;

        if (s_activity[idx] > 255.0f) s_activity[idx] = 255.0f;
    }
}

gesture_result_t gesture_kernel_classify(void)
{
    gesture_result_t result;
    memset(&result, 0, sizeof(result));
    result.gesture = GESTURE_NONE;

    apply_threshold();

    int num_labels = connected_components();
    int dominant   = find_largest_blob(num_labels);
    if (dominant == 0) return result;

    extract_blob_features(dominant, &result.features);

    if (result.features.area >= MIN_BLOB_AREA * 4) {
        resize_activity_to_model_input();
        result.gesture = run_inference(&result.confidence);

        if (result.confidence < 0.55f) {
            result.gesture    = GESTURE_NONE;
            result.confidence = 0.0f;
        }
    }

    return result;
}

const float* gesture_kernel_activity_map(void)
{
    return s_activity;
}
