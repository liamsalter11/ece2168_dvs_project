// Copyright Efficient Computer Company 2026. All Rights Reserved.
//
// Gesture recognition kernel — firmware port of the DVS desktop kernel.
// Uses TFLite Micro for inference instead of full TFLite.

#include "gesture_kernel.h"

#include <cstring>
#include <cstdlib>

/* TFLite Micro headers.
 * flatbuffers.h pulls <algorithm> → <cstdio>. The SDK force-include has
 * already #defined printf as eff_uart_printf, causing a noexcept mismatch
 * when stdio.h tries to redeclare it. Suspend the macro for these includes. */
#pragma push_macro("printf")
#undef printf
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#pragma pop_macro("printf")

/* Embedded model data (generated from gesture_model.tflite by cmake) */
extern "C" {
    extern const uint8_t  gesture_model_tflite[];
    extern const unsigned int gesture_model_tflite_len;
}

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

/* TFLM tensor arena */
alignas(16)
static uint8_t s_tensor_arena[TFLM_ARENA_SIZE];

/* ── TFLM state ───────────────────────────────────────────────────── */

/* MobileNetV2 uses: Conv2D, DepthwiseConv2D, Add, Reshape, Mean
   (GlobalAveragePooling2D → reduce_mean), FullyConnected, Softmax,
   Quantize, Dequantize, Pad */
using Resolver = tflite::MicroMutableOpResolver<10>;

static Resolver              s_resolver;
static tflite::MicroInterpreter* s_interp = nullptr;
static bool s_tflm_ok = false;

/* We allocate the interpreter into a static buffer to avoid heap. */
static uint8_t s_interp_buf[sizeof(tflite::MicroInterpreter)];

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

/* decay_activity, apply_threshold, resize_activity_to_model_input are marked
 * __efficient__ so the fabric compiler maps their inner loops onto the CGRA.
 * On the scalar target __efficient__ expands to nothing. */

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

    /* We can have at most N/1 labels; use a static array.
       In practice the number of blobs is tiny. */
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

            /* Boundary pixel? */
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

static gesture_class_t run_tflm_inference(float* confidence)
{
    *confidence = 0.0f;
    if (!s_tflm_ok || s_interp == nullptr) return GESTURE_NONE;

    /* Copy resized frame into the input tensor.
     * Fully-quantised MobileNetV2 expects int8 in [-128, 127].
     * Our activity map is uint8 [0, 255]; subtract 128 to shift range. */
    TfLiteTensor* input = s_interp->input(0);
    for (int i = 0; i < TFLM_INPUT_W * TFLM_INPUT_H * TFLM_INPUT_CH; i++)
        input->data.int8[i] = (int8_t)((int)s_resize_buf[i] - 128);

    if (s_interp->Invoke() != kTfLiteOk) return GESTURE_NONE;

    /* Dequantise output using tensor quantization params → [0, 1] probability */
    const TfLiteTensor* output = s_interp->output(0);
    const float out_scale     = output->params.scale;
    const int   out_zero      = output->params.zero_point;

    int   best_idx = 0;
    float best_prob = (output->data.int8[0] - out_zero) * out_scale;
    for (int i = 1; i < TFLM_NUM_CLASSES; i++) {
        float prob = (output->data.int8[i] - out_zero) * out_scale;
        if (prob > best_prob) { best_prob = prob; best_idx = i; }
    }

    *confidence = best_prob;
    return kKerasToProto[best_idx];
}

/* ── Public API implementation ────────────────────────────────────── */

int gesture_kernel_init(void)
{
    memset(s_activity, 0, sizeof(s_activity));
    memset(s_binary,   0, sizeof(s_binary));
    memset(s_label_map, 0, sizeof(s_label_map));

    tflite::InitializeTarget();

    /* Register ops needed by MobileNetV2-0.5 int8 */
    s_resolver.AddConv2D();
    s_resolver.AddDepthwiseConv2D();
    s_resolver.AddAdd();
    s_resolver.AddReshape();
    s_resolver.AddMean();           /* GlobalAveragePooling2D */
    s_resolver.AddFullyConnected();
    s_resolver.AddSoftmax();
    s_resolver.AddQuantize();
    s_resolver.AddDequantize();
    s_resolver.AddPad();

    const tflite::Model* model = tflite::GetModel(gesture_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        return -1;
    }

    /* Construct interpreter in pre-allocated buffer (avoids heap). */
    s_interp = new (s_interp_buf) tflite::MicroInterpreter(
        model, s_resolver, s_tensor_arena, TFLM_ARENA_SIZE);

    if (s_interp->AllocateTensors() != kTfLiteOk) {
        s_interp = nullptr;
        return -1;
    }

    s_tflm_ok = true;
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
        result.gesture = run_tflm_inference(&result.confidence);

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
