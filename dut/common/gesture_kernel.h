// Copyright Efficient Computer Company 2026. All Rights Reserved.
//
// Gesture recognition kernel for DVS event streams.
//
// Pipeline:
//   1. Accumulate events into activity heatmap with temporal decay
//   2. Threshold to binary mask
//   3. BFS connected-component labeling
//   4. Extract blob shape features
//   5. TFLite Micro inference on 128x128 resize of activity map

#pragma once

#include "protocol.h"
#include <stdint.h>

/* ── Tunable parameters ───────────────────────────────────────────── */

#define ACTIVITY_DECAY_FACTOR   0.92f
#define ACTIVITY_THRESHOLD      40.0f
#define ACTIVITY_INCREMENT      80.0f
#define ACTIVITY_DECREMENT      40.0f
#define MIN_BLOB_AREA           50

#define TFLM_INPUT_W     128
#define TFLM_INPUT_H     128
#define TFLM_INPUT_CH    3
#define TFLM_NUM_CLASSES 5

/* Tensor arena: ~1 MB — sized for MobileNetV2-0.5 at 128x128 */
#define TFLM_ARENA_SIZE  (1024 * 1024)

/* ── Data structures ──────────────────────────────────────────────── */

typedef struct {
    int area;
    int bbox_x, bbox_y;
    int bbox_w, bbox_h;
    float fill_ratio;
    float aspect_ratio;
    int centroid_x, centroid_y;
    float compactness;
} blob_features_t;

typedef struct {
    gesture_class_t gesture;
    float confidence;
    blob_features_t features;
} gesture_result_t;

/* ── Public API ───────────────────────────────────────────────────── */

#ifdef __cplusplus
extern "C" {
#endif

/* Call once at startup. Returns 0 on success, -1 if TFLM init fails. */
int gesture_kernel_init(void);

/* Ingest a batch of DVS events into the activity map. */
void gesture_kernel_ingest(const dvs_event_t* events, uint32_t count);

/* Run full pipeline and return gesture result. */
gesture_result_t gesture_kernel_classify(void);

/* Access raw activity map (for debug). Length = DVS_FRAME_WIDTH * DVS_FRAME_HEIGHT */
const float* gesture_kernel_activity_map(void);

#ifdef __cplusplus
}
#endif
