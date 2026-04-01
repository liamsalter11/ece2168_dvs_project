/**
 * gesture_kernel.h — Hand gesture recognition kernel
 *
 * Classical computer vision pipeline operating on DVS event streams:
 *   1. Accumulate events into an activity heatmap with temporal decay
 *   2. Threshold to binary hand region
 *   3. Connected component labeling to find dominant blob
 *   4. Shape analysis: bounding box, fill ratio, radial profile
 *   5. Convex hull + defect counting for finger estimation
 *   6. Classify gesture from features
 *
 * Designed to be portable to embedded / FPGA HLS targets.
 */

#ifndef GESTURE_KERNEL_H
#define GESTURE_KERNEL_H

#include "protocol.h"
#include <cstdint>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include <memory>

/* ── Tunable parameters ──────────────────────────────────────────── */

#define ACTIVITY_DECAY_FACTOR   0.92f    /* per-frame exponential decay       */
#define ACTIVITY_THRESHOLD      40.0f    /* binarization threshold             */
#define ACTIVITY_INCREMENT      80.0f    /* added per ON event                 */
#define ACTIVITY_DECREMENT      40.0f    /* subtracted per OFF event           */

#define MIN_BLOB_AREA           50       /* ignore blobs smaller than this     */

/* ── Data structures ─────────────────────────────────────────────── */

struct Point {
    int x, y;
};

struct BlobFeatures {
    int area;                  /* pixel count                         */
    int bbox_x, bbox_y;       /* bounding box top-left               */
    int bbox_w, bbox_h;       /* bounding box size                   */
    float fill_ratio;          /* area / bbox_area                    */
    float aspect_ratio;        /* bbox_w / bbox_h                     */
    Point centroid;            /* center of mass                      */
    float compactness;         /* perimeter² / area                   */
};

struct GestureResult {
    gesture_class_t gesture;
    float confidence;          /* 0.0 – 1.0                           */
    BlobFeatures features;
};

/* ── Kernel class ────────────────────────────────────────────────── */

class GestureKernel {
public:
    GestureKernel(int width, int height, const char* tflite_model_path = nullptr);
    ~GestureKernel();

    /**
     * Ingest a batch of DVS events and update the activity map.
     * Call once per received frame.
     */
    void ingest_events(const dvs_event_t* events, uint32_t count);

    /**
     * Run the full gesture recognition pipeline on the current
     * activity map. Returns classification result.
     */
    GestureResult classify();

    /** Direct access to the activity map (for visualization). */
    const float* activity_map() const { return activity_; }

    /** Direct access to the binary mask (for visualization). */
    const uint8_t* binary_mask() const { return binary_; }

private:
    int width_, height_;
    float* activity_;          /* floating-point activity heatmap     */
    uint8_t* binary_;          /* thresholded binary mask             */
    int* label_map_;           /* connected component labels          */

    /* Pipeline stages */
    void decay_activity();
    void apply_threshold();
    int  connected_components();   /* returns number of labels         */
    int  find_largest_blob(int num_labels);
    void extract_blob_features(int label, BlobFeatures& feat);

    uint8_t* resize_buf_;
    std::unique_ptr<tflite::FlatBufferModel> tflite_model_;
    std::unique_ptr<tflite::Interpreter>     tflite_interp_;

    bool            load_tflite_model(const char* path);
    gesture_class_t run_tflite_inference(float& confidence);
    void            resize_activity_to_model_input();
};

#endif /* GESTURE_KERNEL_H */
