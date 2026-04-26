// Gesture inference back-end for Raspberry Pi 4.
//
// Implements the same extern "C" symbol that eff-import provides on the E1x EVK,
// so dut/common/gesture_kernel.cpp is compiled identically on both targets.
//
// Call tflite_backend_init(model_path) once before gesture_kernel_init().

#include "../common/gesture_kernel.h"   // for TFLM_* constants

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

static std::unique_ptr<tflite::FlatBufferModel> s_model;
static std::unique_ptr<tflite::Interpreter>     s_interp;

extern "C" int tflite_backend_init(const char* model_path)
{
    s_model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!s_model) {
        fprintf(stderr, "[tflite] Failed to load model: %s\n", model_path);
        return -1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*s_model, resolver);
    builder(&s_interp);
    if (!s_interp) {
        fprintf(stderr, "[tflite] Failed to build interpreter\n");
        return -1;
    }

    if (s_interp->AllocateTensors() != kTfLiteOk) {
        fprintf(stderr, "[tflite] AllocateTensors() failed\n");
        return -1;
    }

    TfLiteTensor* in  = s_interp->input_tensor(0);
    TfLiteTensor* out = s_interp->output_tensor(0);
    fprintf(stderr, "[tflite] Loaded %s — input [%d,%d,%d,%d] type=%d\n",
            model_path,
            in->dims->data[0], in->dims->data[1],
            in->dims->data[2], in->dims->data[3],
            (int)in->type);
    fprintf(stderr, "[tflite] Output [%d] type=%d\n",
            out->dims->data[1], (int)out->type);

    return 0;
}

// Matches the eff-import convention used in dut/common/gesture_kernel.cpp:
//   input:  int8_t[TFLM_INPUT_W * TFLM_INPUT_H * TFLM_INPUT_CH]  (zp = -128)
//   output: int8_t[TFLM_NUM_CLASSES]                              (zp = -128)
//   return: 0 on success
//
// The TFLite model uses uint8 tensors (kTfLiteUInt8, type=3).
// The kernel pre-shifts pixels by -128 (uint8 → int8) before calling here,
// so we reverse that shift when writing to the uint8 input tensor, and apply
// it when reading from the uint8 output tensor so the caller sees int8/zp=-128.
extern "C" int32_t gesture_run_inference(void* input, void* output)
{
    if (!s_interp) return -1;

    uint8_t* tensor_in = s_interp->typed_input_tensor<uint8_t>(0);
    if (!tensor_in) return -1;

    const int8_t* src = static_cast<const int8_t*>(input);
    const int n_in = TFLM_INPUT_W * TFLM_INPUT_H * TFLM_INPUT_CH;
    for (int i = 0; i < n_in; i++)
        tensor_in[i] = (uint8_t)((int)src[i] + 128);

    if (s_interp->Invoke() != kTfLiteOk) return -1;

    uint8_t* tensor_out = s_interp->typed_output_tensor<uint8_t>(0);
    if (!tensor_out) return -1;

    int8_t* dst = static_cast<int8_t*>(output);
    for (int i = 0; i < TFLM_NUM_CLASSES; i++)
        dst[i] = (int8_t)((int)tensor_out[i] - 128);

    return 0;
}
