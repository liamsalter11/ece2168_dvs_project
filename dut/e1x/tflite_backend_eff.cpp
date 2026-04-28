// TFLite Micro inference backend for E1x EVK (scalar RISC-V, no CGRA).
//
// Implements the same extern "C" interface that eff-import would provide,
// so dut/common/gesture_kernel.cpp is compiled identically on both targets.
//
// Call gesture_tflm_init() once from main() before the first classify call.

#include "../common/gesture_kernel.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cstring>

/* Byte array + length from embed_model.py / gesture_model_data.cc */
extern const uint8_t      gesture_model_tflite[];
extern const unsigned int gesture_model_tflite_len;

static uint8_t s_arena[TFLM_ARENA_SIZE] __attribute__((aligned(4)));

static tflite::MicroMutableOpResolver<7> s_resolver;
static tflite::MicroInterpreter*         s_interp = nullptr;

extern "C" int gesture_tflm_init(void)
{
    s_resolver.AddQuantize();
    s_resolver.AddConv2D();
    s_resolver.AddDepthwiseConv2D();
    s_resolver.AddAdd();
    s_resolver.AddMean();
    s_resolver.AddFullyConnected();
    s_resolver.AddSoftmax();

    const tflite::Model* model = tflite::GetModel(gesture_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION)
        return -1;

    static tflite::MicroInterpreter interp(
        model, s_resolver, s_arena, TFLM_ARENA_SIZE);

    if (interp.AllocateTensors() != kTfLiteOk)
        return -1;

    s_interp = &interp;
    return 0;
}

// Input:  int8_t[TFLM_INPUT_W * TFLM_INPUT_H * TFLM_INPUT_CH], zp = -128
//         gesture_kernel passes (uint8_pixel - 128) so copy directly.
// Output: int8_t[TFLM_NUM_CLASSES], zp = -128
// Return: 0 on success.

#ifdef HW_BUILD
#include <eff/drivers/uart.h>
#endif

static inline void _inf_log(const char* s)
{
#ifdef HW_BUILD
    eff_uart_puts(STDIO_UART, s);
#else
    (void)s;
#endif
}

extern "C" int32_t gesture_run_inference(void* input, void* output)
{
    if (!s_interp) { _inf_log("inf:no_interp\r\n"); return -1; }

    _inf_log("inf:copy_in\r\n");
    const int n_in = TFLM_INPUT_W * TFLM_INPUT_H * TFLM_INPUT_CH;
    memcpy(s_interp->input(0)->data.int8, input, n_in);

    _inf_log("inf:invoke\r\n");
    if (s_interp->Invoke() != kTfLiteOk) { _inf_log("inf:invoke_fail\r\n"); return -1; }

    _inf_log("inf:copy_out\r\n");
    memcpy(output, s_interp->output(0)->data.int8, TFLM_NUM_CLASSES);
    _inf_log("inf:done\r\n");
    return 0;
}
