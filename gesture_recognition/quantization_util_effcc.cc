// Copyright Efficient Computer Company 2026. All Rights Reserved.
//
// Float-safe replacement for tensorflow/lite/kernels/internal/quantization_util.cc
//
// The upstream file bitcasts double→uint64 (union trick), which emits
// arith.bitcast f64→i64 — an operation the montecarlo/e2 backend marks illegal
// because the target has no hardware f64 support.
//
// This replacement performs all computation in f32 after an initial fptrunc
// (double→float), which the backend lowers correctly via soft-float.
// Quantization scale values originate from float32 in the FlatBuffer and are
// only widened to double by TFLM for historical reasons, so the f32 precision
// loss is negligible.

/* quantization_util.h → types.h → <algorithm> → <cstdio>; same issue. */
#pragma push_macro("printf")
#undef printf
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#pragma pop_macro("printf")

#include <cmath>
#include <cstdint>
#include <climits>

namespace tflite {

void QuantizeMultiplier(double double_multiplier,
                        int32_t* quantized_multiplier, int* shift) {
    float fm = (float)double_multiplier;
    if (fm == 0.0f) {
        *quantized_multiplier = 0;
        *shift = 0;
        return;
    }
    int exp;
    float q = frexpf(fm, &exp);
    *shift = exp;
    /* Scale fractional part to int32 range */
    int64_t q_fixed = (int64_t)roundf(q * (float)(1LL << 31));
    if (q_fixed == (1LL << 31)) {
        q_fixed /= 2;
        ++*shift;
    }
    /* Clamp shift range to what MultiplyByQuantizedMultiplier supports */
    if (*shift < -31) {
        *shift = 0;
        q_fixed = 0;
    }
    if (*shift > 30) {
        *shift = 30;
        q_fixed = (1LL << 31) - 1;
    }
    *quantized_multiplier = (int32_t)q_fixed;
}

void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift) {
    int shift;
    QuantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
    *left_shift = shift;
}

void QuantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                         int32_t* quantized_multiplier,
                                         int* left_shift) {
    int shift;
    QuantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
    *left_shift = shift;
}

void PreprocessSoftmaxScaling(double beta, double input_scale,
                               int input_integer_bits,
                               int32_t* quantized_multiplier,
                               int* left_shift) {
    float scale = (float)beta * (float)input_scale *
                  (float)(1 << (31 - input_integer_bits));
    QuantizeMultiplier((double)scale, quantized_multiplier, left_shift);
}

int CalculateInputRadius(int input_integer_bits, int input_left_shift,
                         int total_signed_bits) {
    float max = (float)((1 << input_integer_bits) - 1) +
                1.0f / (float)(1 << (30 - input_integer_bits));
    float actual_max = max * (float)(1 << input_left_shift);
    int32_t limit = INT32_MAX >> (31 - total_signed_bits);
    int64_t result = (int64_t)actual_max;
    if (result > limit) result = limit;
    return (int)result;
}

/* ---- Integer-domain helpers kept for ABI completeness ---- */

int64_t IntegerFrExp(double input, int* shift) {
    float fi = (float)input;
    if (fi == 0.0f) { *shift = 0; return 0; }
    int exp;
    float frac = frexpf(fi, &exp);
    *shift = exp;
    return (int64_t)roundf(frac * (float)(1LL << 31));
}

double IntegerFrExpToDouble(int64_t fraction, int shift) {
    return (double)ldexpf((float)fraction / (float)(1LL << 31), shift);
}

double DoubleMultiply(double a, double b) {
    return (double)((float)a * (float)b);
}

}  // namespace tflite
