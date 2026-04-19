// Copyright Efficient Computer Company 2026. All Rights Reserved.
//
// Replaces several upstream TFLM files that include <complex> (which requires
// atan2l, unavailable with the effcc soft-float target). Provides the symbols
// referenced by MobileNetV2 inference:
//   • kernel_util.cc  → convolution quantization helpers, HaveSameShapes
//   • schema_utils.cc → GetBuiltinCode
//   • error_reporter.cc → ErrorReporter::Report

#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"

#include <stdint.h>
#include <cstdarg>
#include <algorithm>
#include <cmath>
#include <limits>

#include "tensorflow/lite/kernels/internal/quantization_util.h"

namespace tflite {

bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2) {
    return TfLiteIntArrayEqual(input1->dims, input2->dims);
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              TfLiteTensor* output,
                                              double* multiplier) {
    const double input_product_scale =
        static_cast<double>(input->params.scale * filter->params.scale);
    TF_LITE_ENSURE(context, input_product_scale >= 0);
    *multiplier = input_product_scale /
                  static_cast<double>(output->params.scale);
    return kTfLiteOk;
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier) {
    if (bias) {
        const double input_product_scale =
            static_cast<double>(input->params.scale) *
            static_cast<double>(filter->params.scale);
        const double bias_scale  = static_cast<double>(bias->params.scale);
        const double scale_diff  = std::abs(input_product_scale - bias_scale);
        const double output_scale = static_cast<double>(output->params.scale);
        TF_LITE_ENSURE(context, scale_diff / output_scale <= 0.02);
    }
    return GetQuantizedConvolutionMultipler(context, input, filter, output,
                                            multiplier);
}

namespace {
inline TfLiteStatus QuantizeActivation(TfLiteContext* context, float scale,
                                       int32_t zero_point, float f,
                                       int32_t& q) {
    const float tmp = std::round(f / scale);
    const bool ok = (tmp >= static_cast<float>(
                               std::numeric_limits<int32_t>::min()) &&
                     tmp <= static_cast<float>(
                               std::numeric_limits<int32_t>::max()));
    TF_LITE_ENSURE(context, ok);
    q = zero_point + static_cast<int32_t>(tmp);
    return kTfLiteOk;
}

TfLiteStatus CalcActRangeImpl(TfLiteContext* context,
                               TfLiteFusedActivation activation,
                               int32_t qmin, int32_t qmax,
                               TfLiteTensor* output,
                               int32_t* act_min, int32_t* act_max) {
    const float scale  = output->params.scale;
    const int32_t zp   = output->params.zero_point;
    int32_t tmp_q;
    if (activation == kTfLiteActRelu) {
        TF_LITE_ENSURE_OK(context, QuantizeActivation(context, scale, zp, 0.f, tmp_q));
        *act_min = std::max(qmin, tmp_q);
        *act_max = qmax;
    } else if (activation == kTfLiteActRelu6) {
        TF_LITE_ENSURE_OK(context, QuantizeActivation(context, scale, zp, 0.f, tmp_q));
        *act_min = std::max(qmin, tmp_q);
        TF_LITE_ENSURE_OK(context, QuantizeActivation(context, scale, zp, 6.f, tmp_q));
        *act_max = std::min(qmax, tmp_q);
    } else if (activation == kTfLiteActReluN1To1) {
        TF_LITE_ENSURE_OK(context, QuantizeActivation(context, scale, zp, -1.f, tmp_q));
        *act_min = std::max(qmin, tmp_q);
        TF_LITE_ENSURE_OK(context, QuantizeActivation(context, scale, zp, 1.f, tmp_q));
        *act_max = std::min(qmax, tmp_q);
    } else {
        *act_min = qmin;
        *act_max = qmax;
    }
    return kTfLiteOk;
}
}  // namespace

TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                               TfLiteFusedActivation activation,
                                               TfLiteTensor* output,
                                               int32_t* act_min,
                                               int32_t* act_max) {
    int32_t qmin = 0, qmax = 0;
    if (output->type == kTfLiteUInt8) {
        qmin = std::numeric_limits<uint8_t>::min();
        qmax = std::numeric_limits<uint8_t>::max();
    } else if (output->type == kTfLiteInt8) {
        qmin = std::numeric_limits<int8_t>::min();
        qmax = std::numeric_limits<int8_t>::max();
    } else if (output->type == kTfLiteInt16) {
        qmin = std::numeric_limits<int16_t>::min();
        qmax = std::numeric_limits<int16_t>::max();
    } else {
        TF_LITE_ENSURE(context, false);
    }
    return CalcActRangeImpl(context, activation, qmin, qmax, output,
                            act_min, act_max);
}

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias,
    TfLiteTensor* output, const TfLiteFusedActivation& activation,
    int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift) {
    const auto* aq = reinterpret_cast<const TfLiteAffineQuantization*>(
        filter->quantization.params);
    return PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, activation, multiplier, shift,
        output_activation_min, output_activation_max,
        per_channel_multiplier, per_channel_shift, aq->scale->size);
}

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias,
    TfLiteTensor* output, const TfLiteFusedActivation& activation,
    int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int32_t* per_channel_shift,
    int num_channels) {
    TF_LITE_ENSURE_EQ(context, input->quantization.type, kTfLiteAffineQuantization);
    TF_LITE_ENSURE_EQ(context, filter->quantization.type, kTfLiteAffineQuantization);

    const auto* aq = reinterpret_cast<const TfLiteAffineQuantization*>(
        filter->quantization.params);
    TF_LITE_ENSURE(context, aq && aq->scale);
    const bool per_channel = aq->scale->size > 1;
    if (per_channel) {
        TF_LITE_ENSURE(context,
            input->type == kTfLiteInt8 || input->type == kTfLiteInt16);
        TF_LITE_ENSURE(context,
            filter->type == kTfLiteInt8 || filter->type == kTfLiteInt4);
        TF_LITE_ENSURE_EQ(context, aq->scale->size, num_channels);
        TF_LITE_ENSURE_EQ(context, num_channels,
            filter->dims->data[aq->quantized_dimension]);
    }

    const float input_scale  = input->params.scale;
    const float output_scale = output->params.scale;
    for (int i = 0; i < num_channels; ++i) {
        const float fscale = per_channel ? aq->scale->data[i] : aq->scale->data[0];
        const double eff_scale = static_cast<double>(input_scale) *
                                  static_cast<double>(fscale) /
                                  static_cast<double>(output_scale);
        int32_t sig; int ch_shift;
        QuantizeMultiplier(eff_scale, &sig, &ch_shift);
        per_channel_multiplier[i] = sig;
        per_channel_shift[i]      = ch_shift;
    }

    if (input->type == kTfLiteUInt8) {
        double real_multiplier = 0.0;
        TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
            context, input, filter, bias, output, &real_multiplier));
        int exponent;
        QuantizeMultiplier(real_multiplier, multiplier, &exponent);
        *shift = -exponent;
    }
    if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8 ||
        input->type == kTfLiteInt16) {
        TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
            context, activation, output,
            output_activation_min, output_activation_max));
    }
    return kTfLiteOk;
}

}  // namespace tflite

/* ── GetBuiltinCode (replaces tensorflow/compiler/mlir/lite/schema/schema_utils.cc) ── */

namespace tflite {

BuiltinOperator GetBuiltinCode(const OperatorCode* op_code) {
    return std::max(
        op_code->builtin_code(),
        static_cast<BuiltinOperator>(op_code->deprecated_builtin_code()));
}

/* ── ErrorReporter::Report (replaces tensorflow/compiler/mlir/lite/core/api/error_reporter.cc) ── */

int ErrorReporter::Report(const char* format, ...) {
    va_list args;
    va_start(args, format);
    int code = Report(format, args);
    va_end(args);
    return code;
}

int ErrorReporter::ReportError(void*, const char* format, ...) {
    va_list args;
    va_start(args, format);
    int code = Report(format, args);
    va_end(args);
    return code;
}

}  // namespace tflite
