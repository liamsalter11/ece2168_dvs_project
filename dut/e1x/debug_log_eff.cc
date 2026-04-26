// Copyright Efficient Computer Company 2026. All Rights Reserved.
//
// TFLM DebugLog / DebugVsnprintf stubs for the effcc target.
// Route TFLM internal log output through eff_uart_printf.

#pragma push_macro("printf")
#undef printf
#include "tensorflow/lite/micro/micro_log.h"
#pragma pop_macro("printf")

/* pull vsnprintf/va_list without triggering the printf macro conflict */
#pragma push_macro("printf")
#undef printf
#include <cstdarg>
#include <cstdio>
#pragma pop_macro("printf")

#include <eff/uartprintf.h>

extern "C" void DebugLog(const char* s)
{
    eff_uart_printf("%s", s);
}

extern "C" int DebugVsnprintf(char* buf, size_t n, const char* fmt, va_list args)
{
    return vsnprintf(buf, n, fmt, args);
}
