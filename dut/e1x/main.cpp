// Copyright Efficient Computer Company 2026. All Rights Reserved.
//
// Gesture recognition firmware — E1 EVK main entry point.
//
// Hardware setup:
//   SPI_2 / PINMUX_2 → DVS event stream from host PC via FT232H
//                       Connects through Arduino UNO header (level-shifted, 3.3V)
//   STDIO_UART        → Gesture result output (configured by SDK init)
//
// Inference back-end: TFLite Micro (scalar RISC-V).
// CGRA fabric is not present on this EVK unit (FABRIC_PRESENT=0).

#include "gesture_kernel.h"
#include "protocol.h"

#include <cstdio>
#include <cstring>

#ifdef HW_BUILD
#include <eff/drivers/spi.h>
#include <eff/drivers/pinmux.h>
#include <eff/drivers/uart.h>
#include <eff/time.h>
#endif

/* Provided by tflite_backend_eff.cpp */
extern "C" int gesture_tflm_init(void);

/* Override the weak no-op stub in gesture_kernel.cpp */
extern "C" void gesture_dbg_probe(int point)
{
#ifdef HW_BUILD
    static char _pb[24];
    sprintf(_pb, "probe:%d\r\n", point);
    eff_uart_puts(STDIO_UART, _pb);
#else
    (void)point;
#endif
}

/* ── Frame receive ────────────────────────────────────────────────── */

#define EVK_FRAME_EVENT_CAP 2048

static uint8_t s_frame_buf[DVS_HEADER_SIZE + EVK_FRAME_EVENT_CAP * DVS_EVENT_SIZE];

static bool spi_init(void)
{
#ifndef HW_BUILD
    return true;
#else
    eff_pinmux_set(PINMUX_2, PINMUX_SPI);

    eff_spi_slave_cfg_t cfg = EFF_SPI_SLAVE_DEFAULTS;
    cfg.proto     = SPI_SLAVE_DATA_ONLY;
    cfg.xfer_mode = SPI_XFER_READ_ONLY;
    cfg.bus_size  = SPI_BUS_SINGLE;
    cfg.mode      = SPI_MODE_0;

    return eff_spi_slave_init(SPI_2, &cfg) == 0;
#endif
}

static int receive_frame(dvs_packet_header_t* header_out,
                          dvs_event_t*         events_out)
{
#ifndef HW_BUILD
    return -1;
#else
    static char _dbg[32];
    sprintf(_dbg, "xfer...\r\n");
    eff_uart_puts(STDIO_UART, _dbg);

    int8_t rc = eff_spi_slave_xfer(SPI_2, NULL, 0, s_frame_buf, sizeof(s_frame_buf));

    sprintf(_dbg, "rc=%d [%02X %02X]\r\n", (int)rc, s_frame_buf[0], s_frame_buf[1]);
    eff_uart_puts(STDIO_UART, _dbg);

    if (rc == -2)
        return -1;

    if (s_frame_buf[0] != DVS_MAGIC_0 || s_frame_buf[1] != DVS_MAGIC_1)
        return -2;

    header_out->magic[0]    = s_frame_buf[0];
    header_out->magic[1]    = s_frame_buf[1];
    header_out->frame_id    = (uint32_t)s_frame_buf[2]
                            | ((uint32_t)s_frame_buf[3] << 8)
                            | ((uint32_t)s_frame_buf[4] << 16)
                            | ((uint32_t)s_frame_buf[5] << 24);
    header_out->event_count = (uint32_t)s_frame_buf[6]
                            | ((uint32_t)s_frame_buf[7] << 8)
                            | ((uint32_t)s_frame_buf[8] << 16)
                            | ((uint32_t)s_frame_buf[9] << 24);

    uint32_t n = header_out->event_count;
    if (n > EVK_FRAME_EVENT_CAP) n = EVK_FRAME_EVENT_CAP;

    const uint8_t* ep = s_frame_buf + DVS_HEADER_SIZE;
    for (uint32_t i = 0; i < n; i++) {
        const uint8_t* p = ep + i * DVS_EVENT_SIZE;
        events_out[i].x        = p[0];
        events_out[i].y        = p[1];
        events_out[i].polarity = p[2];
        /* timestamp skipped — gesture_kernel_ingest never reads it, and writing
         * a uint32_t at offset 7i+3 (packed struct) traps on RISC-V. */
    }

    return (int)n;
#endif
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(void)
{
#ifdef HW_BUILD
    eff_uart_cfg_t uart_cfg = EFF_UART_DEFAULTS;
    uart_cfg.baud = 108000;
    eff_uart_init(STDIO_UART, uart_cfg);
    sleep_ms(10);
#endif

    if (!spi_init()) {
        eff_uart_puts(STDIO_UART, "SPI init failed\r\n");
        return 1;
    }

    if (gesture_tflm_init() != 0) {
        eff_uart_puts(STDIO_UART, "TFLM init failed\r\n");
        return 1;
    }

    if (gesture_kernel_init() != 0) {
        eff_uart_puts(STDIO_UART, "Kernel init failed\r\n");
        return 1;
    }

    eff_uart_puts(STDIO_UART, "Gesture recognition ready (TFLite Micro)\r\n");

    static dvs_packet_header_t header;
    static dvs_event_t events[EVK_FRAME_EVENT_CAP];
    static char uart_buf[48];

    gesture_class_t last_gesture = GESTURE_NONE;
    int stable_count = 0;

    for (;;) {
        int n = receive_frame(&header, events);
        if (n == -2) {
            sprintf(uart_buf, "BAD_MAGIC %02X %02X\r\n",
                    s_frame_buf[0], s_frame_buf[1]);
            eff_uart_puts(STDIO_UART, uart_buf);
            continue;
        }
        if (n < 0) continue;

        sprintf(uart_buf, "RECV n=%d\r\n", n);
        eff_uart_puts(STDIO_UART, uart_buf);

        gesture_kernel_ingest(events, (uint32_t)n);
        gesture_result_t result = gesture_kernel_classify();

        sprintf(uart_buf, "n=%d b=%d c=%d\r\n",
                n, result.features.area, (int)(result.confidence * 100));
        eff_uart_puts(STDIO_UART, uart_buf);

        if (result.gesture == last_gesture) {
            stable_count++;
        } else {
            stable_count = 0;
            last_gesture = result.gesture;
        }

        if (stable_count == 2 && result.gesture != GESTURE_NONE) {
            sprintf(uart_buf, "GESTURE %d %s %d%%\r\n",
                    (int)result.gesture,
                    gesture_names[result.gesture],
                    (int)(result.confidence * 100));
            eff_uart_puts(STDIO_UART, uart_buf);
        }
    }

    return 0;
}
