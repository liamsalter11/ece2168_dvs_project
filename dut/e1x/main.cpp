// Copyright Efficient Computer Company 2026. All Rights Reserved.
//
// Gesture recognition firmware — E1x EVK main entry point.
//
// Hardware setup:
//   SPI_2 / PINMUX_2 → DVS event stream from host PC via FT232H
//                      (SPI slave, DATA_ONLY proto, mode 0, single-bit, 10 MHz)
//   STDIO_UART       → Gesture result output (re-init at 108000 baud to match
//                      the verified spi_test config)
//
// Inference path: gesture_model.tflite is compiled by eff-import to MLIR at
// build time and linked in as gesture_run_inference(). Runs on the CGRA fabric.

#include "gesture_kernel.h"
#include "protocol.h"

#include <cstdio>
#include <cstring>

#ifdef HW_BUILD
#include <eff.h>
#include <eff/drivers/spi.h>
#include <eff/drivers/pinmux.h>
#include <eff/drivers/uart.h>
#include <eff/time.h>
#endif

/* ── Frame receive ────────────────────────────────────────────────── */

#define EVK_FRAME_EVENT_CAP 2048

static uint8_t s_frame_buf[DVS_HEADER_SIZE + EVK_FRAME_EVENT_CAP * DVS_EVENT_SIZE];

#ifdef HW_BUILD
static void uart_log(const char* s) {
    eff_uart_puts(STDIO_UART, s);
}
#else
static void uart_log(const char* s) { (void)s; }
#endif

static bool spi_init(void) {
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

/* Returns event count on success, -1 on transfer failure, -2 on bad magic. */
static int receive_frame(dvs_packet_header_t* header_out,
                         dvs_event_t*         events_out) {
#ifndef HW_BUILD
    return -1;
#else
    int8_t rc = eff_spi_slave_xfer(SPI_2, NULL, 0,
                                   s_frame_buf, sizeof(s_frame_buf));
    if (rc == -2) {
        /* RX overrun — host sent more than EVK_FRAME_EVENT_CAP. Drop frame. */
        return -1;
    }

    if (s_frame_buf[0] != DVS_MAGIC_0 || s_frame_buf[1] != DVS_MAGIC_1) {
        return -2;
    }

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

    /* Parse events. Skip timestamp: gesture_kernel_ingest never reads it,
     * and a uint32_t write at offset 7i+3 of a packed 7-byte struct is
     * unaligned and traps the RISC-V scalar core. */
    const uint8_t* ep = s_frame_buf + DVS_HEADER_SIZE;
    for (uint32_t i = 0; i < n; i++) {
        const uint8_t* p = ep + i * DVS_EVENT_SIZE;
        events_out[i].x        = p[0];
        events_out[i].y        = p[1];
        events_out[i].polarity = p[2];
    }

    return (int)n;
#endif
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(void) {
#ifdef HW_BUILD
    /* Re-init STDIO UART at the rate the FT232H side is verified to use.
     * (The SDK auto-initializes at 115200 in a constructor; this overrides.) */
    eff_uart_cfg_t uart_cfg = EFF_UART_DEFAULTS;
    uart_cfg.baud = 108000;
    eff_uart_init(STDIO_UART, uart_cfg);
    sleep_ms(10);
#endif

    uart_log("Gesture firmware boot\r\n");

    if (!spi_init()) {
        uart_log("ERR: SPI_2 slave init failed\r\n");
        return 1;
    }

    if (gesture_kernel_init() != 0) {
        uart_log("ERR: gesture kernel init failed\r\n");
        return 1;
    }

    uart_log("Gesture recognition ready (fabric)\r\n");

    static dvs_packet_header_t header;
    static dvs_event_t         events[EVK_FRAME_EVENT_CAP];
    static char                line[64];

    gesture_class_t last_gesture = GESTURE_NONE;
    int stable_count = 0;
    uint32_t frame_count = 0;

    for (;;) {
        int n = receive_frame(&header, events);

        if (n == -2) {
            sprintf(line, "BAD_MAGIC %02X %02X\r\n",
                    s_frame_buf[0], s_frame_buf[1]);
            uart_log(line);
            continue;
        }
        if (n < 0) {
            uart_log("RX_OVERRUN\r\n");
            continue;
        }

        gesture_kernel_ingest(events, (uint32_t)n);
        gesture_result_t result = gesture_kernel_classify();

        /* Heartbeat every 30 frames so we can confirm the loop is alive. */
        if ((++frame_count % 30) == 0) {
            sprintf(line, "frame=%u n=%d area=%d conf=%d\r\n",
                    (unsigned)frame_count, n,
                    result.features.area,
                    (int)(result.confidence * 100));
            uart_log(line);
        }

        if (result.gesture == last_gesture) {
            stable_count++;
        } else {
            stable_count = 0;
            last_gesture = result.gesture;
        }

        if (stable_count == 2 && result.gesture != GESTURE_NONE) {
            sprintf(line, "GESTURE %d %s %d%%\r\n",
                    (int)result.gesture,
                    gesture_names[result.gesture],
                    (int)(result.confidence * 100));
            uart_log(line);
        }
    }

    return 0;
}
