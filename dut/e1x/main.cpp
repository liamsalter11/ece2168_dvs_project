// Copyright Efficient Computer Company 2026. All Rights Reserved.
//
// Gesture recognition firmware — E1 EVK main entry point.
//
// Hardware setup:
//   SPI_2 / PINMUX_2 → DVS event stream from host PC via FT232H
//                       Connects through Arduino UNO header (level-shifted, 3.3V)
//   STDIO_UART        → Gesture result output via printf (configured by SDK init)
//
// Protocol: receives dvs_packet_header_t + dvs_event_t[] over SPI in
// DATA_ONLY slave mode (no command/dummy phase — raw bytes, CS-framed).

#include "gesture_kernel.h"
#include "protocol.h"

#include <cstdio>
#include <cstring>

#ifdef HW_BUILD
#include <eff/drivers/spi.h>
#include <eff/drivers/pinmux.h>
#endif

/* ── Frame receive ────────────────────────────────────────────────── */

/* Cap per-frame events to fit static SRAM buffer. */
#define EVK_FRAME_EVENT_CAP 2048

static uint8_t s_frame_buf[DVS_HEADER_SIZE + EVK_FRAME_EVENT_CAP * DVS_EVENT_SIZE];

static bool spi_init(void)
{
#ifndef HW_BUILD
    return true;
#else
    eff_pinmux_set(PINMUX_2, PINMUX_SPI);

    eff_spi_slave_cfg_t cfg = EFF_SPI_SLAVE_DEFAULTS;
    cfg.proto     = SPI_SLAVE_DATA_ONLY;   /* no command/dummy byte from master */
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
    /* Read the entire frame in one CS-framed transaction. CS deassert after
     * the frame acts as a natural boundary; SPIRST at the start of the next
     * call flushes any bytes the master sent beyond EVK_FRAME_EVENT_CAP. */
    if (eff_spi_slave_xfer(SPI_2, NULL, 0, s_frame_buf, sizeof(s_frame_buf)) < 0)
        return -1;

    if (s_frame_buf[0] != DVS_MAGIC_0 || s_frame_buf[1] != DVS_MAGIC_1)
        return -1;

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
        events_out[i].x         = p[0];
        events_out[i].y         = p[1];
        events_out[i].polarity  = p[2];
        events_out[i].timestamp = (uint32_t)p[3]
                                | ((uint32_t)p[4] << 8)
                                | ((uint32_t)p[5] << 16)
                                | ((uint32_t)p[6] << 24);
    }

    return (int)n;
#endif
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(void)
{
    if (!spi_init()) {
        printf("ERROR: SPI_2 init failed\n");
        return 1;
    }

    if (gesture_kernel_init() != 0) {
        printf("ERROR: gesture init failed\n");
        return 1;
    }

    printf("Gesture recognition ready (LiteRT)\n");

    static dvs_packet_header_t header;
    static dvs_event_t events[EVK_FRAME_EVENT_CAP];

    gesture_class_t last_gesture = GESTURE_NONE;
    int stable_count = 0;

    for (;;) {
        int n = receive_frame(&header, events);
        if (n < 0) {
            continue;
        }

        gesture_kernel_ingest(events, (uint32_t)n);
        gesture_result_t result = gesture_kernel_classify();

        if (result.gesture == last_gesture) {
            stable_count++;
        } else {
            stable_count = 0;
            last_gesture = result.gesture;
        }

        if (stable_count == 2) {
            printf("GESTURE %d %s %d%%\n",
                   (int)result.gesture,
                   gesture_names[result.gesture],
                   (int)(result.confidence * 100));
        }
    }

    return 0;
}
