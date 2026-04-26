// Copyright Efficient Computer Company 2026. All Rights Reserved.
//
// Gesture recognition firmware — E1 EVK main entry point.
//
// Hardware setup:
//   SPI_1        → DVS event stream from host PC (SPI master, PINMUX_1)
//   STDIO_UART   → Gesture result output via printf (configured by SDK init)
//
// Protocol: receives dvs_packet_header_t + dvs_event_t[] over SPI_1.

#include "gesture_kernel.h"
#include "protocol.h"

#include <cstdio>
#include <cstring>

#ifdef HW_BUILD
#include <eff/drivers/spi.h>
#include <eff/drivers/pinmux.h>
#include <eff/arch/e1x/mmio.h>
#include <eff/atc/atcspi200.h>
#endif

/* ── SPI_1 slave transport ────────────────────────────────────────── */

#ifdef HW_BUILD

static bool spi_init(void)
{
    eff_pinmux_set(PINMUX_1, PINMUX_SPI);

    // Init with the high-level driver first (resets the peripheral, sets mode/bus)
    eff_spi_cfg_t cfg = EFF_SPI_DEFAULTS;
    cfg.command_mode = 0;
    cfg.address_mode = 0;
    cfg.bus_size     = SPI_BUS_SINGLE;
    cfg.mode         = SPI_MODE_0;
    if (eff_spi_init(SPI_1, &cfg) != 0) return false;

    // eff_spi_cfg_t has no slave field — set TRANSFMT.SLVMODE (bit 2) directly.
    // The clock is driven by the laptop master; the EVK just receives.
    DEV_SPI1->TRANSFMT |= ATCSPI200_TRANSFMT_SLVMODE_MASK;
    return true;
}

static bool spi_read(uint8_t* buf, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++) {
        while (DEV_SPI1->STATUS & ATCSPI200_STATUS_RXEMPTY_MASK) { /* spin */ }
        buf[i] = (uint8_t)(DEV_SPI1->DATA & 0xFF);
    }
    return true;
}

#else  /* sim / native — stub */

static bool spi_init(void) { return true; }
static bool spi_read(uint8_t* buf, uint32_t len) { memset(buf, 0, len); return true; }

#endif  /* HW_BUILD */

/* ── Frame receive ────────────────────────────────────────────────── */

/* DVS_MAX_EVENTS in protocol.h is sized for the Pi (heap allocation).
 * Cap the EVK static receive buffer to fit SRAM. */
#define EVK_FRAME_EVENT_CAP 2048

static int receive_frame(dvs_packet_header_t* header_out,
                          dvs_event_t*         events_out)
{
    /* 1. Read header */
    uint8_t hdr_bytes[DVS_HEADER_SIZE];
    if (!spi_read(hdr_bytes, DVS_HEADER_SIZE)) return -1;

    if (hdr_bytes[0] != DVS_MAGIC_0 || hdr_bytes[1] != DVS_MAGIC_1)
        return -1;  /* desync — caller should retry */

    header_out->magic[0]    = hdr_bytes[0];
    header_out->magic[1]    = hdr_bytes[1];
    header_out->frame_id    = (uint32_t)hdr_bytes[2]
                            | ((uint32_t)hdr_bytes[3] << 8)
                            | ((uint32_t)hdr_bytes[4] << 16)
                            | ((uint32_t)hdr_bytes[5] << 24);
    header_out->event_count = (uint32_t)hdr_bytes[6]
                            | ((uint32_t)hdr_bytes[7] << 8)
                            | ((uint32_t)hdr_bytes[8] << 16)
                            | ((uint32_t)hdr_bytes[9] << 24);

    uint32_t n = header_out->event_count;
    if (n > EVK_FRAME_EVENT_CAP) n = EVK_FRAME_EVENT_CAP;

    /* 2. Read events */
    static uint8_t evt_bytes[EVK_FRAME_EVENT_CAP * DVS_EVENT_SIZE];
    if (!spi_read(evt_bytes, n * DVS_EVENT_SIZE)) return -1;

    for (uint32_t i = 0; i < n; i++) {
        const uint8_t* p = evt_bytes + i * DVS_EVENT_SIZE;
        events_out[i].x         = p[0];
        events_out[i].y         = p[1];
        events_out[i].polarity  = p[2];
        events_out[i].timestamp = (uint32_t)p[3]
                                | ((uint32_t)p[4] << 8)
                                | ((uint32_t)p[5] << 16)
                                | ((uint32_t)p[6] << 24);
    }

    return (int)n;
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(void)
{
#ifdef HW_BUILD
    if (!spi_init()) {
        printf("ERROR: SPI_1 init failed\n");
        return 1;
    }
#endif

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
            continue;   /* desync — retry */
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
