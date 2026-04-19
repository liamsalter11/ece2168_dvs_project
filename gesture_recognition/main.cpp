// Copyright Efficient Computer Company 2026. All Rights Reserved.
//
// Gesture recognition firmware — E1 EVK main entry point.
//
// Hardware setup:
//   UART_2  → DVS event stream from host PC
//   UART_3  → Debug output (gesture classification results, via stdio)
//
// Protocol: receives dvs_packet_header_t + dvs_event_t[] over UART_2.

#include "gesture_kernel.h"
#include "protocol.h"

#include <cstring>

#ifdef HW_BUILD
#include <eff/drivers/uart.h>
#endif

/* ── UART_2 transport ─────────────────────────────────────────────── */

#ifdef HW_BUILD

static bool uart2_init(void)
{
    eff_uart_cfg_t cfg = EFF_UART_DEFAULTS;
    cfg.baud = 921600;
    return eff_uart_init(UART_2, cfg) == 0;
}

static bool uart2_read(uint8_t* buf, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++) {
        while (eff_uart_rx_empty(UART_2)) { /* spin-wait */ }
        char c;
        if (eff_uart_getc(UART_2, &c) != 0) return false;
        buf[i] = (uint8_t)c;
    }
    return true;
}

#else  /* sim / native — stub */

static bool uart2_init(void) { return true; }
static bool uart2_read(uint8_t* buf, uint32_t len) { memset(buf, 0, len); return true; }

#endif  /* HW_BUILD */

/* ── Frame receive ────────────────────────────────────────────────── */

static int receive_frame(dvs_packet_header_t* header_out,
                          dvs_event_t*         events_out)
{
    /* 1. Read header */
    uint8_t hdr_bytes[DVS_HEADER_SIZE];
    if (!uart2_read(hdr_bytes, DVS_HEADER_SIZE)) return -1;

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
    if (n > DVS_MAX_EVENTS) n = DVS_MAX_EVENTS;

    /* 2. Read events */
    static uint8_t evt_bytes[DVS_MAX_EVENTS * DVS_EVENT_SIZE];
    if (!uart2_read(evt_bytes, n * DVS_EVENT_SIZE)) return -1;

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
    if (!uart2_init()) {
        eff_uart_printf("ERROR: UART_2 init failed\n");
        return 1;
    }
#endif

    if (gesture_kernel_init() != 0) {
        printf("ERROR: TFLM init failed\n");
        return 1;
    }

    printf("Gesture recognition ready (TFLite Micro)\n");

    static dvs_packet_header_t header;
    static dvs_event_t events[DVS_MAX_EVENTS];

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
            printf("GESTURE %d %s %.0f%%\n",
                   (int)result.gesture,
                   gesture_names[result.gesture],
                   result.confidence * 100.0f);
        }
    }

    return 0;
}
