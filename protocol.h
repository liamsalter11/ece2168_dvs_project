/**
 * protocol.h — Shared DVS event protocol definition
 *
 * Wire format (little-endian):
 *
 *   Frame Packet:
 *   ┌──────────────────────────────────────────────┐
 *   │ Magic        (2 bytes)  0xAE 0xDV            │
 *   │ Frame ID     (4 bytes)  uint32               │
 *   │ Event Count  (4 bytes)  uint32               │
 *   │ Events[]     (N × 7 bytes)                   │
 *   │   ├─ x          (1 byte)  uint8              │
 *   │   ├─ y          (1 byte)  uint8              │
 *   │   ├─ polarity   (1 byte)  uint8  0=OFF 1=ON  │
 *   │   └─ timestamp  (4 bytes) uint32 microseconds │
 *   └──────────────────────────────────────────────┘
 *
 *   Total packet size = 10 + (event_count × 7) bytes
 */

#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

#define DVS_MAGIC_0       0xAE
#define DVS_MAGIC_1       0xD7   /* 0xDV approximation */

#define DVS_HEADER_SIZE   10
#define DVS_EVENT_SIZE    7
#define DVS_MAX_EVENTS    65535

#define DVS_FRAME_WIDTH   160
#define DVS_FRAME_HEIGHT  120

#define DVS_SERVER_PORT   9473

#pragma pack(push, 1)

typedef struct {
    uint8_t  magic[2];
    uint32_t frame_id;
    uint32_t event_count;
} dvs_packet_header_t;

typedef struct {
    uint8_t  x;
    uint8_t  y;
    uint8_t  polarity;   /* 0 = brightness decreased (OFF), 1 = brightness increased (ON) */
    uint32_t timestamp;  /* microseconds since stream start */
} dvs_event_t;

#pragma pack(pop)

/* Gesture classification results */
typedef enum {
    GESTURE_NONE       = 0,
    GESTURE_OPEN_HAND  = 1,
    GESTURE_FIST       = 2,
    GESTURE_POINTING   = 3,
    GESTURE_PEACE      = 4,
    GESTURE_THUMBS_UP  = 5,
    GESTURE_COUNT      = 6
} gesture_class_t;

static const char* gesture_names[] = {
    "none",
    "palm",
    "fist",
    "one",
    "peace",
    "thumb_up"
};

#endif /* PROTOCOL_H */
