// Copyright Efficient Computer Company 2026. All Rights Reserved.
//
// DVS event protocol definitions — shared between host emulator and firmware.
//
// Wire format (little-endian):
//   Frame header (10 bytes):
//     magic[2]      0xAE 0xD7
//     frame_id      uint32 LE
//     event_count   uint32 LE
//   Events (event_count × 7 bytes each):
//     x             uint8
//     y             uint8
//     polarity      uint8   0=OFF 1=ON
//     timestamp     uint32 LE  microseconds

#pragma once

#include <stdint.h>

#define DVS_MAGIC_0       0xAE
#define DVS_MAGIC_1       0xD7

#define DVS_HEADER_SIZE   10
#define DVS_EVENT_SIZE    7
/* On-wire cap. Firmware enforces a tighter EVK_FRAME_EVENT_CAP in e1x/main.cpp
 * to fit the static UART receive buffer.  The Pi receiver uses this value
 * directly since it has ample heap. */
#define DVS_MAX_EVENTS    65535

#define DVS_SERVER_PORT   9473   /* TCP port the Pi receiver listens on */

#define DVS_FRAME_WIDTH   160
#define DVS_FRAME_HEIGHT  120

#pragma pack(push, 1)

typedef struct {
    uint8_t  magic[2];
    uint32_t frame_id;
    uint32_t event_count;
} dvs_packet_header_t;

typedef struct {
    uint8_t  x;
    uint8_t  y;
    uint8_t  polarity;
    uint32_t timestamp;
} dvs_event_t;

#pragma pack(pop)

typedef enum {
    GESTURE_NONE       = 0,
    GESTURE_OPEN_HAND  = 1,
    GESTURE_FIST       = 2,
    GESTURE_POINTING   = 3,
    GESTURE_PEACE      = 4,
    GESTURE_THUMBS_UP  = 5,
    GESTURE_COUNT      = 6
} gesture_class_t;

static const char* const gesture_names[] = {
    "none",
    "palm",
    "fist",
    "one",
    "peace",
    "thumb_up"
};
