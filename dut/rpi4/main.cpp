// Gesture recognition receiver — Raspberry Pi 4
//
// Transport: TCP on eth0.  The Windows laptop runs webcam_to_dvs.py and
// connects to the Pi's eth0 address on DVS_SERVER_PORT (9473).
// The receiver binds INADDR_ANY; route traffic to the Pi's eth0 IP from
// the laptop.
//
// Build:   see dut/rpi4/CMakeLists.txt
// Run:     ./gesture_receiver [--model PATH] [--port PORT]
//          (default model: gesture_model.tflite in working directory)

#include "../common/gesture_kernel.h"
#include "../common/protocol.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

// Implemented in tflite_backend.cpp
extern "C" int tflite_backend_init(const char* model_path);

static volatile bool g_running = true;
static void sig_handler(int) { g_running = false; }

/* ── TCP transport ────────────────────────────────────────────────── */

static bool tcp_read_exact(int fd, uint8_t* buf, size_t n)
{
    size_t total = 0;
    while (total < n) {
        ssize_t r = read(fd, buf + total, n - total);
        if (r <= 0) {
            if (r == 0) printf("[tcp] Client disconnected.\n");
            else        perror("[tcp] read()");
            return false;
        }
        total += (size_t)r;
    }
    return true;
}

static int tcp_listen_and_accept(int port)
{
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket()"); return -1; }

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((uint16_t)port);

    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind()"); close(srv); return -1;
    }
    if (listen(srv, 1) < 0) {
        perror("listen()"); close(srv); return -1;
    }

    printf("[tcp] Listening on port %d ...\n", port);

    struct sockaddr_in client_addr = {};
    socklen_t client_len = sizeof(client_addr);
    int client = accept(srv, (struct sockaddr*)&client_addr, &client_len);
    close(srv);   /* only one connection at a time */

    if (client < 0) { perror("accept()"); return -1; }

    int flag = 1;
    setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    printf("[tcp] Connected from %s:%d\n",
           inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
    return client;
}

/* ── Frame receive ────────────────────────────────────────────────── */

static int receive_frame(int fd,
                          dvs_packet_header_t* header_out,
                          dvs_event_t*         events_out,
                          uint32_t             events_cap)
{
    /* 4-byte length prefix written by webcam_to_dvs.py */
    uint8_t len_buf[4];
    if (!tcp_read_exact(fd, len_buf, 4)) return -1;

    uint32_t packet_len;
    memcpy(&packet_len, len_buf, 4);

    if (packet_len < DVS_HEADER_SIZE) return -1;

    /* Read header */
    uint8_t hdr_bytes[DVS_HEADER_SIZE];
    if (!tcp_read_exact(fd, hdr_bytes, DVS_HEADER_SIZE)) return -1;

    if (hdr_bytes[0] != DVS_MAGIC_0 || hdr_bytes[1] != DVS_MAGIC_1) {
        printf("[tcp] Bad magic: 0x%02X 0x%02X\n", hdr_bytes[0], hdr_bytes[1]);
        return -1;
    }

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
    if (n > events_cap) n = events_cap;

    /* Read events raw then deserialize */
    static uint8_t evt_bytes[DVS_MAX_EVENTS * DVS_EVENT_SIZE];
    if (!tcp_read_exact(fd, evt_bytes, n * DVS_EVENT_SIZE)) return -1;

    /* Drain any excess events we're not storing */
    uint32_t remaining = (header_out->event_count - n) * DVS_EVENT_SIZE;
    if (remaining > 0) {
        static uint8_t drain[DVS_EVENT_SIZE];
        for (uint32_t i = 0; i < remaining / DVS_EVENT_SIZE; i++)
            if (!tcp_read_exact(fd, drain, DVS_EVENT_SIZE)) return -1;
    }

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

int main(int argc, char* argv[])
{
    const char* model_path = "gesture_model.tflite";
    int port = DVS_SERVER_PORT;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: %s [--model PATH] [--port PORT]\n", argv[0]);
            return 0;
        }
    }

    signal(SIGINT,  sig_handler);
    signal(SIGTERM, sig_handler);

    if (tflite_backend_init(model_path) != 0) {
        fprintf(stderr, "ERROR: TFLite init failed\n");
        return 1;
    }
    if (gesture_kernel_init() != 0) {
        fprintf(stderr, "ERROR: gesture kernel init failed\n");
        return 1;
    }

    printf("Gesture recognition ready (TFLite v2.18.0)\n");

    int client_fd = tcp_listen_and_accept(port);
    if (client_fd < 0) return 1;

    static dvs_packet_header_t header;
    static dvs_event_t events[DVS_MAX_EVENTS];

    gesture_class_t last_gesture = GESTURE_NONE;
    int stable_count = 0;

    while (g_running) {
        int n = receive_frame(client_fd, &header, events, DVS_MAX_EVENTS);
        if (n < 0) break;   /* disconnect — exit cleanly */

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
            fflush(stdout);
        }
    }

    close(client_fd);
    return 0;
}
