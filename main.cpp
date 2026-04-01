/**
 * main.cpp — DVS Gesture Recognition Receiver
 *
 * Listens for DVS event packets from the Python webcam emulator,
 * feeds them into the gesture recognition kernel, and prints
 * classification results to stdout.
 *
 * Build:
 *     mkdir build && cd build
 *     cmake .. && make
 *
 * Run:
 *     ./gesture_receiver [--port PORT] [--ascii]
 */

#include "spi_receiver.h"
#include "gesture_kernel.h"
#include "protocol.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <chrono>

static volatile bool g_running = true;

static void signal_handler(int)
{
    g_running = false;
}

/* ────────────────────────────────────────────────────────────────── */
/*  ASCII visualization of the activity map (optional)               */
/* ────────────────────────────────────────────────────────────────── */

static void print_ascii_frame(const GestureKernel& kernel,
                               const GestureResult& result,
                               int width, int height)
{
    /* Subsample for terminal output */
    const int term_w = 79;   /* one fewer than terminal width to avoid wrap */
    const int term_h = 30;
    int step_x = (width + term_w - 1) / term_w;
    int step_y = (height + term_h - 1) / term_h;

    const float* activity = kernel.activity_map();
    const char ramp[] = " .:-=+*#%@";
    int ramp_len = sizeof(ramp) - 2;

    printf("\033[H");  /* Move cursor to top-left (ANSI escape) */

    for (int y = 0; y < height; y += step_y) {
        int cols = 0;
        for (int x = 0; x < width && cols < term_w; x += step_x, cols++) {
            float val = activity[y * width + x];
            int idx = (int)(val / 255.0f * ramp_len);
            if (idx < 0) idx = 0;
            if (idx > ramp_len) idx = ramp_len;
            putchar(ramp[idx]);
        }
        putchar('\n');
    }

    /* Status bar */
    printf("─────────────────────────────────────────────────────\n");
    printf("  Gesture: %-12s  Confidence: %.0f%%\n",
           gesture_names[result.gesture],
           result.confidence * 100.0f);
    printf("  Blob area: %5d  Fill: %.2f  Aspect: %.2f\n",
           result.features.area,
           result.features.fill_ratio,
           result.features.aspect_ratio);
    printf("─────────────────────────────────────────────────────\n");
    fflush(stdout);
}

/* ────────────────────────────────────────────────────────────────── */
/*  Main                                                             */
/* ────────────────────────────────────────────────────────────────── */

int main(int argc, char* argv[])
{
    int port = DVS_SERVER_PORT;
    bool ascii_vis = false;
    const char* model_path = nullptr;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ascii") == 0) {
            ascii_vis = true;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--port PORT] [--ascii] [--model PATH]\n", argv[0]);
            printf("  --port PORT   TCP port to listen on (default: %d)\n", DVS_SERVER_PORT);
            printf("  --ascii       Show ASCII visualization in terminal\n");
            printf("  --model PATH  TFLite model file (enables ML inference)\n");
            return 0;
        }
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   DVS Gesture Recognition Receiver                  ║\n");
    printf("║   Frame: %dx%d   Port: %d                      ║\n",
           DVS_FRAME_WIDTH, DVS_FRAME_HEIGHT, port);
#ifdef TFLITE_ENABLED
    if (model_path)
        printf("║   Inference: TFLite (%s)\n", model_path);
    else
        printf("║   Inference: Classical CV (no --model given)        ║\n");
#else
    printf("║   Inference: Classical CV                           ║\n");
#endif
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    /* Initialize receiver and kernel */
    SPIReceiver receiver(port);
    GestureKernel kernel(DVS_FRAME_WIDTH, DVS_FRAME_HEIGHT, model_path);

    if (!receiver.listen_and_accept()) {
        fprintf(stderr, "Failed to start receiver.\n");
        return 1;
    }

    if (ascii_vis) {
        printf("\033[2J");  /* Clear screen */
    }

    DVSFrame frame;
    uint32_t total_frames = 0;
    uint64_t total_events = 0;
    auto t_start = std::chrono::steady_clock::now();
    auto t_last_report = t_start;
    gesture_class_t last_gesture = GESTURE_NONE;
    int gesture_stable_count = 0;

    while (g_running) {
        if (!receiver.receive_frame(frame)) {
            printf("[main] Stream ended.\n");
            break;
        }

        /* Feed events into kernel */
        kernel.ingest_events(frame.events.data(), frame.event_count);

        /* Run gesture classification */
        GestureResult result = kernel.classify();

        total_frames++;
        total_events += frame.event_count;

        /* Gesture stability tracking (require N consecutive same classifications) */
        if (result.gesture == last_gesture) {
            gesture_stable_count++;
        } else {
            gesture_stable_count = 0;
            last_gesture = result.gesture;
        }

        /* Visualization mode */
        if (ascii_vis) {
            print_ascii_frame(kernel, result, DVS_FRAME_WIDTH, DVS_FRAME_HEIGHT);
        } else {
            /* Periodic text report */
            auto now = std::chrono::steady_clock::now();
            auto ms_since_report = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - t_last_report).count();

            if (ms_since_report >= 500) {
                auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - t_start).count();
                float fps = (total_ms > 0) ? (total_frames * 1000.0f / total_ms) : 0;

                printf("[frame %06u]  events=%5u  gesture=%-12s  conf=%.0f%%  "
                       "stable=%d  (%.1f fps)\n",
                       frame.frame_id,
                       frame.event_count,
                       gesture_names[result.gesture],
                       result.confidence * 100.0f,
                       gesture_stable_count,
                       fps);

                t_last_report = now;
            }
        }
    }

    /* Summary */
    auto t_end = std::chrono::steady_clock::now();
    auto total_s = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() / 1000.0f;

    printf("\n═══ Session Summary ═══\n");
    printf("  Frames processed: %u\n", total_frames);
    printf("  Total events:     %llu\n", (unsigned long long)total_events);
    printf("  Duration:         %.1f s\n", total_s);
    printf("  Average FPS:      %.1f\n", (total_s > 0) ? total_frames / total_s : 0);
    printf("  Avg events/frame: %.0f\n",
           (total_frames > 0) ? (float)total_events / total_frames : 0);

    receiver.shutdown();
    return 0;
}
