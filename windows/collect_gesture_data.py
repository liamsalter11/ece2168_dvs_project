#!/usr/bin/env python3
"""
collect_gesture_data.py

Records DVS activity-map frames as training data for the TFLite model.

Pipeline matches what the EVK actually sees at inference time:
  webcam → grayscale → DVS event diff → KernelMirror activity map →
  KernelMirror.model_input_uint8() (Q16 nearest-neighbor downsample to 96x96)

Both the saved PNG and the live preview come from KernelMirror so there's no
training/runtime resize-algorithm mismatch.  Imports DVSEmulator + KernelMirror
from webcam_to_dvs.py so the host-sender pipeline and collector share one
implementation.

Usage:
  pip install opencv-python numpy
  python collect_gesture_data.py --gesture palm
  python collect_gesture_data.py --gesture fist
  python collect_gesture_data.py --gesture one
  python collect_gesture_data.py --gesture peace
  python collect_gesture_data.py --gesture thumb_up

Controls (OpenCV window must be focused):
  SPACE  — toggle recording on/off
  Q      — quit and save

Output:
  data/<gesture>/<gesture>_0001.png, ...
"""

import argparse
import os
import sys
import time
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv-python required: pip install opencv-python")

# Import the shared DVS emulator + EVK kernel mirror from the sender script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from webcam_to_dvs import (
    DVSEmulator, KernelMirror,
    EVK_SPI_MAX_EVENTS, KERNEL_INPUT_SIZE,
)

THRESHOLD     = 8           # match the runtime sender default
WIDTH, HEIGHT = 160, 120

VALID_GESTURES = ('palm', 'fist', 'one', 'peace', 'thumb_up')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gesture', choices=VALID_GESTURES,
                        help='Gesture class to record (omit to run all sequentially)')
    parser.add_argument('--all',    action='store_true',
                        help='Record all gestures sequentially with a pause between each')
    parser.add_argument('--count',   type=int, default=300,
                        help='Target number of frames to save (default: 300)')
    parser.add_argument('--data_dir', default='data',
                        help='Root output directory (default: data/)')
    parser.add_argument('--camera',  type=int, default=0)
    args = parser.parse_args()

    if not args.gesture and not args.all:
        parser.error('specify --gesture GESTURE or --all')

    gestures = list(VALID_GESTURES) if args.all else [args.gesture]

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    for gesture_idx, gesture in enumerate(gestures):
        out_dir = os.path.join(args.data_dir, gesture)
        os.makedirs(out_dir, exist_ok=True)

        existing = [f for f in os.listdir(out_dir) if f.endswith('.png')]
        start_idx = len(existing)
        print(f"\n[collect] ── Gesture {gesture_idx + 1}/{len(gestures)}: '{gesture}' ──")
        print(f"[collect] Saving to {out_dir}/  (existing: {start_idx})")

        # ── Inter-gesture pause screen ────────────────────────────────────────
        if gesture_idx > 0:
            print(f"[collect] Get ready for '{gesture}' — press SPACE when ready")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                pause_display = cv2.resize(frame, (640, 480))
                cv2.putText(pause_display, f"Next: {gesture}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
                cv2.putText(pause_display, "Press SPACE when ready", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                remaining = ', '.join(gestures[gesture_idx:])
                cv2.putText(pause_display, f"Remaining: {remaining}", (20, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                cv2.imshow("DVS Gesture Collector  [left=webcam  right=activity]", pause_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                elif key in (ord('q'), ord('Q'), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    print('[collect] Aborted.')
                    return

        # ── Recording loop ────────────────────────────────────────────────────
        emulator = DVSEmulator(WIDTH, HEIGHT, THRESHOLD)
        mirror   = KernelMirror(WIDTH, HEIGHT)  # out_size defaults to 96
        recording = False
        saved = 0
        last_save_time = 0.0
        SAVE_INTERVAL = 0.1   # save at most 10 fps to keep dataset diverse

        print(f"[collect] Press SPACE to start/stop recording, Q to quit")
        print(f"[collect] Target: {args.count} frames of gesture '{gesture}'")
        print(f"[collect] Saving {KERNEL_INPUT_SIZE}x{KERNEL_INPUT_SIZE} "
              f"PNGs (matches EVK model input)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            # Drive the EVK kernel mirror through the same path the host sender
            # uses: frame-diff → events → ingest.  Apply the same per-frame cap
            # the SPI sender applies, so the activity map reflects what the EVK
            # actually receives.
            events = emulator.process_frame(gray)
            mirror.ingest(events[:EVK_SPI_MAX_EVENTS])

            # Display: webcam (for framing) | model-input view (what the EVK sees)
            cam_display = cv2.resize(frame_resized, (320, 240))
            model_in    = mirror.model_input_uint8()  # 96x96 grayscale
            model_disp  = cv2.applyColorMap(
                cv2.resize(model_in, (320, 240),
                           interpolation=cv2.INTER_NEAREST),
                cv2.COLORMAP_HOT)
            display = np.hstack([cam_display, model_disp])

            # Status overlay
            status = f"REC {saved}/{args.count}" if recording else f"PAUSED {saved}/{args.count}"
            color  = (0, 0, 255) if recording else (0, 200, 200)
            cv2.putText(display, f"Gesture: {gesture}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, status, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, f"events={len(events)}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.putText(display, "SPACE=record  Q=quit", (10, 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("DVS Gesture Collector  [left=webcam  right=EVK model input]",
                       display)

            # Save frame at the model's native resolution.  Saved as 3-channel
            # to match the model input shape (the trainer's preprocess uses the
            # PNG's native channels, and the EVK runtime feeds 3 replicated
            # grayscale channels — saving 3ch keeps the pipelines aligned).
            now = time.monotonic()
            if recording and (now - last_save_time) >= SAVE_INTERVAL:
                idx = start_idx + saved
                fname = os.path.join(out_dir, f"{gesture}_{idx:04d}.png")
                rgb = cv2.merge([model_in, model_in, model_in])
                cv2.imwrite(fname, rgb)
                saved += 1
                last_save_time = now
                if saved >= args.count:
                    print(f"[collect] Reached target of {args.count} frames — done.")
                    break

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                recording = not recording
                print(f"[collect] {'Recording...' if recording else 'Paused'}")
            elif key in (ord('q'), ord('Q'), 27):
                cap.release()
                cv2.destroyAllWindows()
                print(f"[collect] Saved {saved} frames to {out_dir}/")
                print('[collect] Aborted.')
                return

        print(f"[collect] Saved {saved} frames to {out_dir}/")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[collect] All done! Recorded {len(gestures)} gesture(s).")


if __name__ == '__main__':
    main()
