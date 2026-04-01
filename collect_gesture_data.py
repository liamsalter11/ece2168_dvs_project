#!/usr/bin/env python3
"""
collect_gesture_data.py

Records DVS activity accumulation frames as training data for the TFLite model.
Run this instead of webcam_to_dvs.py when collecting data. The script shows a
live window of what the DVS accumulation looks like and saves frames to disk.

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
import time
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("opencv-python required: pip install opencv-python")

# ── DVS emulation (copied from webcam_to_dvs.py) ──────────────────────────────
THRESHOLD       = 15
WIDTH, HEIGHT   = 160, 120
SAVE_SIZE       = 128   # resize before saving (matches model input)

# Activity map constants (must match gesture_kernel.h)
ACTIVITY_INCREMENT  = 80.0
ACTIVITY_DECREMENT  = 40.0
ACTIVITY_DECAY      = 0.92
ACTIVITY_THRESHOLD  = 40.0

VALID_GESTURES = ('palm', 'fist', 'one', 'peace', 'thumb_up')


class DVSActivityMap:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.map = np.zeros((h, w), dtype=np.float32)

    def update(self, frame_gray, prev_gray):
        """Compute DVS events from frame diff and update activity map."""
        self.map *= ACTIVITY_DECAY
        self.map[self.map < 1.0] = 0.0

        if prev_gray is None:
            return

        diff = frame_gray.astype(np.int16) - prev_gray.astype(np.int16)
        on_mask  = diff >  THRESHOLD
        off_mask = diff < -THRESHOLD

        self.map[on_mask]  += ACTIVITY_INCREMENT
        self.map[off_mask] += ACTIVITY_DECREMENT
        np.clip(self.map, 0, 255, out=self.map)

    def as_uint8(self):
        return self.map.astype(np.uint8)

    def as_rgb_uint8(self):
        """Return (SAVE_SIZE, SAVE_SIZE, 3) uint8 — grayscale replicated to 3ch."""
        gray = cv2.resize(self.as_uint8(), (SAVE_SIZE, SAVE_SIZE),
                          interpolation=cv2.INTER_AREA)
        return cv2.merge([gray, gray, gray])


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
        activity = DVSActivityMap(WIDTH, HEIGHT)
        prev_gray = None
        recording = False
        saved = 0
        last_save_time = 0.0
        SAVE_INTERVAL = 0.1   # save at most 10 fps to keep dataset diverse

        print(f"[collect] Press SPACE to start/stop recording, Q to quit")
        print(f"[collect] Target: {args.count} frames of gesture '{gesture}'")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            activity.update(gray, prev_gray)
            prev_gray = gray

            # Build display: side-by-side webcam + activity map
            act_uint8 = activity.as_uint8()
            act_display = cv2.applyColorMap(act_uint8, cv2.COLORMAP_HOT)
            act_display = cv2.resize(act_display, (320, 240))
            cam_display = cv2.resize(frame_resized, (320, 240))
            display = np.hstack([cam_display, act_display])

            # Status overlay
            status = f"REC {saved}/{args.count}" if recording else f"PAUSED {saved}/{args.count}"
            color  = (0, 0, 255) if recording else (0, 200, 200)
            cv2.putText(display, f"Gesture: {gesture}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, status, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, "SPACE=record  Q=quit", (10, 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("DVS Gesture Collector  [left=webcam  right=activity]", display)

            # Save frame
            now = time.monotonic()
            if recording and (now - last_save_time) >= SAVE_INTERVAL:
                idx = start_idx + saved
                fname = os.path.join(out_dir, f"{gesture}_{idx:04d}.png")
                cv2.imwrite(fname, activity.as_rgb_uint8())
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
