#!/usr/bin/env python3
"""
gesture_receiver.py — Windows-native DVS gesture receiver (third DUT).

Mirrors the Pi (dut/rpi4/main.cpp) and E1x (dut/e1x/main.cpp) firmware end
to end:
  TCP DVS-event ingest  →  KernelMirror activity map (same Q16 resize)  →
  blob detection (threshold + morphology + 4-connected components)        →
  ai_edge_litert TFLite inference  →  stable-2-frame argmax  →  print

Listens for the same wire-format packets webcam_to_dvs.py and tools/dvs_clip.py
already emit, so you can drive all three DUTs from a single sender.

Use cases:
  - End-to-end runs entirely on Windows (sender + receiver on localhost).
  - "Reference" classifier when comparing against the EVK — Python's
    ai_edge_litert is bit-equivalent to the Pi's C++ TFLite Reference int8
    runtime, so any divergence between Windows and EVK isolates the
    eff-import / fabric path.

Usage:
  pip install ai-edge-litert opencv-python numpy
  python windows/gesture_receiver.py --model gesture_model.tflite

Then in another terminal:
  python windows/webcam_to_dvs.py --transport tcp --host 127.0.0.1
  # or replay a recorded clip:
  python tools/dvs_clip.py replay clip.dvsclip --transport tcp --host 127.0.0.1
"""

import argparse
import os
import socket
import struct
import sys
import time

import numpy as np

# cv2 is imported lazily inside find_largest_blob_area so --help works on
# machines that don't have OpenCV installed.

# Reuse the per-event activity-map / Q16 nearest-neighbor downsample from
# webcam_to_dvs.py so the runtime resize matches the EVK exactly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from webcam_to_dvs import KernelMirror, KERNEL_INPUT_SIZE       # noqa: E402

# ── Constants mirroring dut/common/gesture_kernel.h / .cpp ────────────────────

ACTIVITY_THRESHOLD   = 40.0
MIN_BLOB_AREA        = 50
INFERENCE_AREA_GATE  = MIN_BLOB_AREA * 4    # 200 — same gate as the C++
CONFIDENCE_THRESHOLD = 0.40                  # matches gesture_kernel.cpp
TFLM_NUM_CLASSES     = 5

# Wire protocol (dut/common/protocol.h)
DVS_MAGIC_0     = 0xAE
DVS_MAGIC_1     = 0xD7
DVS_HEADER_SIZE = 10
DVS_EVENT_SIZE  = 7
DEFAULT_PORT    = 9473

# Activity-map domain (must match DVS_FRAME_WIDTH/HEIGHT)
WIDTH, HEIGHT = 160, 120

# Keras alphabetical class order maps to the protocol enum:
#   keras 0 fist     → GESTURE_FIST       = 2
#   keras 1 one      → GESTURE_POINTING   = 3
#   keras 2 palm     → GESTURE_OPEN_HAND  = 1
#   keras 3 peace    → GESTURE_PEACE      = 4
#   keras 4 thumb_up → GESTURE_THUMBS_UP  = 5
KERAS_NAMES         = ["fist", "one", "palm", "peace", "thumb_up"]
KERAS_TO_PROTO_IDX  = [2, 3, 1, 4, 5]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_largest_blob_area(activity_map):
    """Threshold + 3x3 morph close + 4-connected CC labelling → largest area.
    Mirrors gesture_kernel.cpp's apply_threshold + connected_components +
    find_largest_blob.  Returns 0 if no blob meets MIN_BLOB_AREA.
    """
    try:
        import cv2
    except ImportError:
        raise SystemExit("opencv-python required: pip install opencv-python")
    binary = (activity_map >= ACTIVITY_THRESHOLD).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
    if n_labels <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(areas.max())
    return best if best >= MIN_BLOB_AREA else 0


def recvall(sock, n):
    """Read exactly n bytes or return None if the connection closed."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


# ── TFLite backend ────────────────────────────────────────────────────────────

class TFLiteBackend:
    def __init__(self, model_path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            raise SystemExit("pip install ai-edge-litert")
        self.interp = Interpreter(model_path=model_path)
        self.interp.allocate_tensors()
        self.inp_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]
        # Sanity-check the model matches what the kernel expects.
        shape = list(self.inp_det["shape"])
        if shape != [1, KERNEL_INPUT_SIZE, KERNEL_INPUT_SIZE, 3]:
            print(f"[receiver] WARN: model input shape {shape} != "
                  f"expected [1,{KERNEL_INPUT_SIZE},{KERNEL_INPUT_SIZE},3]",
                  file=sys.stderr)
        print(f"[tflite] Loaded {model_path} — input {shape} "
              f"dtype={self.inp_det['dtype'].__name__}")
        print(f"[tflite] Output {list(self.out_det['shape'])} "
              f"dtype={self.out_det['dtype'].__name__}")

    def run(self, model_input_uint8):
        """model_input_uint8 is (H, W) uint8 grayscale from KernelMirror.
        Returns (best_idx, confidence, logits[5]).  Logits are 0..255 raw."""
        rgb = np.repeat(model_input_uint8[:, :, None], 3, axis=2)
        x = rgb.reshape(self.inp_det["shape"]).astype(self.inp_det["dtype"])
        self.interp.set_tensor(self.inp_det["index"], x)
        self.interp.invoke()
        out = self.interp.get_tensor(self.out_det["index"])[0]
        logits = [int(v) for v in out]
        best = int(np.argmax(out))
        confidence = float(out[best]) / 255.0
        return best, confidence, logits


# ── Connection handler ────────────────────────────────────────────────────────

def handle_client(conn, addr, mirror, backend):
    print(f"[receiver] accepted {addr}")
    last_gesture = -1
    stable_count = 0
    frame_count = 0
    last_log = time.monotonic()

    try:
        while True:
            len_buf = recvall(conn, 4)
            if not len_buf:
                break
            (plen,) = struct.unpack("<I", len_buf)
            packet = recvall(conn, plen)
            if not packet:
                break
            if len(packet) < DVS_HEADER_SIZE:
                continue
            if packet[0] != DVS_MAGIC_0 or packet[1] != DVS_MAGIC_1:
                print(f"[receiver] BAD_MAGIC {packet[0]:02x} {packet[1]:02x}")
                continue

            event_count = struct.unpack("<I", packet[6:10])[0]

            # Decode events into the (x, y, pol, ts) tuples KernelMirror wants.
            events = []
            base = DVS_HEADER_SIZE
            for i in range(event_count):
                off = base + i * DVS_EVENT_SIZE
                if off + 3 > len(packet):
                    break
                events.append((packet[off], packet[off + 1], packet[off + 2], 0))
            mirror.ingest(events)

            area = find_largest_blob_area(mirror.activity)
            gesture_idx = -1
            confidence = 0.0
            logits = [0] * TFLM_NUM_CLASSES
            if area >= INFERENCE_AREA_GATE:
                gesture_idx, confidence, logits = backend.run(mirror.model_input_uint8())
                if confidence < CONFIDENCE_THRESHOLD:
                    gesture_idx = -1

            frame_count += 1
            if (frame_count % 30) == 0:
                print(f"f={frame_count} n={event_count} a={area} "
                      f"L:{'/'.join(str(x) for x in logits)}")
                last_log = time.monotonic()

            if gesture_idx == last_gesture:
                stable_count += 1
            else:
                stable_count = 0
                last_gesture = gesture_idx
            if stable_count == 2 and gesture_idx >= 0:
                print(f"GESTURE {KERAS_TO_PROTO_IDX[gesture_idx]} "
                      f"{KERAS_NAMES[gesture_idx]} {int(confidence * 100)}%")
    finally:
        conn.close()
        print(f"[receiver] {addr} closed")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Windows-native DVS gesture receiver (third DUT).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_model = os.path.join(project_root, "gesture_model.tflite")
    p.add_argument("--model", default=default_model,
                   help="Path to gesture_model.tflite")
    p.add_argument("--port",  type=int, default=DEFAULT_PORT,
                   help="TCP listen port")
    p.add_argument("--bind",  default="0.0.0.0",
                   help="Listen address (use 127.0.0.1 to restrict to localhost)")
    args = p.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"ERROR: model file not found: {args.model}")

    backend = TFLiteBackend(args.model)
    mirror  = KernelMirror(WIDTH, HEIGHT)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(f"Gesture recognition ready (ai_edge_litert)")
    print(f"[tcp] Listening on {args.bind}:{args.port} ...")

    try:
        while True:
            conn, addr = srv.accept()
            # Reset activity map between connections so old state doesn't bleed.
            mirror.activity[:] = 0.0
            handle_client(conn, addr, mirror, backend)
    except KeyboardInterrupt:
        print("\n[receiver] shutting down")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
