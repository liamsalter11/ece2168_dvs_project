#!/usr/bin/env python3
"""
webcam_to_dvs.py — Webcam to DVS Event Stream Emulator

Captures webcam frames, computes temporal contrast (brightness changes)
between consecutive frames, and emits DVS-style (x, y, polarity, timestamp)
events to a receiver.

Two transports are supported:
  tcp  — TCP socket with 4-byte length prefix (Raspberry Pi 4 / any TCP receiver)
  spi  — SPI via FT232H using pyftdi (E1x EVK; EVK is SPI slave, no length prefix)

Usage:
    python webcam_to_dvs.py [--host HOST] [--port PORT]
                            [--transport {tcp,spi}]
                            [--ftdi-url URL] [--spi-freq HZ]
                            [--threshold THRESH] [--width W] [--height H]
                            [--fps FPS] [--source {webcam,synthetic}]
                            [--visualize]
"""

import argparse
import struct
import socket
import time
import sys
import signal
import numpy as np

# Protocol constants (must match protocol.h)
DVS_MAGIC        = bytes([0xAE, 0xD7])
DVS_HEADER_FMT   = "<2sII"       # magic(2) + frame_id(4) + event_count(4)
DVS_EVENT_FMT    = "<BBbI"       # x(1) + y(1) + polarity(1) + timestamp(4)
DVS_HEADER_SIZE  = struct.calcsize(DVS_HEADER_FMT)
DVS_EVENT_SIZE   = struct.calcsize(DVS_EVENT_FMT)
DVS_MAX_EVENTS   = 65535

DEFAULT_WIDTH    = 160
DEFAULT_HEIGHT   = 120
DEFAULT_PORT     = 9473
DEFAULT_FTDI_URL = "ftdi://ftdi:232h/1"
DEFAULT_SPI_FREQ = 10_000_000   # 10 MHz — comfortably above typical event stream bandwidth


class DVSEmulator:
    """Converts conventional frames into DVS-style temporal contrast events."""

    def __init__(self, width: int, height: int, threshold: int = 15):
        self.width = width
        self.height = height
        self.threshold = threshold
        self.prev_frame = None
        self.start_time = time.monotonic()

    def process_frame(self, gray_frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Compare current grayscale frame against previous frame.
        Returns list of (x, y, polarity, timestamp_us) events.
        """
        timestamp_us = int((time.monotonic() - self.start_time) * 1_000_000)

        if self.prev_frame is None:
            self.prev_frame = gray_frame.copy()
            return []

        diff = gray_frame.astype(np.int16) - self.prev_frame.astype(np.int16)

        on_mask  = diff >  self.threshold
        off_mask = diff < -self.threshold

        events = []

        ys, xs = np.where(on_mask)
        for x, y in zip(xs, ys):
            events.append((int(x), int(y), 1, timestamp_us))

        ys, xs = np.where(off_mask)
        for x, y in zip(xs, ys):
            events.append((int(x), int(y), 0, timestamp_us))

        if len(events) > DVS_MAX_EVENTS:
            events = events[:DVS_MAX_EVENTS]

        self.prev_frame = gray_frame.copy()
        return events


def pack_frame_packet(frame_id: int, events: list[tuple[int, int, int, int]]) -> bytes:
    """Serialize a frame's events into the binary wire protocol (no transport framing)."""
    header = struct.pack(DVS_HEADER_FMT, DVS_MAGIC, frame_id, len(events))
    body = b"".join(
        struct.pack(DVS_EVENT_FMT, x, y, pol, ts)
        for x, y, pol, ts in events
    )
    return header + body


# ── Transport classes ──────────────────────────────────────────────────────────

class TCPSender:
    """TCP sender with 4-byte length prefix (matches rpi4/main.cpp receive_frame)."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self, timeout: float = 30.0):
        deadline = time.monotonic() + timeout
        print(f"[tcp] Connecting to {self.host}:{self.port} ...")
        while time.monotonic() < deadline:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.connect((self.host, self.port))
                print(f"[tcp] Connected!")
                return
            except ConnectionRefusedError:
                self.sock.close()
                time.sleep(0.5)
        raise ConnectionError(f"Could not connect to {self.host}:{self.port} within {timeout}s")

    def send_packet(self, data: bytes):
        length_prefix = struct.pack("<I", len(data))
        self.sock.sendall(length_prefix + data)

    def close(self):
        if self.sock:
            self.sock.close()


class SPISender:
    """
    SPI sender via FT232H using pyftdi (matches e1x/main.cpp receive_frame).

    The E1x is configured as SPI slave; the FT232H acts as master and drives
    the clock.  No length prefix is sent — the EVK reads the 10-byte DVS header
    directly, then reads event_count * 7 bytes of event data.
    """

    def __init__(self, ftdi_url: str, freq: int, mode: int = 0):
        self.ftdi_url = ftdi_url
        self.freq = freq
        self.mode = mode
        self._port = None

    def connect(self, timeout: float = 30.0):
        try:
            from pyftdi.spi import SpiController
        except ImportError:
            print("ERROR: pyftdi not found. Install with: pip install pyftdi")
            sys.exit(1)

        print(f"[spi] Opening {self.ftdi_url} at {self.freq/1e6:.1f} MHz ...")
        ctrl = SpiController()
        ctrl.configure(self.ftdi_url)
        self._port = ctrl.get_port(cs=0, freq=self.freq, mode=self.mode)
        print(f"[spi] FT232H ready (CS0, SPI mode {self.mode})")

    def send_packet(self, data: bytes):
        self._port.write(data)

    def close(self):
        pass  # pyftdi cleans up when the object is GC'd


# ── Main loops ─────────────────────────────────────────────────────────────────

def run_with_webcam(args, sender):
    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV not found. Install with: pip install opencv-python")
        sys.exit(1)

    emulator = DVSEmulator(args.width, args.height, args.threshold)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Use --source synthetic for testing.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    frame_id = 0
    target_dt = 1.0 / args.fps

    print(f"[sender] Streaming {args.width}x{args.height} @ {args.fps} FPS "
          f"(threshold={args.threshold})")
    print(f"[sender] Press Ctrl+C to stop")

    try:
        while True:
            t0 = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                print("[sender] Webcam read failed, retrying...")
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (args.width, args.height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            events = emulator.process_frame(gray)
            packet = pack_frame_packet(frame_id, events)

            try:
                sender.send_packet(packet)
            except (BrokenPipeError, ConnectionResetError):
                print("[sender] Connection lost.")
                break

            if frame_id % 30 == 0:
                print(f"  frame={frame_id:06d}  events={len(events):5d}  "
                      f"packet={len(packet)} bytes")

            if args.visualize:
                vis = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                for x, y, pol, ts in events:
                    if 0 <= y < args.height and 0 <= x < args.width:
                        vis[y, x] = (0, 255, 0) if pol == 1 else (0, 0, 255)
                cv2.imshow("DVS Events (green=ON, red=OFF)", vis)
                cv2.imshow("Grayscale", gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1

            elapsed = time.monotonic() - t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    finally:
        cap.release()
        sender.close()
        if args.visualize:
            cv2.destroyAllWindows()


def run_with_synthetic(args, sender):
    emulator = DVSEmulator(args.width, args.height, args.threshold)

    frame_id = 0
    target_dt = 1.0 / args.fps
    cx, cy = args.width // 2, args.height // 2
    radius = min(args.width, args.height) // 6
    angle = 0.0

    print(f"[sender] Synthetic mode: {args.width}x{args.height} @ {args.fps} FPS")
    print(f"[sender] Generating moving blob pattern")

    try:
        while True:
            t0 = time.monotonic()

            frame = np.zeros((args.height, args.width), dtype=np.uint8)
            bx = int(cx + radius * 2 * np.cos(angle))
            by = int(cy + radius * 2 * np.sin(angle))
            yy, xx = np.ogrid[:args.height, :args.width]
            mask = (xx - bx)**2 + (yy - by)**2 <= radius**2
            frame[mask] = 200

            bx2 = int(cx + radius * np.cos(-angle * 1.5))
            by2 = int(cy + radius * np.sin(-angle * 1.5))
            mask2 = (xx - bx2)**2 + (yy - by2)**2 <= (radius // 2)**2
            frame[mask2] = 150

            events = emulator.process_frame(frame)
            packet = pack_frame_packet(frame_id, events)

            try:
                sender.send_packet(packet)
            except (BrokenPipeError, ConnectionResetError):
                print("[sender] Connection lost.")
                break

            if frame_id % 30 == 0:
                print(f"  frame={frame_id:06d}  events={len(events):5d}  "
                      f"packet={len(packet)} bytes")

            frame_id += 1
            angle += 0.05

            elapsed = time.monotonic() - t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    finally:
        sender.close()


def main():
    parser = argparse.ArgumentParser(
        description="Webcam to DVS Event Stream Emulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Transport selection
    parser.add_argument("--transport", choices=["tcp", "spi"], default="tcp",
                        help="Output transport: tcp (RPi 4) or spi (E1x EVK via FT232H)")

    # TCP options
    parser.add_argument("--host", default="127.0.0.1",
                        help="[tcp] Receiver host address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="[tcp] Receiver port")

    # SPI options
    parser.add_argument("--ftdi-url", default=DEFAULT_FTDI_URL,
                        help="[spi] pyftdi device URL")
    parser.add_argument("--spi-freq", type=int, default=DEFAULT_SPI_FREQ,
                        help="[spi] SPI clock frequency in Hz")

    # Common
    parser.add_argument("--threshold", type=int, default=15,
                        help="Brightness change threshold to emit event (0-255)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help="Frame width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                        help="Frame height")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target frame rate")
    parser.add_argument("--source", choices=["webcam", "synthetic"], default="webcam",
                        help="Video source")
    parser.add_argument("--visualize", action="store_true",
                        help="Show local DVS event visualization (requires OpenCV)")

    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    if args.transport == "spi":
        sender = SPISender(args.ftdi_url, args.spi_freq)
    else:
        sender = TCPSender(args.host, args.port)

    sender.connect()

    if args.source == "webcam":
        run_with_webcam(args, sender)
    else:
        run_with_synthetic(args, sender)


if __name__ == "__main__":
    main()
