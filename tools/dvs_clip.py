#!/usr/bin/env python3
"""
dvs_clip.py — Record once, replay to both DUTs.

Captures a webcam session into a .dvsclip file (the wire-format DVS packets
plus their inter-frame timing), then replays it to either the Pi over TCP
or the E1x EVK over SPI at the recorded cadence.  Lets you compare the two
backends on bit-identical input — you can re-run the same clip dozens of
times and get the exact same event stream every time.

Usage:
  # Record a 30 s clip from the webcam:
  python tools/dvs_clip.py record --output palm.dvsclip --duration 30

  # Replay to the Pi over TCP:
  python tools/dvs_clip.py replay palm.dvsclip --transport tcp --host <PI_IP>

  # Replay to the E1x EVK over SPI (FT232H master):
  python tools/dvs_clip.py replay palm.dvsclip --transport spi \\
      --ftdi-url "ftdi://ftdi:232h:FTAD4WXF/1"

  # Inspect a clip without playing it:
  python tools/dvs_clip.py inspect palm.dvsclip

File format ("DVSCLP01"):
  Header (12 bytes):
    char[8]  "DVSCLP01"
    uint16   width  (LE)
    uint16   height (LE)
  Records (repeating until EOF):
    uint64   rel_us (LE) — microseconds since record start
    uint32   packet_len (LE)
    bytes    packet of length packet_len  (raw DVS wire-protocol packet)

The packet bytes are exactly what would be sent over TCP / SPI in live mode
— same magic, frame_id, event_count, events.  Per-frame events are capped
to EVK_SPI_MAX_EVENTS at record time so both DUTs receive identical bytes
even though the Pi could ingest more.
"""

import argparse
import os
import struct
import sys
import time

# Pull DVSEmulator + the senders from the live-streaming script.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "windows"))
from webcam_to_dvs import (                                       # noqa: E402
    DVSEmulator, pack_frame_packet,
    TCPSender, SPISender,
    EVK_SPI_MAX_EVENTS,
)

DVS_SERVER_PORT = 9473    # matches DVS_SERVER_PORT in dut/common/protocol.h

CLIP_MAGIC      = b"DVSCLP01"
HEADER_FMT      = "<8sHH"        # magic, width, height
HEADER_SIZE     = struct.calcsize(HEADER_FMT)
RECORD_HDR_FMT  = "<QI"          # rel_us, packet_len
RECORD_HDR_SIZE = struct.calcsize(RECORD_HDR_FMT)


# ── Record ────────────────────────────────────────────────────────────────────

def cmd_record(args):
    try:
        import cv2
    except ImportError:
        raise SystemExit("opencv-python required: pip install opencv-python")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("ERROR: could not open webcam (try --camera 1)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    emu = DVSEmulator(args.width, args.height, args.threshold)

    out = open(args.output, "wb")
    out.write(struct.pack(HEADER_FMT, CLIP_MAGIC, args.width, args.height))
    bytes_written = HEADER_SIZE

    print(f"[record] {args.width}x{args.height} @ {args.fps} FPS, threshold={args.threshold}")
    print(f"[record] writing {args.output}")
    if args.duration:
        print(f"[record] stopping after {args.duration} s")
    print(f"[record] Ctrl-C to stop early")

    target_dt = 1.0 / args.fps
    t_start = time.monotonic()
    frame_id = 0

    try:
        while True:
            t0 = time.monotonic()
            elapsed = t0 - t_start
            if args.duration and elapsed >= args.duration:
                break

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (args.width, args.height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            events = emu.process_frame(gray)

            # Cap at the EVK's per-frame limit so the recorded packets are
            # what BOTH DUTs would receive in live mode — keeps replay fair.
            if len(events) > EVK_SPI_MAX_EVENTS:
                events = events[:EVK_SPI_MAX_EVENTS]

            packet = pack_frame_packet(frame_id, events)
            rel_us = int(elapsed * 1_000_000)
            out.write(struct.pack(RECORD_HDR_FMT, rel_us, len(packet)))
            out.write(packet)
            bytes_written += RECORD_HDR_SIZE + len(packet)

            if frame_id % 30 == 0:
                print(f"  [{elapsed:5.2f}s] frame={frame_id:5d} events={len(events):4d} "
                      f"file={bytes_written/1024:.1f} KB")

            frame_id += 1
            dt = time.monotonic() - t0
            if dt < target_dt:
                time.sleep(target_dt - dt)
    except KeyboardInterrupt:
        print("\n[record] stopped by Ctrl-C")
    finally:
        cap.release()
        out.close()

    print(f"[record] done. {frame_id} frames, {bytes_written/1024:.1f} KB → {args.output}")


# ── Read ──────────────────────────────────────────────────────────────────────

def open_clip(path):
    """Open a .dvsclip and yield (kind, ...) tuples.

    First yield: ('HEADER', width, height).
    Subsequent yields: ('FRAME', rel_us, packet_bytes) until EOF.
    """
    f = open(path, "rb")
    hdr = f.read(HEADER_SIZE)
    if len(hdr) != HEADER_SIZE:
        f.close()
        raise SystemExit(f"ERROR: {path} too short to contain a header")
    magic, width, height = struct.unpack(HEADER_FMT, hdr)
    if magic != CLIP_MAGIC:
        f.close()
        raise SystemExit(f"ERROR: {path} not a dvsclip (got magic {magic!r}, "
                         f"expected {CLIP_MAGIC!r})")
    yield ("HEADER", width, height)
    while True:
        rh = f.read(RECORD_HDR_SIZE)
        if len(rh) == 0:
            break
        if len(rh) != RECORD_HDR_SIZE:
            raise SystemExit(f"ERROR: truncated record header in {path}")
        rel_us, plen = struct.unpack(RECORD_HDR_FMT, rh)
        packet = f.read(plen)
        if len(packet) != plen:
            raise SystemExit(f"ERROR: truncated record body in {path}")
        yield ("FRAME", rel_us, packet)
    f.close()


# ── Replay ────────────────────────────────────────────────────────────────────

def cmd_replay(args):
    if args.transport == "tcp":
        sender = TCPSender(args.host, args.port)
    elif args.transport == "spi":
        sender = SPISender(args.ftdi_url, args.spi_freq)
    else:
        raise SystemExit("ERROR: --transport tcp|spi required")
    sender.connect()

    iterator = open_clip(args.input)
    kind, width, height = next(iterator)
    assert kind == "HEADER"
    print(f"[replay] {args.input}: {width}x{height}")
    print(f"[replay] streaming → {args.transport} (Ctrl-C to abort)")

    speed = max(args.speed, 0.01)
    t_start = time.monotonic()
    frame_id = 0
    last_print = 0
    try:
        for rec in iterator:
            _, rel_us, packet = rec
            target = t_start + (rel_us / 1_000_000.0) / speed
            now = time.monotonic()
            if target > now:
                time.sleep(target - now)
            sender.send_packet(packet)
            frame_id += 1
            if frame_id - last_print >= 30:
                print(f"  [{(time.monotonic() - t_start):5.2f}s]  frame={frame_id:5d}")
                last_print = frame_id
    except KeyboardInterrupt:
        print("\n[replay] stopped by Ctrl-C")
    except (BrokenPipeError, ConnectionResetError):
        print("\n[replay] connection lost, stopping")
    finally:
        sender.close()

    print(f"[replay] done. {frame_id} frames sent")


# ── Inspect ───────────────────────────────────────────────────────────────────

def cmd_inspect(args):
    iterator = open_clip(args.input)
    kind, width, height = next(iterator)
    assert kind == "HEADER"

    frames = 0
    total_packet_bytes = 0
    last_rel_us = 0
    interframe_us = []
    event_counts = []

    for rec in iterator:
        _, rel_us, packet = rec
        frames += 1
        total_packet_bytes += len(packet)
        if frames > 1:
            interframe_us.append(rel_us - last_rel_us)
        last_rel_us = rel_us
        # event_count lives at bytes [6:10] of every DVS packet header.
        if len(packet) >= 10:
            event_counts.append(int.from_bytes(packet[6:10], "little"))

    duration_s = last_rel_us / 1_000_000.0
    avg_events = (sum(event_counts) / len(event_counts)) if event_counts else 0
    avg_dt_ms = (sum(interframe_us) / len(interframe_us) / 1000.0) if interframe_us else 0

    print(f"clip:        {args.input}")
    print(f"  resolution:    {width}x{height}")
    print(f"  frames:        {frames}")
    print(f"  duration:      {duration_s:.2f} s")
    if duration_s > 0:
        print(f"  fps (avg):     {frames / duration_s:.1f}")
    if event_counts:
        print(f"  events/frame:  avg={avg_events:.0f}, "
              f"min={min(event_counts)}, max={max(event_counts)}")
    if interframe_us:
        print(f"  inter-frame:   avg={avg_dt_ms:.1f} ms, "
              f"min={min(interframe_us)/1000:.1f}, "
              f"max={max(interframe_us)/1000:.1f}")
    print(f"  packet bytes:  {total_packet_bytes/1024:.1f} KB total")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Record and replay deterministic DVS event clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Capture webcam → DVS events → .dvsclip file")
    rec.add_argument("--output", required=True, metavar="FILE",
                     help="Output .dvsclip path")
    rec.add_argument("--width",  type=int, default=160)
    rec.add_argument("--height", type=int, default=120)
    rec.add_argument("--fps",    type=int, default=30)
    rec.add_argument("--threshold", type=int, default=8,
                     help="DVS frame-diff threshold (matches webcam_to_dvs default)")
    rec.add_argument("--duration", type=float, default=None,
                     help="Stop after N seconds (default: until Ctrl-C)")
    rec.add_argument("--camera",   type=int, default=0)

    rep = sub.add_parser("replay", help="Stream a clip to a DUT (TCP or SPI)")
    rep.add_argument("input", help="Input .dvsclip path")
    rep.add_argument("--transport", choices=["tcp", "spi"], required=True)
    rep.add_argument("--host", default="raspberrypi.local",
                     help="(tcp) Pi hostname or IP")
    rep.add_argument("--port", type=int, default=DVS_SERVER_PORT,
                     help="(tcp) Pi listen port")
    rep.add_argument("--ftdi-url", default="ftdi://ftdi:232h/1",
                     help="(spi) pyftdi device URL — run "
                          "`python -m pyftdi.ftdi` to list devices")
    rep.add_argument("--spi-freq", type=int, default=10_000_000,
                     help="(spi) SPI clock in Hz")
    rep.add_argument("--speed", type=float, default=1.0,
                     help="Playback speed multiplier (e.g. 0.5 = half speed, "
                          "2.0 = 2x).  Useful when the DUT can't keep up.")

    ins = sub.add_parser("inspect", help="Print stats about a clip without playing")
    ins.add_argument("input", help="Input .dvsclip path")

    args = p.parse_args()
    if args.cmd == "record":
        cmd_record(args)
    elif args.cmd == "replay":
        cmd_replay(args)
    elif args.cmd == "inspect":
        cmd_inspect(args)


if __name__ == "__main__":
    main()
