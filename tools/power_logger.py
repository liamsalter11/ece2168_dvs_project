#!/usr/bin/env python3
"""
power_logger.py — Capture and summarize the E1x EVK's power-measurement stream.

The EVK's on-board programmer streams continuous current/voltage/power readings
for four shunts (SYS, 1V8, VDDIO, VDDVAR) on a dedicated USB CDC serial port,
plus two always-on digital channels (AON4, AON5) for software-driven region
markers.

The script runs in two phases:
  1. Capture — drain raw serial bytes to disk as fast as possible, no parsing.
                Uses pyserial on Windows; falls back to a raw file read on Linux.
  2. Analyze — replay the captured file and print average power + integrated
                energy per rail, optionally split by AON region.

Usage examples:

  # Linux, default udev-rule symlink, 30 s capture:
  python tools/power_logger.py --duration 30

  # Windows, EVK enumerated as COM7:
  python tools\\power_logger.py --port COM7 --duration 30

  # Save the raw CSV alongside the summary, split by inference region (AON4):
  python tools/power_logger.py --duration 60 --output run.csv --region-pin aon4

  # Analyze a previously-captured CSV without touching the EVK:
  python tools/power_logger.py --analyze-only run.csv --region-pin aon4

Mark inference regions in the firmware so AON4 toggles reflect them
(see dut/e1x/main.cpp — already wired around gesture_kernel_classify):

    eff_pinmux_set(PINMUX_AON, PINMUX_GPIO);
    eff_gpio_dir_set(GPIO_AON, GPIO_PIN_4, EFF_GPIO_OUT);
    ...
    eff_gpio_set(GPIO_AON, GPIO_PIN_4);    // enter region
    gesture_kernel_classify();
    eff_gpio_clear(GPIO_AON, GPIO_PIN_4);  // exit region

CSV columns (sent on connect):
    timestamp(us),
    current_sys(mA),  voltage_sys(mV),  power_sys(mW),
    current_1v8(mA),  voltage_1v8(mV),  power_1v8(mW),
    current_io(mA),   voltage_io(mV),   power_io(mW),
    current_var(mA),  voltage_var(mV),  power_var(mW),
    aon4, aon5
"""

import argparse
import os
import signal
import sys
import time
import tempfile


DEFAULT_PORT_LINUX = "/dev/eff-power"

# Rails we care about, mapped to the CSV column suffix used in the EVK header.
RAILS = [
    ("SYS",    "sys",  "whole board (DC/DCs, LEDs, peripherals)"),
    ("1V8",    "1v8",  "chip + housekeeping MCU + on-board MRAM"),
    ("VDDIO",  "io",   "chip I/O ring + on-chip DC/DCs"),
    ("VDDVAR", "var",  "scalar core + Fabric + peripheral subsystems"),
]

# Fallback layout used when no header line is captured.  Order matches the
# format documented in the EVK Getting Started guide (Collecting Energy Data).
# The programmer emits a header line once per port-open; if another tool
# already grabbed it, our capture won't include it.  The column ORDER is
# stable across firmware versions, so positional parsing works as a fallback.
FALLBACK_COLS = [
    "timestamp(us)",
    "current_sys(mA)",  "voltage_sys(mV)",  "power_sys(mW)",
    "current_1v8(mA)",  "voltage_1v8(mV)",  "power_1v8(mW)",
    "current_io(mA)",   "voltage_io(mV)",   "power_io(mW)",
    "current_var(mA)",  "voltage_var(mV)",  "power_var(mW)",
    "aon4",             "aon5",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture the E1x EVK power-measurement CSV stream and "
                    "report average power and total energy per rail.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port", default=None,
                   help="Serial port. Linux default /dev/eff-power, "
                        "Windows requires e.g. COM7.")
    p.add_argument("--duration", type=float, default=None,
                   help="Capture window in seconds. Omit to run until Ctrl-C.")
    p.add_argument("--output", default=None,
                   help="Keep the captured CSV at this path. Without this, a "
                        "temp file is used and discarded after analysis.")
    p.add_argument("--region-pin", choices=["aon4", "aon5"], default=None,
                   help="If the firmware toggles this pin around regions of "
                        "interest, split summary stats into in-region vs "
                        "out-of-region.")
    p.add_argument("--analyze-only", default=None, metavar="FILE",
                   help="Skip capture; analyze a previously-saved CSV.")
    p.add_argument("--live", action="store_true",
                   help="Open a live matplotlib dashboard (power trace + "
                        "per-inference energy bars + running stats). Best for "
                        "demos.  Requires matplotlib + pyserial.")
    return p.parse_args()


# ── Capture ────────────────────────────────────────────────────────────────────

def capture_to_file(port, out_path, duration_s):
    """
    Drain the serial port to out_path in big chunks.  No parsing is done in
    the hot path, so we keep up with high-rate streams (the EVK can emit
    several hundred KB/s).  Returns the number of bytes written.
    """
    try:
        import serial
    except ImportError:
        if sys.platform.startswith("win"):
            raise SystemExit(
                "pyserial is required on Windows.  Install with:\n"
                "  pip install pyserial"
            )
        # On Linux we can fall back to raw file reads of /dev/ttyACM*.
        return _capture_linux_raw(port, out_path, duration_s)

    if not port:
        port = DEFAULT_PORT_LINUX if not sys.platform.startswith("win") else None
    if not port:
        raise SystemExit("ERROR: no port specified.  Pass --port COM<N> "
                         "(Windows) or --port /dev/ttyACM<N> (Linux).")

    print(f"[power] opening {port} via pyserial", file=sys.stderr)
    # Baud is irrelevant for USB CDC.  Set a short timeout so .read() returns
    # promptly when the buffer is empty (lets us check duration / Ctrl-C).
    ser = serial.Serial(port, baudrate=115200, timeout=0.05)

    interrupted = {"flag": False}
    def _on_sigint(signum, frame):
        interrupted["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    bytes_written = 0
    last_progress = time.monotonic()
    t_start = time.monotonic()

    print("[power] capturing... (Ctrl-C to stop)", file=sys.stderr)

    with open(out_path, "wb") as f:
        while not interrupted["flag"]:
            if duration_s is not None and (time.monotonic() - t_start) >= duration_s:
                break
            chunk = ser.read(65536)
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
            now = time.monotonic()
            if now - last_progress >= 1.0:
                rate = bytes_written / max(now - t_start, 1e-3) / 1024.0
                print(f"  [{now - t_start:5.1f}s]  {bytes_written/1024:.1f} KB "
                      f"({rate:.1f} KB/s)",
                      end="\r", file=sys.stderr, flush=True)
                last_progress = now

    ser.close()
    print(file=sys.stderr)
    print(f"[power] capture done: {bytes_written/1024:.1f} KB", file=sys.stderr)
    return bytes_written


def _capture_linux_raw(port, out_path, duration_s):
    """Fallback for Linux when pyserial isn't installed."""
    if not os.path.exists(port):
        raise SystemExit(
            f"ERROR: {port} does not exist.  Plug in the EVK and check that "
            f"the udev rule is installed (sudo eff-setup-udev), or pass "
            f"--port /dev/ttyACM<N>."
        )

    interrupted = {"flag": False}
    def _on_sigint(signum, frame):
        interrupted["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    print(f"[power] opening {port} as raw file (no pyserial)", file=sys.stderr)
    bytes_written = 0
    t_start = time.monotonic()
    with open(port, "rb", buffering=0) as ser, open(out_path, "wb") as f:
        # Use os.read for non-blocking-ish chunked drains.
        import fcntl
        flags = fcntl.fcntl(ser.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(ser.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
        while not interrupted["flag"]:
            if duration_s is not None and (time.monotonic() - t_start) >= duration_s:
                break
            try:
                chunk = os.read(ser.fileno(), 65536)
            except BlockingIOError:
                time.sleep(0.005)
                continue
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
    return bytes_written


# ── Analyze ────────────────────────────────────────────────────────────────────

def parse_header(line):
    """Map header column names to indices.  Tolerates whitespace drift."""
    cols = [c.strip() for c in line.split(",")]
    idx = {c: i for i, c in enumerate(cols)}
    needed = ["timestamp(us)"]
    for _, suf, _ in RAILS:
        needed += [f"current_{suf}(mA)", f"voltage_{suf}(mV)",
                   f"power_{suf}(mW)"]
    needed += ["aon4", "aon5"]
    missing = [c for c in needed if c not in idx]
    if missing:
        raise SystemExit(
            f"ERROR: power CSV header missing expected columns: {missing}\n"
            f"Got header: {line!r}"
        )
    return idx, cols


class Accumulator:
    """Trapezoidal integration of (power × dt) → energy, plus running stats."""
    __slots__ = ("samples", "total_us", "energy_uJ", "min_mW", "max_mW",
                 "_last_t", "_last_p")

    def __init__(self):
        self.samples = 0
        self.total_us = 0
        self.energy_uJ = 0.0
        self.min_mW = float("inf")
        self.max_mW = float("-inf")
        self._last_t = None
        self._last_p = None

    def add(self, t_us, p_mW):
        if p_mW < self.min_mW: self.min_mW = p_mW
        if p_mW > self.max_mW: self.max_mW = p_mW
        if self._last_t is not None:
            dt = t_us - self._last_t
            if dt > 0:
                avg = 0.5 * (p_mW + self._last_p)
                self.energy_uJ += avg * dt / 1000.0
                self.total_us += dt
        self._last_t = t_us
        self._last_p = p_mW
        self.samples += 1

    def avg_mW(self):
        if self.total_us == 0:
            return 0.0
        return self.energy_uJ * 1000.0 / self.total_us

    def energy_mJ(self):
        return self.energy_uJ / 1000.0

    def duration_s(self):
        return self.total_us / 1_000_000.0


def fmt_summary(label, acc):
    if acc.samples < 2:
        return f"  {label:<8s}  (no samples)"
    return (f"  {label:<8s}  "
            f"avg={acc.avg_mW():7.2f} mW  "
            f"min={acc.min_mW:7.2f}  max={acc.max_mW:7.2f}  "
            f"energy={acc.energy_mJ():9.3f} mJ over {acc.duration_s():.3f} s "
            f"({acc.samples} samples)")


def analyze_file(path, region_pin):
    """Stream-parse the captured CSV file and emit the summary."""
    rail_acc_all = {name: Accumulator() for name, _, _ in RAILS}
    rail_acc_in  = {name: Accumulator() for name, _, _ in RAILS}
    rail_acc_out = {name: Accumulator() for name, _, _ in RAILS}

    region_transitions = 0
    last_aon = None

    with open(path, "r", encoding="ascii", errors="replace", newline="") as f:
        # The EVK programmer emits the header line once per port-open.  If
        # another tool grabbed the port first (or the EVK was already
        # streaming when we attached), no header lands in our capture — fall
        # back to the documented column layout in that case.
        idx = None
        cols = None
        # Read leading non-data lines using readline() (avoids the for-loop's
        # internal buffer that disables tell()/seek()).
        pre_lines = []
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if "timestamp(us)" in s:
                idx, cols = parse_header(s)
                break
            # Anything that starts with a digit is a data row — header missed.
            if s and (s[0].isdigit() or s[0] == "-"):
                f.seek(pos)
                break
            pre_lines.append(s)
        if idx is None or cols is None:
            cols = FALLBACK_COLS
            idx = {c: i for i, c in enumerate(cols)}
            print(f"[power] no header line in capture — assuming documented "
                  f"{len(cols)}-column layout", file=sys.stderr)

        ts_idx = idx["timestamp(us)"]
        a4_idx = idx["aon4"]
        a5_idx = idx["aon5"]
        # Pre-resolve power column indices (avoid dict lookup in hot loop).
        pow_idx = [(name, idx[f"power_{suf}(mW)"]) for name, suf, _ in RAILS]

        while True:
            line = f.readline()
            if not line:
                break
            parts = line.rstrip("\r\n").split(",")
            if len(parts) < len(cols):
                continue
            try:
                t_us = int(parts[ts_idx])
            except ValueError:
                continue
            aon4 = parts[a4_idx].strip()
            aon5 = parts[a5_idx].strip()
            aon = aon4 if region_pin == "aon4" else aon5
            in_region = (aon == "1") if region_pin else False
            if region_pin and last_aon is not None and aon != last_aon:
                region_transitions += 1
            last_aon = aon

            for name, pi in pow_idx:
                try:
                    p_mW = float(parts[pi])
                except ValueError:
                    continue
                rail_acc_all[name].add(t_us, p_mW)
                if region_pin:
                    if in_region:
                        rail_acc_in[name].add(t_us, p_mW)
                    else:
                        rail_acc_out[name].add(t_us, p_mW)

    print()
    print("=" * 72)
    print(" Power summary")
    print("=" * 72)
    for name, _suf, desc in RAILS:
        print(f"\n{name}  ({desc})")
        print(fmt_summary("all", rail_acc_all[name]))
        if region_pin:
            print(fmt_summary("in-rgn", rail_acc_in[name]))
            print(fmt_summary("out-rgn", rail_acc_out[name]))

    if region_pin:
        print(f"\n[{region_pin}] {region_transitions} transitions detected.")
        if rail_acc_in["VDDVAR"].samples > 1:
            n_regions = max(1, region_transitions // 2)
            mean_us = rail_acc_in["VDDVAR"].total_us / n_regions
            mean_mJ = rail_acc_in["VDDVAR"].energy_mJ() / n_regions
            print(f"[{region_pin}] est. {n_regions} regions; mean per region: "
                  f"{mean_us:.0f} µs, {mean_mJ:.4f} mJ (VDDVAR)")


# ── Live dashboard ────────────────────────────────────────────────────────────

def run_live(port, output_path, region_pin):
    """
    Open a matplotlib window that shows VDDVAR + SYS power as a scrolling line,
    per-inference VDDVAR energy as a bar chart, and live running stats.  The
    raw CSV stream is still written to output_path so the run can be replayed
    with --analyze-only afterwards.
    """
    try:
        import serial
    except ImportError:
        raise SystemExit("pyserial required: pip install pyserial")
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        raise SystemExit("matplotlib required for --live: pip install matplotlib")

    import threading
    import queue as _queue
    from collections import deque

    if not port:
        port = (DEFAULT_PORT_LINUX if not sys.platform.startswith("win")
                else None)
    if not port:
        raise SystemExit("ERROR: --port required on Windows (e.g. --port COM7)")

    print(f"[power] live dashboard on {port}", file=sys.stderr)
    ser = serial.Serial(port, baudrate=115200, timeout=0.05)

    # Use the documented column layout — no waiting on a header that may not
    # arrive (the programmer only emits it once per port-open).
    cols = FALLBACK_COLS
    idx = {c: i for i, c in enumerate(cols)}
    ts_idx  = idx["timestamp(us)"]
    a4_idx  = idx["aon4"]
    pv_idx  = idx["power_var(mW)"]
    ps_idx  = idx["power_sys(mW)"]
    p18_idx = idx["power_1v8(mW)"]
    pio_idx = idx["power_io(mW)"]

    sample_q = _queue.Queue(maxsize=4000)
    stop_event = threading.Event()

    def reader():
        buf = b""
        with open(output_path, "wb") as raw_f:
            while not stop_event.is_set():
                chunk = ser.read(65536)
                if chunk:
                    raw_f.write(chunk)
                    buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line = buf[:nl]
                    buf = buf[nl + 1:]
                    s = line.decode("ascii", errors="replace").strip()
                    if not s or "timestamp(us)" in s:
                        continue
                    parts = s.split(",")
                    if len(parts) < len(cols):
                        continue
                    try:
                        t_us  = int(parts[ts_idx])
                        aon   = parts[a4_idx].strip()
                        p_var = float(parts[pv_idx])
                        p_sys = float(parts[ps_idx])
                        p_1v8 = float(parts[p18_idx])
                        p_io  = float(parts[pio_idx])
                    except ValueError:
                        continue
                    try:
                        sample_q.put_nowait(
                            (t_us, p_var, p_sys, p_1v8, p_io, aon))
                    except _queue.Full:
                        pass  # drop on burst

    th = threading.Thread(target=reader, daemon=True)
    th.start()

    HIST_N = 800       # ~10s at 80Hz, ~4s at 200Hz
    t_buf    = deque(maxlen=HIST_N)
    pvar_buf = deque(maxlen=HIST_N)
    psys_buf = deque(maxlen=HIST_N)

    rail_acc = {name: Accumulator() for name, _, _ in RAILS}

    last_aon = None
    region_start_e_mJ_var = None  # VDDVAR energy at region start
    region_start_e_mJ_sys = None  # SYS energy at region start
    INF_HIST = 30
    inf_energies_var = deque(maxlen=INF_HIST)
    inf_energies_sys = deque(maxlen=INF_HIST)

    t0_us = None

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5))
    fig.canvas.manager.set_window_title("E1x EVK live power")
    fig.suptitle("E1x EVK live power", fontsize=15)

    line_var, = ax1.plot([], [], lw=1.6, color="#fc9c1d",
                         label="VDDVAR (compute)")
    line_sys, = ax1.plot([], [], lw=1.0, color="#5fa8d3",
                         label="SYS (whole board)")
    ax1.set_ylabel("Power (mW)")
    ax1.set_xlabel("time (s)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Per-inference energy (mJ) — VDDVAR")
    ax2.set_xlabel(f"most recent inferences (max {INF_HIST})")
    ax2.grid(True, axis="y", alpha=0.3)

    txt_status = fig.text(
        0.012, 0.012, "waiting for samples…",
        fontsize=10, family="monospace",
        verticalalignment="bottom",
    )
    plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.13,
                        hspace=0.35)

    def update(_frame):
        nonlocal last_aon, region_start_e_mJ_var, region_start_e_mJ_sys, t0_us

        # Drain everything the reader queued since the last frame.
        drained = 0
        while True:
            try:
                t_us, p_var, p_sys, p_1v8, p_io, aon = sample_q.get_nowait()
            except _queue.Empty:
                break
            drained += 1
            if t0_us is None:
                t0_us = t_us
            rail_acc["VDDVAR"].add(t_us, p_var)
            rail_acc["SYS"].add(t_us, p_sys)
            rail_acc["1V8"].add(t_us, p_1v8)
            rail_acc["VDDIO"].add(t_us, p_io)

            t_buf.append((t_us - t0_us) / 1e6)
            pvar_buf.append(p_var)
            psys_buf.append(p_sys)

            # Per-inference region edges (default AON4 unless region_pin says 5)
            tracked_aon = aon  # AON4
            if region_pin == "aon5":
                # Caller asked for AON5; we don't have it in the tuple, but
                # the rest of the loop already ignores aon for non-aon4.
                tracked_aon = None
            if last_aon is not None and tracked_aon is not None and tracked_aon != last_aon:
                if tracked_aon == "1":
                    region_start_e_mJ_var = rail_acc["VDDVAR"].energy_mJ()
                    region_start_e_mJ_sys = rail_acc["SYS"].energy_mJ()
                else:
                    if region_start_e_mJ_var is not None:
                        inf_energies_var.append(
                            rail_acc["VDDVAR"].energy_mJ() - region_start_e_mJ_var)
                        inf_energies_sys.append(
                            rail_acc["SYS"].energy_mJ() - region_start_e_mJ_sys)
                    region_start_e_mJ_var = None
                    region_start_e_mJ_sys = None
            if tracked_aon is not None:
                last_aon = tracked_aon

        # Trace plot
        line_var.set_data(list(t_buf), list(pvar_buf))
        line_sys.set_data(list(t_buf), list(psys_buf))
        if t_buf:
            ax1.set_xlim(t_buf[0], max(t_buf[-1], t_buf[0] + 0.1))
            ymax = max(max(pvar_buf, default=1.0),
                       max(psys_buf, default=1.0)) * 1.15
            ax1.set_ylim(0, ymax)

        # Bar chart of per-inference VDDVAR energy
        ax2.clear()
        ax2.set_ylabel("Per-inference energy (mJ) — VDDVAR")
        ax2.set_xlabel(f"most recent inferences (max {INF_HIST})")
        ax2.grid(True, axis="y", alpha=0.3)
        if inf_energies_var:
            xs = list(range(len(inf_energies_var)))
            ax2.bar(xs, list(inf_energies_var), color="#fc9c1d")
            mean_e = sum(inf_energies_var) / len(inf_energies_var)
            ax2.axhline(mean_e, color="white", linestyle="--", lw=1, alpha=0.6,
                        label=f"mean = {mean_e:.4f} mJ")
            ax2.legend(loc="upper right")
            ax2.set_title(
                f"{len(inf_energies_var)} inferences shown — "
                f"mean VDDVAR = {mean_e:.4f} mJ"
            )

        # Status footer
        if rail_acc["VDDVAR"].samples > 1:
            cur_var = pvar_buf[-1] if pvar_buf else 0.0
            cur_sys = psys_buf[-1] if psys_buf else 0.0
            last_inf_var = inf_energies_var[-1] if inf_energies_var else 0.0
            txt_status.set_text(
                f"VDDVAR   now={cur_var:6.2f} mW   avg={rail_acc['VDDVAR'].avg_mW():6.2f} mW   "
                f"total={rail_acc['VDDVAR'].energy_mJ():9.2f} mJ\n"
                f"SYS      now={cur_sys:6.2f} mW   avg={rail_acc['SYS'].avg_mW():6.2f} mW   "
                f"total={rail_acc['SYS'].energy_mJ():9.2f} mJ\n"
                f"inferences detected: {len(inf_energies_var)}   "
                f"last per-inference VDDVAR energy: {last_inf_var:.4f} mJ"
            )

        return [line_var, line_sys]

    # Keep a reference to the animation so it isn't garbage-collected (matplotlib
    # silently stops drawing if the FuncAnimation goes out of scope).
    ani = animation.FuncAnimation(
        fig, update, interval=200, blit=False, cache_frame_data=False)

    try:
        plt.show()
    finally:
        stop_event.set()
        th.join(timeout=1.0)
        try:
            ser.close()
        except Exception:
            pass

    # Final summary printed to console after the window closes.
    print("\n[power] window closed — final summary:", file=sys.stderr)
    analyze_file(output_path, region_pin)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.analyze_only:
        analyze_file(args.analyze_only, args.region_pin)
        return

    # Decide where to land the capture.
    if args.output:
        capture_path = args.output
        keep = True
    else:
        tf = tempfile.NamedTemporaryFile(prefix="evk_power_",
                                         suffix=".csv", delete=False)
        capture_path = tf.name
        tf.close()
        keep = False

    try:
        if args.live:
            run_live(args.port, capture_path, args.region_pin)
            if keep:
                print(f"Raw samples written to {capture_path}", file=sys.stderr)
            return
        capture_to_file(args.port, capture_path, args.duration)
        analyze_file(capture_path, args.region_pin)
        if keep:
            print(f"\nRaw samples written to {capture_path}", file=sys.stderr)
    finally:
        if not keep and not args.live and os.path.exists(capture_path):
            try:
                os.unlink(capture_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
