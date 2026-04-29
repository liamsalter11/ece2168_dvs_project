# DVS Gesture Recognition

Emulated event-driven gesture recognition system. A Python script on Windows converts webcam frames into DVS-style (Dynamic Vision Sensor) temporal contrast events and streams them over TCP. Two hardware DUTs receive the events, run a TFLite gesture model, and print results to their consoles.

## System Overview

```
┌──────────────────────────┐        TCP / port 9473       ┌──────────────────────────────┐
│  Windows (webcam sender) │ ─── DVS event packets ──▶   │  DUT receiver                │
│                          │      (x, y, pol, ts)         │                              │
│  webcam → frame diff     │                              │  deserialize frames          │
│  → DVS events → TCP      │                              │  → activity heatmap (160×120)│
└──────────────────────────┘                              │  → blob detect + resize      │
                                                          │  → TFLite inference          │
                                                          │  → gesture label (console)   │
                                                          └──────────────────────────────┘
```

### DUT comparison

| | Raspberry Pi 4 | Efficient Computer E1x EVK |
|---|---|---|
| Transport | TCP over Ethernet (`eth0`) | SPI_1 (slave, `PINMUX_1`) — laptop drives clock via FT232H |
| Inference | TFLite v2.18.0 interpreter | `eff-import` compiled MLIR, CGRA-accelerated |
| Output | `printf` to SSH/serial console | `printf` via STDIO_UART |
| Build system | CMake + FetchContent | CMake + effcc SDK |

Shared code: `dut/common/gesture_kernel.cpp` and `dut/common/protocol.h` compile identically on both targets.

## Recognized Gestures

| Label | Description |
|---|---|
| `palm` | Open hand, all fingers spread |
| `fist` | Closed hand |
| `one` | Single finger pointing |
| `peace` | Two fingers (V sign) |
| `thumb_up` | Thumbs up |

---

## Repository Layout

```
windows/                  Python sender scripts (run on Windows)
dut/
  common/                 Shared kernel + protocol (compiled on both DUTs)
    protocol.h            Wire format, gesture enums
    gesture_kernel.h/cpp  DVS accumulation + blob detect + inference pipeline
  rpi4/                   Raspberry Pi 4 receiver
    main.cpp              TCP transport + main loop
    tflite_backend.cpp    gesture_run_inference() via TFLite v2.18.0
    CMakeLists.txt
  e1x/                    E1x EVK firmware
    main.cpp              UART transport + main loop
    CMakeLists.txt        eff-import model compilation + effcc build
    kernel_util_eff.cc    Replaces <complex>-using TFLM file (no atan2l on effcc)
    quantization_util_effcc.cc  Replaces f64-bitcast TFLM file
    debug_log_eff.cc      Routes TFLM DebugLog through eff_uart_printf
    compat.c              Software fmaf() for targets without hardware FMA
    embed_model.py        Converts .tflite to C byte array (legacy TFLM path)
```

---

## Step 1 — Collect training data (Windows)

```bash
cd windows
pip install -r requirements.txt

# Record all five gestures in one session (SPACE = start/stop, Q = quit)
python collect_gesture_data.py --all

# Or one gesture at a time:
python collect_gesture_data.py --gesture palm
python collect_gesture_data.py --gesture fist
python collect_gesture_data.py --gesture one
python collect_gesture_data.py --gesture peace
python collect_gesture_data.py --gesture thumb_up
```

Aim for **300 frames per gesture**. Keep your hand moving — DVS only responds to motion.

## Step 2 — Train the model (Windows or WSL)

```bash
python train_gesture_model.py --data_dir data/
```

Outputs `gesture_model.tflite` (~955 KB, int8 MobileNetV2-0.5) and `gesture_labels.txt` in the project root.

> If training data is under OneDrive, use the full path:
> ```bash
> python train_gesture_model.py --data_dir /mnt/c/Users/<user>/OneDrive/.../data/
> ```

---

## Step 3 — Build and run on Raspberry Pi 4

### Build

Run **on the Pi** (native build):

```bash
cd dut/rpi4
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

The first configure downloads ~400 MB of TFLite v2.18.0 source via FetchContent. Subsequent builds are fast.

> **Note:** XNNPACK is disabled (`TFLITE_ENABLE_XNNPACK=OFF`) in the CMakeLists because it pulls in the `kleidiai` ARM optimization library which requires a separate download. This has no effect on inference correctness.

To cross-compile from a Linux/WSL host instead:

```bash
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
cd dut/rpi4
mkdir build-cross && cd build-cross
cmake -DCMAKE_TOOLCHAIN_FILE=../../rpi4.cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
scp gesture_receiver ../../gesture_model.tflite pi@<PI_IP>:~/
```

### Run

**Always run from the project root** so the default model path (`gesture_model.tflite`) resolves:

```bash
cd /home/liam/ece2168_dvs_project
./dut/rpi4/build/gesture_receiver
```

Options:
```
--model PATH   TFLite model file (default: gesture_model.tflite)
--port  PORT   TCP listen port   (default: 9473)
```

Confirm TFLite loaded — you should see on stderr:
```
[tflite] Loaded gesture_model.tflite — input [1,128,128,3] type=9
[tflite] Output [5] type=9
Gesture recognition ready (TFLite v2.18.0)
[tcp] Listening on port 9473 ...
```

### Stream DVS events from Windows

```bash
# In windows/ on the laptop — Raspberry Pi 4 (TCP):
python webcam_to_dvs.py --transport tcp --host <PI_IP>
python webcam_to_dvs.py --transport tcp --host <PI_IP> --source synthetic

# E1x EVK (SPI via FT232H):
python webcam_to_dvs.py --transport spi
python webcam_to_dvs.py --transport spi --ftdi-url ftdi://ftdi:232h/1 --spi-freq 10000000
```

The FT232H is the SPI master; the E1x is configured as SPI slave. The default SPI clock is 10 MHz. `--ftdi-url` follows pyftdi device URL syntax — run `python -m pyftdi.ftdi` to list connected devices.

Gesture output on the Pi console (printed after 2 stable consecutive classifications):
```
GESTURE 1 palm 87%
GESTURE 2 fist 91%
```

---

## Step 4 — Build and run on E1x EVK

Requires the Efficient Computer SDK at `~/effcc/sdk` and the `litert_effcc` Python package (provides `eff-import`).

```bash
cd dut/e1x
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Override the model path if it is not at the default location:
```bash
cmake -DDVS_MODEL_SRC=/path/to/gesture_model.tflite ..
```

Flash `gesture_recognition_fabric.hex` (CGRA-accelerated) or `gesture_recognition_scalar.hex` (scalar RISC-V).

Connect UART_2 to the Windows laptop and run `webcam_to_dvs.py` with `--host` pointing at the EVK serial adapter. Gesture results appear on STDIO_UART.

---

## Tuning

**Python sender** (`windows/webcam_to_dvs.py`):
- `--threshold N` — brightness-change threshold (default 15; lower = more events)
- `--fps N` — target frame rate (default 30)
- `--width W --height H` — frame resolution (default 160×120)

**Gesture kernel** (`dut/common/gesture_kernel.h` — recompile after editing):
- `ACTIVITY_DECAY_FACTOR` (0.92) — how quickly old events fade
- `ACTIVITY_THRESHOLD` (40.0) — binary mask cutoff
- `ACTIVITY_INCREMENT` / `ACTIVITY_DECREMENT` (80 / 40) — energy per ON/OFF event
- `MIN_BLOB_AREA` (50 px) — ignore small noise blobs
- Blob area ≥ 200 px required before running inference
- Confidence ≥ 55% required to report a gesture label

## Wire Protocol (`dut/common/protocol.h`)

Each DVS event is 7 bytes, packed little-endian:

| Field | Type | Bytes | Description |
|---|---|---|---|
| x | uint8 | 1 | Pixel column |
| y | uint8 | 1 | Pixel row |
| polarity | uint8 | 1 | 0 = OFF (dimmer), 1 = ON (brighter) |
| timestamp | uint32 | 4 | Microseconds since stream start |

Events are batched per frame under a 10-byte header (magic `0xAE 0xD7` + frame_id + event_count). Each TCP message is prefixed with a 4-byte length.
