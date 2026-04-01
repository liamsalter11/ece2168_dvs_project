# DVS Gesture Recognition — Emulated Pipeline

Emulated event-driven gesture recognition system. A Python script converts webcam
frames into DVS-style (Dynamic Vision Sensor) temporal contrast events and streams
them over TCP. A C++ receiver accumulates the events and runs a TFLite ML model to
classify hand gestures. The same C++ binary targets PC (x86), Raspberry Pi 4, and
Efficient Computer E1 EVK for energy comparison.

## Architecture

```
┌─────────────────────┐       TCP/9473        ┌──────────────────────────────┐
│  Python Sender      │ ───── DVS packets ──▶  │  C++ Receiver                │
│  (Windows)          │   (x,y,pol,timestamp)  │  (WSL / RPi4 / E1 EVK)       │
│                     │                        │                              │
│  webcam → grayscale │                        │  deserialize packets         │
│  → frame diff       │                        │  → activity heatmap (160×120)│
│  → threshold        │                        │  → binary mask + blob detect │
│  → DVS events       │                        │  → resize to 128×128×3       │
│  → pack & send      │                        │  → TFLite inference          │
└─────────────────────┘                        │  → gesture label             │
                                               └──────────────────────────────┘
```

## Recognized Gestures

| Label      | Description                        |
|------------|------------------------------------|
| `palm`     | Open hand, all fingers spread      |
| `fist`     | Closed hand                        |
| `one`      | Single finger pointing             |
| `peace`    | Two fingers (V sign)               |
| `thumb_up` | Thumbs up                          |

## DVS Event Format (`protocol.h`)

Each event is 9 bytes, packed little-endian:

| Field     | Type   | Bytes | Description                     |
|-----------|--------|-------|---------------------------------|
| x         | uint16 | 2     | Pixel column                    |
| y         | uint16 | 2     | Pixel row                       |
| polarity  | uint8  | 1     | 0=OFF (dimmer), 1=ON (brighter) |
| timestamp | uint32 | 4     | Microseconds since stream start |

Events are batched per frame inside a 10-byte header (magic + frame_id + count),
and each TCP message is prefixed with a 4-byte length.

## Quick Start

### 1. Collect training data (Windows, needs webcam)

```bash
# All gestures in one session (recommended):
python collect_gesture_data.py --all

# Or one at a time:
python collect_gesture_data.py --gesture palm
python collect_gesture_data.py --gesture fist
python collect_gesture_data.py --gesture one
python collect_gesture_data.py --gesture peace
python collect_gesture_data.py --gesture thumb_up
```

Controls: **SPACE** to start/stop recording, **Q** to quit.
Aim for 300 frames per gesture. **Keep your hand moving** — DVS only responds to motion.

### 2. Train the TFLite model (WSL, uses GPU if available)

```bash
python train_gesture_model.py --data_dir data/
```

Outputs `gesture_model.tflite` (~955 KB, int8 quantized) and `gesture_labels.txt`.

> **Note:** If training data lives under OneDrive, use the full path to avoid
> cloud-only file errors:
> ```bash
> python train_gesture_model.py --data_dir /mnt/c/Users/<user>/OneDrive/.../data/
> ```

### 3. Build the C++ receiver (WSL)

First build downloads ~400 MB of TensorFlow Lite source — subsequent builds are fast.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Build without TFLite (classical CV only, fast configure):
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_TFLITE=OFF ..
```

### 4. Run — receiver first, then sender

**Always run the receiver from the project root** so the model path resolves:

```bash
# Terminal 1 — WSL (receiver)
cd /home/<user>/ece2168
./build/gesture_receiver --model gesture_model.tflite --ascii

# Terminal 2 — Windows (sender)
python webcam_to_dvs.py                        # live webcam
python webcam_to_dvs.py --source synthetic     # no webcam needed
```

Confirm TFLite loaded — you should see on stderr:
```
[tflite] Loaded gesture_model.tflite — input [1,128,128,3] type=3
```

## Project Structure

```
ece2168/
├── protocol.h                  # Shared wire format + gesture enum
├── main.cpp                    # Entry point, ASCII visualizer, --model arg
├── spi_receiver.h/cpp          # TCP packet receiver (SPI-ready interface)
├── gesture_kernel.h/cpp        # DVS accumulation + TFLite inference pipeline
├── CMakeLists.txt              # Build config (FetchContent TFLite v2.18.0)
├── Makefile                    # Classical-CV-only quick build (no TFLite)
├── webcam_to_dvs.py            # Webcam → DVS event emulator (Python sender)
├── collect_gesture_data.py     # Training data recorder
├── train_gesture_model.py      # MobileNetV2 fine-tune → TFLite export
├── gesture_model.tflite        # Trained model (generated, not in git)
├── gesture_labels.txt          # Label index mapping (generated)
└── data/                       # Training images (not in git)
    ├── palm/
    ├── fist/
    ├── one/
    ├── peace/
    └── thumb_up/
```

## Tuning

**Python sender:**
- `--threshold N` — Brightness change threshold (default: 15). Lower = more events.
- `--fps N` — Target frame rate (default: 30).
- `--width W --height H` — Frame resolution (default: 160×120).

**C++ kernel (`gesture_kernel.h`, recompile after editing):**
- `ACTIVITY_DECAY_FACTOR` — How quickly old events fade (default: 0.92).
- `ACTIVITY_THRESHOLD` — Binary mask cutoff (default: 40.0).
- `ACTIVITY_INCREMENT` — Energy added per ON event (default: 80.0).
- `MIN_BLOB_AREA` — Ignore small noise blobs (default: 50 px).

**TFLite inference thresholds (`gesture_kernel.cpp`):**
- Blob area ≥ 200 px required before running inference (avoids blank-frame false positives).
- Confidence ≥ 55% required to report a gesture label.

## Cross-Compilation (Raspberry Pi 4)

```bash
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
mkdir build-rpi4 && cd build-rpi4
cmake -DCMAKE_TOOLCHAIN_FILE=../rpi4.cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
scp gesture_receiver gesture_model.tflite pi@<PI_IP>:~/
```
