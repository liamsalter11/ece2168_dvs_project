#!/usr/bin/env python3
"""
embed_model.py <input.tflite> <output.cc>

Converts a TFLite flatbuffer into a C source file containing the model
as a const uint8_t array, suitable for TFLite Micro (no file I/O needed).
"""

import sys

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input.tflite> <output.cc>", file=sys.stderr)
    sys.exit(1)

src = sys.argv[1]
dst = sys.argv[2]

with open(src, "rb") as f:
    data = f.read()

with open(dst, "w") as out:
    out.write("// Auto-generated from: {}\n".format(src))
    out.write("// Do not edit.\n")
    out.write("#include <cstdint>\n\n")
    out.write("extern const uint8_t gesture_model_tflite[] = {\n")
    for i in range(0, len(data), 16):
        chunk = data[i : i + 16]
        out.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
    out.write("};\n\n")
    out.write(f"extern const unsigned int gesture_model_tflite_len = {len(data)}u;\n")
