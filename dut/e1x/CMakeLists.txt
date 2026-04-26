# @OPT_TAGS=
# @TYPE_TAGS=ml
#
# Gesture recognition firmware — E1 EVK
#
# Uses litert_effcc (eff-import) to compile gesture_model.tflite into native
# MLIR at build time.  The compiled model function is linked directly into the
# firmware — no interpreter, no embedded byte array, full fabric acceleration.
#
# Build targets produced:
#   gesture_recognition_fabric   → .hex / .bin  for flashing the E1 EVK
#   gesture_recognition_scalar   → .hex / .bin  (scalar RISC-V, no fabric accel)
#
# Prerequisites:
#   - effcc SDK at ~/effcc/sdk
#   - litert_effcc Python package (provides eff-import)
#   - gesture_model.tflite in this directory (train with train_gesture_model.py)

cmake_minimum_required(VERSION 3.16)

project(gesture_recognition)

file(REAL_PATH "$ENV{HOME}/effcc/sdk" EFF_SDK_ROOT_DIR)
include(${EFF_SDK_ROOT_DIR}/setup_sdk.cmake)
add_subdirectory(${EFF_SDK_ROOT_DIR} sdk)

# ── Model path ────────────────────────────────────────────────────────────────

set(DVS_MODEL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/gesture_model.tflite
    CACHE FILEPATH "Path to pre-trained gesture_model.tflite")

if(NOT EXISTS "${DVS_MODEL_SRC}")
    message(FATAL_ERROR
        "gesture_model.tflite not found at ${DVS_MODEL_SRC}.\n"
        "Train the model first (from the dvs_project root):\n"
        "  pip install tensorflow\n"
        "  python train_gesture_model.py --data_dir data/ \\\n"
        "      --output gesture_recognition/gesture_model.tflite\n"
        "Or point to an existing model: cmake -DDVS_MODEL_SRC=<path>")
endif()

# ── Compile model to MLIR using eff-import ────────────────────────────────────
#
# eff-import converts gesture_model.tflite → TOSA MLIR → rip dialect MLIR.
# The resulting .mlir file is passed as a SOURCE to add_eff_app; effcc compiles
# it like any other C/MLIR translation unit.  The compiled entry point is:
#
#   int32_t gesture_run_inference(void* input, void* output);
#
# input:  int8_t[128*128*3]  (uint8 pixel − 128 for zero-point shift)
# output: int8_t[5]          (add 128, divide by 255 to get probability)

find_program(EFF_IMPORT_BIN eff-import REQUIRED
    DOC "eff-import tool from the litert_effcc Python package")

set(MODEL_MLIR ${CMAKE_CURRENT_BINARY_DIR}/gesture_inference.mlir)

add_custom_command(
    OUTPUT  ${MODEL_MLIR}
    COMMAND ${EFF_IMPORT_BIN}
            ${DVS_MODEL_SRC}
            --entrypoint-name gesture_run_inference
            -o ${MODEL_MLIR}
    DEPENDS ${DVS_MODEL_SRC}
    COMMENT "Compiling gesture_model.tflite → MLIR (eff-import)"
    VERBATIM
)
add_custom_target(gesture_model_compiled DEPENDS ${MODEL_MLIR})

# ── Firmware app ──────────────────────────────────────────────────────────────

add_eff_app(gesture_recognition
    TYPE exe
    ARCHS all
    TARGETS fabric scalar
    SOURCE
        main.cpp
        ../common/gesture_kernel.cpp
        compat.c
        ${MODEL_MLIR}
)

foreach(sub IN ITEMS fabric scalar)
    if(TARGET gesture_recognition_${sub})
        add_dependencies(gesture_recognition_${sub} gesture_model_compiled)
        target_include_directories(gesture_recognition_${sub} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../common
            ${CMAKE_CURRENT_BINARY_DIR}
        )
        # compat.c must not use LTO so fmaf() becomes a real native symbol the LTO linker can resolve.
        set_source_files_properties(../common/gesture_kernel.cpp
            PROPERTIES COMPILE_OPTIONS "-fno-lto")
        set_source_files_properties(compat.c PROPERTIES COMPILE_OPTIONS "-fno-lto")
    endif()
endforeach()
