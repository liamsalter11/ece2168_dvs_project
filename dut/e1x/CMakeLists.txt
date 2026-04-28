# @OPT_TAGS=
# @TYPE_TAGS=ml
#
# Gesture recognition firmware — E1x EVK
#
# Pipeline:
#   gesture_model.tflite  →  eff-import  →  MLIR
#   MLIR + main.cpp + gesture_kernel.cpp  →  effcc  →  fabric / scalar HEX
#
# eff-import lowers the model through TOSA into the rip dialect; effcc maps
# the resulting kernels onto the CGRA fabric.  The compiled entry point is:
#
#   int32_t gesture_run_inference(void* input, void* output);
#
# input:  int8_t[128*128*3]  (uint8 pixel − 128 for zero-point shift)
# output: int8_t[5]          (add 128, divide by 255 to get probability)
#
# Build targets produced (under build/):
#   gesture_recognition/fabric/gesture_recognition.hex   — flash to BRAM
#   gesture_recognition/scalar/gesture_recognition.hex   — scalar fallback
#   spi_test/scalar/spi_test.hex                         — known-good SPI loopback
#
# Prerequisites:
#   - effcc installed at ~/effcc (release v26.1.1.0)
#   - eff_sdk at ~/effcc/sdk patched with Harrison's SPI slave additions
#   - litert_effcc Python package installed in .venv (provides eff-import)
#   - gesture_model.tflite in project root or pointed at via -DDVS_MODEL_SRC

cmake_minimum_required(VERSION 3.16)

project(gesture_recognition)

# Pin to E1x (chip on the EVK).
if(NOT DEFINED EFF_ARCH)
    set(EFF_ARCH e1x)
endif()

file(REAL_PATH "$ENV{HOME}/effcc/sdk" EFF_SDK_ROOT_DIR)
include(${EFF_SDK_ROOT_DIR}/setup_sdk.cmake)
add_subdirectory(${EFF_SDK_ROOT_DIR} sdk)

# ── Model path ────────────────────────────────────────────────────────────────

set(DVS_MODEL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gesture_model.tflite
    CACHE FILEPATH "Path to pre-trained gesture_model.tflite")

if(NOT EXISTS "${DVS_MODEL_SRC}")
    message(FATAL_ERROR
        "gesture_model.tflite not found at ${DVS_MODEL_SRC}.\n"
        "Train the model first or pass -DDVS_MODEL_SRC=<path>")
endif()

# ── Compile model to MLIR using eff-import ────────────────────────────────────

find_program(EFF_IMPORT_BIN eff-import REQUIRED
    HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin
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
    ARCHS e1x
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
        # compat.c (software fmaf) and the host-side kernel must keep their
        # native symbols visible to LTO so the linker can resolve fmaf calls
        # emitted by the compiled MLIR model.
        set_source_files_properties(../common/gesture_kernel.cpp
            PROPERTIES COMPILE_OPTIONS "-fno-lto")
        set_source_files_properties(compat.c
            PROPERTIES COMPILE_OPTIONS "-fno-lto")
    endif()
endforeach()

# ── SPI loopback test (known-good baseline) ───────────────────────────────────

add_eff_app(spi_test
    TYPE exe
    ARCHS e1x
    TARGETS scalar
    SOURCE spi_test.c
)
