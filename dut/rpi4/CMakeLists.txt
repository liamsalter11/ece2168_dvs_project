# Gesture recognition receiver — Raspberry Pi 4
#
# Uses full TFLite v2.18.0 (same model as E1x EVK; inference back-end
# is in tflite_backend.cpp which provides gesture_run_inference() so
# dut/common/gesture_kernel.cpp compiles identically on both targets).
#
# Native build (on the Pi itself):
#   mkdir build && cd build
#   cmake -DCMAKE_BUILD_TYPE=Release ..
#   make -j$(nproc)
#
# Cross-compile from x86/WSL:
#   sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
#   mkdir build-cross && cd build-cross
#   cmake -DCMAKE_TOOLCHAIN_FILE=../../rpi4.cmake -DCMAKE_BUILD_TYPE=Release ..
#   make -j$(nproc)
#   scp gesture_receiver ../../gesture_model.tflite pi@<PI_IP>:~/
#
# Run (from the directory containing gesture_model.tflite):
#   ./gesture_receiver [--model gesture_model.tflite] [--port 9473]

cmake_minimum_required(VERSION 3.16)
project(gesture_receiver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -O2)

# ── TFLite v2.18.0 via FetchContent ──────────────────────────────────────────
# First configure downloads ~400 MB; subsequent builds are fast.

include(FetchContent)

set(FLATBUFFERS_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL      OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING            OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING       OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL      OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_INSTALL    OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_GPU        OFF CACHE BOOL "" FORCE)
# XNNPACK pulls in kleidiai (ARM download) — disable for offline/Pi builds
set(TFLITE_ENABLE_XNNPACK   OFF CACHE BOOL "" FORCE)
set(XNNPACK_ENABLE_KLEIDIAI OFF CACHE BOOL "" FORCE)

FetchContent_Declare(tensorflow
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
    GIT_TAG        v2.18.0
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(tensorflow)

add_subdirectory(
    ${tensorflow_SOURCE_DIR}/tensorflow/lite
    ${CMAKE_BINARY_DIR}/tflite_build
    EXCLUDE_FROM_ALL
)

# ── Executable ───────────────────────────────────────────────────────────────

add_executable(gesture_receiver
    main.cpp
    tflite_backend.cpp
    ../common/gesture_kernel.cpp
)

target_include_directories(gesture_receiver PRIVATE
    ../common
    ${tensorflow_SOURCE_DIR}
)

target_link_libraries(gesture_receiver PRIVATE tensorflow-lite)
