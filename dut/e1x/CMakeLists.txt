# @OPT_TAGS=
# @TYPE_TAGS=ml
#
# Gesture recognition firmware — E1 EVK
#
# The CGRA fabric is not present on this EVK unit (FABRIC_PRESENT=0).
# Inference runs via TFLite Micro on the scalar RISC-V core instead.
#
# Three source files replace problematic upstream TFLM files that the effcc
# backend cannot compile:
#   quantization_util_effcc.cc  — replaces f64-bitcast quantization_util.cc
#   kernel_util_eff.cc          — replaces <complex>-using kernel_util.cc
#   debug_log_eff.cc            — routes TFLM log output to eff_uart_printf
# These are copied over the TFLite source before the micro library is built.

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
        "Train the model first:\n"
        "  python train_gesture_model.py --data_dir data/")
endif()

# ── Embed model as byte array ─────────────────────────────────────────────────

find_program(PYTHON python3 python REQUIRED DOC "Python 3 interpreter")

set(MODEL_DATA_CC ${CMAKE_CURRENT_BINARY_DIR}/gesture_model_data.cc)
add_custom_command(
    OUTPUT  ${MODEL_DATA_CC}
    COMMAND ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/embed_model.py
            ${DVS_MODEL_SRC} ${MODEL_DATA_CC}
    DEPENDS ${DVS_MODEL_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/embed_model.py
    COMMENT "Embedding gesture_model.tflite as C byte array"
    VERBATIM
)
add_custom_target(gesture_model_embedded DEPENDS ${MODEL_DATA_CC})

# ── TFLite Micro ─────────────────────────────────────────────────────────────
# tflite-micro is a standalone repo with no cmake support. We fetch the source,
# run its Makefile to download third-party headers (flatbuffers, ruy, gemmlowp),
# and then build the library ourselves from the source files we need.

include(FetchContent)

set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING  OFF CACHE BOOL "" FORCE)

FetchContent_Declare(tflite_micro
    GIT_REPOSITORY https://github.com/tensorflow/tflite-micro.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(tflite_micro)

# Download third-party headers (flatbuffers, ruy, gemmlowp) via the TFLM
# Makefile if they are not already present.
set(TFLM_DOWNLOADS ${tflite_micro_SOURCE_DIR}/tensorflow/lite/micro/tools/make/downloads)
if(NOT EXISTS "${TFLM_DOWNLOADS}/flatbuffers")
    message(STATUS "Downloading TFLM third-party headers …")
    execute_process(COMMAND pip install numpy Pillow -q)
    execute_process(
        COMMAND ${CMAKE_MAKE_PROGRAM}
                -f tensorflow/lite/micro/tools/make/Makefile
                third_party_downloads
        WORKING_DIRECTORY ${tflite_micro_SOURCE_DIR}
        RESULT_VARIABLE _dl_result
    )
    if(NOT _dl_result EQUAL 0)
        message(FATAL_ERROR
            "TFLM third_party_downloads failed.\n"
            "Run manually inside ${tflite_micro_SOURCE_DIR}:\n"
            "  pip install numpy Pillow\n"
            "  make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads")
    endif()
endif()

# Copy our compat files over the TFLM sources that effcc cannot compile.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization_util_effcc.cc
    ${tflite_micro_SOURCE_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc
    COPYONLY
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_util_eff.cc
    ${tflite_micro_SOURCE_DIR}/tensorflow/lite/kernels/kernel_util.cc
    COPYONLY
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/debug_log_eff.cc
    ${tflite_micro_SOURCE_DIR}/tensorflow/lite/micro/micro_log.cc
    COPYONLY
)

# Core interpreter sources (exclude tests, fakes, mocks, recording helpers)
file(GLOB _tflm_micro_srcs
    "${tflite_micro_SOURCE_DIR}/tensorflow/lite/micro/*.cc")
list(FILTER _tflm_micro_srcs EXCLUDE REGEX ".*_test\\.cc$")
list(FILTER _tflm_micro_srcs EXCLUDE REGEX ".*/fake_[^/]*\\.cc$")
list(FILTER _tflm_micro_srcs EXCLUDE REGEX ".*/mock_[^/]*\\.cc$")
list(FILTER _tflm_micro_srcs EXCLUDE REGEX ".*/recording_[^/]*\\.cc$")
list(FILTER _tflm_micro_srcs EXCLUDE REGEX ".*/test_helper[^/]*\\.cc$")
list(FILTER _tflm_micro_srcs EXCLUDE REGEX ".*/testing_helper[^/]*\\.cc$")

# Arena allocator sources
file(GLOB _tflm_arena_srcs
    "${tflite_micro_SOURCE_DIR}/tensorflow/lite/micro/arena_allocator/*.cc")
list(FILTER _tflm_arena_srcs EXCLUDE REGEX ".*_test\\.cc$")
list(FILTER _tflm_arena_srcs EXCLUDE REGEX ".*/recording_[^/]*\\.cc$")

# Op kernels: only those registered in tflite_backend_eff.cpp
set(TFLM_K ${tflite_micro_SOURCE_DIR}/tensorflow/lite/micro/kernels)
set(_tflm_kernel_srcs
    ${TFLM_K}/conv.cc             ${TFLM_K}/conv_common.cc
    ${TFLM_K}/depthwise_conv.cc   ${TFLM_K}/depthwise_conv_common.cc
    ${TFLM_K}/add.cc              ${TFLM_K}/add_common.cc
    ${TFLM_K}/pooling.cc          ${TFLM_K}/pooling_common.cc
    ${TFLM_K}/fully_connected.cc  ${TFLM_K}/fully_connected_common.cc
    ${TFLM_K}/reshape.cc          ${TFLM_K}/reshape_common.cc
    ${TFLM_K}/softmax.cc          ${TFLM_K}/softmax_common.cc
    ${TFLM_K}/activations.cc      ${TFLM_K}/activations_common.cc
    ${TFLM_K}/quantize.cc         ${TFLM_K}/quantize_common.cc
    ${TFLM_K}/dequantize.cc       ${TFLM_K}/dequantize_common.cc
    ${TFLM_K}/reduce.cc           ${TFLM_K}/reduce_common.cc
    ${TFLM_K}/kernel_util.cc
)

# Internal kernel utilities (pure C++ reference implementations)
set(TFLM_I ${tflite_micro_SOURCE_DIR}/tensorflow/lite/kernels)
set(TFLM_C ${tflite_micro_SOURCE_DIR}/tensorflow/lite/core)
set(_tflm_internal_srcs
    ${TFLM_I}/kernel_util.cc
    ${TFLM_I}/internal/common.cc
    ${TFLM_I}/internal/portable_tensor_utils.cc
    ${TFLM_I}/internal/quantization_util.cc
    ${TFLM_I}/internal/runtime_shape.cc
    ${TFLM_I}/internal/tensor_ctypes.cc
    ${TFLM_I}/internal/tensor_utils.cc
    ${TFLM_C}/api/flatbuffer_conversions.cc
    ${TFLM_C}/c/common.cc
    ${tflite_micro_SOURCE_DIR}/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc
    ${tflite_micro_SOURCE_DIR}/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc
    ${tflite_micro_SOURCE_DIR}/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc
    ${tflite_micro_SOURCE_DIR}/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc
    ${tflite_micro_SOURCE_DIR}/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cc
)

add_library(tensorflow-lite-micro STATIC
    ${_tflm_micro_srcs}
    ${_tflm_arena_srcs}
    ${_tflm_kernel_srcs}
    ${_tflm_internal_srcs}
)

target_include_directories(tensorflow-lite-micro PUBLIC
    ${tflite_micro_SOURCE_DIR}
    ${TFLM_DOWNLOADS}/flatbuffers/include
    ${TFLM_DOWNLOADS}/gemmlowp
    ${TFLM_DOWNLOADS}/ruy
    ${EFF_SDK_ROOT_DIR}/include
)

target_compile_definitions(tensorflow-lite-micro PUBLIC
    TF_LITE_STATIC_MEMORY
)
target_compile_options(tensorflow-lite-micro PRIVATE -fno-lto)

# ── Firmware app ──────────────────────────────────────────────────────────────

add_eff_app(gesture_recognition
    TYPE exe
    ARCHS e1x
    TARGETS scalar
    SOURCE
        main.cpp
        ../common/gesture_kernel.cpp
        tflite_backend_eff.cpp
        compat.c
        ${MODEL_DATA_CC}
)

if(TARGET gesture_recognition_scalar)
    add_dependencies(gesture_recognition_scalar gesture_model_embedded)
    target_include_directories(gesture_recognition_scalar PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_BINARY_DIR}
        ${tflite_micro_SOURCE_DIR}
        ${TFLM_DOWNLOADS}/flatbuffers/include
    )
    target_link_libraries(gesture_recognition_scalar
        tensorflow-lite-micro
        m
    )
    set_source_files_properties(../common/gesture_kernel.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-lto")
    set_source_files_properties(tflite_backend_eff.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-lto")
    set_source_files_properties(compat.c PROPERTIES COMPILE_OPTIONS "-fno-lto")
endif()

# ── SPI loopback test (unchanged) ─────────────────────────────────────────────

add_eff_app(spi_test
    TYPE exe
    ARCHS e1x
    TARGETS scalar
    SOURCE spi_test.c
)
