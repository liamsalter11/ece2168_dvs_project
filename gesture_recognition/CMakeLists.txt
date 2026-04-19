# @OPT_TAGS=
# @TYPE_TAGS=ml
#
# Gesture recognition firmware — E1 EVK
#
# Uses TFLite Micro (TFLM) for int8-quantized MobileNetV2-0.5 inference.
# The gesture_model.tflite from the DVS project is embedded as a byte array
# so no filesystem is needed on the target.
#
# Build targets produced:
#   gesture_recognition_fabric   → .hex / .bin  for flashing the E1 EVK
#   gesture_recognition_scalar   → .hex / .bin  (scalar RISC-V, no fabric accel)
#
# Prerequisites: cmake 3.16+, python3 (for model embedding), internet
#   access at configure time (FetchContent downloads tflite-micro + flatbuffers).

cmake_minimum_required(VERSION 3.16)
include(FetchContent)

# ── Model path ────────────────────────────────────────────────────────────────

set(DVS_MODEL_SRC
    /home/liam/Documents/ece2168_dvs_project/gesture_model.tflite
    CACHE FILEPATH "Path to pre-trained gesture_model.tflite")

if(NOT EXISTS "${DVS_MODEL_SRC}")
    message(FATAL_ERROR "gesture_model.tflite not found at ${DVS_MODEL_SRC}.\n"
            "Override with -DDVS_MODEL_SRC=<path>")
endif()

# ── Download standalone TFLite Micro ──────────────────────────────────────────
#
# We use the standalone tflite-micro repo (NOT the full tensorflow repo).
# It is much smaller and focused solely on the embedded micro runtime.
# FetchContent_Populate downloads only; we do NOT process its CMakeLists.txt
# to avoid it resetting our cross-compiler settings via project().

FetchContent_Declare(
    tflite-micro
    GIT_REPOSITORY https://github.com/tensorflow/tflite-micro.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    GIT_SUBMODULES ""
)

FetchContent_GetProperties(tflite-micro)
if(NOT tflite-micro_POPULATED)
    FetchContent_Populate(tflite-micro)
endif()

set(TFLM_ROOT ${tflite-micro_SOURCE_DIR})

# ── Download FlatBuffers (headers only) ───────────────────────────────────────
#
# tflite-micro's third_party/flatbuffers/ contains only Bazel BUILD files.
# We fetch the matching FlatBuffers release directly for its C++ headers.

FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG        v25.9.23
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    GIT_SUBMODULES ""
)

FetchContent_GetProperties(flatbuffers)
if(NOT flatbuffers_POPULATED)
    FetchContent_Populate(flatbuffers)
endif()

# ── Download gemmlowp (headers only, for fixedpoint/) ────────────────────────
#
# tensorflow/lite/kernels/internal/common.h includes fixedpoint/fixedpoint.h
# which lives in the gemmlowp repo.  We fetch the exact commit TFLM pins.

FetchContent_Declare(
    gemmlowp
    GIT_REPOSITORY https://github.com/google/gemmlowp.git
    GIT_TAG        fda83bdc38b118cc6b56753bd540caa49e570745
    GIT_SHALLOW    FALSE
    GIT_PROGRESS   TRUE
    GIT_SUBMODULES ""
)

FetchContent_GetProperties(gemmlowp)
if(NOT gemmlowp_POPULATED)
    FetchContent_Populate(gemmlowp)
endif()

# ── Download ruy (headers only, for ruy/profiler/) ───────────────────────────
#
# third_party/ruy in tflite-micro contains only a Bazel BUILD stub.
# We fetch the real ruy headers at the commit TFLM pins.

FetchContent_Declare(
    ruy
    GIT_REPOSITORY https://github.com/google/ruy.git
    GIT_TAG        54774a7a2cf85963777289193629d4bd42de4a59
    GIT_SHALLOW    FALSE
    GIT_PROGRESS   TRUE
    GIT_SUBMODULES ""
)

FetchContent_GetProperties(ruy)
if(NOT ruy_POPULATED)
    FetchContent_Populate(ruy)
endif()

# ── Build TFLite Micro static library ─────────────────────────────────────────
#
# We collect sources explicitly to avoid TFLM's own CMake resetting our
# cross-compiler. All platform-specific backends (cmsis_nn, ethos_u, xtensa,
# esp_nn, hexagon) and all test/bench/example files are excluded.
# The reference kernels give us portable C++ ops for all required ops.

file(GLOB_RECURSE TFLM_SRCS
    "${TFLM_ROOT}/tensorflow/lite/micro/*.cc"
    "${TFLM_ROOT}/tensorflow/lite/c/common.cc"
    "${TFLM_ROOT}/tensorflow/lite/core/c/common.cc"
    "${TFLM_ROOT}/tensorflow/lite/core/api/flatbuffer_conversions.cc"
    "${TFLM_ROOT}/tensorflow/lite/core/api/op_resolver.cc"
    "${TFLM_ROOT}/tensorflow/lite/core/api/tensor_utils.cc"
    "${TFLM_ROOT}/tensorflow/lite/schema/schema_utils.cc"
    # Non-micro kernel support files (quantization helpers, tensor utilities)
    "${TFLM_ROOT}/tensorflow/lite/kernels/internal/common.cc"
    "${TFLM_ROOT}/tensorflow/lite/kernels/internal/tensor_ctypes.cc"
    "${TFLM_ROOT}/tensorflow/lite/kernels/internal/portable_tensor_utils.cc"
    # kernel_util.cc excluded: includes <complex> which needs atan2l (no long
    # double on this target). Replaced by kernel_util_eff.cc below.
)

# Exclude tests, benchmarks, examples, build tools, and vendor-specific backends.
# Also exclude quantization_util.cc (uses f64 bitcast illegal on this target —
# replaced by quantization_util_effcc.cc).
foreach(_pat IN ITEMS
        "_test\\.cc$"
        "/test_[^/]+\\.cc$"
        "/testing/"
        "/integration_tests/"
        "/benchmarks?/"
        "/examples?/"
        "/tools/"
        "/python"
        "/debug_log"
        "/cmsis_nn/"
        "/ethos_u/"
        "/esp_nn/"
        "/hexagon/"
        "/ceva/"
        "/xtensa/"
        "/arc/"
        "/arc_custom/"
        "/arc_mli/"
        "/arc_emsdp/"
        "/bluepill/"
        "/chre/"
        "/cortex_m_generic/"
        "/docs/"
        "/models/"
        "/riscv32_generic/"
        "/cortex_m_corstone_300/"
        "quantization_util\\.cc$"
        "/kernels/mul"
        "/kernels/lstm"
        )
    list(FILTER TFLM_SRCS EXCLUDE REGEX "${_pat}")
endforeach()

if(NOT TFLM_SRCS)
    message(FATAL_ERROR
        "No TFLM source files found under ${TFLM_ROOT}.\n"
        "Check that FetchContent downloaded the tflite-micro repo correctly.")
endif()

# kernel_util_eff.cc replaces kernel_util.cc (which pulls in <complex>).
list(APPEND TFLM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/kernel_util_eff.cc)

add_library(tensorflow-lite-micro STATIC ${TFLM_SRCS})

target_include_directories(tensorflow-lite-micro PUBLIC
    # Root of the tflite-micro repo  — for #include "tensorflow/lite/micro/..."
    ${TFLM_ROOT}
    # FlatBuffers C++ headers (flatbuffers/flatbuffers.h)
    ${flatbuffers_SOURCE_DIR}/include
    # ruy headers (third_party/ruy in TFLM is a Bazel stub only)
    ${ruy_SOURCE_DIR}
    # gemmlowp provides fixedpoint/fixedpoint.h used by common.h
    ${gemmlowp_SOURCE_DIR}
)

target_compile_definitions(tensorflow-lite-micro PUBLIC
    TF_LITE_STATIC_MEMORY
    NDEBUG
)

# Cross-compile the library for the target with the effcc toolchain.
# Note: effcc++ does not accept -fno-exceptions/-fno-rtti; omit them.
target_compile_options(tensorflow-lite-micro PRIVATE
    --target=${EFF_PLUGIN}
    -O3
    -Wno-unused-function
    -Wno-unused-variable
    -Wno-sign-compare
)
target_link_options(tensorflow-lite-micro PRIVATE --target=${EFF_PLUGIN})

# ── Embed gesture model as a C byte array ─────────────────────────────────────

set(MODEL_CC ${CMAKE_CURRENT_BINARY_DIR}/gesture_model_data.cc)

add_custom_command(
    OUTPUT  ${MODEL_CC}
    COMMAND python3
            ${CMAKE_CURRENT_SOURCE_DIR}/embed_model.py
            ${DVS_MODEL_SRC}
            ${MODEL_CC}
    DEPENDS ${DVS_MODEL_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/embed_model.py
    COMMENT "Embedding gesture_model.tflite as C array"
    VERBATIM
)
add_custom_target(gesture_model_embedded DEPENDS ${MODEL_CC})

# ── Firmware app ──────────────────────────────────────────────────────────────
#
# TARGETS fabric / scalar only:
#   - fabric  → uses E1x CGRA accelerator (full hardware run)
#   - scalar  → plain RISC-V core, no fabric (good for SW-only test)
# The sim target is excluded because TFLM is compiled for the hardware ISA;
# a separate sim-compatible TFLM build would be needed for cocotb simulation.

add_eff_app(gesture_recognition
    TYPE exe
    ARCHS all
    TARGETS fabric scalar
    SOURCE
        main.cpp
        gesture_kernel.cpp
        quantization_util_effcc.cc
        debug_log_eff.cc
        ${MODEL_CC}
)

foreach(sub IN ITEMS fabric scalar)
    if(TARGET gesture_recognition_${sub})
        add_dependencies(gesture_recognition_${sub} gesture_model_embedded)

        target_include_directories(gesture_recognition_${sub} PRIVATE
            ${TFLM_ROOT}
            ${flatbuffers_SOURCE_DIR}/include
            ${TFLM_ROOT}/third_party/ruy
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}
        )

        target_link_libraries(gesture_recognition_${sub}
            tensorflow-lite-micro
            m
        )
    endif()
endforeach()
