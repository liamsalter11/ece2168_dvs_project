cmake_minimum_required(VERSION 3.16)
project(gesture_receiver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -O2)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(gesture_receiver
    main.cpp
    spi_receiver.cpp
    gesture_kernel.cpp
)

target_include_directories(gesture_receiver PRIVATE
    ${CMAKE_SOURCE_DIR}/..
)

# ── TFLite via FetchContent ───────────────────────────────────────────────────
include(FetchContent)

# Suppress noise from TFLite's dependency builds
set(FLATBUFFERS_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL        OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING              OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING         OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL        OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_INSTALL      OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_GPU          OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    tensorflow
    GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
    GIT_TAG        v2.18.0
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(tensorflow)

# Build the TFLite standalone cmake (lives at tensorflow/lite/CMakeLists.txt)
add_subdirectory(
    ${tensorflow_SOURCE_DIR}/tensorflow/lite
    ${CMAKE_BINARY_DIR}/tflite_build
    EXCLUDE_FROM_ALL
)

target_link_libraries(gesture_receiver PRIVATE tensorflow-lite)
target_include_directories(gesture_receiver PRIVATE
    ${tensorflow_SOURCE_DIR}
)
