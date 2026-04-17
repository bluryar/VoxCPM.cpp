cmake_minimum_required(VERSION 3.14)

project(voxcpm
    VERSION 0.1.0
    DESCRIPTION "VoxCPM TTS Inference Engine with GGML"
    LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Threads REQUIRED)

if(EMSCRIPTEN AND VOXCPM_BUILD_WASM)
    add_compile_options(
        -msimd128
        -matomics
        -mbulk-memory
        -pthread
    )
endif()

# Backend options
option(VOXCPM_VULKAN "Enable Vulkan backend support through ggml" OFF)
option(VOXCPM_CUDA "Enable CUDA backend support through ggml" OFF)
option(VOXCPM_NATIVE "Optimize ggml for the current CPU" ON)
option(VOXCPM_BUILD_EXAMPLES "Build example binaries" ON)
option(VOXCPM_BUILD_BENCHMARK "Build benchmark binaries" ON)
option(VOXCPM_BUILD_TESTS "Build tests" ON)
option(VOXCPM_BUILD_WASM "Build the Emscripten WASM target" OFF)
option(VOXCPM_ENABLE_MP3 "Enable MP3 response_format support" ON)
option(VOXCPM_ENABLE_OPUS "Enable Opus response_format support via ffmpeg fallback" ON)

# =============================================================================
# Dependencies
# =============================================================================

find_program(VOXCPM_FFMPEG_EXECUTABLE ffmpeg)
if(VOXCPM_ENABLE_OPUS AND NOT VOXCPM_FFMPEG_EXECUTABLE)
    message(WARNING "ffmpeg was not found; disabling VOXCPM_ENABLE_OPUS")
    set(VOXCPM_ENABLE_OPUS OFF)
endif()

# GGML
set(GGML_DIR ${CMAKE_SOURCE_DIR}/third_party/ggml)
set(GGML_VULKAN ${VOXCPM_VULKAN} CACHE BOOL "ggml: use Vulkan" FORCE)
set(GGML_CUDA ${VOXCPM_CUDA} CACHE BOOL "ggml: use CUDA" FORCE)
set(GGML_NATIVE ${VOXCPM_NATIVE} CACHE BOOL "ggml: optimize the build for the current system" FORCE)
add_subdirectory(${GGML_DIR} ggml)

# nlohmann_json is only required for tests / trace tooling.
if(VOXCPM_BUILD_TESTS)
    set(JSON_DIR ${CMAKE_SOURCE_DIR}/third_party/json)
    if(EXISTS ${JSON_DIR}/CMakeLists.txt)
        add_subdirectory(${JSON_DIR} json)
    else()
        include(FetchContent)
        FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
        FetchContent_MakeAvailable(json)
    endif()
endif()

# =============================================================================
# VoxCPM Library
# =============================================================================

add_library(voxcpm STATIC
    third_party/miniaudio/miniaudio.c
    src/audio_io.cpp
    src/audio-vae.cpp
    src/backend.cpp
    src/components.cpp
    src/context.cpp
    src/graph-cache.cpp
    src/fsq.cpp
    src/output.cpp
    src/imatrix.cpp
    src/localenc.cpp
    src/locdit.cpp
    src/minicpm.cpp
    src/quantize.cpp
    src/runtime.cpp
    src/server_common.cpp
    src/state.cpp
    src/tokenizer.cpp
    src/unified_cfm.cpp
    src/voxcpm.cpp
    src/weight-store.cpp
)

target_include_directories(voxcpm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${GGML_DIR}/include
        ${GGML_DIR}/src
        ${CMAKE_SOURCE_DIR}/third_party/miniaudio
        ${CMAKE_SOURCE_DIR}/third_party/json/include
)

target_link_libraries(voxcpm
    PUBLIC
        ggml
        ggml-base
        Threads::Threads
)

target_compile_definitions(voxcpm
    PUBLIC
        VOXCPM_ENABLE_MP3=$<IF:$<BOOL:${VOXCPM_ENABLE_MP3}>,1,0>
        VOXCPM_ENABLE_OPUS=$<IF:$<BOOL:${VOXCPM_ENABLE_OPUS}>,1,0>
    PRIVATE
        $<$<CONFIG:Debug>:VOXCPM_DEBUG>
)

# =============================================================================
# Examples
# =============================================================================

if(VOXCPM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(VOXCPM_BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()

# =============================================================================
# Tests
# =============================================================================

if(VOXCPM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(EMSCRIPTEN AND VOXCPM_BUILD_WASM)
    add_subdirectory(wasm)
endif()

# =============================================================================
# Install
# =============================================================================

install(TARGETS voxcpm
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/voxcpm
    DESTINATION include
)
