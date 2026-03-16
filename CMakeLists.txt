cmake_minimum_required(VERSION 3.14)

project(voxcpm
    VERSION 0.1.0
    DESCRIPTION "VoxCPM TTS Inference Engine with GGML"
    LANGUAGES CXX
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

# Backend options
option(VOXCPM_VULKAN "Enable Vulkan backend support through ggml" OFF)
option(VOXCPM_NATIVE "Optimize ggml for the current CPU" ON)

# =============================================================================
# Dependencies
# =============================================================================

# GGML
set(GGML_DIR ${CMAKE_SOURCE_DIR}/third_party/ggml)
set(GGML_VULKAN ${VOXCPM_VULKAN} CACHE BOOL "ggml: use Vulkan" FORCE)
set(GGML_NATIVE ${VOXCPM_NATIVE} CACHE BOOL "ggml: optimize the build for the current system" FORCE)
add_subdirectory(${GGML_DIR} ggml)

# nlohmann_json for trace loading
set(JSON_DIR ${CMAKE_SOURCE_DIR}/third_party/json)
if(EXISTS ${JSON_DIR}/CMakeLists.txt)
    add_subdirectory(${JSON_DIR} json)
else()
    # Fetch if not available
    include(FetchContent)
    FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
    FetchContent_MakeAvailable(json)
endif()

# =============================================================================
# VoxCPM Library
# =============================================================================

add_library(voxcpm STATIC
    src/audio-vae.cpp
    src/backend.cpp
    src/components.cpp
    src/context.cpp
    src/fsq.cpp
    src/imatrix.cpp
    src/localenc.cpp
    src/locdit.cpp
    src/minicpm.cpp
    src/quantize.cpp
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
)

target_link_libraries(voxcpm
    PUBLIC
        ggml
        ggml-base
        nlohmann_json::nlohmann_json
)

target_compile_definitions(voxcpm
    PRIVATE
        $<$<CONFIG:Debug>:VOXCPM_DEBUG>
)

# =============================================================================
# Examples
# =============================================================================

add_subdirectory(examples)
add_subdirectory(benchmark)

# =============================================================================
# Tests
# =============================================================================

option(VOXCPM_BUILD_TESTS "Build tests" ON)
if(VOXCPM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
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
