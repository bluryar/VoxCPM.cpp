# Tests CMakeLists.txt

# Use Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.5.0.tar.gz
)
FetchContent_MakeAvailable(Catch2)

add_executable(test_audio_vae test_audio_vae.cpp)
target_link_libraries(test_audio_vae PRIVATE voxcpm Catch2::Catch2WithMain)

add_test(NAME test_audio_vae COMMAND test_audio_vae)

# Test data directory
target_compile_definitions(test_audio_vae PRIVATE
    VOXCPM_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/VoxCPM/trace"
)

# FSQ tests
add_executable(test_fsq test_fsq.cpp)
target_link_libraries(test_fsq PRIVATE voxcpm Catch2::Catch2WithMain)

add_test(NAME test_fsq COMMAND test_fsq)

# Components tests
add_executable(test_components test_components.cpp)
target_link_libraries(test_components PRIVATE voxcpm Catch2::Catch2WithMain)

add_test(NAME test_components COMMAND test_components)

# MiniCPM tests
add_executable(test_minicpm test_minicpm.cpp)
target_link_libraries(test_minicpm PRIVATE voxcpm Catch2::Catch2WithMain)

add_test(NAME test_minicpm COMMAND test_minicpm)

# LocEnc tests
add_executable(test_localenc test_localenc.cpp)
target_link_libraries(test_localenc PRIVATE voxcpm Catch2::Catch2WithMain)

add_test(NAME test_localenc COMMAND test_localenc)

# LocDiT tests
add_executable(test_locdit test_locdit.cpp)
target_link_libraries(test_locdit PRIVATE voxcpm Catch2::Catch2WithMain)

add_test(NAME test_locdit COMMAND test_locdit)

# UnifiedCFM tests
add_executable(test_unified_cfm test_unified_cfm.cpp)
target_link_libraries(test_unified_cfm PRIVATE voxcpm Catch2::Catch2WithMain)

add_test(NAME test_unified_cfm COMMAND test_unified_cfm)

# VoxCPM prefill/decode tests
add_executable(test_voxcpm test_voxcpm.cpp)
target_link_libraries(test_voxcpm PRIVATE voxcpm Catch2::Catch2WithMain)

add_test(NAME test_voxcpm COMMAND test_voxcpm)

# Tokenizer tests
add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE voxcpm Catch2::Catch2WithMain)

add_test(NAME test_tokenizer COMMAND test_tokenizer)

# Quantization tests
add_executable(test_quantize test_quantize.cpp)
target_link_libraries(test_quantize PRIVATE voxcpm Catch2::Catch2WithMain)

add_test(NAME test_quantize COMMAND test_quantize)

# imatrix tests
add_executable(test_imatrix test_imatrix.cpp)
target_link_libraries(test_imatrix PRIVATE voxcpm Catch2::Catch2WithMain)

add_test(NAME test_imatrix COMMAND test_imatrix)
