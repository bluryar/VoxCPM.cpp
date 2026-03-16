# Examples CMakeLists.txt

add_executable(voxcpm_tts voxcpm_tts.cpp)
target_link_libraries(voxcpm_tts PRIVATE voxcpm)

add_executable(voxcpm_quantize voxcpm_quantize.cpp)
target_link_libraries(voxcpm_quantize PRIVATE voxcpm)

add_executable(voxcpm_imatrix voxcpm_imatrix.cpp)
target_link_libraries(voxcpm_imatrix PRIVATE voxcpm)
