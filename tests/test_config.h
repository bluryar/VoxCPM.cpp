#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

namespace voxcpm {
namespace test {

inline std::string default_model_path() {
#ifdef VOXCPM_DEFAULT_MODEL_PATH
    return VOXCPM_DEFAULT_MODEL_PATH;
#else
    return "../models/voxcpm1.5.gguf";
#endif
}

inline std::string default_trace_dir() {
#ifdef VOXCPM_DEFAULT_TRACE_DIR
    return VOXCPM_DEFAULT_TRACE_DIR;
#else
    return "../tests/fixtures/trace";
#endif
}

inline std::string default_voice_dir() {
#ifdef VOXCPM_DEFAULT_VOICE_DIR
    return VOXCPM_DEFAULT_VOICE_DIR;
#else
    return "../runtime/voices";
#endif
}

inline std::string get_model_path() {
    const char* env = std::getenv("VOXCPM_MODEL_PATH");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    return default_model_path();
}

inline std::string get_trace_dir() {
    const char* env = std::getenv("VOXCPM_TRACE_DIR");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    return default_trace_dir();
}

inline std::string get_trace_path(const std::string& trace_name) {
    return (std::filesystem::path(get_trace_dir()) / trace_name).string();
}

inline std::string get_voice_dir() {
    const char* env = std::getenv("VOXCPM_VOICE_DIR");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    return default_voice_dir();
}

}  // namespace test
}  // namespace voxcpm
