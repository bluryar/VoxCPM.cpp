#ifndef VOXCPM_IMATRIX_H
#define VOXCPM_IMATRIX_H

#include "voxcpm/common.h"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

namespace voxcpm {

class VoxCPMBackend;

struct VoxCPMImatrixStats {
    std::vector<float> values;
    std::vector<int64_t> counts;
};

struct VoxCPMCalibrationSample {
    std::string text;
    std::string prompt_audio_path;
    std::string prompt_text;
};

class VoxCPMImatrixCollector {
public:
    VoxCPMImatrixCollector() = default;

    void set_chunk_size(uint32_t chunk_size);
    void set_datasets(std::vector<std::string> datasets);
    void add_dataset(const std::string& dataset);
    void mark_chunk_processed();

    void observe_graph(ggml_cgraph* graph, VoxCPMBackend& backend);
    bool load_from_file(const std::string& input_path);
    bool save_to_file(const std::string& output_path) const;
    void print_statistics(std::ostream& os, size_t top_k = 12) const;

    size_t entry_count() const { return stats_.size(); }
    int32_t chunks_count() const { return chunks_count_; }
    uint32_t chunk_size() const { return chunk_size_; }
    const std::vector<std::string>& datasets() const { return datasets_; }
    const std::unordered_map<std::string, VoxCPMImatrixStats>& stats() const { return stats_; }

private:
    std::unordered_map<std::string, VoxCPMImatrixStats> stats_;
    std::vector<std::string> datasets_;
    uint32_t chunk_size_ = 1;
    int32_t chunks_count_ = 0;
};

std::vector<VoxCPMCalibrationSample> load_text_calibration_file(const std::string& path, int max_samples);
std::vector<VoxCPMCalibrationSample> load_calibration_dataset_file(const std::string& path, int max_samples);

}  // namespace voxcpm

#endif  // VOXCPM_IMATRIX_H
