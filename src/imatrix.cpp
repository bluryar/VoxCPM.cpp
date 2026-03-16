#include "voxcpm/imatrix.h"

#include "voxcpm/backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <string>

namespace voxcpm {

namespace {

static const char* const LLM_KV_IMATRIX_DATASETS = "imatrix.datasets";
static const char* const LLM_KV_IMATRIX_CHUNK_COUNT = "imatrix.chunk_count";
static const char* const LLM_KV_IMATRIX_CHUNK_SIZE = "imatrix.chunk_size";

std::string filter_tensor_name(const char* name) {
    if (!name) {
        return {};
    }

    const char* first_hash = std::strchr(name, '#');
    if (!first_hash) {
        return std::string(name);
    }

    const char* body = first_hash + 1;
    const char* second_hash = std::strchr(body, '#');
    if (!second_hash) {
        return std::string(body);
    }
    return std::string(body, second_hash - body);
}

float load_scalar_as_f32(const uint8_t* ptr, ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float*>(ptr);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(ptr));
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(ptr));
        default:
            return 0.0f;
    }
}

bool is_supported_activation_type(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

bool remove_suffix(std::string* value, const std::string& suffix) {
    if (!value || suffix.size() > value->size()) {
        return false;
    }
    if (value->compare(value->size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
    }
    value->erase(value->size() - suffix.size());
    return true;
}

int load_legacy_imatrix(const std::string& input_path,
                        std::unordered_map<std::string, VoxCPMImatrixStats>* out_stats,
                        std::vector<std::string>* out_datasets,
                        uint32_t* out_chunk_size,
                        int32_t* out_chunks_count) {
    VOXCPM_ASSERT(out_stats != nullptr);
    VOXCPM_ASSERT(out_datasets != nullptr);
    VOXCPM_ASSERT(out_chunk_size != nullptr);
    VOXCPM_ASSERT(out_chunks_count != nullptr);

    std::ifstream in(input_path.c_str(), std::ios::binary);
    if (!in) {
        throw Error(ErrorCode::FileNotFound, "failed to open imatrix file: " + input_path);
    }

    int n_entries = 0;
    in.read(reinterpret_cast<char*>(&n_entries), sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        throw Error(ErrorCode::InvalidFormat, "no imatrix data in file: " + input_path);
    }

    out_stats->clear();
    out_datasets->clear();
    *out_chunk_size = 1;
    *out_chunks_count = 0;

    for (int i = 0; i < n_entries; ++i) {
        int len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (in.fail() || len < 1) {
            throw Error(ErrorCode::InvalidFormat, "failed reading imatrix entry name length");
        }

        std::vector<char> name_bytes(static_cast<size_t>(len) + 1, 0);
        in.read(name_bytes.data(), len);
        if (in.fail()) {
            throw Error(ErrorCode::InvalidFormat, "failed reading imatrix entry name");
        }

        const std::string name(name_bytes.data());
        int ncall = 0;
        int nval = 0;
        in.read(reinterpret_cast<char*>(&ncall), sizeof(ncall));
        in.read(reinterpret_cast<char*>(&nval), sizeof(nval));
        if (in.fail() || nval < 1) {
            throw Error(ErrorCode::InvalidFormat, "failed reading imatrix entry size for " + name);
        }

        VoxCPMImatrixStats& stat = (*out_stats)[name];
        stat.values.resize(static_cast<size_t>(nval));
        stat.counts.assign(1, std::max(0, ncall));
        in.read(reinterpret_cast<char*>(stat.values.data()), nval * sizeof(float));
        if (in.fail()) {
            throw Error(ErrorCode::InvalidFormat, "failed reading imatrix entry payload for " + name);
        }
    }

    int chunks_count = 0;
    if (in.peek() != EOF) {
        in.read(reinterpret_cast<char*>(&chunks_count), sizeof(chunks_count));
        int dataset_len = 0;
        in.read(reinterpret_cast<char*>(&dataset_len), sizeof(dataset_len));
        if (!in.fail() && dataset_len > 0) {
            std::vector<char> dataset_bytes(static_cast<size_t>(dataset_len));
            in.read(dataset_bytes.data(), dataset_len);
            if (!in.fail()) {
                out_datasets->emplace_back(dataset_bytes.begin(), dataset_bytes.end());
            }
        }
    }

    *out_chunks_count = chunks_count;
    return chunks_count;
}

struct RankedTensorStats {
    std::string name;
    int64_t mats = 0;
    int64_t total_calls = 0;
    int64_t values_per_mat = 0;
    double total_sqr = 0.0;
    double mean_sqr = 0.0;
    double peak_sqr = 0.0;
};

std::vector<std::string> split_tsv_line(const std::string& line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (start <= line.size()) {
        const size_t pos = line.find('\t', start);
        if (pos == std::string::npos) {
            fields.push_back(line.substr(start));
            break;
        }
        fields.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return fields;
}

}  // namespace

void VoxCPMImatrixCollector::set_chunk_size(uint32_t chunk_size) {
    chunk_size_ = std::max<uint32_t>(1, chunk_size);
}

void VoxCPMImatrixCollector::set_datasets(std::vector<std::string> datasets) {
    datasets_ = std::move(datasets);
}

void VoxCPMImatrixCollector::add_dataset(const std::string& dataset) {
    if (dataset.empty()) {
        return;
    }
    if (std::find(datasets_.begin(), datasets_.end(), dataset) == datasets_.end()) {
        datasets_.push_back(dataset);
    }
}

void VoxCPMImatrixCollector::mark_chunk_processed() {
    ++chunks_count_;
}

void VoxCPMImatrixCollector::observe_graph(ggml_cgraph* graph, VoxCPMBackend& backend) {
    if (!graph) {
        return;
    }

    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor* node = ggml_graph_node(graph, i);
        if (!node || node->op != GGML_OP_MUL_MAT) {
            continue;
        }

        ggml_tensor* weights = node->src[0];
        ggml_tensor* activations = node->src[1];
        if (!weights || !activations) {
            continue;
        }
        if (!is_supported_activation_type(activations->type)) {
            continue;
        }

        const std::string weight_name = filter_tensor_name(weights->name);
        if (weight_name.empty()) {
            continue;
        }

        std::vector<uint8_t> activation_bytes;
        const uint8_t* data = nullptr;
        if (activations->buffer == nullptr && activations->data != nullptr) {
            data = static_cast<const uint8_t*>(activations->data);
        } else {
            activation_bytes.resize(ggml_nbytes(activations));
            backend.tensor_get(activations, activation_bytes.data(), 0, activation_bytes.size());
            data = activation_bytes.data();
        }

        VoxCPMImatrixStats& entry = stats_[weight_name];
        const int64_t n_mat = weights->ne[2] * weights->ne[3];
        if (entry.values.empty()) {
            entry.values.assign(static_cast<size_t>(activations->ne[0] * n_mat), 0.0f);
            entry.counts.assign(1, 0);
        } else if (entry.values.size() != static_cast<size_t>(activations->ne[0] * n_mat)) {
            throw Error(
                ErrorCode::InvalidArgument,
                "inconsistent imatrix activation size for " + weight_name);
        }

        for (int64_t i3 = 0; i3 < activations->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < activations->ne[2]; ++i2) {
                const int64_t mat_id = (i3 % std::max<int64_t>(1, weights->ne[3])) * std::max<int64_t>(1, weights->ne[2]) +
                                       (i2 % std::max<int64_t>(1, weights->ne[2]));
                const int64_t mat_start = mat_id * activations->ne[0];

                for (int64_t row = 0; row < activations->ne[1]; ++row) {
                    const uint8_t* row_ptr =
                        data + row * activations->nb[1] + i2 * activations->nb[2] + i3 * activations->nb[3];
                    for (int64_t col = 0; col < activations->ne[0]; ++col) {
                        const float value = load_scalar_as_f32(row_ptr + col * activations->nb[0], activations->type);
                        entry.values[static_cast<size_t>(mat_start + col)] += value * value;
                    }
                }
            }
        }

        entry.counts[0] += std::max<int64_t>(1, ggml_nrows(activations) / std::max<int64_t>(1, n_mat));
    }
}

bool VoxCPMImatrixCollector::load_from_file(const std::string& input_path) {
    if (input_path.empty()) {
        throw Error(ErrorCode::InvalidArgument, "imatrix input path must not be empty");
    }

    stats_.clear();
    datasets_.clear();
    chunk_size_ = 1;
    chunks_count_ = 0;

    ggml_context* ctx_raw = nullptr;
    gguf_init_params params = {
        .no_alloc = false,
        .ctx = &ctx_raw,
    };
    UniqueGGUFContext gguf_ctx(gguf_init_from_file(input_path.c_str(), params));
    UniqueContext ctx(ctx_raw);

    if (!gguf_ctx || !ctx) {
        load_legacy_imatrix(input_path, &stats_, &datasets_, &chunk_size_, &chunks_count_);
        return !stats_.empty();
    }

    const int32_t n_tensors = static_cast<int32_t>(gguf_get_n_tensors(gguf_ctx.get()));
    if (n_tensors < 1) {
        throw Error(ErrorCode::InvalidFormat, "no imatrix tensor data in file: " + input_path);
    }

    const int dataset_idx = gguf_find_key(gguf_ctx.get(), LLM_KV_IMATRIX_DATASETS);
    const int chunk_count_idx = gguf_find_key(gguf_ctx.get(), LLM_KV_IMATRIX_CHUNK_COUNT);
    const int chunk_size_idx = gguf_find_key(gguf_ctx.get(), LLM_KV_IMATRIX_CHUNK_SIZE);
    if (dataset_idx < 0 || chunk_count_idx < 0 || chunk_size_idx < 0) {
        throw Error(ErrorCode::InvalidFormat, "missing imatrix metadata in file: " + input_path);
    }

    chunk_size_ = gguf_get_val_u32(gguf_ctx.get(), chunk_size_idx);
    chunks_count_ = static_cast<int32_t>(gguf_get_val_u32(gguf_ctx.get(), chunk_count_idx));

    const std::string sums_suffix = ".in_sum2";
    const std::string counts_suffix = ".counts";
    std::map<std::string, std::pair<ggml_tensor*, ggml_tensor*> > pairs;

    for (ggml_tensor* cur = ggml_get_first_tensor(ctx.get()); cur; cur = ggml_get_next_tensor(ctx.get(), cur)) {
        std::string name = cur->name;
        if (name.empty()) {
            continue;
        }
        if (remove_suffix(&name, sums_suffix)) {
            pairs[std::move(name)].first = cur;
        } else if (remove_suffix(&name, counts_suffix)) {
            pairs[std::move(name)].second = cur;
        }
    }

    for (const auto& item : pairs) {
        const std::string& name = item.first;
        const ggml_tensor* sums = item.second.first;
        const ggml_tensor* counts = item.second.second;
        if (!sums || !counts) {
            throw Error(ErrorCode::InvalidFormat, "mismatched imatrix sums/counts for tensor: " + name);
        }
        if (sums->type != GGML_TYPE_F32 || counts->type != GGML_TYPE_F32) {
            throw Error(ErrorCode::InvalidFormat, "imatrix tensors must be stored as F32: " + name);
        }
        if (counts->ne[0] != 1 || counts->ne[1] != sums->ne[1]) {
            throw Error(ErrorCode::InvalidFormat, "mismatched imatrix tensor shape for: " + name);
        }

        VoxCPMImatrixStats& stat = stats_[name];
        stat.values.resize(static_cast<size_t>(ggml_nelements(sums)));
        std::memcpy(stat.values.data(), sums->data, stat.values.size() * sizeof(float));
        stat.counts.resize(static_cast<size_t>(ggml_nelements(counts)));
        const float* count_values = static_cast<const float*>(counts->data);
        for (size_t i = 0; i < stat.counts.size(); ++i) {
            stat.counts[i] = static_cast<int64_t>(std::llround(count_values[i]));
        }
    }

    const int64_t n_datasets = static_cast<int64_t>(gguf_get_arr_n(gguf_ctx.get(), dataset_idx));
    datasets_.reserve(static_cast<size_t>(std::max<int64_t>(0, n_datasets)));
    for (int64_t i = 0; i < n_datasets; ++i) {
        datasets_.push_back(gguf_get_arr_str(gguf_ctx.get(), dataset_idx, static_cast<size_t>(i)));
    }

    return !stats_.empty();
}

bool VoxCPMImatrixCollector::save_to_file(const std::string& output_path) const {
    if (output_path.empty()) {
        throw Error(ErrorCode::InvalidArgument, "imatrix output path must not be empty");
    }
    if (stats_.empty()) {
        throw Error(ErrorCode::InvalidArgument, "no imatrix statistics collected");
    }

    std::vector<std::string> to_store;
    size_t data_size = 0;
    to_store.reserve(stats_.size());
    for (const auto& item : stats_) {
        to_store.push_back(item.first);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * item.second.values.size(), GGML_MEM_ALIGN);
        data_size += GGML_PAD(ggml_tensor_overhead() + sizeof(float) * item.second.counts.size(), GGML_MEM_ALIGN);
    }
    std::sort(to_store.begin(), to_store.end());

    ggml_init_params params = {
        .mem_size = data_size,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    UniqueContext ctx(ggml_init(params));
    if (!ctx) {
        throw Error(ErrorCode::OutOfMemory, "failed to allocate imatrix ggml context");
    }

    UniqueGGUFContext gguf_ctx(gguf_init_empty());
    if (!gguf_ctx) {
        throw Error(ErrorCode::OutOfMemory, "failed to allocate imatrix gguf context");
    }

    std::vector<const char*> dataset_ptrs;
    dataset_ptrs.reserve(datasets_.size());
    for (const std::string& dataset : datasets_) {
        dataset_ptrs.push_back(dataset.c_str());
    }

    gguf_set_val_str(gguf_ctx.get(), "general.type", "imatrix");
    if (!dataset_ptrs.empty()) {
        gguf_set_arr_str(gguf_ctx.get(), "imatrix.datasets", dataset_ptrs.data(), dataset_ptrs.size());
    }
    gguf_set_val_u32(gguf_ctx.get(), "imatrix.chunk_count", static_cast<uint32_t>(std::max(0, chunks_count_)));
    gguf_set_val_u32(gguf_ctx.get(), "imatrix.chunk_size", chunk_size_);

    for (const std::string& name : to_store) {
        const VoxCPMImatrixStats& stat = stats_.at(name);
        const int32_t nval = static_cast<int32_t>(stat.values.size());
        const int32_t nmat = static_cast<int32_t>(stat.counts.size());
        if (nval <= 0 || nmat <= 0) {
            continue;
        }

        ggml_tensor* in_sum2 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, nval / nmat, nmat);
        ggml_tensor* counts = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, nmat);
        ggml_format_name(in_sum2, "%s.in_sum2", name.c_str());
        ggml_format_name(counts, "%s.counts", name.c_str());

        std::memcpy(in_sum2->data, stat.values.data(), stat.values.size() * sizeof(float));
        for (int32_t i = 0; i < nmat; ++i) {
            static_cast<float*>(counts->data)[i] = static_cast<float>(stat.counts[static_cast<size_t>(i)]);
        }

        gguf_add_tensor(gguf_ctx.get(), in_sum2);
        gguf_add_tensor(gguf_ctx.get(), counts);
    }

    return gguf_write_to_file(gguf_ctx.get(), output_path.c_str(), false);
}

void VoxCPMImatrixCollector::print_statistics(std::ostream& os, size_t top_k) const {
    if (stats_.empty()) {
        os << "imatrix statistics: no entries\n";
        return;
    }

    size_t total_values = 0;
    int64_t total_calls = 0;
    size_t zero_count_entries = 0;
    std::vector<RankedTensorStats> ranked;
    ranked.reserve(stats_.size());

    for (const auto& item : stats_) {
        const std::string& name = item.first;
        const VoxCPMImatrixStats& stat = item.second;
        const int64_t mats = static_cast<int64_t>(std::max<size_t>(1, stat.counts.size()));
        const int64_t calls = std::accumulate(stat.counts.begin(), stat.counts.end(), int64_t{0});
        const int64_t values_per_mat = static_cast<int64_t>(stat.values.size() / static_cast<size_t>(mats));

        double total_sqr = 0.0;
        double peak_sqr = 0.0;
        for (float value : stat.values) {
            total_sqr += static_cast<double>(value);
            peak_sqr = std::max(peak_sqr, static_cast<double>(value));
        }
        const double denom = static_cast<double>(std::max<int64_t>(1, calls) * std::max<int64_t>(1, values_per_mat));
        const double mean_sqr = total_sqr / denom;

        total_values += stat.values.size();
        total_calls += calls;
        if (calls <= 0) {
            ++zero_count_entries;
        }

        ranked.push_back(RankedTensorStats{
            name,
            mats,
            calls,
            values_per_mat,
            total_sqr,
            mean_sqr,
            peak_sqr,
        });
    }

    std::sort(ranked.begin(), ranked.end(), [](const RankedTensorStats& a, const RankedTensorStats& b) {
        if (a.total_sqr != b.total_sqr) {
            return a.total_sqr > b.total_sqr;
        }
        return a.name < b.name;
    });

    os << "imatrix statistics\n";
    os << "  entries=" << stats_.size()
       << ", chunk_size=" << chunk_size_
       << ", chunk_count=" << chunks_count_
       << ", total_values=" << total_values
       << ", total_calls=" << total_calls
       << ", zero_count_entries=" << zero_count_entries << "\n";

    os << "  datasets=";
    if (datasets_.empty()) {
        os << "(none)\n";
    } else {
        for (size_t i = 0; i < datasets_.size(); ++i) {
            if (i > 0) {
                os << ", ";
            }
            os << datasets_[i];
        }
        os << "\n";
    }

    os << "  top_tensors_by_total_sqr:\n";
    const size_t n_to_show = std::min(top_k, ranked.size());
    for (size_t i = 0; i < n_to_show; ++i) {
        const RankedTensorStats& row = ranked[i];
        std::ostringstream total_sqr_ss;
        std::ostringstream mean_sqr_ss;
        std::ostringstream peak_sqr_ss;
        total_sqr_ss << std::scientific << std::setprecision(4) << row.total_sqr;
        mean_sqr_ss << std::scientific << std::setprecision(4) << row.mean_sqr;
        peak_sqr_ss << std::scientific << std::setprecision(4) << row.peak_sqr;
        os << "    [" << (i + 1) << "] " << row.name
           << " | mats=" << row.mats
           << " | values_per_mat=" << row.values_per_mat
           << " | calls=" << row.total_calls
           << " | total_sqr=" << total_sqr_ss.str()
           << " | mean_sqr=" << mean_sqr_ss.str()
           << " | peak_sqr=" << peak_sqr_ss.str() << "\n";
    }
}

std::vector<VoxCPMCalibrationSample> load_text_calibration_file(const std::string& path, int max_samples) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw Error(ErrorCode::FileNotFound, "failed to open text file: " + path);
    }

    std::vector<VoxCPMCalibrationSample> samples;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        samples.push_back(VoxCPMCalibrationSample{line, {}, {}});
        if (max_samples > 0 && static_cast<int>(samples.size()) >= max_samples) {
            break;
        }
    }

    if (samples.empty()) {
        throw Error(ErrorCode::InvalidArgument, "no usable text samples found in " + path);
    }
    return samples;
}

std::vector<VoxCPMCalibrationSample> load_calibration_dataset_file(const std::string& path, int max_samples) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw Error(ErrorCode::FileNotFound, "failed to open dataset file: " + path);
    }

    std::vector<VoxCPMCalibrationSample> samples;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }

        const std::vector<std::string> fields = split_tsv_line(line);
        if (fields.size() != 1 && fields.size() != 3) {
            throw Error(
                ErrorCode::InvalidFormat,
                "invalid dataset row at line " + std::to_string(line_number) + ": expected 1 or 3 TSV fields");
        }
        if (fields[0].empty()) {
            throw Error(
                ErrorCode::InvalidFormat,
                "invalid dataset row at line " + std::to_string(line_number) + ": text must not be empty");
        }

        VoxCPMCalibrationSample sample;
        sample.text = fields[0];
        if (fields.size() == 3) {
            sample.prompt_text = fields[1];
            sample.prompt_audio_path = fields[2];
            if (sample.prompt_text.empty() || sample.prompt_audio_path.empty()) {
                throw Error(
                    ErrorCode::InvalidFormat,
                    "invalid dataset row at line " + std::to_string(line_number) +
                        ": prompt_text and prompt_audio must both be present");
            }
            if (!std::filesystem::exists(sample.prompt_audio_path)) {
                throw Error(
                    ErrorCode::FileNotFound,
                    "dataset prompt audio does not exist at line " + std::to_string(line_number) +
                        ": " + sample.prompt_audio_path);
            }
        }

        samples.push_back(std::move(sample));
        if (max_samples > 0 && static_cast<int>(samples.size()) >= max_samples) {
            break;
        }
    }

    if (samples.empty()) {
        throw Error(ErrorCode::InvalidArgument, "no usable dataset samples found in " + path);
    }
    return samples;
}

}  // namespace voxcpm
