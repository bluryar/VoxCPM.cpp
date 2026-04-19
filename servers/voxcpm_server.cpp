#include "voxcpm/audio_io.h"
#include "voxcpm/server_common.h"

#include "httplib.h"
#include <nlohmann/json.hpp>

#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace voxcpm {
namespace {

using json = nlohmann::json;

struct Options {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_path;
    std::string model_name;
    std::string voice_dir;
    std::string api_key;
    BackendType backend = BackendType::CPU;
    int threads = 4;
    int max_queue = 8;
    int output_sample_rate = 0;
    bool disable_auth = false;
};

struct RequestContext {
    std::string voice_id;
    std::string input;
    AudioResponseFormat format = AudioResponseFormat::Mp3;
    double speed = 1.0;
    bool sse = false;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

BackendType parse_backend_type(const std::string& value) {
    if (value == "cpu") return BackendType::CPU;
    if (value == "cuda") return BackendType::CUDA;
    if (value == "vulkan") return BackendType::Vulkan;
    if (value == "auto") return BackendType::Auto;
    fail("Unsupported backend: " + value);
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = std::stoi(require_value("--port"));
        } else if (arg == "--model-path") {
            options.model_path = require_value("--model-path");
        } else if (arg == "--model-name") {
            options.model_name = require_value("--model-name");
        } else if (arg == "--voice-dir") {
            options.voice_dir = require_value("--voice-dir");
        } else if (arg == "--backend") {
            options.backend = parse_backend_type(require_value("--backend"));
        } else if (arg == "--threads") {
            options.threads = std::stoi(require_value("--threads"));
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--disable-auth") {
            options.disable_auth = true;
        } else if (arg == "--max-queue") {
            options.max_queue = std::stoi(require_value("--max-queue"));
        } else if (arg == "--output-sample-rate") {
            options.output_sample_rate = std::stoi(require_value("--output-sample-rate"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: voxcpm-server --model-path MODEL.gguf --model-name NAME --voice-dir DIR [options]\n"
                << "Options:\n"
                << "  --host HOST           Default: 127.0.0.1\n"
                << "  --port PORT           Default: 8080\n"
                << "  --backend TYPE        cpu|cuda|vulkan|auto\n"
                << "  --threads N           Default: 4\n"
                << "  --max-queue N         Default: 8\n"
                << "  --output-sample-rate HZ  Optional output resample rate before encoding\n"
                << "  --api-key KEY         Required unless --disable-auth\n"
                << "  --disable-auth\n";
            std::exit(0);
        } else {
            fail("Unknown argument: " + arg);
        }
    }

    if (options.model_path.empty()) fail("--model-path is required");
    if (options.model_name.empty()) fail("--model-name is required");
    if (options.voice_dir.empty()) fail("--voice-dir is required");
    if (!options.disable_auth && options.api_key.empty()) fail("--api-key is required unless --disable-auth is set");
    if (options.port < 1 || options.port > 65535) fail("--port must be between 1 and 65535");
    if (options.threads < 1) fail("--threads must be >= 1");
    if (options.max_queue < 0) fail("--max-queue must be >= 0");
    if (options.output_sample_rate < 0) fail("--output-sample-rate must be >= 0");
    return options;
}

void respond_json(httplib::Response& res, int status, const json& body) {
    res.status = status;
    res.set_header("Content-Type", "application/json");
    res.set_content(body.dump(), "application/json");
}

void respond_error(httplib::Response& res,
                   int status,
                   const std::string& message,
                   const std::string& type,
                   const std::string& code) {
    respond_json(res, status, {{"error", {{"message", message}, {"type", type}, {"code", code}}}});
}

json metadata_to_json(const VoiceMetadata& metadata) {
    return {
        {"id", metadata.id},
        {"prompt_text", metadata.prompt_text},
        {"prompt_audio_length", metadata.prompt_audio_length},
        {"sample_rate", metadata.sample_rate},
        {"patch_size", metadata.patch_size},
        {"feat_dim", metadata.feat_dim},
        {"created_at", metadata.created_at},
        {"updated_at", metadata.updated_at},
    };
}

bool authorize(const Options& options, const httplib::Request& req, httplib::Response& res) {
    if (options.disable_auth) {
        return true;
    }

    const auto auth_header = req.get_header_value("Authorization");
    const std::string prefix = "Bearer ";
    if (auth_header.rfind(prefix, 0) != 0 || auth_header.substr(prefix.size()) != options.api_key) {
        respond_error(res, 401, "Invalid or missing bearer token.", "authentication_error", "invalid_api_key");
        return false;
    }
    return true;
}

RequestContext parse_request(const json& body, const Options& options) {
    RequestContext ctx;
    if (!body.contains("model") || !body["model"].is_string()) {
        fail("`model` is required and must be a string");
    }
    if (body["model"].get<std::string>() != options.model_name) {
        fail("Requested model does not match the configured server model");
    }
    if (!body.contains("input") || !body["input"].is_string()) {
        fail("`input` is required and must be a string");
    }
    ctx.input = body["input"].get<std::string>();
    if (ctx.input.empty() || ctx.input.size() > 4096) {
        fail("`input` must be between 1 and 4096 characters");
    }

    if (body.contains("instructions") && !body["instructions"].is_null()) {
        const std::string instructions = body["instructions"].is_string() ? body["instructions"].get<std::string>() : "";
        if (!instructions.empty()) {
            throw std::invalid_argument("`instructions` is not supported by VoxCPM v1");
        }
    }

    if (!body.contains("voice")) {
        fail("`voice` is required");
    }
    if (body["voice"].is_string()) {
        ctx.voice_id = body["voice"].get<std::string>();
    } else if (body["voice"].is_object() && body["voice"].contains("id") && body["voice"]["id"].is_string()) {
        ctx.voice_id = body["voice"]["id"].get<std::string>();
    } else {
        fail("`voice` must be a string or an object with an `id` field");
    }
    if (!is_valid_voice_id(ctx.voice_id)) {
        fail("`voice` must be a valid voice id");
    }

    const std::string response_format = body.value("response_format", std::string("mp3"));
    ctx.format = parse_audio_response_format(response_format);

    ctx.speed = body.value("speed", 1.0);
    if (!(ctx.speed >= 0.25 && ctx.speed <= 4.0)) {
        fail("`speed` must be between 0.25 and 4.0");
    }

    const std::string stream_format = body.value("stream_format", std::string("audio"));
    if (stream_format == "audio") {
        ctx.sse = false;
    } else if (stream_format == "sse") {
        ctx.sse = true;
    } else {
        fail("`stream_format` must be `audio` or `sse`");
    }

    return ctx;
}

int effective_output_sample_rate(const Options& options, int source_sample_rate) {
    return options.output_sample_rate > 0 ? options.output_sample_rate : source_sample_rate;
}

std::vector<float> prepare_response_waveform(std::vector<float> waveform,
                                             int source_sample_rate,
                                             int output_sample_rate,
                                             double speed) {
    if (source_sample_rate <= 0) {
        fail("Invalid source sample rate for synthesized audio");
    }
    if (output_sample_rate <= 0) {
        fail("Invalid output sample rate for synthesized audio");
    }

    if (output_sample_rate != source_sample_rate) {
        waveform = resample_audio_to_rate(waveform, source_sample_rate, output_sample_rate);
    }
    if (speed != 1.0) {
        waveform = resample_audio_linear(waveform, speed);
    }
    return waveform;
}

void ensure_voice_dir_exists(const std::string& path) {
    std::error_code ec;
    if (std::filesystem::create_directories(path, ec) || std::filesystem::exists(path)) {
        return;
    }
    fail("Failed to create --voice-dir: " + path + " (" + ec.message() + ")");
}

class BoundedSynthesisQueue {
public:
    explicit BoundedSynthesisQueue(int max_waiting)
        : max_waiting_(max_waiting) {}

    class Permit {
    public:
        explicit Permit(BoundedSynthesisQueue* queue = nullptr)
            : queue_(queue) {}

        Permit(const Permit&) = delete;
        Permit& operator=(const Permit&) = delete;

        Permit(Permit&& other) noexcept
            : queue_(other.queue_) {
            other.queue_ = nullptr;
        }

        Permit& operator=(Permit&& other) noexcept {
            if (this != &other) {
                release();
                queue_ = other.queue_;
                other.queue_ = nullptr;
            }
            return *this;
        }

        ~Permit() {
            release();
        }

    private:
        void release() {
            if (queue_ != nullptr) {
                queue_->release();
                queue_ = nullptr;
            }
        }

        BoundedSynthesisQueue* queue_ = nullptr;
    };

    std::optional<Permit> acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (active_ && waiting_ >= max_waiting_) {
            return std::nullopt;
        }
        if (active_) {
            ++waiting_;
            cv_.wait(lock, [&]() { return !active_; });
            --waiting_;
        }
        active_ = true;
        return Permit(this);
    }

private:
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ = false;
        cv_.notify_one();
    }

    int max_waiting_ = 0;
    bool active_ = false;
    int waiting_ = 0;
    std::mutex mutex_;
    std::condition_variable cv_;
};

}  // namespace
}  // namespace voxcpm

int main(int argc, char** argv) {
    using namespace voxcpm;

    try {
        const Options options = parse_args(argc, argv);
        ensure_voice_dir_exists(options.voice_dir);
        VoxCPMServiceCore core(options.model_path, options.backend, options.threads);
        core.load();
        VoiceStore voice_store(options.voice_dir);
        BoundedSynthesisQueue queue(options.max_queue);

        httplib::Server server;
        server.new_task_queue = [] { return new httplib::ThreadPool(8); };

        server.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
            respond_json(res, 200, {{"status", "ok"}});
        });

        server.Post("/v1/voices", [&](const httplib::Request& req, httplib::Response& res) {
            if (!authorize(options, req, res)) {
                return;
            }

            try {
                if (!req.is_multipart_form_data()) {
                    fail("Expected multipart/form-data");
                }
                if (!req.form.has_field("id")) {
                    fail("Missing multipart field `id`");
                }
                if (!req.form.has_field("text")) {
                    fail("Missing multipart field `text`");
                }
                if (!req.form.has_file("audio")) {
                    fail("Missing multipart file `audio`");
                }

                const std::string id = req.form.get_field("id");
                const std::string text = req.form.get_field("text");
                if (!is_valid_voice_id(id)) {
                    fail("Invalid voice id");
                }
                if (voice_store.has_voice(id)) {
                    respond_error(res, 409, "Voice id already exists.", "invalid_request_error", "voice_exists");
                    return;
                }

                const auto file = req.form.get_file("audio");
                const DecodedAudio decoded = decode_audio_from_memory(file.content.data(), file.content.size());
                const std::vector<float> mono = convert_to_mono(decoded);
                PromptFeatures features = core.encode_prompt_audio(id, text, mono, decoded.sample_rate);
                voice_store.save_voice(features);
                respond_json(res, 201, metadata_to_json(voice_store.load_metadata(id)));
            } catch (const std::exception& e) {
                respond_error(res, 400, e.what(), "invalid_request_error", "bad_request");
            }
        });

        server.Get(R"(/v1/voices/([A-Za-z0-9._-]+))", [&](const httplib::Request& req, httplib::Response& res) {
            if (!authorize(options, req, res)) {
                return;
            }

            try {
                const std::string id = req.matches[1];
                respond_json(res, 200, metadata_to_json(voice_store.load_metadata(id)));
            } catch (const std::exception&) {
                respond_error(res, 404, "Voice id not found.", "invalid_request_error", "voice_not_found");
            }
        });

        server.Delete(R"(/v1/voices/([A-Za-z0-9._-]+))", [&](const httplib::Request& req, httplib::Response& res) {
            if (!authorize(options, req, res)) {
                return;
            }

            try {
                const std::string id = req.matches[1];
                voice_store.delete_voice(id);
                respond_json(res, 200, {{"id", id}, {"deleted", true}});
            } catch (const std::exception&) {
                respond_error(res, 404, "Voice id not found.", "invalid_request_error", "voice_not_found");
            }
        });

        server.Post("/v1/audio/speech", [&](const httplib::Request& req, httplib::Response& res) {
            if (!authorize(options, req, res)) {
                return;
            }

            try {
                const json body = json::parse(req.body);
                const RequestContext ctx = parse_request(body, options);
                const int response_sample_rate = effective_output_sample_rate(options, core.sample_rate());

                if (!audio_response_format_supported(ctx.format)) {
                    respond_error(res,
                                  501,
                                  std::string("this build does not include ") + audio_response_format_name(ctx.format) +
                                      " encoder support",
                                  "server_error",
                                  "not_implemented");
                    return;
                }

                const auto permit = queue.acquire();
                if (!permit.has_value()) {
                    respond_error(res, 503, "Synthesis queue is full.", "server_error", "queue_full");
                    return;
                }

                PromptFeatures prompt;
                try {
                    prompt = voice_store.load_voice(ctx.voice_id);
                } catch (const std::exception&) {
                    respond_error(res, 400, "Unknown voice id.", "invalid_request_error", "voice_not_found");
                    return;
                }

                if (ctx.sse) {
                    std::vector<std::string> events;
                    SynthesisRequest request;
                    request.text = ctx.input;
                    request.prompt = std::move(prompt);
                    request.chunk_callback = [&](const std::vector<float>& chunk_waveform) {
                        const std::vector<float> prepared = prepare_response_waveform(chunk_waveform,
                                                                                      core.sample_rate(),
                                                                                      response_sample_rate,
                                                                                      ctx.speed);
                        const std::vector<uint8_t> encoded = encode_audio(ctx.format, prepared, response_sample_rate);
                        const std::string encoded64 = base64_encode(encoded.data(), encoded.size());
                        const json payload = {
                            {"type", "audio.delta"},
                            {"delta", encoded64},
                            {"format", audio_response_format_name(ctx.format)},
                        };
                        events.push_back("event: audio.delta\ndata: " + payload.dump() + "\n\n");
                    };
                    core.synthesize(request);
                    events.push_back("event: audio.completed\ndata: {\"type\":\"audio.completed\"}\n\n");
                    res.set_chunked_content_provider(
                        "text/event-stream",
                        [events = std::move(events), index = size_t{0}](size_t, httplib::DataSink& sink) mutable {
                            if (index >= events.size()) {
                                sink.done();
                                return true;
                            }
                            sink.write(events[index].data(), events[index].size());
                            ++index;
                            return true;
                        });
                    return;
                }

                SynthesisRequest request;
                request.text = ctx.input;
                request.prompt = std::move(prompt);
                SynthesisResult result = core.synthesize(request);
                result.waveform = prepare_response_waveform(std::move(result.waveform),
                                                            result.sample_rate,
                                                            response_sample_rate,
                                                            ctx.speed);
                const std::vector<uint8_t> payload = encode_audio(ctx.format, result.waveform, response_sample_rate);
                res.set_header("Content-Type", audio_content_type(ctx.format));
                res.set_content(reinterpret_cast<const char*>(payload.data()), payload.size(), audio_content_type(ctx.format));
            } catch (const std::invalid_argument& e) {
                respond_error(res, 400, e.what(), "invalid_request_error", "unsupported_parameter");
            } catch (const std::exception& e) {
                respond_error(res, 400, e.what(), "invalid_request_error", "bad_request");
            }
        });

        std::cerr << "Listening on http://" << options.host << ":" << options.port << "\n";
        if (!server.listen(options.host, options.port)) {
            std::cerr << "Failed to start server\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
