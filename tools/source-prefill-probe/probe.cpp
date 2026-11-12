#include "core/clean_exit.h"
#include "ggml.h"
#include "lighton4b_physical_q6_tiled.h"
#include "lighton4b_source_prefill_chunks.h"
#include "lighton4b_physical_native_ops.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
static void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
static std::vector<char> read(const char * path, size_t bytes) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "input open");
    std::vector<char> out(bytes);
    in.read(out.data(), bytes);
    require(size_t(in.gcount()) == bytes && in.peek() == EOF, "exact input bytes");
    return out;
}
static int run(int argc, char ** argv) {
    try {
        require(argc == 15, "weights input type width rows tokens mode output-prefix threads chunk modalities "
                            "positions gamma embedding required");
        const int type = std::stoi(argv[3]), width = std::stoi(argv[4]), rows = std::stoi(argv[5]),
                  tokens = std::stoi(argv[6]);
        const std::string mode = argv[7];
        const int chunk = std::stoi(argv[10]);
        require(chunk >= 1 && chunk <= 16, "closed chunk");
        require(std::stoi(argv[9]) == 1 || std::stoi(argv[9]) == 4, "closed threads");
        require((type == GGML_TYPE_Q6_K) && width > 0 && width <= 16384 && width % 256 == 0 && rows > 0 &&
                    rows <= 16384 && (tokens >= 1 && tokens <= 512),
                "closed tensor shape/type");
        require(mode == "source-q6" || mode == "source-plan", "closed schedule mode");
        const size_t packed = ggml_row_size(ggml_type(type), width) * size_t(rows);
        require(packed <= 64 * 1024 * 1024, "bounded quantized matrix");
        auto bytes = read(argv[1], packed), input_bytes = read(argv[2], size_t(width) * tokens * sizeof(float));
        std::vector<float> input(size_t(width) * tokens);
        std::memcpy(input.data(), input_bytes.data(), input_bytes.size());
        require(std::all_of(input.begin(), input.end(), [](float x) { return std::isfinite(x); }),
                "finite actual input");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        require(bool(backend), "CPU backend");
        ggml_backend_cpu_set_n_threads(backend.get(), std::stoi(argv[9]));
        auto * device = ggml_backend_get_device(backend.get());
        require(device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU, "actual CPU device");
        std::unique_ptr<ggml_context, decltype(&ggml_free)> weights(ggml_init({ 65536, nullptr, true }), ggml_free);
        require(bool(weights), "bounded weights context");
        auto * weight = ggml_new_tensor_2d(weights.get(), ggml_type(type), width, rows);
        ggml_set_name(weight, "authenticated.weight");
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> storage(
            ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), packed), ggml_backend_buffer_free);
        require(bool(storage), "aligned raw storage");
        void * data = ggml_backend_buffer_get_base(storage.get());
        require(data, "raw storage base");
        std::memcpy(data, bytes.data(), packed);
        weight->data = data;
        std::ifstream modality_file(argv[11], std::ios::binary | std::ios::ate);
        require(bool(modality_file) && modality_file.tellg() >= 0 && modality_file.tellg() <= 8192,
                "bounded modalities");
        modality_file.seekg(0);
        std::vector<int> modalities;
        for (int value; modality_file >> value;) {
            require(modalities.size() < size_t(tokens), "modality count bound");
            modalities.push_back(value);
        }
        require(modality_file.eof() && modalities.size() == size_t(tokens), "exact modality count");
        auto position_bytes = read(argv[12], size_t(tokens) * 3 * sizeof(float));
        std::vector<float> positions(size_t(tokens) * 3), preserved(size_t(tokens) * 3);
        std::memcpy(positions.data(), position_bytes.data(), position_bytes.size());
        std::vector<qwen35_ocr::lighton4b_prefill_chunk> plan;
        if (mode == "source-plan")
            require(qwen35_ocr::lighton4b_source_prefill_plan(modalities, plan), "generic source plan");
        else
            for (int offset = 0; offset < tokens; offset += chunk)
                plan.push_back({ offset, std::min(chunk, tokens - offset), 0, tokens });
        auto gamma_bytes = read(argv[13], size_t(width) * sizeof(float)),
             embedding_bytes = read(argv[14], size_t(width) * tokens * sizeof(float));
        std::vector<float> gamma(width), embedding(size_t(width) * tokens), normalized(size_t(width) * tokens);
        std::memcpy(gamma.data(), gamma_bytes.data(), gamma_bytes.size());
        std::memcpy(embedding.data(), embedding_bytes.data(), embedding_bytes.size());
        for (const auto & part : plan) {
            std::vector<float> value;
            require(qwen35_ocr::lighton4b_physical_rms(embedding.data() + size_t(part.offset) * width, gamma.data(),
                                                       width, part.tokens, std::stoi(argv[9]), value),
                    "actual physical input normalization");
            std::copy(value.begin(), value.end(), normalized.begin() + size_t(part.offset) * width);
        }
        std::ofstream norm_file(std::string(argv[8]) + ".normalized.f32", std::ios::binary);
        norm_file.write(reinterpret_cast<char *>(normalized.data()), normalized.size() * sizeof(float));
        require(bool(norm_file), "normalization write");
        norm_file.close();
        require(!std::memcmp(normalized.data(), input.data(), input_bytes.size()),
                "whole actual source normalization closure");
        std::ofstream plan_file(std::string(argv[8]) + ".plan.tsv");
        plan_file << "offset\ttokens\tlogical_offset\tlogical_tokens\n";
        std::vector<float> result(size_t(rows) * tokens);
        auto started = std::chrono::steady_clock::now();
        for (const auto & part : plan) {
            const int offset = part.offset, count = part.tokens;
            plan_file << offset << "\t" << count << "\t" << part.logical_offset << "\t" << part.logical_tokens << "\n";
            std::vector<float> sliced;
            require(qwen35_ocr::lighton4b_prefill_chunk_positions(positions.data(), tokens, part, sliced),
                    "original position slice");
            for (int axis = 0; axis < 3; ++axis)
                std::copy_n(sliced.data() + axis * count, count, preserved.data() + size_t(axis) * tokens + offset);
            if (count >= 8) {
                std::vector<float> candidate;
                require(qwen35_ocr::lighton4b_physical_q6_tiled_cpu(bytes.data(), packed,
                                                                    normalized.data() + size_t(offset) * width, width,
                                                                    rows, count, std::stoi(argv[9]), candidate),
                        "source-compatible Q6 helper");
                std::memcpy(result.data() + size_t(offset) * rows, candidate.data(),
                            size_t(rows) * count * sizeof(float));
                continue;
            }
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 32 * 1024 * 1024, nullptr, false }),
                                                                    ggml_free);
            require(bool(ctx), "bounded graph context");
            auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, count);
            std::memcpy(x->data, normalized.data() + size_t(offset) * width, size_t(count) * width * sizeof(float));
            auto * y = ggml_mul_mat(ctx.get(), weight, x);
            std::fill_n(static_cast<float *>(y->data), size_t(rows) * count, std::numeric_limits<float>::quiet_NaN());
            ggml_mul_mat_set_prec(y, GGML_PREC_F32);
            require(ggml_backend_dev_supports_op(device, y), "actual CPU supports selected buffer operation");
            auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
            ggml_build_forward_expand(graph, y);
            require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "selected matmul compute");
            auto * values = static_cast<float *>(y->data);
            require(std::all_of(values, values + size_t(rows) * count, [](float v) { return std::isfinite(v); }),
                    "finite projection output");
            std::memcpy(result.data() + size_t(offset) * rows, values, size_t(rows) * count * sizeof(float));
        }
        require(bool(plan_file), "plan write");
        plan_file.close();
        require(!std::memcmp(positions.data(), preserved.data(), position_bytes.size()),
                "all original position bits preserved");
        std::ofstream position_file(std::string(argv[8]) + ".positions.f32", std::ios::binary);
        position_file.write(reinterpret_cast<char *>(preserved.data()), position_bytes.size());
        require(bool(position_file), "positions write");
        position_file.close();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        auto * values = result.data();
        std::ofstream output(std::string(argv[8]) + ".f32", std::ios::binary);
        output.write(reinterpret_cast<char *>(values), size_t(rows) * tokens * sizeof(float));
        require(bool(output), "projection output write");
        output.close();
        auto * buffer = weight->buffer;
        std::ofstream meta(std::string(argv[8]) + ".json");
        meta << "{\"mode\":" << std::quoted(mode) << ",\"type\":" << type << ",\"width\":" << width
             << ",\"rows\":" << rows << ",\"tokens\":" << tokens << ",\"chunk\":" << chunk
             << ",\"threads\":" << std::stoi(argv[9]) << ",\"quantized_bytes\":" << packed
             << ",\"buffer\":" << std::quoted(buffer ? ggml_backend_buffer_name(buffer) : "<none>")
             << ",\"buffer_type\":"
             << std::quoted(buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer)) : "<none>")
             << ",\"extra\":" << (weight->extra ? "true" : "false")
             << ",\"host\":" << (buffer && ggml_backend_buffer_is_host(buffer) ? "true" : "false")
             << ",\"operator_seconds_including_plan_io\":" << std::setprecision(17) << seconds << "}\n";
        require(bool(meta), "metadata write");
        std::cout << "SOURCE_PREFILL_OPERATOR PASS " << mode << " " << tokens << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
