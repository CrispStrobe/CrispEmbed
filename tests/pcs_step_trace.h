// Explicit test-only PCS stage observer. Never installed or built by default.
#pragma once
#include <cerrno>
#include <stdexcept>
#include <utility>
#include "crispembed_diff.h"

struct pcs_step_trace {
    bool enabled = core_env::on("PCS_STEP_CAPTURE");
    ggml_context * context;
    int rows;
    crispembed_diff::Ref reference;
    bool replay = core_env::on("PCS_STEP_REPLAY");
    std::vector<std::pair<std::string, ggml_tensor *>> tensors;
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> inputs;
    std::map<std::string, std::vector<float>> cpu_values;
    std::map<std::string, int> cpu_widths;
    explicit pcs_step_trace(ggml_context * ctx, int count) : context(ctx), rows(count) {
        if (replay) {
            const char * path = std::getenv("PCS_STEP_REPLAY_REF");
            if (!enabled || !path || !reference.load(path))
                throw std::runtime_error("PCS operator replay requires capture and explicit reference");
        }
    }
    ggml_tensor * record(const std::string & name, ggml_tensor * tensor) {
        if (enabled) tensors.emplace_back(name, tensor);
        // Capture the real result, then feed the independent reference result
        // to its consumers. Thus each following operation sees reference inputs
        // rather than accumulated native error. This is a test-only replay,
        // never composed inference or decoded quality evidence.
        if (replay && tensor->type == GGML_TYPE_F32 && reference.has(name)) {
            const auto data = reference.get_f32(name);
            const auto shape = reference.shape(name);
            if (data.second != (size_t)ggml_nelements(tensor) || shape.size() > 4)
                throw std::runtime_error("PCS replay element count differs");
            for (size_t dim = 0; dim < 4; ++dim)
                if (tensor->ne[dim] != (dim < shape.size() ? shape[dim] : 1))
                    throw std::runtime_error("PCS replay tensor axes differ");
            auto * replacement = ggml_dup_tensor(context, tensor);
            ggml_set_input(replacement);
            inputs.emplace_back(replacement, std::vector<float>(data.first, data.first + data.second));
            return replacement;
        }
        return tensor;
    }
    void set_inputs() {
        for (auto & item : inputs)
            ggml_backend_tensor_set(item.first, item.second.data(), 0, item.second.size() * sizeof(float));
    }
    float scalar(const std::string & name, int row, int column, int width, float value) {
        if (!enabled) return value;
        if (row < 0 || row >= rows || column < 0 || column >= width || width > 4096)
            throw std::runtime_error("PCS CPU capture coordinate invalid");
        auto & values = cpu_values[name];
        if (values.empty()) {
            values.resize(rows * width);
            cpu_widths[name] = width;
        }
        if (cpu_widths[name] != width) throw std::runtime_error("PCS CPU capture width changed");
        values[row * width + column] = value;
        if (replay && reference.has(name)) {
            const auto shape = reference.shape(name);
            const auto source = reference.get_f32(name);
            if (shape != std::vector<int64_t>{ width, rows } || source.second != values.size())
                throw std::runtime_error("PCS CPU replay axes differ");
            return source.first[row * width + column];
        }
        return value;
    }
    std::vector<float> cpu(const std::string & name, std::vector<float> values, int row) {
        for (size_t column = 0; column < values.size(); ++column)
            values[column] = scalar(name, row, (int)column, (int)values.size(), values[column]);
        return values;
    }
    void build(ggml_context * context, ggml_cgraph * graph) {
        if (!enabled) return;
        for (auto & item : tensors) {
            // A contiguous reshape is still a view. Marking only that view
            // OUTPUT does not retain its unmarked backing allocation after
            // the final inference consumer. A future DUP consumer holds the
            // source alive, then owns its own retained, non-view output.
            // Never return this snapshot to the inference graph.
            if (item.second->type == GGML_TYPE_F32) {
                item.second = ggml_dup(context, item.second);
                if (item.second->view_src || !ggml_is_contiguous(item.second))
                    throw std::runtime_error("PCS snapshot does not own contiguous storage");
            } else if (item.second->view_src || item.second->op != GGML_OP_NONE) {
                throw std::runtime_error("PCS integer capture must be an original input leaf");
            }
            ggml_set_name(item.second, item.first.c_str());
            ggml_set_output(item.second);
            ggml_build_forward_expand(graph, item.second);
        }
    }
    void dump() {
        if (!enabled) return;
        const char * root = std::getenv("PCS_STEP_CAPTURE_PATH");
        if (!root || !*root) throw std::runtime_error("PCS step capture path missing");
        if (tensors.size() > 512) throw std::runtime_error("PCS step inventory exceeds bound");
        const std::string directory(root);
        FILE * manifest = fopen((directory + "/manifest.tsv").c_str(), "wx");
        if (!manifest) throw std::runtime_error("PCS step manifest exists or cannot be created");
        size_t total = 0;
        for (const auto & item : tensors) {
            const auto * tensor = item.second;
            if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_I32)
                throw std::runtime_error("PCS step tensor must be F32 or I32");
            total += ggml_nbytes(tensor);
            if (ggml_nbytes(tensor) > 4 * 1024 * 1024 || total > 64 * 1024 * 1024)
                throw std::runtime_error("PCS step capture bytes exceed bound");
            std::vector<unsigned char> bytes(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
            FILE * output = fopen((directory + "/" + item.first + ".bin").c_str(), "wx");
            if (!output) throw std::runtime_error("PCS step output exists or cannot be created");
            const bool written = fwrite(bytes.data(), 1, bytes.size(), output) == bytes.size();
            const int closed = fclose(output);
            if (!written || closed) throw std::runtime_error("PCS step output write failed");
            fprintf(manifest, "%s\t%s\t%lld\t%lld\t%lld\t%lld\n", item.first.c_str(),
                    tensor->type == GGML_TYPE_I32 ? "i32" : "f32", (long long)tensor->ne[0], (long long)tensor->ne[1],
                    (long long)tensor->ne[2], (long long)tensor->ne[3]);
        }
        for (const auto & item : cpu_values) {
            FILE * output = fopen((directory + "/" + item.first + ".bin").c_str(), "wx");
            if (!output) throw std::runtime_error("PCS CPU capture output exists");
            const bool written =
                fwrite(item.second.data(), sizeof(float), item.second.size(), output) == item.second.size();
            const int closed = fclose(output);
            if (!written || closed) throw std::runtime_error("PCS CPU output write failed");
            fprintf(manifest, "%s\tf32\t%d\t%d\t1\t1\n", item.first.c_str(), cpu_widths.at(item.first), rows);
        }
        if (fclose(manifest)) throw std::runtime_error("PCS step manifest close failed");
    }
};
