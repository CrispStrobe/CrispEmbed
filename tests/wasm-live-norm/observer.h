// TEST OVERLAY ONLY: immediate scheduler observation, no numeric graph nodes.
#pragma once
#include "ggml-backend.h"
#include "ggml.h"
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
struct live_norm_observer {
    struct entry {
        ggml_tensor * tensor;
        std::string name;
        int hits = 0;
    };
    std::vector<entry> entries;
    std::string architecture, directory;
    int width = 0, layers = 0, heads = 0, intermediate = 0, norms = 0;
    float epsilon = 0;
    bool cpu = false, pre_ln = false, enabled = false, failed = false;
    void configure(const char * arch, int h, int l, int n, int f, bool pre, float eps, bool is_cpu, bool on) {
        architecture = arch;
        width = h;
        layers = l;
        heads = n;
        intermediate = f;
        pre_ln = pre;
        epsilon = eps;
        cpu = is_cpu;
        enabled = on;
    }
    bool active() const { return !directory.empty(); }
    static bool callback(ggml_tensor * tensor, bool ask, void * userdata) {
        auto & state = *static_cast<live_norm_observer *>(userdata);
        bool wanted = false;
        auto reject = [&]() {
            state.failed = true;
            return false;
        };
        for (auto & row : state.entries) {
            if (row.tensor != tensor) continue;
            wanted = true;
            if (ask) continue;
            if (row.hits++ != 0 || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) return reject();
            const size_t bytes = ggml_nbytes(tensor);
            if (bytes == 0 || bytes > 12 * 1536 * sizeof(float)) return reject();
            std::vector<char> buffer(bytes);
            ggml_backend_tensor_get(tensor, buffer.data(), 0, bytes);
            std::ofstream file(state.directory + "/" + row.name + ".f32", std::ios::binary);
            file.write(buffer.data(), buffer.size());
            file.close();
            if (!file) return reject();
        }
        return ask ? wanted : true;
    }
    void begin(ggml_backend_sched_t scheduler) {
        const char * path = std::getenv("CRISPEMBED_LIVE_NORM_CAPTURE");
        directory = path ? path : "";
        norms = 0;
        failed = false;
        entries.clear();
        ggml_backend_sched_set_eval_callback(scheduler, active() ? callback : nullptr, active() ? this : nullptr);
    }
    bool add(const std::string & name, ggml_tensor * tensor) {
        if (!active()) return true;
        if (!tensor || tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || tensor->ne[1] != 12 ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1 || (tensor->ne[0] != 384 && tensor->ne[0] != 1536))
            return false;
        for (const auto & row : entries)
            if (row.name == name) return false;
        entries.push_back({ tensor, name, 0 });
        return true;
    }
    bool add_norm(ggml_tensor * input, ggml_tensor * output) {
        if (!active()) return true;
        if (!input || !output || norms >= 13 || output->src[0] != input || input->ne[0] != 384 ||
            output->ne[0] != 384 || output->op != (enabled ? GGML_OP_MAP_CUSTOM1 : GGML_OP_NORM))
            return false;
        char label[32];
        std::snprintf(label, sizeof(label), "norm%02d", norms);
        if (!add(std::string(label) + "_input", input) || !add(std::string(label) + "_output", output)) return false;
        ++norms;
        return true;
    }
    bool finish() const {
        if (!active() || failed || norms != 13 || entries.size() != 49) return false;
        for (const auto & row : entries)
            if (row.hits != 1) return false;
        std::ofstream out(directory + "/capture.json");
        out << "{\"schema\":\"crispembed-wasm-live-norm-capture-v1\",\"architecture\":" << std::quoted(architecture)
            << ",\"width\":" << width << ",\"layers\":" << layers << ",\"heads\":" << heads
            << ",\"intermediate\":" << intermediate << ",\"pre_ln\":" << (pre_ln ? "true" : "false")
            << ",\"cpu\":" << (cpu ? "true" : "false") << ",\"enabled\":" << (enabled ? "true" : "false")
            << ",\"epsilon_bits\":";
        uint32_t bits;
        static_assert(sizeof(bits) == sizeof(epsilon), "F32 required");
        std::memcpy(&bits, &epsilon, sizeof(bits));
        out << bits << ",\"connected_norm_nodes\":" << norms << ",\"custom_norm_nodes\":" << (enabled ? norms : 0)
            << ",\"files\":[";
        bool first = true;
        for (const auto & row : entries) {
            if (!first) out << ',';
            first = false;
            out << "{\"name\":" << std::quoted(row.name) << ",\"hits\":" << row.hits << ",\"shape\":["
                << row.tensor->ne[1] << ',' << row.tensor->ne[0] << "],\"bytes\":" << ggml_nbytes(row.tensor) << '}';
        }
        out << "]}\n";
        out.close();
        return bool(out);
    }
};
