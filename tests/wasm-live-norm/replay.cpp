// Same-input actual GGML norm and unchanged candidate, TEST ONLY.
#include "core/clean_exit.h"
#include "core/wasm_native_order_norm.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "observer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define EXPORTED EMSCRIPTEN_KEEPALIVE
#else
#define EXPORTED
#endif
struct owner {
    ggml_context * context = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~owner() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
        if (backend) ggml_backend_free(backend);
    }
};
extern "C" EXPORTED int live_norm_replay(int mode, const float * x, float * y, int count, int threads) {
    if ((mode != 0 && mode != 1) || !x || !y || count != 384 * 12 || (threads != 1 && threads != 2)) return -1;
    for (int i = 0; i < count; ++i)
        if (!std::isfinite(x[i])) return -2;
    owner o;
    o.context = ggml_init({ 2 * 1024 * 1024, nullptr, true });
    o.backend = ggml_backend_cpu_init();
    if (!o.context || !o.backend) return -3;
    ggml_backend_cpu_set_n_threads(o.backend, threads);
    auto * input = ggml_new_tensor_2d(o.context, GGML_TYPE_F32, 384, 12);
    const wasm_native_order_norm::parameters parameters;
    auto * result =
        mode ? wasm_native_order_norm::make(o.context, input, &parameters) : ggml_norm(o.context, input, 1e-12f);
    if (!result || result->src[0] != input) return -4;
    auto * graph = ggml_new_graph(o.context);
    ggml_build_forward_expand(graph, result);
    o.buffer = ggml_backend_alloc_ctx_tensors(o.context, o.backend);
    if (!o.buffer) return -5;
    ggml_backend_tensor_set(input, x, 0, size_t(count) * 4);
    if (ggml_backend_graph_compute(o.backend, graph) != GGML_STATUS_SUCCESS) return -6;
    std::vector<float> pending(count);
    ggml_backend_tensor_get(result, pending.data(), 0, size_t(count) * 4);
    for (float v : pending)
        if (!std::isfinite(v)) return -7;
    std::copy(pending.begin(), pending.end(), y);
    return count;
}
static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
static int body(int argc, char ** argv) {
    try {
        // Registration must distinguish disconnected, wrong-op and wrong-width graphs.
        owner o;
        o.context = ggml_init({ 2 * 1024 * 1024, nullptr, true });
        require(o.context, "Control context");
        auto * input = ggml_new_tensor_2d(o.context, GGML_TYPE_F32, 384, 12);
        auto * norm = ggml_norm(o.context, input, 1e-12f);
        auto * other = ggml_new_tensor_2d(o.context, GGML_TYPE_F32, 384, 12);
        live_norm_observer state;
        state.directory = "test-only-not-written";
        require(!state.add_norm(other, norm) && state.norms == 0, "Disconnected node admitted");
        state.enabled = true;
        require(!state.add_norm(input, norm) && state.norms == 0, "Old norm mislabeled candidate");
        state.enabled = false;
        require(state.add_norm(input, norm) && state.norms == 1 && !state.finish(), "Incomplete capture admitted");
        require(!wasm_native_order_norm::eligible(true, 384, 6, 12, 1536, false, 1e-5f) &&
                    !wasm_native_order_norm::eligible(true, 384, 5, 12, 1536, false, 1e-12f),
                "Geometry sensitivity missing");
        if (argc == 1) {
            std::puts("PASS connected/opcode/incomplete/epsilon/geometry sensitivity");
            return 0;
        }
        require(argc == 5, "Expected input.f32 output.f32 mode threads");
        const int mode = std::stoi(argv[3]), threads = std::stoi(argv[4]);
        std::vector<float> input_data(384 * 12), output(input_data.size()), repeat(input_data.size());
        std::ifstream in(argv[1], std::ios::binary);
        in.read(reinterpret_cast<char *>(input_data.data()), input_data.size() * 4);
        require(bool(in) && in.peek() == std::char_traits<char>::eof(), "Exact input extent required");
        const auto before = input_data;
        require(
            live_norm_replay(mode, input_data.data(), output.data(), output.size(), threads) == int(output.size()) &&
                live_norm_replay(mode, input_data.data(), repeat.data(), repeat.size(), threads) == int(repeat.size()),
            "Connected replay failed");
        require(std::memcmp(output.data(), repeat.data(), output.size() * 4) == 0 &&
                    std::memcmp(before.data(), input_data.data(), input_data.size() * 4) == 0,
                "Replay repeat/input neutrality");
        std::ofstream out(argv[2], std::ios::binary);
        out.write(reinterpret_cast<char *>(output.data()), output.size() * 4);
        require(bool(out), "Output write failed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "Live norm replay: %s\n", e.what());
        return 2;
    }
}
int main(int argc, char ** argv) {
    core_util::clean_exit(body(argc, argv));
}
