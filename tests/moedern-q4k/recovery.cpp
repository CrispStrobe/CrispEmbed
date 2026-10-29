// Hosted model controls, deliberately excluded from the ordinary library.
#include "moedern_text_api.h"
#include "core/clean_exit.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static int body(int argc, char ** argv) {
    require(argc == 2, "recovery MODEL.gguf");
    std::unique_ptr<crispembed_moedern_context, decltype(&crispembed_moedern_free)> ctx(
        crispembed_moedern_init(argv[1], 2), crispembed_moedern_free);
    require(bool(ctx), crispembed_moedern_error(nullptr));
    const auto * result = crispembed_moedern_encode(ctx.get(), "Wie funktioniert die Photosynthese?", 0);
    require(result && result->dimension == 128 && result->n_tokens == 32, "first query rejected");
    std::vector<float> anchor(result->all_vectors, result->all_vectors + result->n_tokens * 128);
    for (const char * gate : { "CRISPEMBED_MOEDERN_Q4_K_PADDED", "CRISPEMBED_MOEDERN_FAST", "CRISPEMBED_MOEDERN_QUANT",
                               "CRISPEMBED_MOEDERN_FAMILY" }) {
        const char * values[] = { "0", "", nullptr };
        for (const char * value : values) {
            if (value)
                setenv(gate, value, 1);
            else
                unsetenv(gate);
            require(!crispembed_moedern_encode(ctx.get(), "x", 0) && crispembed_moedern_error(ctx.get())[0],
                    "disabled quantized context accepted input");
            require(!crispembed_moedern_init(argv[1], 2), "disabled quantized init accepted model");
            setenv(gate, "1", 1);
            result = crispembed_moedern_encode(ctx.get(), "Wie funktioniert die Photosynthese?", 0);
            require(result && result->n_tokens == 32 &&
                        std::memcmp(anchor.data(), result->all_vectors, anchor.size() * sizeof(float)) == 0,
                    "gate recovery changed final vector bits");
        }
    }
    require(!crispembed_moedern_encode(ctx.get(), "x", 7), "bad role accepted");
    result = crispembed_moedern_encode(ctx.get(), "Wie funktioniert die Photosynthese?", 0);
    require(result && std::memcmp(anchor.data(), result->all_vectors, anchor.size() * sizeof(float)) == 0,
            "invalid-input recovery changed final vector bits");
    std::cout << "PASS12 gate toggles/init rejections and bit-exact context recovery; invalid role recovery\n";
    return 0;
}
int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(body(argc, argv));
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        core_util::clean_exit(1);
    }
}
