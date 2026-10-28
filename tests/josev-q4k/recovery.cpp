// Test-only loaded mixed-model gate/recovery checks; no production callbacks.
#include "core/clean_exit.h"
#include "core/decision_schema.h"
#include "josev2b_composed.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <utility>
namespace {
void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
bool exact(const josev2b_head::output & a, const josev2b_head::output & b) {
    if (a.fields.size() != b.fields.size()) return false;
    for (size_t i = 0; i < a.fields.size(); ++i) {
        for (auto values : { std::make_pair(&a.fields[i].logits, &b.fields[i].logits),
                             std::make_pair(&a.fields[i].probabilities, &b.fields[i].probabilities) })
            if (values.first->size() != values.second->size() ||
                std::memcmp(values.first->data(), values.second->data(), values.first->size() * sizeof(float)))
                return false;
    }
    return true;
}
int run(int argc, char ** argv) {
    require(argc == 3, "recovery MODEL REQUESTS");
    std::ifstream f(argv[2]);
    require(bool(f) && std::filesystem::file_size(argv[2]) <= 1024 * 1024, "bounded request packet required");
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const auto requests = core_decision::Parser(text).parse();
    require(requests.kind == core_decision::Value::array && requests.list.size() == 4, "four frozen requests required");
    require(requests.list[0].get("request_json").kind == core_decision::Value::string, "raw request string required");
    const auto & request = requests.list[0].get("request_json").text;
    std::string error;
    auto ctx = josev2b::composed_context::load_packed(argv[1], 2, error);
    require(bool(ctx), error.c_str());
    josev2b::composed_result baseline;
    require(ctx->decide(request, baseline, error), error.c_str());
    const char * values[] = { nullptr, "", "0" };
    for (const char * gate : { "CRISPEMBED_JOSEV_QUANT", "CRISPEMBED_QWEN35_QUANT" }) {
        for (const char * value : values) {
            if (value)
                setenv(gate, value, 1);
            else
                unsetenv(gate);
            auto rejected = josev2b::composed_context::load_packed(argv[1], 2, error);
            require(!rejected && !error.empty(), "disabled mixed-model init accepted");
            josev2b::composed_result output = baseline;
            require(!ctx->decide(request, output, error) && output.response_json.empty() &&
                        output.native_fields.fields.empty() && !error.empty(),
                    "failed gate left stale output");
            setenv(gate, "1", 1);
        }
    }
    josev2b::composed_result recovered = baseline;
    require(!ctx->decide("{", recovered, error) && recovered.response_json.empty() &&
                recovered.native_fields.fields.empty(),
            "invalid JSON left stale output");
    require(ctx->decide(request, recovered, error) && recovered.response_json == baseline.response_json &&
                exact(recovered.native_fields, baseline.native_fields),
            "gate/input recovery changed raw final bits");
    std::cout << "PASS6 unset/empty/zero gate init/run negatives; failed-output clearing and bit-exact recovery\n";
    return 0;
}
} // namespace
int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(run(argc, argv));
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        core_util::clean_exit(1);
    }
}
