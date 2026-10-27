// Experimental original-reference joint-head component; no backbone or public ABI.
#pragma once
#include <cstddef>
#include <string>
#include <vector>
namespace josev2b_head {
constexpr int hidden_size = 2048, width = 512, max_tokens = 512, max_fields = 16, max_options = 128;
constexpr const char * original_receipt_sha256 = "51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090";
constexpr const char * original_worker_source = "b2edf8c83c9aca1f8a9210198d47798ac13828b8";
struct span {
    int start = 0, end = 0;
};
struct field {
    std::string id;
    int type = 0; // Publisher QUESTION_TYPES: noul=0, choice=1, score=2.
    span question;
    std::vector<span> options;
    // Exact raw means of the TIED output embedding rows, in option order.
    // No normalization/temperature and no substitute contextual embedding.
    std::vector<float> lexical_means;
};
struct input {
    int tokens = 0;
    std::vector<float> hidden; // Original contiguous [1,T,2048], no padding.
    std::vector<field> fields; // Publisher input order.
};
struct field_output {
    std::string id;
    std::vector<float> logits, probabilities;
};
struct output {
    std::vector<field_output> fields;
    std::string backend;
};
struct context;
// Explicit backend: "cpu" or an exact registered CUDA device such as "CUDA0".
// No backend fallback. Both init/run require core_env::on("CRISPEMBED_JOSEV_HEAD").
context * init(const char * path, const char * reviewed_receipt, const char * reviewed_worker, const char * backend,
               int threads, std::string & error);
void free(context * ctx);
const char * backend_name(const context * ctx);
bool validate_input(const input & values, std::string & error); // Pure bounded contract.
bool run(context * ctx, const input & values, output & result, std::string & error);
} // namespace josev2b_head
