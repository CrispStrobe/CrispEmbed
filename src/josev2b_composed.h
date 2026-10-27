#pragma once
#include "josev2b_head.h"
#include <memory>
#include <string>
namespace josev2b {
// Bounded CPU/F32 batch-one SystemOne text composition. Internal component
// paths are explicit; this is not a public alias or a packed model loader.
struct composed_result {
    std::string response_json;
    josev2b_head::output native_fields;
};
class composed_context {
public:
    static std::unique_ptr<composed_context> load(const char * backbone, const char * head, const char * tokenizer,
                                                  int threads, std::string & error);
    ~composed_context();
    composed_context(const composed_context &) = delete;
    composed_context & operator=(const composed_context &) = delete;
    bool decide(const std::string & request_json, std::string & response_json, std::string & error);
    bool decide(const std::string & request_json, composed_result & result, std::string & error);

private:
    struct implementation;
    composed_context();
    std::unique_ptr<implementation> impl_;
};
} // namespace josev2b
