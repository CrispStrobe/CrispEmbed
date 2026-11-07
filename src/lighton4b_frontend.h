#pragma once
#include "core/gguf_loader.h"
#include <string>
#include <vector>
#include <unordered_map>
namespace lighton4b {
// Independent format5 adapter; historical qwen35 frontend source stays unchanged.
class frontend_tokenizer {
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int32_t> tokens_, merges_;
    std::vector<int> added_, special_, stops_;
    int eos_ = -1;
    bool enabled() const;

public:
    bool load(gguf_context *);
    bool encode(const std::string &, std::vector<int> &) const;
    bool decode(const std::vector<int> &, std::string &) const;
    bool image_prompt(const std::string &, int, std::vector<int> &, std::vector<int> &, std::string &) const;
    bool is_stop(int id) const;
    int eos() const { return eos_; }
};
} // namespace lighton4b
