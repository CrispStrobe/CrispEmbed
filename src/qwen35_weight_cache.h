#pragma once

#include <cstddef>
#include <map>
#include <memory>

struct ggml_tensor;

namespace qwen35_ocr {
// Owns converted immutable decoder matrices for exactly one loaded model.
// No eviction: returned storage remains valid until clear()/destruction.
// Do not share an instance between models or simultaneous recognition calls.
class converted_weight_cache {
public:
    static constexpr std::size_t default_budget = 256ull * 1024 * 1024;
    static constexpr std::size_t maximum_budget = 1024ull * 1024 * 1024;
    static std::size_t environment_budget();
    explicit converted_weight_cache(std::size_t budget = environment_budget());
    ~converted_weight_cache();
    converted_weight_cache(const converted_weight_cache &) = delete;
    converted_weight_cache & operator=(const converted_weight_cache &) = delete;
    // Null means use the original materialization path, including invalid/nonfinite
    // data and allocation failure. Literal gate=0 always bypasses existing entries.
    const float * get(ggml_tensor * source, int threads);
    void clear();
    std::size_t bytes() const { return bytes_; }
    std::size_t budget() const { return budget_; }
    std::size_t entries() const { return entries_.size(); }
    std::size_t hits() const { return hits_; }

private:
    struct entry;
    std::map<const ggml_tensor *, std::unique_ptr<entry>> entries_;
    std::size_t budget_ = 0, bytes_ = 0, hits_ = 0;
};
} // namespace qwen35_ocr
