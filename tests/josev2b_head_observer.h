// Test variant only. Never expose these symbols through production libraries.
#pragma once
#include "josev2b_head.h"
#include <functional>
#include <cstdint>
namespace josev2b_head {
namespace test {
struct stage {
    std::string label;
    size_t occurrence = 0;
    std::vector<int64_t> shape;
    std::vector<float> values;
};
using observer = std::function<void(const stage &)>;
bool run_observed(context *, const input &, output &, const observer &, std::string &);
bool metadata_controls(std::string & error);
} // namespace test
} // namespace josev2b_head
