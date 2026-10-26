// Test-only interface, absent from the ordinary prototype archive/shared lib.
#pragma once

#include "migan.h"

#include <array>
#include <functional>

namespace migan {
namespace test {

// Shape uses ggml [W,H,C,N] axes; payload is contiguous NCHW float32. Every
// observed tensor is marked ggml_set_output before allocation and execution.
using observer = std::function<bool(const std::string &, const float *, size_t, const std::array<int64_t, 4> &)>;
bool run_observed(context * ctx, const float * input_nchw, size_t input_count, float * raw_nchw, size_t output_count,
                  const observer & observe, size_t & graph_buffer_bytes, std::string & error);

} // namespace test
} // namespace migan
