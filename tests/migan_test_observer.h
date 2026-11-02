// Test-only interface, absent from the ordinary prototype archive/shared lib.
#pragma once

#include "migan.h"

#include <array>
#include <functional>

namespace migan {
namespace test {

// Shape uses ggml [W,H,C,N] axes; payload is contiguous NCHW float32. Every
// observed tensor and its allocation-owner chain are marked ggml_set_output
// before allocation and execution.
using observer = std::function<bool(const std::string &, const float *, size_t, const std::array<int64_t, 4> &)>;
bool observer_retention_regression(std::string & error);
// Per-node device time (each node as its own one-node graph; diagnostic only).
using op_timer = std::function<void(const char * op, const int64_t * ne, const int64_t * src0_ne,
                                    const char * src0_type, double ms)>;
bool profile_ops(context * ctx, const float * input_nchw, size_t batch, const op_timer & timer, std::string & error);
bool run_observed(context * ctx, const float * input_nchw, size_t input_count, float * raw_nchw, size_t output_count,
                  const observer & observe, size_t & graph_buffer_bytes, std::string & error);

} // namespace test
} // namespace migan
