#pragma once
// Test-only declarations; no production callback or trace storage.
#include "josev2b_text_frontend.h"
#include "qwen35_decoder.h"
#include "josev2b_head_observer.h"
namespace josev2b {
namespace composed_test {
void observe_head(const josev2b_head::test::stage & stage);
void observe(const encoded_text &, const std::vector<float> & embedding, const qwen35_ocr::decoder_outputs &,
             const josev2b_head::input &, const josev2b_head::output &, const std::string & response);
} // namespace composed_test
} // namespace josev2b
