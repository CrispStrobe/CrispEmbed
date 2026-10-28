#pragma once
#include "qwen35_decoder.h"
#include <memory>

namespace qwen35_ocr {
// Internal CPU candidate. Stored matrix types reach ggml_mul_mat unchanged;
// small normalization vectors and the qualified CPU recurrence stay F32.
// All request KV/conv/recurrent state lives in decoder_outputs::cache, exactly
// as on decoder_cpu. No process-global weight or graph cache is introduced.
class graph_decoder {
public:
    graph_decoder();
    ~graph_decoder();
    graph_decoder(const graph_decoder &) = delete;
    graph_decoder & operator=(const graph_decoder &) = delete;
    void clear();
    bool run(const decoder_weights & weights, const attention_config & config, const float * hidden,
             const float * positions, int tokens, int threads, const decoder_cache * previous,
             decoder_outputs & output);
    // Internal component boundary for hosted stored-type projection controls.
    bool project(const std::vector<ggml_tensor *> & matrices, const float * hidden, int width, int tokens, int threads,
                 std::vector<std::vector<float>> & output);

private:
    struct implementation;
    std::unique_ptr<implementation> impl_;
};
} // namespace qwen35_ocr
