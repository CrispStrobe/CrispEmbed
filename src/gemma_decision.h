#pragma once
#include "crispembed.h"
#include <string>
#include <vector>

struct gemma_decision_context;
gemma_decision_context * gemma_decision_init(const char * path, int threads);
void gemma_decision_free(gemma_decision_context * ctx);
bool gemma_decision_encode(gemma_decision_context * ctx, const std::string & text, std::vector<float> & embedding,
                           std::vector<int32_t> * ids = nullptr, std::vector<float> * raw_tokens = nullptr);
bool gemma_decision_scores(gemma_decision_context * ctx, const float * state, const float * question,
                           const float * options, int n_options, int kind, float * logits, float * probabilities);
std::string gemma_decision_run(gemma_decision_context * ctx, const std::string & request);
const char * gemma_decision_error(const gemma_decision_context * ctx);
void gemma_decision_hparams(const gemma_decision_context * ctx, crispembed_hparams * out);
const char * gemma_decision_token(const gemma_decision_context * ctx, int id);
