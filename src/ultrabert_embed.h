#pragma once
#include "crispembed.h"
#include <string>
#include <vector>

// Experimental CPU/F32 pooled profile. No original arrays or audit hooks here.
struct ultrabert_context;
ultrabert_context * ultrabert_init(const char * path, int threads);
void ultrabert_free(ultrabert_context * ctx);
bool ultrabert_encode(ultrabert_context * ctx, const std::string & text, std::vector<float> & pooled,
                      std::vector<int32_t> * ids = nullptr, std::vector<float> * hidden = nullptr);
void ultrabert_hparams(crispembed_hparams * out);
const char * ultrabert_token(const ultrabert_context * ctx, int id);
const char * ultrabert_query_prefix();
