// UltraBERT-only model-free tokenizer context configuration.
#pragma once
#include "tokenizer.h"
#include "core/env_gate.h"
namespace ultrabert_profile {
inline void configure_tokenizer_mode(BPETokenizer & raw) {
    raw.set_gpt2_regex_pretok(true);
    raw.set_ultrabert_unicode_gpt2(core_env::on("CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN"));
}
} // namespace ultrabert_profile
