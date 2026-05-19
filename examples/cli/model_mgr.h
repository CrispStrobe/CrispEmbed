#pragma once
// model_mgr.h — Auto-download model manager for CrispEmbed.
//
// Resolves model names (e.g., "octen-0.6b") to local GGUF paths,
// downloading from HuggingFace if needed.

#include <string>

namespace crispembed_mgr {

// Cache directory: $CRISPEMBED_CACHE_DIR or the per-user default.
std::string cache_dir();

// Resolve a model argument to a local file path.
// If arg is an existing file, returns it directly.
// If arg is a known model name, downloads from HF if not cached.
// For models under a restricted license (cc-by-nc-*, gemma, other) the caller
// must either be on a TTY (in which case the user is prompted to accept) or
// pass `accepted_license` matching the model's SPDX tag (or "all"). Without
// acceptance the download is refused even when `auto_download` is true.
// Falls back to the `CRISPEMBED_ACCEPT_LICENSE` env var if `accepted_license`
// is empty.
// Returns empty string on failure.
std::string resolve_model(const std::string & arg, bool auto_download = false,
                          const std::string & accepted_license = "");

// List available model names
void list_models();

// Registry accessors
int n_models();
const char * model_name(int i);
const char * model_desc(int i);
const char * model_filename(int i);
const char * model_size(int i);
const char * model_license(int i);       // SPDX-style tag, e.g. "apache-2.0",
                                          // "mit", "cc-by-nc-4.0", "gemma"
const char * model_card_url(int i);       // upstream HuggingFace model card

// Returns true if the given SPDX tag designates a restricted license that
// the user must explicitly accept before redistribution / download.
// Currently: anything matching "cc-by-nc*", "gemma", "llama*", "other".
bool license_requires_acceptance(const char * spdx);

// Prompt prefixes for optimal retrieval quality.
// Returns nullptr if the model doesn't use prefixes.
const char * get_query_prefix(const char * model_name);
const char * get_passage_prefix(const char * model_name);

}  // namespace crispembed_mgr
