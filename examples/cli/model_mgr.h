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
// Returns empty string on failure.
std::string resolve_model(const std::string & arg, bool auto_download = false);

// List available model names
void list_models();

// Registry accessors
int n_models();
const char * model_name(int i);
const char * model_desc(int i);
const char * model_filename(int i);
const char * model_size(int i);

}  // namespace crispembed_mgr
