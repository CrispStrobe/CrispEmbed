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
// For models under a restricted license (cc-by-nc-*, gemma, lfm1.0, other) the caller
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
const char * model_license(int i);  // SPDX-style tag, e.g. "apache-2.0",
                                    // "mit", "cc-by-nc-4.0", "gemma", "lfm1.0"
const char * model_card_url(int i); // upstream HuggingFace model card

// Scripts the model's recognition dictionary can emit ("latin+cjk+kana"),
// scanned from the shipped GGUF by tools/scan_model_languages.py. Returns
// nullptr or "" when the model was not scanned — which means UNKNOWN, not
// "no coverage". Only OCR recognizers carry it. Coverage is necessary but
// not sufficient for quality; see docs/LANGUAGES.md.
const char * model_languages(int i);

// Returns true if the given SPDX tag designates a restricted license that
// the user must explicitly accept before redistribution / download.
// Currently: anything matching "cc-by-nc*", "gemma", "llama*", "lfm1.0", "other".
bool license_requires_acceptance(const char * spdx);

// Acknowledgement gate for face *recognition* models.
//
// Face recognition turns an image into a biometric template. Under the GDPR
// that output is special-category data (Art. 9); building a 1:N identification
// system on top of it is regulated under the EU AI Act (Annex III §1, from
// 2 December 2027). Detection alone (bounding boxes) is not gated.
//
// Returns true when the caller has acknowledged this. Acceptance comes from
// `accepted_flag` (`--accept-biometric`), the `CRISPEMBED_ACCEPT_BIOMETRIC`
// env var, or an interactive [y/N] prompt on a TTY. Off a TTY without
// acknowledgement the call fails closed and returns false.
//
// This is a deliberate speed bump and an audit trail, not a security control:
// CrispEmbed is MIT-licensed and the check is trivially removable. It exists so
// that biometric processing is never something a user starts by accident.
//
// On every success path this also calls crispembed_accept_biometric_use(), so
// the library-level gate in crispembed_face_init() does not ask a second time.
bool accept_biometric_use(const char * model_label, bool accepted_flag);

// Prompt prefixes for optimal retrieval quality.
// Returns nullptr if the model doesn't use prefixes.
const char * get_query_prefix(const char * model_name);
const char * get_passage_prefix(const char * model_name);

} // namespace crispembed_mgr
