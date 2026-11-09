// SPDX-License-Identifier: MIT
// Versioned native extraction snapshots. No model/runtime initialization.
#pragma once

#include "marker_document.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

namespace crisp_marker {

struct extraction_snapshot {
    document value;
    // Public identity values, such as source/model/configuration SHA-256.
    // Never store credentials or source/model paths here.
    std::map<std::string, std::string> identity;
    int first_page = 0, last_page = -1, next_page = 0;
    bool complete = false;
};

struct artifact_limits {
    uint64_t max_file_bytes = 256ull * 1024 * 1024;
    uint64_t max_string_bytes = 4ull * 1024 * 1024;
    uint64_t max_asset_bytes = 128ull * 1024 * 1024;
    size_t max_pages = 1000;
    size_t max_blocks_per_page = 4096;
    // Aggregate entries across all vectors/maps, including spans and warnings.
    size_t max_collection_entries = 1000000;
    size_t max_depth = 32;
};

// Both functions throw std::runtime_error/length_error/invalid_argument on
// malformed content or I/O failure. No partially decoded document is returned.
// save_extraction writes an exclusively created sibling file, then atomically
// replaces the destination after validation, writing, flushing and closing.
// Existing symlinks/directories are rejected. No artifact member becomes a path.
void save_extraction(const std::string & path, const extraction_snapshot & snapshot,
                     const artifact_limits & limits = {});
extraction_snapshot load_extraction(const std::string & path, const artifact_limits & limits = {});

// Streaming file identity for source/model inputs. Throws on failed reads;
// does not require an extraction-format file and never loads a model.
// Explicitly selected input/model/library symlinks resolve to their targets.
std::string extraction_file_sha256(const std::string & path);

} // namespace crisp_marker
