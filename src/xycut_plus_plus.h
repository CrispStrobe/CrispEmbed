// xycut_plus_plus.h — XY-Cut++ reading order (Liu, Li, Wei; arXiv 2504.10258).
//
// Native translation of the get_layout_ordering function closure vendored by
// SoMarkAI/OXR @ c1a1dcb3 (oxr/pipeline/_vendor/xycut_plus_plus.py), itself
// from liushuai35/PaddleXrc @ 7873ab38 paddlex/inference/pipelines/
// layout_parsing/utils.py. Copyright (c) 2024 PaddlePaddle Authors, Apache-2.0.
// Changes: C++ data structures; only the default pipeline flags
// (is_only_xycut=False, is_add_mgs/cmm/pre_cut/adaptive=True,
// is_single_stage=False, min_gap_x=min_gap_y=-1, is_only_x=False) are
// translated, plus the OXR remove_overlaps switch. "doc_title" entries are
// rejected because their branches are not translated; OXR never emits them.
//
// numpy argsort ties: the generic (non-SIMD) numpy introsort is reproduced.
// numpy builds that dispatch 64-bit argsort to AVX-512 x86-simd-sort can
// order equal keys differently; that only affects distinct boxes sharing
// both a segment and identical (x_min, y_min).
#pragma once

#include <array>
#include <string>
#include <vector>

namespace xycut_plus_plus {

struct entry {
    std::array<long long, 4> bbox{}; // integer [x_min, y_min, x_max, y_max]
    std::string label;               // PaddleX block label, e.g. "text"
};

// Returns input positions in final reading order (sorted by sub_index).
// Throws std::invalid_argument on unsupported labels or malformed boxes.
std::vector<int> get_layout_ordering(const std::vector<entry> & entries,
                                     const std::vector<std::string> & no_mask_labels, bool remove_overlaps = true);

// numpy.argsort(kind="quicksort") of an int64 vector, generic code path.
std::vector<long long> numpy_argsort(const std::vector<long long> & values);

} // namespace xycut_plus_plus
