// oxr_document_internal.h — shared declarations of the OXR document port.
// See oxr_document.h for provenance (SoMarkAI/OXR @ c1a1dcb3, Apache-2.0 +
// OXR additional terms).
#pragma once

#include "core/py_str.h"
#include "oxr_document.h"

#include <array>
#include <optional>

namespace oxr_document {
namespace text {
using py_str::u32;
size_t latex_command(const u32 & s, size_t i);
bool is_escaped(const u32 & s, size_t index);
std::optional<u32> fenced_search(const u32 & s, int header_mode, const char * literal, bool ci);
std::optional<u32> anchored_wrapper(const u32 & s, const u32 & open, const u32 & close, bool lazy);
u32 space_latex_row_breaks(const u32 & latex);
u32 space_markdown_math(const u32 & markdown);
std::array<u32, 3> extract_formula_numbering(const u32 & text);
u32 clean_formula_markdown_delimiters(const u32 & latex);
u32 move_formula_numbering_outside_math(const u32 & text);
u32 postprocess_text(const u32 & raw);
std::array<u32, 3> postprocess_formula(const u32 & raw); // content, before, after
u32 postprocess_code(const u32 & raw);
u32 postprocess_chemical(const u32 & raw);
u32 strip_table_code_fence(const u32 & raw);
} // namespace text
} // namespace oxr_document
