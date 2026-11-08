// Additive, model-free OCR review serialization. No inference or image mutation.
#pragma once

#include "ocr_orchestrator.h"
#include "core/json.h"
#include "core/token_utf8.h"

#include <cmath>
#include <iomanip>
#include <locale>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ocr_review {

inline std::string valid_utf8(const std::string & value) {
    if (value.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("OCR review string exceeds the C ABI byte limit");
    return core_utf8::token_utf8_lossy(value.data(), static_cast<int>(value.size()));
}

inline void text(std::ostream & out, const std::string & value) {
    out << '"' << core_json::json_escape_strict(valid_utf8(value)) << '"';
}

// Preserve malformed decoder bytes alongside replacement-decoded display text.
inline void document_text(std::ostream & out, const std::string & value, const char * field) {
    const auto safe = valid_utf8(value);
    text(out, safe);
    if (safe == value) return;
    out << ',';
    text(out, std::string(field) + "_bytes_hex");
    out << ":\"";
    static const char hex[] = "0123456789abcdef";
    for (unsigned char byte : value) out << hex[byte >> 4] << hex[byte & 15];
    out << '"';
}

inline void number(std::ostream & out, double value) {
    if (std::isfinite(value))
        out << value;
    else
        out << "null";
}

inline bool valid_score(float value) {
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

inline void score(std::ostream & out, float value) {
    if (valid_score(value))
        number(out, value);
    else
        out << "null";
}

inline void optional_stage(std::ostream & out, bool compiled, bool configured, bool loaded) {
    out << "{\"compiled\":" << compiled << ",\"configured\":" << configured << ",\"loaded\":" << loaded << '}';
}

inline void box(std::ostream & out, float x, float y, float width, float height) {
    out << '[';
    number(out, x);
    out << ',';
    number(out, y);
    out << ',';
    number(out, width);
    out << ',';
    number(out, height);
    out << ']';
}

template <typename T> inline void indices(std::ostream & out, const std::vector<T> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << static_cast<long long>(values[i]);
    }
    out << ']';
}

// A malformed or partial engine order must never discard recognized text.
inline std::vector<int> reading_order(size_t count, const std::vector<int> & engine_order) {
    std::vector<bool> seen(count, false);
    std::vector<int> out;
    out.reserve(count);
    for (int index : engine_order) {
        if (index < 0 || static_cast<size_t>(index) >= count || seen[index]) continue;
        seen[index] = true;
        out.push_back(index);
    }
    for (size_t i = 0; i < count; ++i)
        if (!seen[i]) out.push_back(static_cast<int>(i));
    return out;
}

inline const char * source_type(ocr_orchestrator::source_type source) {
    using S = ocr_orchestrator::source_type;
    switch (source) {
    case S::screenshot:
        return "screenshot";
    case S::scanned_doc:
        return "scanned_document";
    case S::photo:
        return "photo";
    default:
        return "auto";
    }
}

inline bool detection_score_available(ocr_orchestrator::engine engine) {
    using E = ocr_orchestrator::engine;
    return engine == E::dbnet_trocr || engine == E::surya || engine == E::ppocrv6;
}

inline const char * sequence_unit(ocr_orchestrator::engine engine) {
    using E = ocr_orchestrator::engine;
    if (engine == E::tesseract || engine == E::tesseract_fraktur) return "recognizer_character";
    return "engine_token";
}

inline const char * route_name(ocr_region_router::destination destination) {
    using D = ocr_region_router::destination;
    switch (destination) {
    case D::table:
        return "table";
    case D::formula:
        return "formula";
    case D::fallback:
        return "fallback";
    default:
        return "text";
    }
}

// full_text may include the optional punctuation restorer. The original
// recognition text is retained separately to keep that provenance visible.
inline std::string render(const ocr_orchestrator::result & value, const ocr_orchestrator::capabilities & capabilities,
                          const std::string & full_text, const std::string & markdown) {
    std::ostringstream out;
    out.imbue(std::locale::classic()); // JSON decimals must not follow OS locale.
    out << std::setprecision(9) << std::boolalpha;
    const bool input_valid = value.input_width > 0 && value.input_height > 0;
    const char * status = !input_valid            ? "input_unreadable"
                          : value.regions.empty() ? "no_text"
                          : value.accepted        ? "recognized"
                                                  : "review_required";
    out << "{\"schema\":\"crispembed.ocr.review\",\"schema_version\":1,\"status\":";
    text(out, status);
    out << ",\"accepted\":" << value.accepted << ",\"selected_stage\":" << value.selected_stage;
    out << ",\"engine\":";
    std::string selected_engine;
    for (const auto & stage : value.stage_metrics)
        if (stage.index == value.selected_stage) selected_engine = stage.engine;
    if (selected_engine.empty())
        out << "null";
    else
        text(out, selected_engine);
    out << ",\"source_type\":";
    text(out, source_type(value.used_type));
    out << ",\"geometry\":{\"units\":\"pixels\",\"origin\":\"top_left\",\"bbox_format\":\"xywh\""
        << ",\"input_width\":" << value.input_width << ",\"input_height\":" << value.input_height
        << ",\"effective_width\":" << value.page_width << ",\"effective_height\":" << value.page_height
        << ",\"recognition_width\":" << value.recognition_width
        << ",\"recognition_height\":" << value.recognition_height
        << ",\"regions_space\":\"recognition_image\",\"layout_space\":\"effective_input\""
        << ",\"coordinates_valid_for_input\":" << value.coordinates_valid_for_input
        << ",\"cleanup_applied\":" << value.cleanup_applied
        << ",\"super_resolution_applied\":" << value.super_resolution_applied << '}';
    out << ",\"text\":";
    document_text(out, full_text, "text");
    out << ",\"recognition_text\":";
    document_text(out, value.full_text, "recognition_text");
    out << ",\"text_postprocessed\":" << (full_text != value.full_text) << ",\"markdown\":";
    document_text(out, markdown, "markdown");
    out << ",\"mean_engine_confidence\":";
    score(out, value.mean_confidence);
    out << ",\"confidence_calibrated\":false,\"detected_language\":";
    if (value.detected_lang.empty())
        out << "null";
    else
        text(out, value.detected_lang);
    out << ",\"language_confidence\":";
    if (value.detected_lang.empty())
        out << "null";
    else
        score(out, value.lang_confidence);
    out << ",\"capabilities\":{\"layout_configured\":" << capabilities.layout
        << ",\"tables_configured\":" << capabilities.tables << ",\"formulas_configured\":" << capabilities.formulas
        << ",\"image_text_fallback\":" << capabilities.image_text_fallback << ",\"language_identification\":";
    optional_stage(out, capabilities.lid_compiled, capabilities.lid_configured, capabilities.lid_loaded);
    out << ",\"truecase\":";
    optional_stage(out, capabilities.truecase_compiled, capabilities.truecase_configured, capabilities.truecase_loaded);
    out << ",\"punctuation\":";
    optional_stage(out, capabilities.punctuation_compiled, capabilities.punctuation_configured,
                   capabilities.punctuation_loaded);
    out << '}';
    out << ",\"reading_order\":";
    indices(out, reading_order(value.regions.size(), value.reading_order));
    out << ",\"regions\":[";
    for (size_t i = 0; i < value.regions.size(); ++i) {
        if (i) out << ',';
        const auto & region = value.regions[i];
        // PP-OCRv6's legacy orchestrator copies its detection score into both
        // confidence fields; it is not a measured recognition probability.
        const bool has_recognition = value.used_engine != ocr_orchestrator::engine::ppocrv6 &&
                                     valid_score(region.rec_confidence) &&
                                     (!region.char_conf.empty() || region.rec_confidence > 0.0f);
        const bool has_detection = detection_score_available(value.used_engine) && valid_score(region.confidence);
        out << "{\"id\":";
        text(out, "region-" + std::to_string(i));
        out << ",\"index\":" << i << ",\"text\":";
        document_text(out, region.text, "text");
        out << ",\"bbox\":";
        box(out, region.box.x, region.box.y, region.box.w, region.box.h);
        out << ",\"confidence\":";
        if (has_recognition)
            score(out, region.rec_confidence);
        else if (has_detection)
            score(out, region.confidence);
        else
            out << "null";
        out << ",\"confidence_source\":";
        text(out, has_recognition ? "recognizer" : has_detection ? "detector" : "unavailable");
        out << ",\"engine_score\":";
        score(out, region.confidence);
        out << ",\"recognition_confidence\":";
        if (has_recognition)
            score(out, region.rec_confidence);
        else
            out << "null";
        out << ",\"sequence_confidences\":[";
        for (size_t j = 0; j < region.char_conf.size(); ++j) {
            if (j) out << ',';
            score(out, region.char_conf[j]);
        }
        out << "],\"sequence_unit\":";
        text(out, sequence_unit(value.used_engine));
        // The legacy ABI has no offsets mapping tokens/codepoints to text.
        // Length equality alone is insufficient to certify character alignment.
        out << ",\"character_alignment_verified\":false,\"orientation\":{\"corrected\":" << region.orientation_corrected
            << ",\"angle\":" << region.orientation_angle << ",\"confidence\":";
        score(out, region.orientation_confidence);
        out << "}}";
    }
    out << "],\"stages\":[";
    for (size_t i = 0; i < value.stage_metrics.size(); ++i) {
        if (i) out << ',';
        const auto & stage = value.stage_metrics[i];
        out << "{\"index\":" << stage.index << ",\"engine\":";
        text(out, stage.engine);
        out << ",\"elapsed_ms\":";
        number(out, stage.elapsed_ms);
        // The historical metric records the requested recipe, not whether
        // cleanup successfully emitted a transformed image.
        out << ",\"cleanup_requested\":" << stage.cleanup_applied << ",\"accepted\":" << stage.accepted
            << ",\"text_bytes\":" << stage.text_chars << ",\"mean_engine_confidence\":";
        score(out, stage.mean_confidence);
        out << '}';
    }
    out << "],\"layout\":[";
    for (size_t i = 0; i < value.layout.size(); ++i) {
        if (i) out << ',';
        const auto & region = value.layout[i];
        out << "{\"index\":" << i << ",\"label\":";
        text(out, region.label_name ? region.label_name : "unknown");
        out << ",\"raw_label\":";
        if (region.raw_label)
            text(out, region.raw_label);
        else
            out << "null";
        out << ",\"canonical_exact\":" << region.canonical_exact << ",\"confidence\":";
        score(out, region.score);
        out << ",\"bbox\":";
        box(out, region.x1, region.y1, region.x2 - region.x1, region.y2 - region.y1);
        out << '}';
    }
    out << "],\"tables\":[";
    for (size_t i = 0; i < value.tables.size(); ++i) {
        if (i) out << ',';
        const auto & table = value.tables[i];
        out << "{\"layout_index\":" << table.layout_index << ",\"confidence\":";
        score(out, table.confidence);
        out << ",\"bbox\":";
        box(out, table.x1, table.y1, table.x2 - table.x1, table.y2 - table.y1);
        out << ",\"html\":";
        document_text(out, table.html, "html");
        out << ",\"confidence_source\":\"layout_detector\"";
        out << '}';
    }
    out << "],\"formulas\":[";
    for (size_t i = 0; i < value.formulas.size(); ++i) {
        if (i) out << ',';
        const auto & formula = value.formulas[i];
        out << "{\"layout_index\":" << formula.layout_index << ",\"confidence\":";
        score(out, formula.confidence);
        out << ",\"bbox\":";
        box(out, formula.x1, formula.y1, formula.x2 - formula.x1, formula.y2 - formula.y1);
        out << ",\"latex\":";
        document_text(out, formula.latex, "latex");
        out << '}';
    }
    out << "],\"routing\":{\"decisions\":[";
    for (size_t i = 0; i < value.routing.layout_decisions.size(); ++i) {
        if (i) out << ',';
        const auto & decision = value.routing.layout_decisions[i];
        out << "{\"layout_index\":" << decision.layout_index << ",\"destination\":";
        text(out, route_name(decision.dest));
        out << ",\"also_text\":" << decision.also_text << '}';
    }
    out << "],\"text_to_layout\":";
    indices(out, value.routing.text_to_layout);
    out << ",\"suppress_text\":";
    indices(out, value.routing.suppress_text);
    out << "},\"warnings\":[";
    bool warning_written = false;
    auto warning = [&](const char * message) {
        if (warning_written) out << ',';
        text(out, message);
        warning_written = true;
    };
    if (capabilities.lid_configured && !capabilities.lid_compiled)
        warning("Language identification was requested but this library was built without its optional backend.");
    else if (capabilities.lid_configured && !capabilities.lid_loaded)
        warning("Language identification was requested but its model could not be loaded.");
    if (capabilities.truecase_configured && !capabilities.truecase_compiled)
        warning("Truecasing was requested but this library was built without its optional backend.");
    else if (capabilities.truecase_configured && !capabilities.truecase_loaded)
        warning("Truecasing was requested but its model could not be loaded.");
    if (capabilities.punctuation_configured && !capabilities.punctuation_loaded)
        warning("Punctuation restoration was requested but its model could not be loaded.");
    if (!input_valid) warning("Input image dimensions are unavailable; recognition may not have run.");
    if (value.regions.empty())
        warning("No text returned. This does not distinguish a blank page from an engine failure.");
    if (!value.accepted && !value.regions.empty())
        warning("No stage met its acceptance criteria; review the selected output.");
    if (!value.coordinates_valid_for_input && !value.regions.empty())
        warning("Region coordinates are not verified against the original input. Use an explicitly prepared image with "
                "internal cleanup and super-resolution disabled for overlays.");
    if (value.cleanup_applied && !value.layout.empty())
        warning("Layout and recognition coordinates may use different image spaces after cleanup.");
    if (full_text != value.full_text)
        warning("Joined text was postprocessed; region text and sequence scores retain recognition output.");
    out << "]}";
    return out.str();
}

} // namespace ocr_review
