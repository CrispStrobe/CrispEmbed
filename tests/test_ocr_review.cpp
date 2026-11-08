#include "ocr_review.h"
#include "core/json_value.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <limits>
#include <locale>

static int failures = 0;
#define CHECK(condition)                                                                                               \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition);                                          \
            ++failures;                                                                                                \
        }                                                                                                              \
    } while (0)

using value = core_json_value::value;

static const value & field(const value & object, const char * name) {
    const auto * found = object.get(name);
    if (!found) throw std::runtime_error(std::string("Missing JSON field: ") + name);
    return *found;
}

struct comma_decimal : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
};

static ocr_orchestrator::result fixture() {
    ocr_orchestrator::result result;
    result.input_width = result.page_width = result.recognition_width = 600;
    result.input_height = result.page_height = result.recognition_height = 800;
    result.coordinates_valid_for_input = true;
    result.selected_stage = 2;
    result.accepted = true;
    result.used_type = ocr_orchestrator::source_type::scanned_doc;
    result.used_engine = ocr_orchestrator::engine::tesseract;
    result.full_text = "Invoice\nBetrag: 42,00 €\nمرحبا";
    result.markdown = "# Invoice\n\nBetrag: 42,00 €";
    result.detected_lang = "de";
    result.lang_confidence = 0.85f;
    result.mean_confidence = 0.8f;
    result.stage_metrics = { { 1, "dbnet_trocr", 3.5f, false, false, 3, 0.5f },
                             { 2, "tesseract", 15.25f, false, true, 24, 0.8f } };
    for (int i = 0; i < 3; ++i) {
        ocr_pipeline::ocr_result region{};
        region.box = { 10.0f, 20.0f + i * 20.0f, 200.0f, 15.0f, 0.9f, 0.0f, {}, {} };
        region.text = i == 0 ? "Invoice" : i == 1 ? "Betrag: 42,00 €" : "مرحبا";
        region.confidence = 0.95f;
        region.rec_confidence = 0.8f;
        region.char_conf = { 0.8f, 0.5f, 0.0f };
        result.regions.push_back(region);
    }
    result.reading_order = { 2, 0, 2, -1, 100 };
    result.layout.push_back({ 10, 20, 220, 90, 0.75f, layout_detect::label_id::table, "table" });
    result.tables.push_back({ 0, 0.7f, 10, 20, 220, 90, "<table><tr><td>42 &amp; 7</td></tr></table>" });
    result.formulas.push_back({ 0, 0.6f, 10, 120, 220, 180, "\\frac{a}{b}" });
    result.routing.layout_decisions.push_back({ 0, ocr_region_router::destination::table, false });
    result.routing.text_to_layout = { 0, 0, -1 };
    result.routing.suppress_text = { 1, 1, 0 };
    return result;
}

static value snapshot(const ocr_orchestrator::result & result, const std::string & text = "") {
    const auto serialized = ocr_review::render(result, { true, true, true, false },
                                               text.empty() ? result.full_text : text, result.markdown);
    return core_json_value::parser(serialized).parse();
}

static int crispembed_test_main() {
    try {
        auto result = fixture();
        const auto original_order = result.reading_order;
        auto json = snapshot(result);
        CHECK(field(json, "schema").s == "crispembed.ocr.review");
        CHECK(field(json, "schema_version").i == 1);
        CHECK(field(json, "status").s == "recognized");
        CHECK(field(json, "engine").s == "tesseract");
        CHECK(field(json, "selected_stage").i == 2);
        CHECK(field(json, "text").s == result.full_text);
        CHECK(field(json, "recognition_text").s == result.full_text);
        CHECK(field(json, "confidence_calibrated").b == false);
        CHECK(field(json, "reading_order").a.size() == 3);
        CHECK(field(json, "reading_order").a[0].i == 2);
        CHECK(field(json, "reading_order").a[1].i == 0);
        CHECK(field(json, "reading_order").a[2].i == 1);
        CHECK(result.reading_order == original_order);
        CHECK(field(field(json, "geometry"), "coordinates_valid_for_input").b);
        CHECK(field(field(json, "capabilities"), "tables_configured").b);
        CHECK(field(json, "regions").a.size() == 3);
        CHECK(field(field(json, "regions").a[0], "confidence_source").s == "recognizer");
        CHECK(!field(field(json, "regions").a[0], "character_alignment_verified").b);
        CHECK(field(field(json, "regions").a[0], "sequence_unit").s == "recognizer_character");
        CHECK(field(field(json, "tables").a[0], "html").s == result.tables[0].html);
        CHECK(field(field(json, "formulas").a[0], "latex").s == result.formulas[0].latex);
        CHECK(field(json, "stages").a.size() == 2);
        CHECK(field(field(json, "stages").a[1], "elapsed_ms").number() == 15.25);

        // Quotes, control bytes, UTF-8 and embedded NUL survive JSON encoding.
        result.regions[0].text = std::string("A\0B", 3) + "\"\t\n\\ <tag> € 日本語";
        json = snapshot(result);
        CHECK(field(field(json, "regions").a[0], "text").s == result.regions[0].text);

        // Invalid decoder bytes remain recoverable while JSON stays UTF-8.
        result.regions[0].text = std::string("A") + "\xff\xc0\x80";
        json = snapshot(result);
        CHECK(field(field(json, "regions").a[0], "text").s == "A\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd");
        CHECK(field(field(json, "regions").a[0], "text_bytes_hex").s == "41ffc080");

        // The PP-OCRv6 legacy rec field is actually a detector-score copy.
        result.used_engine = ocr_orchestrator::engine::ppocrv6;
        json = snapshot(result);
        CHECK(field(field(json, "regions").a[0], "confidence_source").s == "detector");
        CHECK(field(field(json, "regions").a[0], "recognition_confidence").kind == value::null_t);

        // A VLM's legacy placeholder score of one is not proof of certainty.
        result.used_engine = ocr_orchestrator::engine::qwen2vl;
        result.regions[0].confidence = 1.0f;
        result.regions[0].rec_confidence = 0.0f;
        result.regions[0].char_conf.clear();
        json = snapshot(result);
        CHECK(field(field(json, "regions").a[0], "confidence").kind == value::null_t);
        CHECK(field(field(json, "regions").a[0], "confidence_source").s == "unavailable");
        CHECK(field(field(json, "regions").a[0], "sequence_unit").s == "engine_token");

        // Zero is a legitimate measured recognition score, not absence.
        result.regions[0].char_conf = { 0.0f };
        json = snapshot(result);
        CHECK(field(field(json, "regions").a[0], "confidence").number() == 0.0);
        CHECK(field(field(json, "regions").a[0], "confidence_source").s == "recognizer");

        // Non-finite/invalid engine values must not create invalid JSON.
        result.mean_confidence = std::numeric_limits<float>::quiet_NaN();
        result.regions[0].box.x = std::numeric_limits<float>::infinity();
        result.regions[0].rec_confidence = -0.3f;
        result.regions[0].char_conf = { std::numeric_limits<float>::infinity(), -1.0f, 2.0f };
        result.stage_metrics[0].elapsed_ms = std::numeric_limits<float>::infinity();
        json = snapshot(result);
        CHECK(field(json, "mean_engine_confidence").kind == value::null_t);
        CHECK(field(field(json, "regions").a[0], "bbox").a[0].kind == value::null_t);
        for (const auto & item : field(field(json, "regions").a[0], "sequence_confidences").a)
            CHECK(item.kind == value::null_t);

        // JSON remains standards-compliant when the process uses comma decimals.
        const auto locale = std::locale();
        std::locale::global(std::locale(locale, new comma_decimal));
        json = snapshot(fixture());
        std::locale::global(locale);
        CHECK(field(field(json, "stages").a[1], "elapsed_ms").number() == 15.25);

        result = fixture();
        result.coordinates_valid_for_input = false;
        result.cleanup_applied = true;
        result.accepted = false;
        json = snapshot(result, "Restored text.");
        CHECK(field(json, "status").s == "review_required");
        CHECK(!field(field(json, "geometry"), "coordinates_valid_for_input").b);
        CHECK(field(json, "text_postprocessed").b);
        CHECK(field(json, "recognition_text").s == result.full_text);
        CHECK(field(json, "warnings").a.size() == 4);

        result = {};
        json = snapshot(result);
        CHECK(field(json, "status").s == "input_unreadable");
        CHECK(field(json, "detected_language").kind == value::null_t);
        CHECK(field(json, "language_confidence").kind == value::null_t);
        CHECK(field(json, "engine").kind == value::null_t);
        result.input_width = result.input_height = 100;
        json = snapshot(result);
        CHECK(field(json, "status").s == "no_text");
        CHECK(!field(json, "warnings").a.empty());
        CHECK(ocr_review::reading_order(3, {}) == std::vector<int>({ 0, 1, 2 }));
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Unexpected exception: %s\n", error.what());
        ++failures;
    }
    std::printf("OCR review snapshot: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
