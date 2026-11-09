// SPDX-License-Identifier: MIT
// Public lifecycle ABI controls using an owned PDFium-shaped fixture library.
// These prove state/durability behavior, not PDF rendering or OCR qualification.
#include "crispembed.h"
#include "marker_artifact.h"
#include "core/clean_exit.h"
#include "core/json.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MARKER_FIXTURE_LIBRARY
#error "The lifecycle test requires its explicitly selected fixture library"
#endif

namespace fs = std::filesystem;
namespace {
int checks = 0;
const char * source_text = "%PDF-1.7\nMARKER_LIFECYCLE_FIXTURE\nowned source\n";
const char * failing_text = "%PDF-1.7\nMARKER_LIFECYCLE_FIXTURE\nFAIL_SECOND_PAGE\n";
using context = std::unique_ptr<crispembed_marker_context, decltype(&crispembed_marker_free)>;

void check(bool ok, const char * label) {
    ++checks;
    if (!ok) throw std::runtime_error(label);
}
void success(int ok, crispembed_marker_context * ctx, const char * label) {
    if (!ok) std::fprintf(stderr, "%s: %s\n", label, crispembed_marker_error(ctx));
    check(ok == 1, label);
}
void error_has(crispembed_marker_context * ctx, const char * fragment) {
    const char * error = crispembed_marker_error(ctx);
    if (!error || std::string(error).find(fragment) == std::string::npos)
        std::fprintf(stderr, "Expected '%s', got '%s'\n", fragment, error ? error : "(null)");
    check(error && std::string(error).find(fragment) != std::string::npos, "specific lifecycle diagnostic");
}
void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
struct workspace {
    fs::path path;
    workspace() {
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        for (int attempt = 0; attempt < 100; ++attempt) {
            path = fs::temp_directory_path() /
                   ("marker-lifecycle-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            if (fs::create_directory(path)) return;
        }
        throw std::runtime_error("cannot create the owned test directory");
    }
    ~workspace() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

std::string bytes(const fs::path & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read the owned fixture");
    return { std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
}
void write(const fs::path & path, const std::string & content) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(content.data(), static_cast<std::streamsize>(content.size()));
    file.close();
    if (!file) throw std::runtime_error("cannot write the owned fixture");
}
crispembed_marker_config config() {
    auto value = crispembed_marker_default_config();
    value.mode = 1;
    value.pdfium_path = MARKER_FIXTURE_LIBRARY;
    value.extract_images = 0;
    return value;
}
context create(const crispembed_marker_config & cfg) {
    context ctx(crispembed_marker_init(&cfg), crispembed_marker_free);
    if (!ctx) std::fprintf(stderr, "Marker init: %s\n", crispembed_marker_error(nullptr));
    check(ctx != nullptr, "model-free Marker context initializes");
    return ctx;
}
std::string render(crispembed_marker_context * ctx, const char * format = "markdown") {
    const char * value = crispembed_marker_render(ctx, format);
    check(value != nullptr, "completed units remain renderable");
    return value;
}
std::string string_field(const std::string & json, const char * key) {
    const auto at = core_json::json_find_key_value(json, key);
    std::string out;
    size_t end = 0;
    check(at != std::string::npos && core_json::json_decode_string(json, at, out, end), "valid progress string field");
    return out;
}
void progress(crispembed_marker_context * ctx, const char * state, int pages, int total, int next) {
    const char * raw = crispembed_marker_progress(ctx);
    check(raw != nullptr, "progress is available");
    const std::string json(raw);
    check(string_field(json, "schema") == "crispembed.marker-progress/1", "versioned progress schema");
    check(string_field(json, "state") == state, "expected lifecycle state");
    check(core_json::json_extract_number_structural(json, "completed_pages", -1) == pages &&
              core_json::json_extract_number_structural(json, "total_pages", -1) == total &&
              core_json::json_extract_number_structural(json, "next_page", -1) == next,
          "completed page count and cursor are exact");
}
crisp_marker::extraction_snapshot checkpoint(const fs::path & path, int pages, int next, bool complete, int first = 0) {
    const auto snapshot = crisp_marker::load_extraction(path.u8string());
    check(snapshot.value.pages.size() == static_cast<size_t>(pages) && snapshot.next_page == next &&
              snapshot.complete == complete && snapshot.first_page == first,
          "on-disk checkpoint records exactly the completed units");
    check(snapshot.value.metadata.at("extraction_phase") == "raw", "resumable checkpoints remain raw");
    for (int i = 0; i < pages; ++i) {
        check(snapshot.value.pages[static_cast<size_t>(i)].index == first + i, "durable physical page order");
        check(!snapshot.value.pages[static_cast<size_t>(i)].blocks.empty(), "each fixture page retains text blocks");
    }
    return snapshot;
}
bool contains(const std::string & text, const char * part) {
    return text.find(part) != std::string::npos;
}

void gate_controls(const fs::path & root, const fs::path & source) {
    auto ctx = create(config());
    const auto target = root / "gate.cemk";
    const char * const off_values[] = { nullptr, "", "0", "false", "off", "no", "FaLsE", "OfF", "nO" };
    for (const char * gate : { "CRISPEMBED_MARKER", "CRISPEMBED_MARKER_ARTIFACTS", "CRISPEMBED_MARKER_INCREMENTAL" }) {
        for (const char * value : off_values) {
            set_env(gate, value);
            check(crispembed_marker_start(ctx.get(), source.u8string().c_str(), target.u8string().c_str(), 0) == 0,
                  "incremental start rejects off gate values");
            error_has(ctx.get(), gate);
            check(crispembed_marker_step(ctx.get(), 1) == CRISPEMBED_MARKER_JOB_ERROR,
                  "incremental step rejects off gate values");
            error_has(ctx.get(), gate);
            check(!fs::exists(target), "gate rejection creates no checkpoint");
        }
        set_env(gate, "1");
    }
    check(crispembed_marker_start(nullptr, "x", "y", 0) == 0, "null start rejects");
    check(crispembed_marker_step(nullptr, 1) == CRISPEMBED_MARKER_JOB_ERROR, "null step rejects");
    check(crispembed_marker_progress(nullptr) == nullptr, "null progress rejects");
    crispembed_marker_cancel(nullptr);
    check(crispembed_marker_start(ctx.get(), source.u8string().c_str(), target.u8string().c_str(), 2) == 0,
          "invalid resume selector rejects");
    error_has(ctx.get(), "resume=0/1");
    auto no_library = config();
    no_library.pdfium_path = nullptr;
    auto missing = create(no_library);
    check(crispembed_marker_start(missing.get(), source.u8string().c_str(), target.u8string().c_str(), 0) == 0,
          "incremental PDFs require an explicit library");
    error_has(missing.get(), "explicit PDFium");
}

void lifecycle(const fs::path & root, const fs::path & source) {
    const auto raw = root / "lifecycle.cemk";
    const auto view = root / "view.cemk";
    auto ctx = create(config());
    progress(ctx.get(), "idle", 0, 0, 0);
    success(crispembed_marker_start(ctx.get(), source.u8string().c_str(), raw.u8string().c_str(), 0), ctx.get(),
            "start");
    progress(ctx.get(), "running", 0, 3, 0);
    const auto empty = checkpoint(raw, 0, 0, false);
    check(empty.identity.at("build_sha256").size() == 64 && empty.identity.at("build_sha256") != "unrecorded",
          "test exercises a source-identified native build");
    check(empty.identity.at("source_sha256") == crisp_marker::extraction_file_sha256(source.u8string()) &&
              empty.identity.at("pdfium_sha256") == crisp_marker::extraction_file_sha256(MARKER_FIXTURE_LIBRARY) &&
              empty.identity.at("model_sha256") == "none",
          "checkpoint records actual source/library hashes and model-free identity");
    const auto empty_bytes = bytes(raw);
    for (int budget : { 0, -1, 65 }) {
        check(crispembed_marker_step(ctx.get(), budget) == CRISPEMBED_MARKER_JOB_ERROR, "invalid page budget rejects");
        error_has(ctx.get(), "page budget");
        check(bytes(raw) == empty_bytes, "invalid budget leaves the checkpoint unchanged");
        success(crispembed_marker_start(ctx.get(), source.u8string().c_str(), raw.u8string().c_str(), 1), ctx.get(),
                "resume after bad budget");
    }
    check(crispembed_marker_step(ctx.get(), 1) == CRISPEMBED_MARKER_JOB_RUNNING, "one-page budget remains running");
    progress(ctx.get(), "running", 1, 3, 1);
    const auto first = checkpoint(raw, 1, 1, false);
    check(first.value.pages[0].printed_label == "iv" && first.value.pages[0].label_source == "pdfium-page-label",
          "UTF-16 page label survives raw checkpointing");
    const auto first_bytes = bytes(raw);
    const auto first_output = render(ctx.get());
    check(contains(first_output, "First owned page") && !contains(first_output, "Second owned page"),
          "partial view has only committed pages");
    check(bytes(raw) == first_bytes, "rendering does not rewrite the raw checkpoint");
    check(crispembed_marker_save_extraction(ctx.get(), raw.u8string().c_str()) == 0,
          "processed view cannot replace its raw checkpoint");
    error_has(ctx.get(), "separately");
    check(bytes(raw) == first_bytes, "failed view save preserves raw checkpoint bytes");

    crispembed_marker_cancel(ctx.get());
    check(core_json::json_extract_bool(crispembed_marker_progress(ctx.get()), "cancel_requested", false),
          "cancel request is observable");
    check(crispembed_marker_step(ctx.get(), 64) == CRISPEMBED_MARKER_JOB_CANCELLED,
          "cancellation stops at the page boundary");
    progress(ctx.get(), "cancelled", 1, 3, 1);
    check(bytes(raw) == first_bytes && render(ctx.get()) == first_output, "cancellation retains durable output");
    check(crispembed_marker_step(ctx.get(), 1) == CRISPEMBED_MARKER_JOB_CANCELLED,
          "cancelled jobs require explicit resume");
    ctx.reset();

    ctx = create(config());
    success(crispembed_marker_start(ctx.get(), source.u8string().c_str(), raw.u8string().c_str(), 1), ctx.get(),
            "fresh-context resume");
    progress(ctx.get(), "running", 1, 3, 1);
    check(!core_json::json_extract_bool(crispembed_marker_progress(ctx.get()), "cancel_requested", true),
          "resume clears cancellation");
    check(crispembed_marker_step(ctx.get(), 1) == CRISPEMBED_MARKER_JOB_RUNNING, "resume extracts the next page only");
    const auto second = checkpoint(raw, 2, 2, false);
    check(second.value.pages[1].printed_label == u8"\u7ae0-2", "non-ASCII BMP page label survives");
    check(second.value.pages[0].blocks[0].text == first.value.pages[0].blocks[0].text,
          "resume preserves previous raw page text");
    check(crispembed_marker_step(ctx.get(), 64) == CRISPEMBED_MARKER_JOB_COMPLETE,
          "budget larger than remaining work completes");
    progress(ctx.get(), "complete", 3, 3, 3);
    const auto done = checkpoint(raw, 3, 3, true);
    check(done.value.pages[2].printed_label == u8"\U0001f4d6-3",
          "UTF-16 surrogate pair becomes the exact astral UTF-8 label");
    const auto complete_bytes = bytes(raw);
    const auto complete_output = render(ctx.get());
    check(contains(complete_output, "First owned page") && contains(complete_output, "Second owned page") &&
              contains(complete_output, "Third owned page"),
          "completed view includes every owned page");
    const auto json = render(ctx.get(), "json");
    check(contains(json, "\"printed_page_label\":\"iv\"") && contains(json, u8"\U0001f4d6-3"),
          "public JSON retains exact page labels");
    check(crispembed_marker_step(ctx.get(), 1) == CRISPEMBED_MARKER_JOB_COMPLETE && bytes(raw) == complete_bytes,
          "completed job is idempotent");
    success(crispembed_marker_save_extraction(ctx.get(), view.u8string().c_str()), ctx.get(),
            "save processed artifact");
    const auto saved_view = crisp_marker::load_extraction(view.u8string());
    check(saved_view.complete && saved_view.value.metadata.at("extraction_phase") == "processed" &&
              saved_view.identity == done.identity,
          "processed artifact keeps complete identity and phase");

    auto standalone_cfg = config();
    standalone_cfg.pdfium_path = nullptr;
    standalone_cfg.dpi += 1;
    auto standalone = create(standalone_cfg);
    success(crispembed_marker_open_extraction(standalone.get(), view.u8string().c_str()), standalone.get(),
            "standalone artifact open");
    check(render(standalone.get()) == complete_output && crispembed_marker_asset_count(standalone.get()) == 0,
          "processed view reopens without a model or PDFium path");
    check(crispembed_marker_open_extraction(standalone.get(), raw.u8string().c_str()) == 0,
          "raw view rejects mismatched processing settings");
    error_has(standalone.get(), "matching processing settings");
    check(render(standalone.get()) == complete_output, "failed artifact open preserves the previous view");

    auto corrupted = complete_bytes;
    check(corrupted.size() > 88, "owned checkpoint has the versioned integrity envelope");
    corrupted[24] = corrupted[24] == '0' ? '1' : '0';
    const auto bad = root / "corrupted.cemk";
    write(bad, corrupted);
    check(crispembed_marker_start(ctx.get(), source.u8string().c_str(), bad.u8string().c_str(), 1) == 0,
          "corrupted resume rejects");
    error_has(ctx.get(), "integrity mismatch");
    check(crispembed_marker_open_extraction(standalone.get(), bad.u8string().c_str()) == 0,
          "corrupted standalone open rejects");
    error_has(standalone.get(), "integrity mismatch");
    check(render(standalone.get()) == complete_output && bytes(raw) == complete_bytes,
          "corruption does not replace valid state");

    check(crispembed_marker_start(ctx.get(), source.u8string().c_str(), raw.u8string().c_str(), 0) == 0,
          "fresh start will not overwrite a checkpoint");
    error_has(ctx.get(), "already exists");
    check(crispembed_marker_start(ctx.get(), source.u8string().c_str(), view.u8string().c_str(), 1) == 0,
          "processed artifact cannot be resumed as raw");
    error_has(ctx.get(), "raw extraction checkpoint");
    success(crispembed_marker_start(ctx.get(), source.u8string().c_str(), raw.u8string().c_str(), 1), ctx.get(),
            "resume complete checkpoint");
    check(crispembed_marker_step(ctx.get(), 1) == CRISPEMBED_MARKER_JOB_COMPLETE && bytes(raw) == complete_bytes,
          "complete resume does not duplicate a page");
}

void view_provenance(const fs::path & root, const fs::path & source) {
    auto legacy = create(config());
    success(crispembed_marker_convert(legacy.get(), source.u8string().c_str()), legacy.get(),
            "legacy provenance reference converts");
    const auto baseline = root / "legacy-provenance.cemk";
    success(crispembed_marker_save_extraction(legacy.get(), baseline.u8string().c_str()), legacy.get(),
            "legacy provenance reference saves");
    const auto expected = crisp_marker::load_extraction(baseline.u8string()).value.metadata;
    const char * fields[] = { "implementation",     "software_version",   "marker_source_revision", "schema",
                              "surya_model_parity", "text_quality_check", "layout_backend" };
    for (const char * key : fields)
        check(expected.count(key) && !expected.at(key).empty(), "legacy provenance field is present");
    check(expected.at("surya_model_parity") == "not_claimed" && expected.at("layout_backend") == "pdftext_heuristics" &&
              !expected.count("model_file"),
          "text-only reference does not claim model inference");
    const auto check_provenance = [&](const crisp_marker::extraction_snapshot & snapshot) {
        for (const char * key : fields) {
            const auto actual = snapshot.value.metadata.find(key);
            if (actual == snapshot.value.metadata.end() || actual->second != expected.at(key))
                std::fprintf(stderr, "Incremental provenance field differs: %s\n", key);
            check(actual != snapshot.value.metadata.end() && actual->second == expected.at(key),
                  "raw checkpoints and processed views preserve legacy provenance");
        }
    };

    auto incremental = create(config());
    const auto raw = root / "provenance-raw.cemk";
    success(crispembed_marker_start(incremental.get(), source.u8string().c_str(), raw.u8string().c_str(), 0),
            incremental.get(), "provenance job starts");
    check_provenance(crisp_marker::load_extraction(raw.u8string()));
    for (int budget : { 1, 64 }) {
        const auto expected_status = budget == 1 ? CRISPEMBED_MARKER_JOB_RUNNING : CRISPEMBED_MARKER_JOB_COMPLETE;
        check(crispembed_marker_step(incremental.get(), budget) == expected_status,
              "provenance probe reaches partial and complete states");
        const auto view = root / ("provenance-view-" + std::to_string(budget) + ".cemk");
        success(crispembed_marker_save_extraction(incremental.get(), view.u8string().c_str()), incremental.get(),
                "provenance view saves");
        const auto snapshot = crisp_marker::load_extraction(view.u8string());
        check_provenance(snapshot);
        check_provenance(crisp_marker::load_extraction(raw.u8string()));
        check(!snapshot.value.metadata.count("model_file") && !snapshot.value.metadata.count("elapsed_seconds"),
              "incremental text views do not invent a model or uninterrupted timing");
        const std::string metadata = crispembed_marker_metadata(incremental.get());
        auto standalone_config = config();
        standalone_config.pdfium_path = nullptr;
        standalone_config.dpi += 1;
        auto standalone = create(standalone_config);
        success(crispembed_marker_open_extraction(standalone.get(), view.u8string().c_str()), standalone.get(),
                "provenance view reopens without PDFium");
        check(metadata == crispembed_marker_metadata(standalone.get()),
              "partial and complete artifact reopen preserves all metadata bytes");

        auto raw_reader = create(config());
        success(crispembed_marker_open_extraction(raw_reader.get(), raw.u8string().c_str()), raw_reader.get(),
                "raw checkpoint opens with matching processing settings");
        const auto raw_view = root / ("provenance-raw-view-" + std::to_string(budget) + ".cemk");
        success(crispembed_marker_save_extraction(raw_reader.get(), raw_view.u8string().c_str()), raw_reader.get(),
                "raw checkpoint processing saves a view");
        check_provenance(crisp_marker::load_extraction(raw_view.u8string()));
    }
}

void identity_controls(const fs::path & root, const fs::path & source) {
    const auto raw = root / "identity.cemk";
    auto ctx = create(config());
    success(crispembed_marker_start(ctx.get(), source.u8string().c_str(), raw.u8string().c_str(), 0), ctx.get(),
            "identity start");
    check(crispembed_marker_step(ctx.get(), 1) == CRISPEMBED_MARKER_JOB_RUNNING, "identity fixture commits one page");
    const auto original = bytes(raw);
    auto changed = config();
    ++changed.dpi;
    auto other = create(changed);
    check(crispembed_marker_start(other.get(), source.u8string().c_str(), raw.u8string().c_str(), 1) == 0,
          "configuration mismatch rejects resume");
    error_has(other.get(), "does not match");
    changed = config();
    changed.first_page = 1;
    other = create(changed);
    check(crispembed_marker_start(other.get(), source.u8string().c_str(), raw.u8string().c_str(), 1) == 0,
          "selection mismatch rejects resume");
    error_has(other.get(), "does not match");
    set_env("CRISPEMBED_MARKER_REFERENCES", "0");
    check(crispembed_marker_start(ctx.get(), source.u8string().c_str(), raw.u8string().c_str(), 1) == 0,
          "processing gate mismatch rejects resume");
    error_has(ctx.get(), "does not match");
    set_env("CRISPEMBED_MARKER_REFERENCES", "1");

    const auto source_bytes = bytes(source);
    write(source, source_bytes + "changed source bytes\n");
    check(crispembed_marker_start(ctx.get(), source.u8string().c_str(), raw.u8string().c_str(), 1) == 0,
          "source mismatch rejects resume");
    error_has(ctx.get(), "does not match");
    check(crispembed_marker_step(ctx.get(), 1) == CRISPEMBED_MARKER_JOB_ERROR, "active job rejects changed input");
    error_has(ctx.get(), "identity changed");
    check(bytes(raw) == original && contains(render(ctx.get()), "First owned page"),
          "identity error retains committed units");
    write(source, source_bytes);

    for (const char * key : { "model_sha256", "pdfium_sha256", "build_sha256" }) {
        auto altered = crisp_marker::load_extraction(raw.u8string());
        altered.identity[key] = std::string(64, '0');
        const auto mismatch = root / (std::string(key) + ".cemk");
        crisp_marker::save_extraction(mismatch.u8string(), altered);
        check(crispembed_marker_start(ctx.get(), source.u8string().c_str(), mismatch.u8string().c_str(), 1) == 0,
              "model/library/build identity mismatch rejects without model weights");
        error_has(ctx.get(), "does not match");
    }
    check(bytes(raw) == original, "failed identity probes never rewrite the valid checkpoint");
}

void failure_controls(const fs::path & root, const fs::path & source, const fs::path & failing) {
    const auto raw = root / "failed-page.cemk";
    auto ctx = create(config());
    success(crispembed_marker_start(ctx.get(), failing.u8string().c_str(), raw.u8string().c_str(), 0), ctx.get(),
            "failure start");
    check(crispembed_marker_step(ctx.get(), 64) == CRISPEMBED_MARKER_JOB_ERROR,
          "second-page fixture failure propagates");
    error_has(ctx.get(), "page 1");
    progress(ctx.get(), "error", 1, 3, 1);
    checkpoint(raw, 1, 1, false);
    const auto committed = bytes(raw);
    const auto partial = render(ctx.get());
    check(contains(partial, "First owned page") && !contains(partial, "Second owned page"),
          "failed page leaves the first page inspectable");
    check(crispembed_marker_step(ctx.get(), 1) == CRISPEMBED_MARKER_JOB_ERROR, "failed job requires explicit resume");
    error_has(ctx.get(), "resumed");
    ctx.reset();
    ctx = create(config());
    success(crispembed_marker_start(ctx.get(), failing.u8string().c_str(), raw.u8string().c_str(), 1), ctx.get(),
            "failed job resumes in fresh context");
    check(crispembed_marker_step(ctx.get(), 1) == CRISPEMBED_MARKER_JOB_ERROR && bytes(raw) == committed,
          "retrying the failing page cannot erase or duplicate completed units");

    const auto blocked = root / "blocked.cemk";
    const auto preserved = root / "preserved.cemk";
    success(crispembed_marker_start(ctx.get(), source.u8string().c_str(), blocked.u8string().c_str(), 0), ctx.get(),
            "checkpoint failure start");
    check(crispembed_marker_step(ctx.get(), 1) == CRISPEMBED_MARKER_JOB_RUNNING,
          "checkpoint failure fixture commits page one");
    const auto before = bytes(blocked);
    fs::rename(blocked, preserved);
    fs::create_directory(blocked);
    check(crispembed_marker_step(ctx.get(), 1) == CRISPEMBED_MARKER_JOB_ERROR,
          "failed checkpoint write rejects the current page");
    error_has(ctx.get(), "regular");
    progress(ctx.get(), "error", 1, 3, 1);
    const auto rollback = render(ctx.get());
    check(contains(rollback, "First owned page") && !contains(rollback, "Second owned page") &&
              bytes(preserved) == before,
          "checkpoint write failure rolls back to the durable page boundary");
    fs::remove(blocked);
    fs::rename(preserved, blocked);
    success(crispembed_marker_start(ctx.get(), source.u8string().c_str(), blocked.u8string().c_str(), 1), ctx.get(),
            "resume after checkpoint destination repair");
    check(crispembed_marker_step(ctx.get(), 64) == CRISPEMBED_MARKER_JOB_COMPLETE,
          "repaired destination completes remaining pages");
    checkpoint(blocked, 3, 3, true);
}

void alias_controls(const fs::path & root, const fs::path & source) {
    const auto model = root / "model-sentinel.gguf";
    write(model, "owned model-path sentinel, never loaded in text mode\n");
    auto cfg = config();
    const std::string model_name = model.u8string();
    cfg.model_path = model_name.c_str();
    auto ctx = create(cfg);
    success(crispembed_marker_convert(ctx.get(), source.u8string().c_str()), ctx.get(), "alias fixture conversion");
    std::vector<fs::path> paths{ source, source.parent_path() / "." / source.filename(), model,
                                 fs::u8path(MARKER_FIXTURE_LIBRARY) };
    for (const auto & target : { source, model }) {
        const auto alias = root / ("hardlink-" + target.filename().u8string());
        std::error_code error;
        fs::create_hard_link(target, alias, error);
        if (!error)
            paths.push_back(alias);
        else
            std::fprintf(stderr, "SKIP: filesystem does not support owned hardlink fixture: %s\n",
                         error.message().c_str());
    }
    for (const auto & path : paths) {
        const auto digest = crisp_marker::extraction_file_sha256(path.u8string());
        check(crispembed_marker_start(ctx.get(), source.u8string().c_str(), path.u8string().c_str(), 0) == 0,
              "checkpoint refuses source/model/library output aliases");
        error_has(ctx.get(), "must not replace");
        check(crispembed_marker_save_extraction(ctx.get(), path.u8string().c_str()) == 0,
              "artifact save refuses source/model/library output aliases");
        error_has(ctx.get(), "must not replace");
        check(crisp_marker::extraction_file_sha256(path.u8string()) == digest,
              "protected alias bytes remain unchanged");
    }
}

void legacy_and_selection(const fs::path & root, const fs::path & source, const fs::path & failing) {
    auto cfg = config();
    cfg.first_page = cfg.last_page = 1;
    auto ctx = create(cfg);
    const auto raw = root / "selected.cemk";
    success(crispembed_marker_start(ctx.get(), source.u8string().c_str(), raw.u8string().c_str(), 0), ctx.get(),
            "nonzero selection start");
    check(crispembed_marker_step(ctx.get(), 64) == CRISPEMBED_MARKER_JOB_COMPLETE,
          "bounded selection completes exactly one page");
    progress(ctx.get(), "complete", 1, 1, 2);
    checkpoint(raw, 1, 2, true, 1);
    check(contains(render(ctx.get()), "Second owned page") && !contains(render(ctx.get()), "First owned page"),
          "physical page selection is preserved");
    ctx.reset();

    set_env("CRISPEMBED_MARKER_ARTIFACTS", "0");
    set_env("CRISPEMBED_MARKER_INCREMENTAL", "0");
    set_env("CRISPEMBED_MARKER_REFERENCES", "0");
    ctx = create(config());
    success(crispembed_marker_convert(ctx.get(), source.u8string().c_str()), ctx.get(),
            "legacy conversion without new gates");
    check(contains(render(ctx.get()), "Third owned page"), "legacy conversion still processes all pages");
    check(!contains(render(ctx.get(), "json"), "printed_page_label"), "page-label projection remains default-off");
    check(crispembed_marker_convert(ctx.get(), failing.u8string().c_str()) == 0,
          "legacy conversion propagates later page failure");
    error_has(ctx.get(), "page 1");
    check(crispembed_marker_render(ctx.get(), "json") == nullptr && crispembed_marker_metadata(ctx.get()) == nullptr &&
              crispembed_marker_asset_count(ctx.get()) == 0,
          "legacy failure clears all prior and partial output");
    success(crispembed_marker_convert(ctx.get(), source.u8string().c_str()), ctx.get(), "legacy conversion recovers");
    set_env("CRISPEMBED_MARKER_ARTIFACTS", "1");
    set_env("CRISPEMBED_MARKER_INCREMENTAL", "1");
    set_env("CRISPEMBED_MARKER_REFERENCES", "1");
}

void active_library_identity(const fs::path & root, const fs::path & source) {
#if defined(__unix__) || defined(__APPLE__)
    // Replace an owned copy atomically: writing into a mapped library's inode
    // could corrupt live code. Windows locks loaded DLLs, so this filesystem
    // replacement scenario is exercised only on platforms that permit it.
    const auto original_library = fs::u8path(MARKER_FIXTURE_LIBRARY);
    const auto library = root / ("fixture-copy" + original_library.extension().u8string());
    fs::copy_file(original_library, library);
    const auto raw = root / "live-library.cemk";
    const auto other_raw = root / "changed-library.cemk";
    auto cfg = config();
    const std::string library_name = library.u8string();
    cfg.pdfium_path = library_name.c_str();
    auto active = create(cfg);
    success(crispembed_marker_start(active.get(), source.u8string().c_str(), raw.u8string().c_str(), 0), active.get(),
            "owned library identity start");
    const auto retained = bytes(raw);
    const auto replacement = root / "replacement-library";
    write(replacement, bytes(library) + "\nowned trailing identity change\n");
    fs::rename(replacement, library);
    auto fresh = create(cfg);
    check(crispembed_marker_start(fresh.get(), source.u8string().c_str(), other_raw.u8string().c_str(), 0) == 0,
          "fresh job cannot attribute a still-loaded library to replacement bytes");
    error_has(fresh.get(), "active PDFium library identity");
    check(!fs::exists(other_raw), "live library mismatch creates no checkpoint");
    check(crispembed_marker_step(active.get(), 1) == CRISPEMBED_MARKER_JOB_ERROR,
          "original job rejects a replaced PDFium file");
    error_has(active.get(), "identity changed");
    check(bytes(raw) == retained, "library replacement preserves the last durable checkpoint");
#else
    (void)root;
    (void)source;
#endif
}

int test_main() {
    try {
        workspace owned;
        const auto source = owned.path / "source.pdf";
        const auto failing = owned.path / "failing.pdf";
        write(source, source_text);
        write(failing, failing_text);
        for (const char * name : { "CRISPEMBED_MARKER", "CRISPEMBED_MARKER_ARTIFACTS", "CRISPEMBED_MARKER_INCREMENTAL",
                                   "CRISPEMBED_MARKER_REFERENCES" })
            set_env(name, "1");
        for (const char * name : { "CRISPEMBED_MARKER_REPETITION_CHECK", "CRISPEMBED_MARKER_SOURCE_LINKS",
                                   "CRISPEMBED_MARKER_MULTILINGUAL_NORMALIZE", "CRISPEMBED_TELEOCR",
                                   "CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE" })
            set_env(name, "0");
        check(fs::is_regular_file(fs::u8path(MARKER_FIXTURE_LIBRARY)), "explicit owned PDFium fixture exists");
        gate_controls(owned.path, source);
        lifecycle(owned.path, source);
        view_provenance(owned.path, source);
        identity_controls(owned.path, source);
        failure_controls(owned.path, source, failing);
        alias_controls(owned.path, source);
        legacy_and_selection(owned.path, source, failing);
        active_library_identity(owned.path, source);
        std::printf("Marker lifecycle ABI: %d checks passed (owned fixture; no rendering/OCR qualification)\n", checks);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL after %d lifecycle checks: %s\n", checks, error.what());
        return 1;
    }
}
} // namespace

int main() {
    core_util::clean_exit(test_main());
}
