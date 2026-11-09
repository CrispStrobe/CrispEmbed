// SPDX-License-Identifier: MIT
// Owned, model-free artifact fixtures. No external document or checkpoint.
#include "marker_artifact.h"
#include "core/clean_exit.h"
#include "core/sha256.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace crisp_marker;
namespace fs = std::filesystem;

namespace {
int checks = 0, failures = 0;
void check(bool condition, const char * name) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", name);
    }
}
void throws(const std::function<void()> & fn, const char * name, const std::string & fragment = {}) {
    bool caught = false;
    try {
        fn();
    } catch (const std::exception & error) {
        caught = fragment.empty() || std::string(error.what()).find(fragment) != std::string::npos;
        if (!caught) std::fprintf(stderr, "Unexpected error for %s: %s\n", name, error.what());
    }
    check(caught, name);
}

struct workspace {
    fs::path path;
    workspace() {
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        for (size_t attempt = 0; attempt < 100; ++attempt) {
            path = fs::temp_directory_path() /
                   ("crisp-marker-artifact-test-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            if (fs::create_directory(path)) return;
        }
        throw std::runtime_error("could not create the test directory");
    }
    ~workspace() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

std::string bytes(const fs::path & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("fixture read failed");
    return { std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>() };
}
void write(const fs::path & path, const std::string & content) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
    stream.close();
    if (!stream) throw std::runtime_error("fixture write failed");
}
void little(std::string & value, size_t at, uint64_t number, size_t count = 8) {
    for (size_t i = 0; i < count; ++i) value.at(at + i) = static_cast<char>(number >> (8 * i));
}
void reseal(std::string & value) {
    core_sha256::sha256_state digest;
    core_sha256::sha256_update(digest, reinterpret_cast<const uint8_t *>(value.data()), 24);
    core_sha256::sha256_update(digest, reinterpret_cast<const uint8_t *>(value.data() + 88), value.size() - 88);
    value.replace(24, 64, core_sha256::sha256_finish(digest));
}
void replace_one(std::string & value, const std::string & from, const std::string & to) {
    const auto found = value.find(from, 88);
    if (found == std::string::npos || from.size() != to.size()) throw std::runtime_error("fixture marker not found");
    value.replace(found, from.size(), to);
}

bool same_real(double a, double b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}
bool same(const box & a, const box & b) {
    return same_real(a.x0, b.x0) && same_real(a.y0, b.y0) && same_real(a.x1, b.x1) && same_real(a.y1, b.y1);
}
bool same(const point & a, const point & b) {
    return same_real(a.x, b.x) && same_real(a.y, b.y);
}
bool same(const span & a, const span & b) {
    return a.text == b.text && same(a.bbox, b.bbox) && same_real(a.font_size, b.font_size) && a.bold == b.bold &&
           a.italic == b.italic && a.monospace == b.monospace && a.url == b.url;
}
bool same(const line &, const line &);
bool same(const source_range &, const source_range &);
bool same(const footnote_reference &, const footnote_reference &);
bool same(const block &, const block &);
bool same(const page &, const page &);
bool same(const asset &, const asset &);
bool same(const source_record &, const source_record &);
template <typename T> bool same_vector(const std::vector<T> & a, const std::vector<T> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!same(a[i], b[i])) return false;
    return true;
}
bool same(const line & a, const line & b) {
    return same(a.bbox, b.bbox) && same_vector(a.spans, b.spans);
}
bool same(const source_range & a, const source_range & b) {
    return a.source_id == b.source_id && a.source_start == b.source_start && a.source_end == b.source_end &&
           a.text_start == b.text_start && a.text_end == b.text_end && a.transform == b.transform;
}
bool same(const footnote_reference & a, const footnote_reference & b) {
    return a.start == b.start && a.end == b.end && a.label == b.label && a.status == b.status &&
           a.target_id == b.target_id && a.candidates == b.candidates;
}
bool same(const block & a, const block & b) {
    return a.id == b.id && a.type == b.type && a.text == b.text && a.html == b.html && a.raw == b.raw &&
           a.raw_label == b.raw_label && a.source == b.source && same(a.bbox, b.bbox) &&
           same_vector(a.polygon, b.polygon) && same_vector(a.lines, b.lines) && same_vector(a.children, b.children) &&
           a.section_hierarchy == b.section_hierarchy && a.warnings == b.warnings && a.image_name == b.image_name &&
           a.order == b.order && a.heading_level == b.heading_level && a.angle == b.angle &&
           a.generated_tokens == b.generated_tokens && same_real(a.confidence, b.confidence) &&
           a.token_cap_reached == b.token_cap_reached && a.ignored == b.ignored && a.continuation == b.continuation &&
           a.attributes == b.attributes && same_vector(a.source_ranges, b.source_ranges) &&
           a.continuation_target == b.continuation_target && a.footnote_id == b.footnote_id &&
           same_vector(a.references, b.references);
}
bool same(const page & a, const page & b) {
    return a.index == b.index && same_real(a.width, b.width) && same_real(a.height, b.height) &&
           same_vector(a.text_lines, b.text_lines) && same_vector(a.blocks, b.blocks) && a.warnings == b.warnings &&
           a.extraction_method == b.extraction_method && a.raw_layout == b.raw_layout &&
           a.layout_generated_tokens == b.layout_generated_tokens &&
           a.layout_token_cap_reached == b.layout_token_cap_reached && a.printed_label == b.printed_label &&
           a.label_source == b.label_source && a.label_status == b.label_status &&
           a.label_candidates == b.label_candidates;
}
bool same(const asset & a, const asset & b) {
    return a.name == b.name && a.mime_type == b.mime_type && a.bytes == b.bytes;
}
bool same(const source_record & a, const source_record & b) {
    return a.id == b.id && a.text == b.text && a.source == b.source && a.page_index == b.page_index &&
           same(a.bbox, b.bbox) && same_vector(a.polygon, b.polygon);
}
bool same(const document & a, const document & b) {
    return a.source_name == b.source_name && a.backend == b.backend && same_vector(a.pages, b.pages) &&
           same_vector(a.assets, b.assets) && a.warnings == b.warnings && a.metadata == b.metadata &&
           same_vector(a.sources, b.sources);
}
bool same(const extraction_snapshot & a, const extraction_snapshot & b) {
    return same(a.value, b.value) && a.identity == b.identity && a.first_page == b.first_page &&
           a.last_page == b.last_page && a.next_page == b.next_page && a.complete == b.complete;
}

extraction_snapshot fixture() {
    extraction_snapshot out;
    out.first_page = 5;
    out.last_page = 9;
    out.next_page = 7;
    out.identity = { { "source_sha256", std::string(64, 'a') },
                     { "model_sha256", "none" },
                     { "preprocessing_fingerprint", std::string(64, 'b') },
                     { "identity_gate_A", "off" },
                     { "identity_gate_B", "on" } };
    out.value.source_name = "owned-source.pdf";
    out.value.backend = "owned-fixture";
    out.value.metadata = { { "extraction_phase", "raw" }, { std::string("key\0", 4), std::string("value\0\xff", 7) } };
    out.value.warnings = { "A document warning", std::string("binary\0warning", 14) };
    page first;
    first.index = 5;
    first.width = 612.5;
    first.height = 792.25;
    first.extraction_method = "owned-pdftext";
    first.raw_layout = std::string("layout\0\xff\x80", 9);
    first.layout_generated_tokens = 77;
    first.layout_token_cap_reached = true;
    first.warnings = { "A page warning" };
    first.printed_label = "iv";
    first.label_source = "pdf-page-labels";
    first.label_status = "explicit";
    first.label_candidates = { "iv", "4" };
    line row;
    row.bbox = { -0.0, 10.5, 320.75, 40.125 };
    span segment;
    segment.text = std::string("A\0B\xff", 4);
    segment.bbox = { 0.25, 11.5, 120.5, 30.125 };
    segment.font_size = 12.75;
    segment.bold = segment.italic = segment.monospace = true;
    segment.url = "https://example.invalid/?a=1&b=2";
    row.spans = { segment };
    first.text_lines = { row };
    block body;
    body.id = "block-a";
    body.type = "ListItem";
    body.text = std::string("A\0B\xff", 4) + " 中文 [1]";
    body.raw = std::string("RAW_SENTINEL\0\xff\x80", 15);
    body.html = "<p>raw &amp; markup</p>";
    body.raw_label = "owned-original-label";
    body.source = "owned-origin";
    body.bbox = row.bbox;
    body.polygon = { { -0.0, 10.5 }, { 320.75, 10.5 }, { 320.75, 40.125 }, { 0.0, 40.125 } };
    body.lines = { row };
    body.section_hierarchy = { { 1, "heading-a" }, { 3, "heading-b" } };
    body.warnings = { "A block warning" };
    body.image_name = "asset001.png";
    body.order = 6;
    body.heading_level = 3;
    body.angle = 90;
    body.generated_tokens = 23;
    body.confidence = 0.875;
    body.token_cap_reached = body.ignored = body.continuation = true;
    body.attributes = { { "provider.kind", "nested" }, { "raw.bytes", std::string("\0\xff", 2) } };
    body.source_ranges = { { "source-a", 0, 4, 0, 4, "copy" },
                           { "source-gap", 0, 2, 4, 4, "whitespace" },
                           { "source-a", 4, body.text.size(), 4, body.text.size(), "copy" } };
    body.continuation_target = "block-b";
    body.footnote_id = "footnote-a";
    body.references = { { body.text.size() - 3, body.text.size(), "1", "ambiguous", "", { "note-1", "note-2" } } };
    block child;
    child.id = "block-child";
    child.text = "nested detail";
    child.bbox = { 20, 20, 300, 30 };
    child.source = "owned-child";
    body.children = { child };
    first.blocks = { body };
    page second;
    second.index = 6;
    second.width = 600;
    second.height = 800;
    second.printed_label = "v";
    second.extraction_method = "owned-pdftext";
    block tail;
    tail.id = "block-b";
    tail.text = "continuation";
    tail.bbox = { 20, 30, 500, 50 };
    second.blocks = { tail };
    out.value.pages = { first, second };
    out.value.sources = { { "source-a", body.text, "owned-origin", 5, body.bbox, body.polygon },
                          { "source-gap", "\r\n", "owned-origin", 5, body.bbox, body.polygon },
                          { "source-history", "retained earlier extraction", "owned-origin", 5, body.bbox, {} } };
    out.value.assets = { { "asset001.png", "image/png", { 0, 1, 0xff, 0x89, 'P', 'N', 'G', 0 } } };
    return out;
}

void roundtrip(const fs::path & root) {
    const auto path = root / "checkpoint.cemk";
    const auto copy = root / "copy.cemk";
    auto value = fixture();
    auto initial = value;
    initial.next_page = initial.first_page;
    initial.value.pages.clear();
    initial.value.assets.clear();
    initial.value.sources.clear();
    const auto initial_path = root / "initial.cemk";
    save_extraction(initial_path.u8string(), initial);
    check(same(initial, load_extraction(initial_path.u8string())),
          "empty initial checkpoint round-trips before page one");
    save_extraction(path.u8string(), value);
    const auto restored = load_extraction(path.u8string());
    check(same(value, restored), "every IR field and binary string round-trips in a partial snapshot");
    check(!restored.complete && restored.next_page == 7 && restored.value.metadata.at("extraction_phase") == "raw",
          "partial checkpoint and raw phase remain explicit");
    save_extraction(copy.u8string(), restored);
    check(bytes(path) == bytes(copy), "portable encoding is deterministic after a round trip");
    check(bytes(path).compare(0, 8, std::string("CRISPMK\0", 8)) == 0 && bytes(path)[8] == 1,
          "version-one header has fixed magic and little-endian version");
    value.last_page = 6;
    value.complete = true;
    save_extraction(copy.u8string(), value);
    check(load_extraction(copy.u8string()).complete &&
              load_extraction(copy.u8string()).value.metadata.at("extraction_phase") == "raw",
          "complete raw checkpoint does not imply a processed view");
    value.value.metadata["extraction_phase"] = "processed";
    save_extraction(path.u8string(), value);
    check(same(value, load_extraction(path.u8string())), "complete finalized snapshot atomically replaces checkpoint");
    const auto known = bytes(path);
    check(extraction_file_sha256(path.u8string()) == core_sha256::bytes(known.data(), known.size()),
          "streaming file hash covers exact artifact bytes");
    const auto abc = root / "abc";
    write(abc, "abc");
    check(extraction_file_sha256(abc.u8string()) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "streaming identity uses the standard SHA-256 digest");
    for (const auto & entry : fs::directory_iterator(root))
        check(entry.path().filename().u8string().find(".crispembed-marker-") != 0,
              "successful save leaves no temporary sibling");
}

void malformed(const fs::path & root) {
    const auto path = root / "malformed.cemk";
    save_extraction(path.u8string(), fixture());
    const std::string original = bytes(path);
    auto reject = [&](std::string value, const char * name, const std::string & message = {}) {
        write(path, value);
        throws([&] { (void)load_extraction(path.u8string()); }, name, message);
    };
    auto value = original;
    value[0] = 'X';
    reject(value, "unknown magic is rejected", "invalid magic");
    value = original;
    little(value, 8, 2, 4);
    reject(value, "unknown versions are rejected", "unsupported format version");
    value = original;
    little(value, 12, 1, 4);
    reject(value, "unknown flags are rejected", "unsupported format flags");
    value = original;
    little(value, 16, std::numeric_limits<uint64_t>::max());
    reject(value, "oversized advertised payload rejected before allocation", "file byte limit");
    value = original;
    replace_one(value, "RAW_SENTINEL", "RAW_SENTINEQ");
    reject(value, "payload corruption fails SHA-256 verification", "integrity mismatch");
    value = original;
    value[24] = 'z';
    reject(value, "invalid digest encoding is rejected", "digest encoding");
    reject(original.substr(0, original.size() - 1), "truncated payload is rejected");
    reject(original + "extra", "trailing bytes are rejected", "trailing data");
    value = original;
    value[88 + 12] = 2;
    reseal(value);
    reject(value, "invalid boolean rejected despite a valid integrity hash", "invalid boolean");
    value = original;
    little(value, 88 + 13, std::numeric_limits<uint64_t>::max());
    reseal(value);
    reject(value, "malicious collection count rejected before allocation", "collection limit");
    value = original;
    little(value, 88 + 13 + 8, 129);
    reseal(value);
    reject(value, "oversized identity key rejected before allocation", "string byte limit");
    value = original;
    replace_one(value, "identity_gate_B", "identity_gate_A");
    reseal(value);
    reject(value, "duplicate identity key is rejected", "duplicate map key");
    value = original;
    replace_one(value, "source-a", "source-x");
    reseal(value);
    reject(value, "unresolved source reference is rejected", "unresolved source identity");
    value = original;
    const auto block_text = fixture().value.pages[0].blocks[0].text;
    auto changed_text = block_text;
    changed_text[0] = 'Z';
    replace_one(value, block_text, changed_text);
    reseal(value);
    reject(value, "forged copy mapping rejected despite a valid integrity hash", "invalid source copy range");
    value = original;
    replace_one(value, "asset001.png", "../evil_.png");
    reseal(value);
    reject(value, "embedded image path traversal rejected", "unsafe image member");
    value = original;
    little(value, 88 + 8, 8, 4);
    reseal(value);
    reject(value, "inconsistent next-page checkpoint rejected", "page checkpoint");
}

void limits_and_atomicity(const fs::path & root) {
    const auto path = root / "durable.cemk";
    const auto original = fixture();
    save_extraction(path.u8string(), original);
    const auto known = bytes(path);
    auto save_reject = [&](const extraction_snapshot & value, const artifact_limits & limits, const char * name) {
        throws([&] { save_extraction(path.u8string(), value, limits); }, name);
        check(bytes(path) == known, "failed save preserves the previous complete bytes");
    };
    artifact_limits limits;
    limits.max_pages = 1;
    save_reject(original, limits, "writer enforces page limit");
    throws([&] { (void)load_extraction(path.u8string(), limits); }, "reader enforces page limit");
    limits = {};
    limits.max_asset_bytes = 7;
    save_reject(original, limits, "writer enforces aggregate asset limit");
    throws([&] { (void)load_extraction(path.u8string(), limits); }, "reader enforces aggregate asset limit");
    limits = {};
    limits.max_string_bytes = 4;
    save_reject(original, limits, "writer enforces string limit");
    throws([&] { (void)load_extraction(path.u8string(), limits); }, "reader enforces string limit");
    limits = {};
    limits.max_collection_entries = 10;
    save_reject(original, limits, "writer enforces aggregate collection budget");
    throws([&] { (void)load_extraction(path.u8string(), limits); }, "reader enforces aggregate collection budget");
    limits = {};
    limits.max_blocks_per_page = 1;
    save_reject(original, limits, "writer counts nested blocks against the page cap");
    throws([&] { (void)load_extraction(path.u8string(), limits); }, "reader counts nested blocks against the page cap");
    limits = {};
    limits.max_depth = 1;
    save_reject(original, limits, "writer enforces nesting limit");
    throws([&] { (void)load_extraction(path.u8string(), limits); }, "reader enforces nesting limit");
    limits = {};
    limits.max_file_bytes = known.size() - 1;
    limits.max_asset_bytes = 8;
    limits.max_string_bytes = 1024;
    save_reject(original, limits, "writer enforces total encoded size limit");
    throws([&] { (void)load_extraction(path.u8string(), limits); }, "reader enforces advertised file size limit");
    limits = {};
    auto value = original;
    value.value.pages[0].blocks[0].bbox.x0 = std::numeric_limits<double>::quiet_NaN();
    save_reject(value, limits, "non-finite geometry is rejected before replacing the destination");
    value = original;
    value.value.pages[0].blocks[0].source_ranges[0].source_end = value.value.sources[0].text.size() + 1;
    save_reject(value, limits, "out-of-range source span is rejected");
    value = original;
    value.value.pages[0].blocks[0].source_ranges.back().text_end -= 1;
    value.value.pages[0].blocks[0].source_ranges.back().source_end -= 1;
    save_reject(value, limits, "source mappings must cover the complete current text");
    value = original;
    value.value.sources[1].text = "xx";
    save_reject(value, limits, "whitespace mapping cannot remove ordinary source bytes");
    value = original;
    value.value.pages[0].blocks[0].source_ranges[0].transform = "unsupported";
    save_reject(value, limits, "unknown persisted source transform is rejected");
    value = original;
    value.value.pages[0].blocks[0].references[0].end += 1;
    save_reject(value, limits, "out-of-range footnote span is rejected");
    value = original;
    value.value.assets.push_back(value.value.assets[0]);
    save_reject(value, limits, "duplicate assets are rejected");
    value = original;
    value.value.assets[0].name = "../outside.png";
    save_reject(value, limits, "asset traversal is rejected before any file is created");
    value = original;
    value.identity["invalid/key"] = "value";
    save_reject(value, limits, "identity keys are constrained");
    value = original;
    value.identity["gate"] = std::string("value\0hidden", 12);
    save_reject(value, limits, "identity values cannot hide NUL or control bytes");
    value = original;
    value.complete = true;
    save_reject(value, limits, "complete snapshots must reach the declared last page");
    throws([&] { save_extraction((root / "absent" / "out.cemk").u8string(), original); },
           "missing parent reports save failure");
    throws([&] { save_extraction(root.u8string(), original); }, "directory destination is rejected");
    throws([&] { (void)load_extraction(root.u8string()); }, "directory input is rejected");
    throws([&] { (void)load_extraction(std::string("x\0suffix", 8)); }, "embedded NUL in path is rejected");
#ifndef _WIN32
    const auto alias = root / "selected-symlink";
    fs::create_symlink(path.filename(), alias);
    throws([&] { save_extraction(alias.u8string(), original); }, "archive save rejects symlink target");
    throws([&] { (void)load_extraction(alias.u8string()); }, "archive load rejects symlink input");
    check(extraction_file_sha256(alias.u8string()) == extraction_file_sha256(path.u8string()),
          "explicit source/model hash follows selected symlink targets");
    check(bytes(path) == known, "rejected symlink save preserves the actual target");
#endif
    for (const auto & entry : fs::directory_iterator(root))
        check(entry.path().filename().u8string().find(".crispembed-marker-") != 0,
              "failed saves leave no temporary siblings");
}
} // namespace

static int crispembed_test_main() {
    try {
        workspace work;
        roundtrip(work.path);
        malformed(work.path);
        limits_and_atomicity(work.path);
    } catch (const std::exception & error) {
        ++failures;
        std::fprintf(stderr, "UNEXPECTED: %s\n", error.what());
    }
    std::printf("marker artifact: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
int main() {
    core_util::clean_exit(crispembed_test_main());
}
