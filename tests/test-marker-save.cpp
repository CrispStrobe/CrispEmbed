// SPDX-License-Identifier: Apache-2.0
#include "../examples/marker/save_bundle.h"
#include "core/clean_exit.h"
#include <functional>
#include <iostream>
#include <utility>

namespace {
namespace fs = std::filesystem;
int passed = 0, total = 0;

void check(bool condition, const std::string & name) {
    ++total;
    if (condition) {
        ++passed;
        std::cout << "PASS " << name << '\n';
    } else
        std::cerr << "FAIL " << name << '\n';
}

void rejects(const std::string & name, const std::function<void()> & action, const std::string & expected = "") {
    try {
        action();
        check(false, name);
    } catch (const std::exception & error) {
        check(expected.empty() || std::string(error.what()).find(expected) != std::string::npos, name);
    }
}

void put(const fs::path & path, const std::string & text) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!stream) throw std::runtime_error("test fixture write failed");
}

std::string get(const fs::path & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("test fixture read failed");
    return { std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>() };
}

void save(const fs::path & path, const std::vector<std::pair<std::string, std::string>> & assets) {
    std::vector<marker_cli::asset_view> views;
    for (const auto & asset : assets)
        views.push_back({ asset.first, reinterpret_cast<const uint8_t *>(asset.second.data()), asset.second.size() });
    marker_cli::save_bundle(path, "new document", "new metadata", views);
}

struct temporary_directory {
    fs::path path;
    temporary_directory()
        : path(fs::current_path() /
               ("marker-save-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        if (!fs::create_directory(path)) throw std::runtime_error("test directory already exists");
    }
    ~temporary_directory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
    fs::path child(const std::string & name) const {
        const auto p = path / name;
        fs::create_directory(p);
        return p;
    }
};

int marker_save_tests() {
    temporary_directory fixture;
    {
        const auto folder = fixture.child("conflict"), output = folder / "document.md",
                   meta = folder / "document_meta.json";
        put(output, "old document");
        put(meta, "old metadata");
        put(folder / "figure.png", "old figure");
        rejects(
            "conflicting asset is rejected",
            [&] { save(output, { { "new.png", "new" }, { "figure.png", "different" } }); },
            "separate directory per document");
        check(get(output) == "old document" && get(meta) == "old metadata",
              "conflict preserves primary output and metadata");
        check(get(folder / "figure.png") == "old figure", "conflict preserves old asset");
        check(!fs::exists(folder / "new.png"), "all assets preflighted before any creation");
    }
    {
        const auto folder = fixture.child("identical"), output = folder / "document.md", asset = folder / "figure.png";
        const std::string binary("same\0binary\xff", 12);
        put(asset, binary);
        const auto timestamp = fs::last_write_time(asset) - std::chrono::hours(24);
        fs::last_write_time(asset, timestamp);
        save(output, { { "figure.png", binary } });
        check(get(output) == "new document" && get(folder / "document_meta.json") == "new metadata",
              "identical asset permits saving document");
        check(get(asset) == binary && fs::last_write_time(asset) == timestamp, "identical asset reused without write");
    }
    {
        const auto folder = fixture.child("new"), output = folder / "nested" / "document.md";
        const std::string binary("\0\xff\n", 3);
        save(output, { { "figure.png", binary } });
        check(get(output) == "new document" && get(output.parent_path() / "figure.png") == binary,
              "new parent and binary asset saved");
    }
    {
        const auto folder = fixture.child("unsafe"), output = folder / "document.md",
                   meta = folder / "document_meta.json";
        put(output, "old document");
        put(meta, "old metadata");
        const std::vector<std::string> names = { "../escape.png",
                                                 "/absolute.png",
                                                 "back\\slash.png",
                                                 std::string("nul\0.png", 8),
                                                 "new\n.png",
                                                 "",
                                                 ".",
                                                 "..",
                                                 "NUL.png",
                                                 "LPT1.png",
                                                 "dot." };
        for (size_t i = 0; i < names.size(); ++i)
            rejects(
                "unsafe asset name " + std::to_string(i),
                [&] { save(output, { { "first.png", "new" }, { names[i], "bad" } }); }, "unsafe");
        check(!fs::exists(folder / "first.png") && get(output) == "old document" && get(meta) == "old metadata",
              "unsafe names preserve bundle before writes");
        rejects(
            "asset cannot be primary output", [&] { save(output, { { output.filename().string(), "old document" } }); },
            "collides");
        rejects(
            "asset cannot be metadata output", [&] { save(output, { { meta.filename().string(), "old metadata" } }); },
            "collides");
        rejects(
            "duplicate asset paths rejected", [&] { save(output, { { "same.png", "same" }, { "same.png", "same" } }); },
            "destinations collide");
        check(!fs::exists(folder / "same.png"), "duplicate assets rejected before creation");
        rejects("directory output rejected", [&] { save(folder, {}); }, "unsafe");
        rejects("dot output rejected", [&] { save(folder / ".", {}); }, "unsafe");
    }
    {
        const auto folder = fixture.child("symlinks"), target = folder / "target", output = folder / "document.md",
                   meta = folder / "document_meta.json";
        put(target, "outside target");
        std::error_code error;
        fs::create_symlink(target, output, error);
        if (error) {
            check(false, "symlink fixtures available: " + error.message());
        } else {
            rejects("output symlink rejected", [&] { save(output, { { "new.png", "new" } }); }, "symlink");
            check(get(target) == "outside target" && !fs::exists(meta) && !fs::exists(folder / "new.png"),
                  "output symlink target remains untouched");
            fs::remove(output);
            put(output, "old document");
            fs::create_symlink(target, meta);
            rejects("metadata symlink rejected", [&] { save(output, {}); }, "symlink");
            check(get(output) == "old document" && get(target) == "outside target",
                  "metadata symlink rejection precedes primary write");
            fs::remove(meta);
            put(meta, "old metadata");
            fs::create_symlink(target, folder / "figure.png");
            rejects(
                "matching asset symlink rejected", [&] { save(output, { { "figure.png", "outside target" } }); },
                "symlink");
            check(get(output) == "old document" && get(meta) == "old metadata",
                  "asset symlink rejection preserves bundle");
            const auto real = fixture.child("real"), link = folder / "parent";
            fs::create_directory_symlink(real, link);
            rejects("parent directory symlink rejected", [&] { save(link / "document.md", {}); }, "symlink");
            check(fs::is_empty(real), "symlink parent target remains untouched");
        }
    }
    {
        const auto folder = fixture.child("hardlinks"), output = folder / "document.md",
                   meta = folder / "document_meta.json", asset = folder / "figure.png";
        put(output, "old document");
        put(meta, "old metadata");
        fs::create_hard_link(output, asset);
        rejects(
            "asset hardlink to output rejected", [&] { save(output, { { "figure.png", "old document" } }); },
            "collides");
        fs::remove(asset);
        fs::remove(meta);
        fs::create_hard_link(output, meta);
        rejects("metadata hardlink to output rejected", [&] { save(output, {}); }, "destinations collide");
        check(get(output) == "old document" && get(meta) == "old document", "hardlink aliases preserve source bytes");
        fs::remove(meta);
        put(meta, "old metadata");
        put(asset, "same");
        fs::create_hard_link(asset, folder / "second.png");
        rejects(
            "duplicate asset hardlinks rejected",
            [&] { save(output, { { "figure.png", "same" }, { "second.png", "same" } }); }, "destinations collide");
        check(get(output) == "old document" && get(meta) == "old metadata", "duplicate hardlinks preserve bundle");
    }
    std::cout << passed << '/' << total << " Marker save checks passed\n";
    return passed == total ? 0 : 1;
}
} // namespace

int main() {
    int result = 1;
    try {
        result = marker_save_tests();
    } catch (const std::exception & error) {
        std::cerr << "Marker save tests: " << error.what() << '\n';
    }
    core_util::clean_exit(result);
}
