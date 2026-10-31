// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace marker_cli {
namespace fs = std::filesystem;

struct asset_view {
    std::string name;
    const uint8_t * data;
    size_t size;
};

inline fs::path destination(const fs::path & path) {
    const auto name = path.filename();
    if (name.empty() || name == "." || name == ".." ||
        path.native().find(fs::path::value_type(0)) != fs::path::string_type::npos)
        throw std::runtime_error("unsafe Marker output filename");
    const auto absolute = fs::absolute(path);
    auto candidate = absolute;
    while (true) {
        std::error_code error;
        const auto status = fs::symlink_status(candidate, error);
        if (error && error != std::errc::no_such_file_or_directory)
            throw std::runtime_error("cannot inspect Marker output destination");
        if (!error && fs::exists(status)) {
            if (fs::is_symlink(status)) throw std::runtime_error("Marker output destinations must not use symlinks");
            if (candidate == absolute ? !fs::is_regular_file(status) : !fs::is_directory(status))
                throw std::runtime_error("Marker output destination or parent has an unsafe file type");
        }
        if (candidate == candidate.parent_path()) break;
        candidate = candidate.parent_path();
    }
    return absolute;
}

inline bool same_destination(const fs::path & left, const fs::path & right) {
    const auto a = fs::absolute(left).lexically_normal(), b = fs::absolute(right).lexically_normal();
#ifdef _WIN32
    if (CompareStringOrdinal(a.c_str(), -1, b.c_str(), -1, TRUE) == CSTR_EQUAL) return true;
#else
    if (a == b) return true;
#endif
    std::error_code error;
    const bool same = fs::equivalent(a, b, error);
    if (error && error != std::errc::no_such_file_or_directory)
        throw std::runtime_error("cannot compare Marker output destinations");
    return !error && same;
}

inline bool safe_asset_name(const std::string & name) {
    if (name.empty() || name.size() > 255 || name == "." || name == ".." || name.back() == '.') return false;
    for (unsigned char c : name)
        if (!(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9') && c != '_' && c != '-' &&
            c != '.')
            return false;
    auto stem = name.substr(0, name.find('.'));
    for (auto & c : stem)
        if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
    if (stem == "CON" || stem == "PRN" || stem == "AUX" || stem == "NUL") return false;
    return !(stem.size() == 4 && (stem.substr(0, 3) == "COM" || stem.substr(0, 3) == "LPT") && stem[3] >= '1' &&
             stem[3] <= '9');
}

inline bool asset_matches(const fs::path & path, const asset_view & asset) {
    destination(path);
    std::error_code error;
    const auto size = fs::file_size(path, error);
    if (error == std::errc::no_such_file_or_directory) return false;
    if (error) throw std::runtime_error("cannot read existing Marker asset");
    if (size != asset.size) return false;
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read existing Marker asset");
    std::array<char, 65536> bytes;
    for (size_t offset = 0; offset < asset.size;) {
        const size_t count = std::min(bytes.size(), asset.size - offset);
        stream.read(bytes.data(), static_cast<std::streamsize>(count));
        if (stream.gcount() != static_cast<std::streamsize>(count) ||
            !std::equal(bytes.begin(), bytes.begin() + count, asset.data + offset,
                        [](char a, uint8_t b) { return static_cast<uint8_t>(a) == b; }))
            return false;
        offset += count;
    }
    return stream.peek() == std::char_traits<char>::eof() && !stream.bad();
}

// Exclusively create a new file. Existing files, including symlinks, are never
// opened for writing. Return false only when another directory entry exists.
inline bool write_exclusive(const fs::path & path, const uint8_t * data, size_t size) {
#ifdef _WIN32
    const int descriptor = _wopen(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
    const int descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
#endif
    if (descriptor < 0) {
        if (errno == EEXIST) return false;
        throw std::runtime_error("cannot create Marker output file");
    }
    bool failed = false;
    size_t offset = 0;
    while (offset < size) {
        const auto count = static_cast<unsigned int>(std::min<size_t>(size - offset, 1 << 20));
#ifdef _WIN32
        const auto written = _write(descriptor, data + offset, count);
#else
        const auto written = write(descriptor, data + offset, count);
#endif
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            failed = true;
            break;
        }
        offset += static_cast<size_t>(written);
    }
#ifdef _WIN32
    if (_close(descriptor) != 0) failed = true;
#else
    if (close(descriptor) != 0) failed = true;
#endif
    if (failed) {
        std::error_code ignored;
        fs::remove(path, ignored);
        throw std::runtime_error("cannot finish Marker output file");
    }
    return true;
}

inline void atomic_output(const fs::path & path, const std::string & content) {
    destination(path);
    static std::atomic<uint64_t> serial{ 0 };
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    fs::path temporary;
    bool created = false;
    for (int attempt = 0; attempt < 100 && !created; ++attempt) {
        temporary = path.parent_path() / (".marker-save-" + std::to_string(stamp) + "-" + std::to_string(serial++));
        created = write_exclusive(temporary, reinterpret_cast<const uint8_t *>(content.data()), content.size());
    }
    if (!created) throw std::runtime_error("cannot create temporary Marker output");
    try {
        destination(path);
#ifdef _WIN32
        if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
            throw std::runtime_error("cannot replace Marker output file");
#else
        fs::rename(temporary, path);
#endif
    } catch (...) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
}

inline void save_bundle(const fs::path & output, const std::string & text, const std::string & metadata,
                        const std::vector<asset_view> & assets) {
    const auto path = destination(output);
    const auto meta = destination(path.parent_path() / (path.stem().string() + "_meta.json"));
    if (same_destination(path, meta)) throw std::runtime_error("Marker output and metadata destinations collide");
    std::vector<fs::path> targets;
    targets.reserve(assets.size());
    for (const auto & asset : assets) {
        if (!safe_asset_name(asset.name) || (!asset.data && asset.size))
            throw std::runtime_error("unsafe native asset filename or bytes");
        const auto target = destination(path.parent_path() / asset.name);
        if (same_destination(target, path) || same_destination(target, meta))
            throw std::runtime_error(
                "Marker asset collides with output or metadata; use a separate directory per document");
        for (const auto & other : targets)
            if (same_destination(target, other))
                throw std::runtime_error("Marker asset destinations collide; use a separate directory per document");
        if (fs::exists(target) && !asset_matches(target, asset))
            throw std::runtime_error(
                "Marker asset already exists with different bytes; use a separate directory per document");
        targets.push_back(target);
    }
    // Every destination is checked before any output is changed. This protects
    // detected conflicts; it does not promise a transaction across I/O failure.
    fs::create_directories(path.parent_path());
    for (size_t i = 0; i < assets.size(); ++i) {
        destination(targets[i]);
        if (!write_exclusive(targets[i], assets[i].data, assets[i].size) && !asset_matches(targets[i], assets[i]))
            throw std::runtime_error("Marker asset changed during save; use a separate directory per document");
    }
    atomic_output(meta, metadata);
    atomic_output(path, text);
}
} // namespace marker_cli
