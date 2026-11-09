// SPDX-License-Identifier: MIT
// Native extraction archive. Uses the repository's own streaming SHA-256.
#include "marker_artifact.h"
#include "core/sha256.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace crisp_marker {
namespace {
constexpr unsigned char magic[8] = { 'C', 'R', 'I', 'S', 'P', 'M', 'K', 0 };
constexpr uint32_t format_version = 1;
constexpr size_t prefix_size = 24, digest_size = 64, header_size = prefix_size + digest_size;
constexpr uint64_t hard_file_limit = 4ull * 1024 * 1024 * 1024;
constexpr size_t hard_depth_limit = 64;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("Marker extraction artifact: " + message);
}

void check_limits(const artifact_limits & limits) {
    if (limits.max_file_bytes < header_size || limits.max_file_bytes > hard_file_limit || !limits.max_string_bytes ||
        limits.max_string_bytes > limits.max_file_bytes || limits.max_asset_bytes > limits.max_file_bytes ||
        !limits.max_pages || !limits.max_blocks_per_page || !limits.max_collection_entries || !limits.max_depth ||
        limits.max_depth > hard_depth_limit)
        fail("invalid resource limits");
}

std::filesystem::path checked_path(const std::string & path) {
    if (path.empty() || path.find('\0') != std::string::npos) fail("invalid file path");
    const auto value = std::filesystem::u8path(path);
    if (value.filename().empty() || value.filename() == "." || value.filename() == "..") fail("invalid file path");
    return value;
}

void check_file_kind(const std::filesystem::path & path, bool may_be_missing) {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (may_be_missing &&
        (status.type() == std::filesystem::file_type::not_found || error == std::errc::no_such_file_or_directory))
        return;
    if (error || !std::filesystem::is_regular_file(status)) fail("path must identify a regular, non-symlink file");
}

FILE * open_read(const std::filesystem::path & path) {
#ifdef _WIN32
    HANDLE handle =
        CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                    OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (handle == INVALID_HANDLE_VALUE) return nullptr;
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle(handle, &info) || GetFileType(handle) != FILE_TYPE_DISK ||
        (info.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY))) {
        CloseHandle(handle);
        return nullptr;
    }
    const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_RDONLY | _O_BINARY);
    if (fd < 0) {
        CloseHandle(handle);
        return nullptr;
    }
    FILE * file = _fdopen(fd, "rb");
    if (!file) _close(fd);
    return file;
#else
    // The descriptor remains open throughout validation and parsing. Atomic
    // replacement by another writer cannot switch the file halfway through.
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) return nullptr;
    struct stat status {};
    if (fstat(fd, &status) != 0 || !S_ISREG(status.st_mode)) {
        close(fd);
        return nullptr;
    }
    FILE * file = fdopen(fd, "rb");
    if (!file) close(fd);
    return file;
#endif
}

void write_exact(FILE * file, const void * data, size_t size) {
    if (size && std::fwrite(data, 1, size, file) != size) fail("could not write the complete file");
}

void read_exact(FILE * file, void * data, size_t size) {
    if (size && std::fread(data, 1, size, file) != size) fail("truncated file or failed read");
}

void put_le(unsigned char * out, uint64_t value, size_t size) {
    for (size_t i = 0; i < size; ++i) out[i] = static_cast<unsigned char>(value >> (8 * i));
}

uint64_t get_le(const unsigned char * in, size_t size) {
    uint64_t value = 0;
    for (size_t i = 0; i < size; ++i) value |= static_cast<uint64_t>(in[i]) << (8 * i);
    return value;
}

struct budget {
    const artifact_limits & limits;
    size_t entries = 0, page_blocks = 0;
    uint64_t assets = 0;

    explicit budget(const artifact_limits & value) : limits(value) {}
    void collection(uint64_t count, size_t maximum) {
        if (count > maximum || count > limits.max_collection_entries - entries) fail("collection limit exceeded");
        entries += static_cast<size_t>(count);
    }
    void asset_bytes(uint64_t count) {
        if (count > limits.max_asset_bytes - assets) fail("asset byte limit exceeded");
        assets += count;
    }
    void block_at(size_t depth) {
        if (depth > limits.max_depth || ++page_blocks > limits.max_blocks_per_page)
            fail("block count or nesting limit exceeded");
    }
};

struct writer : budget {
    FILE * file;
    uint64_t bytes = 0;
    core_sha256::sha256_state digest;

    writer(FILE * output, const artifact_limits & limits) : budget(limits), file(output) {}
    void raw(const void * data, size_t count) {
        if (count > limits.max_file_bytes - header_size - bytes) fail("file byte limit exceeded");
        bytes += count;
        if (file) {
            write_exact(file, data, count);
            core_sha256::sha256_update(digest, static_cast<const uint8_t *>(data), count);
        }
    }
    void u64(uint64_t value) {
        unsigned char encoded[8];
        put_le(encoded, value, sizeof(encoded));
        raw(encoded, sizeof(encoded));
    }
    void integer(int value) {
        if (static_cast<int64_t>(value) < std::numeric_limits<int32_t>::min() ||
            static_cast<int64_t>(value) > std::numeric_limits<int32_t>::max())
            fail("integer is outside the portable range");
        unsigned char encoded[4];
        put_le(encoded, static_cast<uint32_t>(value), sizeof(encoded));
        raw(encoded, sizeof(encoded));
    }
    void offset(size_t value) { u64(value); }
    void boolean(bool value) {
        const unsigned char encoded = value ? 1 : 0;
        raw(&encoded, 1);
    }
    void real(double value) {
        static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                      "extraction artifacts require IEEE-754 binary64");
        if (!std::isfinite(value)) fail("non-finite geometry or score");
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        u64(bits);
    }
    void string(const std::string & value, uint64_t maximum) {
        if (value.size() > std::min(limits.max_string_bytes, maximum)) fail("string byte limit exceeded");
        u64(value.size());
        raw(value.data(), value.size());
    }
    void string(const std::string & value) { string(value, limits.max_string_bytes); }
    void blob(const std::vector<uint8_t> & value) {
        asset_bytes(value.size());
        u64(value.size());
        raw(value.data(), value.size());
    }
    template <typename Values, typename Fn> void sequence(const Values & values, size_t maximum, size_t, Fn fn) {
        collection(values.size(), maximum);
        u64(values.size());
        for (const auto & item : values) fn(item);
    }
    void strings(const std::vector<std::string> & values) {
        sequence(values, limits.max_collection_entries, 8, [&](const auto & item) { string(item); });
    }
    void mapping(const std::map<std::string, std::string> & values, size_t maximum,
                 uint64_t key_bytes = std::numeric_limits<uint64_t>::max(),
                 uint64_t value_bytes = std::numeric_limits<uint64_t>::max()) {
        collection(values.size(), maximum);
        u64(values.size());
        for (const auto & item : values) {
            string(item.first, key_bytes);
            string(item.second, value_bytes);
        }
    }
    void hierarchy(const std::map<int, std::string> & values) {
        collection(values.size(), limits.max_collection_entries);
        u64(values.size());
        for (const auto & item : values) {
            integer(item.first);
            string(item.second);
        }
    }
};

struct reader : budget {
    FILE * file;
    uint64_t remaining;
    core_sha256::sha256_state digest;

    reader(FILE * input, uint64_t count, const artifact_limits & limits)
        : budget(limits), file(input), remaining(count) {}
    void raw(void * data, size_t count) {
        if (count > remaining) fail("field exceeds the remaining payload");
        read_exact(file, data, count);
        remaining -= count;
        core_sha256::sha256_update(digest, static_cast<const uint8_t *>(data), count);
    }
    uint64_t u64() {
        unsigned char encoded[8];
        raw(encoded, sizeof(encoded));
        return get_le(encoded, sizeof(encoded));
    }
    void integer(int & value) {
        unsigned char encoded[4];
        raw(encoded, sizeof(encoded));
        const uint32_t encoded_value = static_cast<uint32_t>(get_le(encoded, sizeof(encoded)));
        const int64_t signed_value = encoded_value <= 0x7fffffffu ? static_cast<int64_t>(encoded_value)
                                                                  : static_cast<int64_t>(encoded_value) - 0x100000000ll;
        if (signed_value < std::numeric_limits<int>::min() || signed_value > std::numeric_limits<int>::max())
            fail("integer is not representable on this platform");
        value = static_cast<int>(signed_value);
    }
    void offset(size_t & value) {
        const uint64_t encoded = u64();
        if (encoded > std::numeric_limits<size_t>::max()) fail("offset is not representable on this platform");
        value = static_cast<size_t>(encoded);
    }
    void boolean(bool & value) {
        unsigned char encoded;
        raw(&encoded, 1);
        if (encoded > 1) fail("invalid boolean encoding");
        value = encoded != 0;
    }
    void real(double & value) {
        const uint64_t bits = u64();
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) fail("non-finite geometry or score");
    }
    void string(std::string & value, uint64_t maximum) {
        const uint64_t count = u64();
        if (count > std::min(limits.max_string_bytes, maximum) || count > remaining || count > value.max_size())
            fail("string byte limit exceeded");
        value.resize(static_cast<size_t>(count));
        raw(value.data(), value.size());
    }
    void string(std::string & value) { string(value, limits.max_string_bytes); }
    void blob(std::vector<uint8_t> & value) {
        const uint64_t count = u64();
        asset_bytes(count);
        if (count > remaining || count > value.max_size()) fail("asset exceeds the remaining payload");
        value.resize(static_cast<size_t>(count));
        raw(value.data(), value.size());
    }
    size_t count(size_t maximum, size_t minimum_bytes) {
        const uint64_t count_value = u64();
        collection(count_value, maximum);
        if (minimum_bytes && count_value > remaining / minimum_bytes) fail("collection exceeds the remaining payload");
        return static_cast<size_t>(count_value);
    }
    template <typename T, typename Fn>
    void sequence(std::vector<T> & values, size_t maximum, size_t minimum_bytes, Fn fn) {
        const size_t size = count(maximum, minimum_bytes);
        // Decode each member before allocating the next. A malicious count
        // cannot cause a large up-front vector allocation from a tiny file.
        for (size_t i = 0; i < size; ++i) {
            T item;
            fn(item);
            values.push_back(std::move(item));
        }
    }
    void strings(std::vector<std::string> & values) {
        sequence(values, limits.max_collection_entries, 8, [&](auto & item) { string(item); });
    }
    void mapping(std::map<std::string, std::string> & values, size_t maximum,
                 uint64_t key_bytes = std::numeric_limits<uint64_t>::max(),
                 uint64_t value_bytes = std::numeric_limits<uint64_t>::max()) {
        const size_t size = count(maximum, 16);
        for (size_t i = 0; i < size; ++i) {
            std::string key, value;
            string(key, key_bytes);
            string(value, value_bytes);
            if (!values.emplace(std::move(key), std::move(value)).second) fail("duplicate map key");
        }
    }
    void hierarchy(std::map<int, std::string> & values) {
        const size_t size = count(limits.max_collection_entries, 12);
        for (size_t i = 0; i < size; ++i) {
            int key;
            std::string value;
            integer(key);
            string(value);
            if (!values.emplace(key, std::move(value)).second) fail("duplicate section hierarchy key");
        }
    }
};

template <typename A, typename V> void transfer_box(A & io, V & value) {
    io.real(value.x0);
    io.real(value.y0);
    io.real(value.x1);
    io.real(value.y1);
}
template <typename A, typename V> void transfer_polygon(A & io, V & values) {
    io.sequence(values, io.limits.max_collection_entries, 16, [&](auto & item) {
        io.real(item.x);
        io.real(item.y);
    });
}
template <typename A, typename V> void transfer_line(A & io, V & value) {
    transfer_box(io, value.bbox);
    io.sequence(value.spans, io.limits.max_collection_entries, 59, [&](auto & item) {
        io.string(item.text);
        transfer_box(io, item.bbox);
        io.real(item.font_size);
        io.boolean(item.bold);
        io.boolean(item.italic);
        io.boolean(item.monospace);
        io.string(item.url);
    });
}
template <typename A, typename V> void transfer_block(A & io, V & value, size_t depth) {
    io.block_at(depth);
    io.string(value.id);
    io.string(value.type);
    io.string(value.text);
    io.string(value.html);
    io.string(value.raw);
    io.string(value.raw_label);
    io.string(value.source);
    transfer_box(io, value.bbox);
    transfer_polygon(io, value.polygon);
    io.sequence(value.lines, io.limits.max_collection_entries, 40, [&](auto & item) { transfer_line(io, item); });
    io.sequence(value.children, io.limits.max_blocks_per_page, 128,
                [&](auto & item) { transfer_block(io, item, depth + 1); });
    io.hierarchy(value.section_hierarchy);
    io.strings(value.warnings);
    io.string(value.image_name);
    io.integer(value.order);
    io.integer(value.heading_level);
    io.integer(value.angle);
    io.integer(value.generated_tokens);
    io.real(value.confidence);
    io.boolean(value.token_cap_reached);
    io.boolean(value.ignored);
    io.boolean(value.continuation);
    io.mapping(value.attributes, io.limits.max_collection_entries);
    io.sequence(value.source_ranges, io.limits.max_collection_entries, 48, [&](auto & item) {
        io.string(item.source_id);
        io.offset(item.source_start);
        io.offset(item.source_end);
        io.offset(item.text_start);
        io.offset(item.text_end);
        io.string(item.transform);
    });
    io.string(value.continuation_target);
    io.string(value.footnote_id);
    io.sequence(value.references, io.limits.max_collection_entries, 48, [&](auto & item) {
        io.offset(item.start);
        io.offset(item.end);
        io.string(item.label);
        io.string(item.status);
        io.string(item.target_id);
        io.strings(item.candidates);
    });
}
template <typename A, typename V> void transfer_document(A & io, V & value) {
    io.string(value.source_name);
    io.string(value.backend);
    io.sequence(value.pages, io.limits.max_pages, 64, [&](auto & item) {
        io.page_blocks = 0;
        io.integer(item.index);
        io.real(item.width);
        io.real(item.height);
        io.sequence(item.text_lines, io.limits.max_collection_entries, 40, [&](auto & row) { transfer_line(io, row); });
        io.sequence(item.blocks, io.limits.max_blocks_per_page, 128, [&](auto & b) { transfer_block(io, b, 1); });
        io.strings(item.warnings);
        io.string(item.extraction_method);
        io.string(item.raw_layout);
        io.integer(item.layout_generated_tokens);
        io.boolean(item.layout_token_cap_reached);
        io.string(item.printed_label);
        io.string(item.label_source);
        io.string(item.label_status);
        io.strings(item.label_candidates);
    });
    io.sequence(value.assets, io.limits.max_collection_entries, 24, [&](auto & item) {
        io.string(item.name);
        io.string(item.mime_type);
        io.blob(item.bytes);
    });
    io.strings(value.warnings);
    io.mapping(value.metadata, io.limits.max_collection_entries);
    io.sequence(value.sources, io.limits.max_collection_entries, 68, [&](auto & item) {
        io.string(item.id);
        io.string(item.text);
        io.string(item.source);
        io.integer(item.page_index);
        transfer_box(io, item.bbox);
        transfer_polygon(io, item.polygon);
    });
}
template <typename A, typename V> void transfer_snapshot(A & io, V & value) {
    io.integer(value.first_page);
    io.integer(value.last_page);
    io.integer(value.next_page);
    io.boolean(value.complete);
    io.mapping(value.identity, 64, 128, 4096);
    transfer_document(io, value.value);
}

bool safe_asset_name(const std::string & name) {
    if (name.empty() || name.size() > 1024 || name == "." || name == "..") return false;
    for (unsigned char c : name)
        if (c < 32 || c == 127 || c == '/' || c == '\\' || c == ':') return false;
    return true;
}

void validate_block(const block & value, const std::map<std::string, const source_record *> & sources, size_t depth,
                    const artifact_limits & limits, uint64_t & source_bytes) {
    if (depth > limits.max_depth) fail("block nesting limit exceeded");
    if (!value.image_name.empty() && !safe_asset_name(value.image_name)) fail("unsafe image member name");
    size_t covered = 0;
    for (const auto & range : value.source_ranges) {
        const auto source = sources.find(range.source_id);
        if (source == sources.end() || range.source_start > range.source_end ||
            range.source_end > source->second->text.size() || range.text_start != covered ||
            range.text_start > range.text_end || range.text_end > value.text.size())
            fail("invalid source range or unresolved source identity");
        const auto & original = source->second->text;
        const size_t inspected = range.source_end - range.source_start;
        if (inspected > limits.max_file_bytes - source_bytes) fail("source range validation byte limit exceeded");
        source_bytes += inspected;
        if (range.transform == "copy") {
            if (range.source_end - range.source_start != range.text_end - range.text_start ||
                value.text.compare(range.text_start, range.text_end - range.text_start, original, range.source_start,
                                   range.source_end - range.source_start) != 0)
                fail("invalid source copy range");
        } else if (range.transform == "whitespace") {
            if (range.text_end - range.text_start > 1 ||
                (range.text_end > range.text_start && value.text[range.text_start] != ' '))
                fail("invalid source whitespace range");
            for (size_t i = range.source_start; i < range.source_end; ++i) {
                const char c = original[i];
                if (c != ' ' && c != '\t' && c != '\n' && c != '\r') fail("invalid source whitespace bytes");
            }
        } else {
            fail("unsupported source range transform");
        }
        covered = range.text_end;
    }
    if (!value.source_ranges.empty() && covered != value.text.size()) fail("incomplete source range coverage");
    for (const auto & reference : value.references)
        if (reference.start > reference.end || reference.end > value.text.size()) fail("invalid footnote text range");
    for (const auto & child : value.children) validate_block(child, sources, depth + 1, limits, source_bytes);
}

void validate_snapshot(const extraction_snapshot & value, const artifact_limits & limits) {
    if (value.first_page < 0 || value.last_page < -1 || value.next_page < value.first_page ||
        (value.last_page >= 0 && (value.last_page < value.first_page ||
                                  static_cast<int64_t>(value.next_page) > static_cast<int64_t>(value.last_page) + 1)) ||
        (value.complete &&
         (value.last_page < 0 || static_cast<int64_t>(value.next_page) != static_cast<int64_t>(value.last_page) + 1)) ||
        static_cast<uint64_t>(value.next_page - value.first_page) != value.value.pages.size())
        fail("inconsistent page checkpoint");
    for (size_t i = 0; i < value.value.pages.size(); ++i) {
        const auto & page = value.value.pages[i];
        if (static_cast<int64_t>(page.index) != static_cast<int64_t>(value.first_page) + static_cast<int64_t>(i) ||
            !std::isfinite(page.width) || !std::isfinite(page.height) || page.width <= 0 || page.height <= 0)
            fail("invalid page identity or dimensions");
    }
    for (const auto & item : value.identity) {
        if (item.first.empty() || item.first.size() > 128 || item.second.size() > 4096) fail("invalid identity field");
        for (unsigned char c : item.first)
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' ||
                  c == '.'))
                fail("identity keys must be ASCII identifiers");
        for (unsigned char c : item.second)
            if (c < 32 || c > 126) fail("identity values must be printable ASCII");
    }
    std::set<std::string> assets;
    for (const auto & item : value.value.assets)
        if (!safe_asset_name(item.name) || !assets.insert(item.name).second) fail("unsafe or duplicate asset name");
    std::map<std::string, const source_record *> sources;
    for (const auto & item : value.value.sources)
        if (item.id.empty() || item.page_index < value.first_page || item.page_index >= value.next_page ||
            !sources.emplace(item.id, &item).second)
            fail("invalid or duplicate source identity");
    uint64_t source_bytes = 0;
    for (const auto & page : value.value.pages)
        for (const auto & item : page.blocks) validate_block(item, sources, 1, limits, source_bytes);
}

class sibling_file {
public:
    std::filesystem::path path;
    FILE * file = nullptr;
    bool committed = false;

    explicit sibling_file(const std::filesystem::path & destination) {
        auto parent = destination.parent_path();
        if (parent.empty()) parent = ".";
#ifdef _WIN32
        static std::atomic<uint64_t> sequence{ 0 };
        int fd = -1;
        for (size_t attempt = 0; attempt < 64 && fd < 0; ++attempt) {
            const std::string name = ".crispembed-marker-" + std::to_string(GetCurrentProcessId()) + "-" +
                                     std::to_string(GetTickCount64()) + "-" + std::to_string(sequence.fetch_add(1));
            path = parent / std::filesystem::u8path(name);
            fd = _wopen(path.c_str(), _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
            if (fd < 0 && errno != EEXIST) fail("cannot create a sibling temporary file");
        }
        if (fd < 0) fail("cannot create a unique sibling temporary file");
        file = _fdopen(fd, "w+b");
        if (!file) {
            _close(fd);
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            fail("cannot open the sibling temporary file");
        }
#else
        std::string pattern = (parent / ".crispembed-marker-XXXXXX").string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        const int fd = mkstemp(buffer.data());
        if (fd < 0) fail("cannot create a private sibling temporary file");
        path = buffer.data();
        file = fdopen(fd, "w+b");
        if (!file) {
            close(fd);
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            fail("cannot open the sibling temporary file");
        }
#endif
    }
    sibling_file(const sibling_file &) = delete;
    sibling_file & operator=(const sibling_file &) = delete;
    ~sibling_file() {
        if (file) std::fclose(file);
        if (!committed) {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    }
    void finish(const std::filesystem::path & destination) {
        if (std::fflush(file) != 0) fail("could not flush the temporary file");
#ifdef _WIN32
        if (_commit(_fileno(file)) != 0) fail("could not sync the temporary file");
#else
        if (fsync(fileno(file)) != 0) fail("could not sync the temporary file");
#endif
        FILE * closing = file;
        file = nullptr;
        if (std::fclose(closing) != 0) fail("could not close the temporary file");
        check_file_kind(destination, true);
#ifdef _WIN32
        if (!MoveFileExW(path.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            fail("could not atomically replace the destination");
#else
        if (std::rename(path.c_str(), destination.c_str()) != 0) fail("could not atomically replace the destination");
#endif
        committed = true;
        // Commit is complete. Do not report a failed save after replacement.
        // Directory syncing is best-effort because some filesystems reject it.
#ifndef _WIN32
        const auto parent = destination.parent_path().empty() ? std::filesystem::path(".") : destination.parent_path();
        const int directory = open(parent.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
        if (directory >= 0) {
            (void)fsync(directory);
            close(directory);
        }
#endif
    }
};

} // namespace

void save_extraction(const std::string & path, const extraction_snapshot & snapshot, const artifact_limits & limits) {
    check_limits(limits);
    const auto destination = checked_path(path);
    check_file_kind(destination, true);
    // Complete validation and size accounting precede any file creation.
    writer preview(nullptr, limits);
    transfer_snapshot(preview, snapshot);
    validate_snapshot(snapshot, limits);
    unsigned char prefix[prefix_size] = {};
    std::memcpy(prefix, magic, sizeof(magic));
    put_le(prefix + 8, format_version, 4);
    put_le(prefix + 16, preview.bytes, 8);
    sibling_file temporary(destination);
    write_exact(temporary.file, prefix, sizeof(prefix));
    const std::string placeholder(digest_size, '0');
    write_exact(temporary.file, placeholder.data(), placeholder.size());
    writer output(temporary.file, limits);
    core_sha256::sha256_update(output.digest, prefix, sizeof(prefix));
    transfer_snapshot(output, snapshot);
    if (output.bytes != preview.bytes) fail("snapshot changed during serialization");
    const std::string digest = core_sha256::sha256_finish(output.digest);
    if (std::fseek(temporary.file, prefix_size, SEEK_SET) != 0) fail("cannot write the integrity header");
    write_exact(temporary.file, digest.data(), digest.size());
    temporary.finish(destination);
}

extraction_snapshot load_extraction(const std::string & path, const artifact_limits & limits) {
    check_limits(limits);
    const auto source = checked_path(path);
    check_file_kind(source, false);
    std::unique_ptr<FILE, decltype(&std::fclose)> file(open_read(source), std::fclose);
    if (!file) fail("cannot open the extraction file");
    unsigned char prefix[prefix_size];
    read_exact(file.get(), prefix, sizeof(prefix));
    if (std::memcmp(prefix, magic, sizeof(magic)) != 0) fail("invalid magic");
    if (get_le(prefix + 8, 4) != format_version) fail("unsupported format version");
    if (get_le(prefix + 12, 4) != 0) fail("unsupported format flags");
    const uint64_t payload_size = get_le(prefix + 16, 8);
    if (payload_size > limits.max_file_bytes - header_size) fail("file byte limit exceeded");
    std::string expected(digest_size, '\0');
    read_exact(file.get(), expected.data(), expected.size());
    if (expected.find_first_not_of("0123456789abcdef") != std::string::npos) fail("invalid integrity digest encoding");
    reader input(file.get(), payload_size, limits);
    core_sha256::sha256_update(input.digest, prefix, sizeof(prefix));
    extraction_snapshot snapshot;
    transfer_snapshot(input, snapshot);
    if (input.remaining) fail("unexpected trailing payload fields");
    if (core_sha256::sha256_finish(input.digest) != expected) fail("SHA-256 integrity mismatch");
    if (std::fgetc(file.get()) != EOF || std::ferror(file.get())) fail("trailing data or failed read");
    validate_snapshot(snapshot, limits);
    return snapshot;
}

std::string extraction_file_sha256(const std::string & path) {
    // Sources, model caches and shared libraries may be explicitly selected
    // through symlinks. Resolve once and hash the target without loading it.
    std::error_code error;
    const auto source = std::filesystem::canonical(checked_path(path), error);
    if (error) fail("cannot resolve the identity source");
    check_file_kind(source, false);
    std::unique_ptr<FILE, decltype(&std::fclose)> file(open_read(source), std::fclose);
    if (!file) fail("cannot open the identity source");
    core_sha256::sha256_state digest;
    std::array<uint8_t, 64 * 1024> buffer{};
    for (;;) {
        const size_t count = std::fread(buffer.data(), 1, buffer.size(), file.get());
        if (count) core_sha256::sha256_update(digest, buffer.data(), count);
        if (count != buffer.size()) {
            if (std::ferror(file.get())) fail("cannot read the complete identity source");
            break;
        }
    }
    return core_sha256::sha256_finish(digest);
}

} // namespace crisp_marker
