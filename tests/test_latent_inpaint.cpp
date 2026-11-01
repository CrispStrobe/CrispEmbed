// Hosted original-vs-native replay. No model/framework download is performed
// by this binary. Reference data and model receipts are verified before init.
#include "latent_inpaint.h"
#include "latent_inpaint_schema.h"
#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "core/sha256.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CRISPEMBED_LATENT_INPAINT_TESTING
#error This reference runner requires the test-only native observer API.
#endif

namespace fs = std::filesystem;
using latent_inpaint::Tensor;

namespace {

constexpr double component_atol = 0.002;
constexpr double component_relative_l2 = 0.003;
constexpr double component_min_cosine = 0.99999;
constexpr int rgb_max_abs = 2;
constexpr double rgb_mean_abs = 0.1;
constexpr uint64_t max_json_bytes = 8 * 1024 * 1024;
constexpr uint64_t max_stage_bytes = 128 * 1024 * 1024;
const std::array<std::string, 4> cases = { "subtitle_bar", "edge_touch", "center", "empty" };

void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}

// A bounded strict JSON reader, local to this test executable. In particular,
// duplicate keys, nonfinite numbers, trailing data and ambiguous path strings
// must not silently select a different reference or relax an acceptance gate.
struct Json {
    enum Kind { null, boolean, number, string, array, object } kind = null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::map<std::string, Json> o;
    Json() = default;
    Json(bool v) : kind(boolean), b(v) {}
    Json(double v) : kind(number), n(v) { require(std::isfinite(v), "nonfinite JSON output"); }
    Json(int v) : Json(double(v)) {}
    Json(size_t v) : Json(double(v)) {}
    Json(const char * v) : kind(string), s(v) {}
    Json(std::string v) : kind(string), s(std::move(v)) {}
    static Json list() {
        Json v;
        v.kind = array;
        return v;
    }
    static Json dict() {
        Json v;
        v.kind = object;
        return v;
    }
    const Json & at(const std::string & key) const {
        require(kind == object, "expected JSON object");
        const auto it = o.find(key);
        require(it != o.end(), "missing JSON key: " + key);
        return it->second;
    }
    Json & operator[](const std::string & key) {
        require(kind == object, "expected JSON output object");
        return o[key];
    }
    const std::string & str() const {
        require(kind == string, "expected JSON string");
        return s;
    }
    double num() const {
        require(kind == number && std::isfinite(n), "expected finite JSON number");
        return n;
    }
    uint64_t uint() const {
        const double value = num();
        require(value >= 0 && value <= 9007199254740991.0 && std::floor(value) == value,
                "expected exact nonnegative JSON integer");
        return uint64_t(value);
    }
    bool flag() const {
        require(kind == boolean, "expected JSON boolean");
        return b;
    }
    const std::vector<Json> & rows() const {
        require(kind == array, "expected JSON array");
        return a;
    }
};

class Parser {
    const std::string & source;
    size_t pos = 0, nodes = 0;
    void space() {
        while (pos < source.size() &&
               (source[pos] == ' ' || source[pos] == '\n' || source[pos] == '\r' || source[pos] == '\t'))
            ++pos;
    }
    char take() {
        require(pos < source.size(), "truncated JSON");
        return source[pos++];
    }
    unsigned hex4() {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = take();
            const int digit = c >= '0' && c <= '9'   ? c - '0'
                              : c >= 'a' && c <= 'f' ? c - 'a' + 10
                              : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                                     : -1;
            require(digit >= 0, "bad JSON Unicode escape");
            value = value * 16 + unsigned(digit);
        }
        return value;
    }
    static void utf8(std::string & out, unsigned v) {
        if (v < 0x80)
            out += char(v);
        else if (v < 0x800) {
            out += char(0xc0 | (v >> 6));
            out += char(0x80 | (v & 63));
        } else if (v < 0x10000) {
            out += char(0xe0 | (v >> 12));
            out += char(0x80 | ((v >> 6) & 63));
            out += char(0x80 | (v & 63));
        } else {
            out += char(0xf0 | (v >> 18));
            out += char(0x80 | ((v >> 12) & 63));
            out += char(0x80 | ((v >> 6) & 63));
            out += char(0x80 | (v & 63));
        }
    }
    std::string quoted() {
        require(take() == '"', "expected JSON quote");
        std::string result;
        for (;;) {
            const unsigned char c = static_cast<unsigned char>(take());
            if (c == '"') return result;
            require(c >= 32, "unescaped JSON control character");
            if (c == '\\') {
                const char escape = take();
                if (escape == '"' || escape == '\\' || escape == '/')
                    result += escape;
                else if (escape == 'b')
                    result += '\b';
                else if (escape == 'f')
                    result += '\f';
                else if (escape == 'n')
                    result += '\n';
                else if (escape == 'r')
                    result += '\r';
                else if (escape == 't')
                    result += '\t';
                else if (escape == 'u') {
                    unsigned v = hex4();
                    if (v >= 0xd800 && v <= 0xdbff) {
                        require(take() == '\\' && take() == 'u', "unpaired JSON surrogate");
                        const unsigned low = hex4();
                        require(low >= 0xdc00 && low <= 0xdfff, "unpaired JSON surrogate");
                        v = 0x10000 + ((v - 0xd800) << 10) + low - 0xdc00;
                    } else
                        require(v < 0xdc00 || v > 0xdfff, "unpaired JSON surrogate");
                    utf8(result, v);
                } else
                    require(false, "invalid JSON escape");
            } else if (c < 0x80)
                result += char(c);
            else {
                int count = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : -1;
                require(count > 0, "invalid JSON UTF-8");
                unsigned cp = c & ((1u << (6 - count)) - 1);
                result += char(c);
                for (int i = 0; i < count; ++i) {
                    const unsigned char d = static_cast<unsigned char>(take());
                    require((d & 0xc0) == 0x80, "invalid JSON UTF-8 continuation");
                    cp = (cp << 6) | (d & 63);
                    result += char(d);
                }
                require(cp >= (count == 1   ? 0x80u
                               : count == 2 ? 0x800u
                                            : 0x10000u) &&
                            cp <= 0x10ffff && !(cp >= 0xd800 && cp <= 0xdfff),
                        "invalid JSON UTF-8 scalar");
            }
            require(result.size() <= 1024 * 1024, "JSON string exceeds limit");
        }
    }
    Json value(int depth) {
        space();
        require(depth <= 64 && ++nodes <= 500000, "JSON nesting/node limit");
        require(pos < source.size(), "truncated JSON value");
        const char c = source[pos];
        if (c == '"') return Json(quoted());
        if (c == '{' || c == '[') {
            const bool dict = take() == '{';
            Json out = dict ? Json::dict() : Json::list();
            const char end = dict ? '}' : ']';
            space();
            if (pos < source.size() && source[pos] == end) {
                ++pos;
                return out;
            }
            for (;;) {
                space();
                if (dict) {
                    std::string key = quoted();
                    space();
                    require(take() == ':', "expected JSON colon");
                    Json child = value(depth + 1);
                    require(out.o.emplace(std::move(key), std::move(child)).second, "duplicate JSON object key");
                } else
                    out.a.push_back(value(depth + 1));
                space();
                const char separator = take();
                if (separator == end) return out;
                require(separator == ',', "expected JSON comma");
            }
        }
        for (const auto & literal : { std::string("true"), std::string("false"), std::string("null") }) {
            if (source.compare(pos, literal.size(), literal) == 0) {
                pos += literal.size();
                return literal == "null" ? Json() : Json(literal == "true");
            }
        }
        const size_t begin = pos;
        if (source[pos] == '-') ++pos;
        require(pos < source.size() && source[pos] >= '0' && source[pos] <= '9', "invalid JSON number");
        if (source[pos] == '0')
            ++pos;
        else
            while (pos < source.size() && source[pos] >= '0' && source[pos] <= '9') ++pos;
        if (pos < source.size() && source[pos] == '.') {
            ++pos;
            const size_t start = pos;
            while (pos < source.size() && source[pos] >= '0' && source[pos] <= '9') ++pos;
            require(pos > start, "missing JSON fraction");
        }
        if (pos < source.size() && (source[pos] == 'e' || source[pos] == 'E')) {
            ++pos;
            if (pos < source.size() && (source[pos] == '+' || source[pos] == '-')) ++pos;
            const size_t start = pos;
            while (pos < source.size() && source[pos] >= '0' && source[pos] <= '9') ++pos;
            require(pos > start, "missing JSON exponent");
        }
        std::istringstream stream(source.substr(begin, pos - begin));
        stream.imbue(std::locale::classic());
        double number = 0;
        stream >> number;
        require(bool(stream) && stream.eof() && std::isfinite(number), "unrepresentable JSON number");
        return Json(number);
    }

public:
    explicit Parser(const std::string & source) : source(source) {
        require(source.size() <= max_json_bytes, "JSON exceeds limit");
    }
    Json parse() {
        Json out = value(0);
        space();
        require(pos == source.size(), "trailing JSON data");
        return out;
    }
};

std::string escape(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\')
            out << '\\' << char(c);
        else if (c < 32)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else
            out << char(c);
    }
    out << '"';
    return out.str();
}

void serialize(std::ostream & out, const Json & v) {
    switch (v.kind) {
    case Json::null:
        out << "null";
        break;
    case Json::boolean:
        out << (v.b ? "true" : "false");
        break;
    case Json::number:
        out << std::setprecision(17) << v.n;
        break;
    case Json::string:
        out << escape(v.s);
        break;
    case Json::array:
        out << '[';
        for (size_t i = 0; i < v.a.size(); ++i) {
            if (i) out << ',';
            serialize(out, v.a[i]);
        }
        out << ']';
        break;
    case Json::object:
        out << '{';
        {
            size_t i = 0;
            for (const auto & item : v.o) {
                if (i++) out << ',';
                out << escape(item.first) << ':';
                serialize(out, item.second);
            }
        }
        out << '}';
        break;
    }
}

void save_json(const fs::path & path, const Json & value) {
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    require(bool(output), "cannot write report: " + path.string());
    output.imbue(std::locale::classic());
    serialize(output, value);
    output << '\n';
    output.close();
    require(bool(output), "cannot finish report: " + path.string());
    fs::rename(temporary, path);
}

void keys(const Json & value, std::initializer_list<const char *> expected) {
    require(value.kind == Json::object && value.o.size() == expected.size(),
            "JSON object has missing or unexpected keys");
    for (const char * key : expected) (void)value.at(key);
}

bool digest_valid(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(),
                                             [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

void relative_name(const std::string & name) {
    require(!name.empty() && name.size() < 4096 && name.front() != '/' && name.back() != '/',
            "invalid relative reference path");
    size_t start = 0;
    for (size_t i = 0; i <= name.size(); ++i) {
        if (i == name.size() || name[i] == '/') {
            const std::string part = name.substr(start, i - start);
            require(!part.empty() && part != "." && part != "..", "reference path traversal or empty component");
            start = i + 1;
        } else
            require((name[i] >= 'a' && name[i] <= 'z') || (name[i] >= 'A' && name[i] <= 'Z') ||
                        (name[i] >= '0' && name[i] <= '9') || name[i] == '_' || name[i] == '-' || name[i] == '.',
                    "unsupported reference path character");
    }
}

fs::path regular_child(const fs::path & root, const std::string & relative, uint64_t limit) {
    relative_name(relative);
    fs::path path = root;
    for (const auto & component : fs::path(relative)) {
        path /= component;
        require(!fs::is_symlink(fs::symlink_status(path)), "symlink in reference path");
    }
    require(fs::is_regular_file(path) && fs::file_size(path) <= limit,
            "missing, nonregular or oversized reference file: " + relative);
    return path;
}

std::vector<uint8_t> read_bytes(const fs::path & path, uint64_t limit) {
    require(!fs::is_symlink(path) && fs::is_regular_file(path), "file is not regular: " + path.string());
    const uint64_t size = fs::file_size(path);
    require(size <= limit, "file exceeds size limit");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "cannot open file: " + path.string());
    if (!bytes.empty()) input.read(reinterpret_cast<char *>(bytes.data()), std::streamsize(bytes.size()));
    require(bool(input) && input.peek() == std::char_traits<char>::eof(),
            "truncated or growing file: " + path.string());
    return bytes;
}

Json read_json(const fs::path & path) {
    const auto bytes = read_bytes(path, max_json_bytes);
    return Parser(std::string(bytes.begin(), bytes.end())).parse();
}

void expect_hash(const fs::path & path, const std::string & expected) {
    require(digest_valid(expected) && core_sha256::file(path.string()) == expected,
            "SHA-256 differs: " + path.string());
}

using Shape = std::vector<int64_t>;
Shape shape(const Json & value, size_t rank) {
    require(value.rows().size() == rank, "tensor rank differs");
    Shape result;
    for (const auto & d : value.rows()) {
        const auto dim = d.uint();
        require(dim > 0 && dim <= 65536, "tensor dimension out of bounds");
        result.push_back(int64_t(dim));
    }
    return result;
}

Json shape_json(const Shape & dimensions) {
    Json result = Json::list();
    for (int64_t d : dimensions) result.a.emplace_back(double(d));
    return result;
}

uint64_t elements(const Shape & dimensions) {
    uint64_t size = 1;
    for (int64_t d : dimensions) {
        require(d > 0 && size <= UINT64_C(0x100000000) / uint64_t(d), "tensor size overflow");
        size *= uint64_t(d);
    }
    return size;
}

struct Record {
    std::string name, filename, sha256;
    Shape dimensions;
    uint64_t bytes = 0;
    double minimum = 0, maximum = 0, l2 = 0;
};

Record record(const Json & row) {
    keys(row, { "name", "file", "shape", "bytes", "sha256", "min", "max", "l2" });
    Record r{ row.at("name").str(),   row.at("file").str(), row.at("sha256").str(), shape(row.at("shape"), 3),
              row.at("bytes").uint(), row.at("min").num(),  row.at("max").num(),    row.at("l2").num() };
    require(!r.name.empty() && r.name.size() <= 512 && digest_valid(r.sha256) && r.minimum <= r.maximum && r.l2 >= 0,
            "invalid floating record identity/statistics");
    require(r.bytes == elements(r.dimensions) * 4 && r.bytes <= max_stage_bytes, "floating record size differs");
    relative_name(r.filename);
    require(r.filename.find('/') == std::string::npos, "stage file must be a direct child");
    return r;
}

Tensor read_tensor(const fs::path & directory, const Record & r) {
    const auto raw = read_bytes(regular_child(directory, r.filename, max_stage_bytes), max_stage_bytes);
    require(raw.size() == r.bytes && core_sha256::bytes(raw.data(), raw.size()) == r.sha256,
            "floating payload hash/size differs: " + r.name);
    Tensor value(int(r.dimensions[0]), int(r.dimensions[1]), int(r.dimensions[2]));
    double minimum = std::numeric_limits<double>::infinity(), maximum = -minimum, squared = 0;
    for (size_t i = 0; i < value.v.size(); ++i) {
        const uint32_t bits = uint32_t(raw[i * 4]) | (uint32_t(raw[i * 4 + 1]) << 8) |
                              (uint32_t(raw[i * 4 + 2]) << 16) | (uint32_t(raw[i * 4 + 3]) << 24);
        std::memcpy(&value.v[i], &bits, sizeof(float));
        const double x = value.v[i];
        require(std::isfinite(x), "nonfinite reference float: " + r.name);
        minimum = std::min(minimum, x);
        maximum = std::max(maximum, x);
        squared += x * x;
    }
    require(minimum == r.minimum && maximum == r.maximum &&
                std::abs(std::sqrt(squared) - r.l2) <= 1e-9 * std::max(1.0, r.l2),
            "reference statistics differ: " + r.name);
    return value;
}

void save_tensor(const fs::path & path, const Tensor & value) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(bool(output), "cannot save divergent tensor");
    for (float v : value.v) {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        const char raw[4] = { char(bits), char(bits >> 8), char(bits >> 16), char(bits >> 24) };
        output.write(raw, 4);
    }
    output.close();
    require(bool(output), "cannot finish divergent tensor");
}

struct Metrics {
    bool finite = true;
    double max_abs = 0, mae = 0, relative_l2 = 0, cosine = 1, native_l2 = 0, reference_l2 = 0;
    size_t max_index = 0;
    bool pass() const {
        return finite && max_abs <= component_atol && relative_l2 <= component_relative_l2 &&
               cosine >= component_min_cosine;
    }
    Json json() const {
        Json r = Json::dict();
        r["finite"] = finite;
        r["pass"] = pass();
        if (finite) {
            r["max_abs"] = max_abs;
            r["mean_abs"] = mae;
            r["relative_l2"] = relative_l2;
            r["cosine"] = cosine;
            r["native_l2"] = native_l2;
            r["reference_l2"] = reference_l2;
            r["max_abs_index"] = max_index;
        }
        return r;
    }
};

Metrics compare(const std::vector<float> & actual, const std::vector<float> & expected) {
    require(!actual.empty() && actual.size() == expected.size(), "comparison element count differs");
    Metrics result;
    double error2 = 0, a2 = 0, b2 = 0, dot = 0, total = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], b = expected[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            result.finite = false;
            return result;
        }
        const double d = a - b;
        if (std::abs(d) > result.max_abs) {
            result.max_abs = std::abs(d);
            result.max_index = i;
        }
        total += std::abs(d);
        error2 += d * d;
        a2 += a * a;
        b2 += b * b;
        dot += a * b;
    }
    result.mae = total / double(actual.size());
    result.native_l2 = std::sqrt(a2);
    result.reference_l2 = std::sqrt(b2);
    // Both zero vectors have cosine one. A nonzero vector vs zero has cosine
    // zero and a bounded-denominator relative error, so it cannot pass by NaN.
    result.relative_l2 = std::sqrt(error2) / std::max(result.reference_l2, 1e-12);
    result.cosine = a2 == 0 || b2 == 0 ? (a2 == 0 && b2 == 0 ? 1 : 0) : std::clamp(dot / std::sqrt(a2 * b2), -1.0, 1.0);
    return result;
}

using Inventory = std::map<std::string, Shape>;

Inventory module_inventory(const std::string & model) {
    const bool pixel = model == "pixelhacker";
    require(pixel || model == "moebius", "unknown inventory model");
    Inventory result;
    auto put = [&](const std::string & name, int c, int side) {
        require(result.emplace(name, Shape{ c, side, side }).second, "duplicate native module inventory");
    };
    result["embedding_layer"] = { 3072, 1, 10 };
    result["diff_model.encoder_hid_proj"] = { 768, 1, 10 };
    result["diff_model.time_proj"] = { 320, 1, 1 };
    result["diff_model.time_embedding"] = { 1280, 1, 1 };
    put("diff_model.conv_in", 320, 64);
    put("diff_model.conv_norm_out", 320, 64);
    put("diff_model.conv_out", 4, 64);
    put("diff_model", 4, 64);
    auto resnet = [&](const std::string & name, int c, int side) {
        put(name + ".conv1", c, side);
        put(name + ".conv2", c, side);
        put(name, c, side);
    };
    auto attention = [&](const std::string & name, int c, int side) {
        const std::string block = name + ".transformer_blocks.0";
        for (const auto & suffix :
             { std::string(""), std::string(".attn1"), std::string(".attn2"), std::string(".ff") })
            put(block + suffix, c, side);
        if (pixel) put(block + ".attn1.g_norm_swish_gate", c, side);
        put(name, c, side);
    };
    const int count = pixel ? 4 : 3;
    const int down[] = { 320, 640, 1280, 1280 }, up_pixel[] = { 1280, 1280, 640, 320 },
              up_moebius[] = { 1280, 640, 320 };
    int side = 64;
    for (int b = 0; b < count; ++b) {
        const std::string prefix = "diff_model.down_blocks." + std::to_string(b);
        for (int r = 0; r < 2; ++r) {
            resnet(prefix + ".resnets." + std::to_string(r), down[b], side);
            if (!pixel || b < 3) attention(prefix + ".attentions." + std::to_string(r), down[b], side);
        }
        if (b < count - 1) {
            side /= 2;
            put(prefix + ".downsamplers.0", down[b], side);
        }
        put(prefix, down[b], side);
    }
    if (pixel) {
        resnet("diff_model.mid_block.resnets.0", 1280, 8);
        resnet("diff_model.mid_block.resnets.1", 1280, 8);
        attention("diff_model.mid_block.attentions.0", 1280, 8);
        put("diff_model.mid_block", 1280, 8);
    }
    for (int b = 0; b < count; ++b) {
        const std::string prefix = "diff_model.up_blocks." + std::to_string(b);
        const int c = pixel ? up_pixel[b] : up_moebius[b];
        for (int r = 0; r < 3; ++r) {
            resnet(prefix + ".resnets." + std::to_string(r), c, side);
            if (!pixel || b > 0) attention(prefix + ".attentions." + std::to_string(r), c, side);
        }
        if (b < count - 1) {
            side *= 2;
            put(prefix + ".upsamplers.0", c, side);
        }
        put(prefix, c, side);
    }
    return result;
}

Inventory stage_inventory(const std::string & model, bool full) {
    Inventory result{ { "sampler.alphas_cumprod", { 1000, 1, 1 } },
                      { "vae.image.moments", { 8, 64, 64 } },
                      { "vae.masked.moments", { 8, 64, 64 } },
                      { "vae.image.latents", { 4, 64, 64 } },
                      { "vae.masked.latents", { 4, 64, 64 } },
                      { "latent.initial", { 4, 64, 64 } },
                      { "vae.decoded", { 3, 512, 512 } } };
    for (int step = 0; step < 19; ++step) {
        for (const auto & label : { std::string("unconditional"), std::string("conditional") })
            result["unet.step" + std::to_string(step) + "." + label] = { 4, 64, 64 };
        result["sampler.noise." + std::to_string(step)] = { 4, 64, 64 };
        result["sampler.step." + std::to_string(step)] = { 4, 64, 64 };
    }
    if (full)
        for (const auto & module : module_inventory(model))
            for (const auto & label : { std::string("unconditional"), std::string("conditional") })
                result["unet.step.0." + label + "." + module.first] = module.second;
    return result;
}

struct Fixture {
    std::string rgb_hash, mask_hash;
    uint64_t fill;
};

Fixture fixture(const std::string & id) {
    const std::string base_rgb = "76f4c84f19ee059f7e39fe49c4e228cad1fb47f977ea9121816472906d9c67da";
    if (id == "subtitle_bar")
        return { "be9d9698ea60757782d128a2310ba3295bc87725a692ed7a496704503d724c2b",
                 "e60406d47135cac5b58c8ca29f5bc69bca8044af46d3eb36ed60b206f0c7785d", 26624 };
    if (id == "edge_touch")
        return { base_rgb, "d570764759b243c9540491fcf818059f1401b6fc89e59bbbfec1fc7eb37a5e62", 22400 };
    if (id == "center") return { base_rgb, "91c7cff5369c293e461698f4986a82b124ba3daeff4e8be19beeebfc1e761c4e", 25600 };
    if (id == "empty") return { base_rgb, "8a39d2abd3999ab73c34db2476849cddf303ce389b35826850f9a700589b4a90", 0 };
    throw std::runtime_error("unknown frozen fixture");
}

std::vector<uint8_t> byte_record(const fs::path & directory, const Json & row, const std::string & filename,
                                 const Shape & expected_shape) {
    keys(row, { "file", "shape", "sha256" });
    require(row.at("file").str() == filename && shape(row.at("shape"), expected_shape.size()) == expected_shape,
            "RGB/mask record filename or shape differs");
    const auto data = read_bytes(regular_child(directory, filename, max_stage_bytes), max_stage_bytes);
    require(data.size() == elements(expected_shape) && digest_valid(row.at("sha256").str()) &&
                core_sha256::bytes(data.data(), data.size()) == row.at("sha256").str(),
            "RGB/mask payload size or hash differs");
    return data;
}

struct Case {
    std::string id, manifest_sha256;
    fs::path directory;
    std::vector<Record> stages;
    Tensor image, mask;
    latent_inpaint::ReferenceNoise noise;
    std::vector<uint8_t> rgb, fill_mask, generated, composited;
};

Case read_case(const fs::path & root, const Json & index, const std::string & model) {
    keys(index, { "id", "file", "sha256" });
    Case result;
    result.id = index.at("id").str();
    const Fixture identity = fixture(result.id);
    require(index.at("file").str() == result.id + "/reference.json", "case manifest path differs");
    const fs::path file = regular_child(root, index.at("file").str(), max_json_bytes);
    expect_hash(file, index.at("sha256").str());
    result.directory = file.parent_path();
    result.manifest_sha256 = index.at("sha256").str();
    const Json doc = read_json(file);
    keys(doc, { "schema",
                "model",
                "source_contract_sha256",
                "checkpoint_sha256",
                "vae_checkpoint_sha256",
                "fixture",
                "settings",
                "inputs",
                "scheduler_alphas",
                "rgb",
                "fill_mask",
                "stages",
                "final_generated_rgb",
                "final_composited_rgb",
                "original_seconds",
                "ordinary_original_forwards",
                "observed_original_forwards",
                "hook_and_random_replay_neutrality_bit_exact",
                "known_rgb_exact",
                "complete",
                "native_qualified" });
    require(doc.at("schema").str() == "crispembed-latent-inpaint-reference-case-v1" && doc.at("model").str() == model &&
                doc.at("source_contract_sha256").str() == latent_inpaint::source_contract_sha256() &&
                doc.at("checkpoint_sha256").str() == latent_inpaint::checkpoint_sha256(model) &&
                doc.at("vae_checkpoint_sha256").str() == latent_inpaint::checkpoint_sha256("sdxl_vae"),
            "reference source/checkpoint contract differs");
    require(doc.at("complete").flag() && !doc.at("native_qualified").flag() && doc.at("known_rgb_exact").flag() &&
                doc.at("hook_and_random_replay_neutrality_bit_exact").flag() &&
                doc.at("ordinary_original_forwards").uint() == 1 && doc.at("observed_original_forwards").uint() == 1 &&
                doc.at("original_seconds").num() > 0,
            "original reference is incomplete or neutrality evidence differs");
    const Json & f = doc.at("fixture");
    keys(f, { "id", "width", "height", "rgb_sha256", "mask_sha256", "fill_pixels" });
    require(f.at("id").str() == result.id && f.at("width").uint() == 512 && f.at("height").uint() == 512 &&
                f.at("rgb_sha256").str() == identity.rgb_hash && f.at("mask_sha256").str() == identity.mask_hash &&
                f.at("fill_pixels").uint() == identity.fill,
            "frozen fixture identity differs");
    const Json & settings = doc.at("settings");
    keys(settings, { "steps", "guidance", "strength", "noise_offset", "timesteps" });
    require(settings.at("steps").uint() == 20 && settings.at("guidance").num() == (model == "moebius" ? 2.0 : 4.5) &&
                settings.at("strength").num() == (model == "moebius" ? 0.99 : 0.999) &&
                settings.at("noise_offset").num() == 0.0357 && settings.at("timesteps").rows().size() == 19,
            "reference sampler options differ");
    const auto native_steps = latent_inpaint::detail::ddim_timesteps(20, model);
    require(native_steps.size() == 19, "native timestep count differs");
    for (size_t i = 0; i < 19; ++i)
        require(settings.at("timesteps").a[i].uint() == 900 - i * 50 && native_steps[i] == 900 - int(i) * 50,
                "reference/native timesteps differ");
    result.rgb = byte_record(result.directory, doc.at("rgb"), "input.rgb", { 512, 512, 3 });
    result.fill_mask = byte_record(result.directory, doc.at("fill_mask"), "fill-mask.u8", { 512, 512 });
    require(core_sha256::bytes(result.rgb.data(), result.rgb.size()) == identity.rgb_hash &&
                core_sha256::bytes(result.fill_mask.data(), result.fill_mask.size()) == identity.mask_hash,
            "fixture payload is not the frozen fixture");
    result.generated =
        byte_record(result.directory, doc.at("final_generated_rgb"), "final.generated.rgb", { 512, 512, 3 });
    result.composited =
        byte_record(result.directory, doc.at("final_composited_rgb"), "final.composited.rgb", { 512, 512, 3 });
    for (size_t p = 0; p < result.fill_mask.size(); ++p) {
        require(result.fill_mask[p] == 0 || result.fill_mask[p] == 255, "nonbinary reference mask");
        for (int ch = 0; ch < 3; ++ch)
            require(result.composited[p * 3 + ch] ==
                        (result.fill_mask[p] ? result.generated[p * 3 + ch] : result.rgb[p * 3 + ch]),
                    "original exact-mask compositing differs");
    }
    const Json & inputs = doc.at("inputs");
    keys(inputs, { "image", "mask", "posterior_image", "posterior_masked", "latent_noise", "offset_noise" });
    std::set<std::string> filenames;
    auto input = [&](const std::string & name, const Shape & dimensions) {
        const Record r = record(inputs.at(name));
        require(r.name == name && r.filename == "input." + name + ".f32" && r.dimensions == dimensions &&
                    filenames.insert(r.filename).second,
                "input floating schema differs");
        return read_tensor(result.directory, r);
    };
    result.image = input("image", { 3, 512, 512 });
    result.mask = input("mask", { 1, 512, 512 });
    result.noise.posterior_image = input("posterior_image", { 4, 64, 64 });
    result.noise.posterior_masked = input("posterior_masked", { 4, 64, 64 });
    result.noise.latent_noise = input("latent_noise", { 4, 64, 64 });
    result.noise.offset_noise = input("offset_noise", { 4, 1, 1 });
    require(result.image.v == latent_inpaint::detail::preprocess_rgb8(result.rgb.data(), result.rgb.size()).v,
            "original/native RGB normalization differs");
    for (size_t p = 0; p < result.mask.v.size(); ++p)
        require(result.mask.v[p] == (result.fill_mask[p] ? 1.0f : 0.0f), "original mask float differs from fixture");
    Inventory expected = stage_inventory(model, result.id == "subtitle_bar");
    require(doc.at("stages").rows().size() == expected.size(),
            "reference stage count differs from complete architecture inventory");
    size_t ordinal = 0;
    for (const Json & row : doc.at("stages").rows()) {
        Record r = record(row);
        const auto found = expected.find(r.name);
        require(found != expected.end() && found->second == r.dimensions,
                "unexpected, duplicate or misshaped reference stage: " + r.name);
        expected.erase(found);
        std::ostringstream filename;
        filename << "stage-" << std::setw(4) << std::setfill('0') << ordinal << ".f32";
        require(r.filename == (ordinal == 0 ? "sampler.alphas_cumprod.f32" : filename.str()) &&
                    filenames.insert(r.filename).second,
                "reference stage file/order differs");
        if (ordinal == 0) {
            std::ostringstream a, b;
            serialize(a, row);
            serialize(b, doc.at("scheduler_alphas"));
            require(r.name == "sampler.alphas_cumprod" && a.str() == b.str(),
                    "scheduler alpha record differs from first stage");
        }
        (void)read_tensor(result.directory, r);
        result.stages.push_back(std::move(r));
        ++ordinal;
    }
    require(expected.empty(), "reference omitted architecture stages");
    return result;
}

std::string checkpoint_pickle(const std::string & model) {
    if (model == "moebius") return "c811caeaf6d7b8581d39bd1345e4c8c0e85b23eed52cb8513e7d3c4e6ded0c18";
    if (model == "pixelhacker") return "03c048230eb6650dcac5e7a03315c692b35aba6edb4257873f3b630715ab2125";
    return "afcd9646b70f3e70472ddfd24806834f447b50acc75cc196d6049ec9fbdd666d";
}

Json expected_metadata(const std::string & model) {
    Json result = Json::dict();
    result["general.architecture"] = model;
    result["general.name"] = model == "sdxl_vae" ? "sdxl-vae-experimental-f32" : model + "-places2-experimental-f32";
    result["general.file_type"] = 0;
    result["general.license"] = latent_inpaint::license(model);
    result["latent_inpaint.schema"] = 1;
    result["latent_inpaint.component"] = model == "sdxl_vae" ? "vae" : "denoiser";
    result["latent_inpaint.model"] = model;
    result["latent_inpaint.image_size"] = 512;
    result["latent_inpaint.source.commit"] = latent_inpaint::source_commit(model);
    result["latent_inpaint.checkpoint.sha256"] = latent_inpaint::checkpoint_sha256(model);
    result["latent_inpaint.checkpoint.revision"] = latent_inpaint::checkpoint_revision(model);
    result["latent_inpaint.contract.sha256"] = latent_inpaint::source_contract_sha256();
    result["latent_inpaint.tensor.layout"] = latent_inpaint::tensor_layout();
    result["latent_inpaint.input.contract"] = latent_inpaint::input_contract();
    return result;
}

std::string hash_region(std::ifstream & input, uint64_t offset, uint64_t size) {
    input.clear();
    input.seekg(std::streamoff(offset));
    require(bool(input), "cannot seek GGUF tensor");
    core_sha256::sha256_state state;
    std::array<uint8_t, 256 * 1024> buffer{};
    while (size) {
        const size_t take = size_t(std::min<uint64_t>(size, buffer.size()));
        input.read(reinterpret_cast<char *>(buffer.data()), std::streamsize(take));
        require(bool(input), "truncated GGUF tensor");
        core_sha256::sha256_update(state, buffer.data(), take);
        size -= take;
    }
    return core_sha256::sha256_finish(state);
}

Json verify_receipt(const fs::path & file, const std::string & digest, const std::string & model) {
    require(!fs::is_symlink(file) && fs::is_regular_file(file) &&
                fs::file_size(file) <= UINT64_C(4) * 1024 * 1024 * 1024,
            "GGUF must be a regular file within the 4GiB experimental profile");
    expect_hash(file, digest);
    const fs::path receipt_path = file.string() + ".json";
    const Json receipt = read_json(receipt_path);
    keys(receipt, { "schema",
                    "model",
                    "component",
                    "source_commit",
                    "source_contract_sha256",
                    "checkpoint_sha256",
                    "checkpoint_bytes",
                    "checkpoint_revision",
                    "checkpoint_pickle_sha256",
                    "gguf_sha256",
                    "gguf_bytes",
                    "original_tensor_count",
                    "export_tensor_count",
                    "metadata",
                    "tensors",
                    "omitted_inference_inert_buffers",
                    "all_f32_tensor_readbacks_exact",
                    "f16_sources_widened_exactly_to_f32",
                    "framework_imported",
                    "inference_performed",
                    "native_qualified",
                    "representative_quality_qualified",
                    "published",
                    "python",
                    "converter_sha256",
                    "helper_sha256",
                    "command" });
    require(receipt.at("schema").str() == "crispembed-latent-inpaint-conversion-v1" &&
                receipt.at("model").str() == model &&
                receipt.at("component").str() == (model == "sdxl_vae" ? "vae" : "denoiser") &&
                receipt.at("source_commit").str() == latent_inpaint::source_commit(model) &&
                receipt.at("source_contract_sha256").str() == latent_inpaint::source_contract_sha256() &&
                receipt.at("checkpoint_sha256").str() == latent_inpaint::checkpoint_sha256(model) &&
                receipt.at("checkpoint_bytes").uint() == latent_inpaint::checkpoint_bytes(model) &&
                receipt.at("checkpoint_revision").str() == latent_inpaint::checkpoint_revision(model) &&
                receipt.at("checkpoint_pickle_sha256").str() == checkpoint_pickle(model),
            "conversion receipt source/checkpoint identity differs");
    require(!receipt.at("command").rows().empty(), "missing replayable conversion command");
    for (const Json & argument : receipt.at("command").rows())
        require(!argument.str().empty() && argument.str().find('\0') == std::string::npos,
                "invalid conversion command argument");
    require(receipt.at("gguf_sha256").str() == digest && receipt.at("gguf_bytes").uint() == fs::file_size(file),
            "conversion receipt does not bind the supplied GGUF");
    require(receipt.at("all_f32_tensor_readbacks_exact").flag() &&
                receipt.at("f16_sources_widened_exactly_to_f32").flag() && !receipt.at("framework_imported").flag() &&
                !receipt.at("inference_performed").flag() && !receipt.at("native_qualified").flag() &&
                !receipt.at("representative_quality_qualified").flag() && !receipt.at("published").flag() &&
                !receipt.at("python").str().empty() && digest_valid(receipt.at("converter_sha256").str()) &&
                digest_valid(receipt.at("helper_sha256").str()),
            "conversion receipt flags or provenance differ");
    const auto schema = model == "sdxl_vae" ? latent_inpaint::vae_schema() : latent_inpaint::denoiser_schema(model);
    const size_t omitted_count = model == "moebius" ? 124 : 0;
    require(receipt.at("export_tensor_count").uint() == schema.size() &&
                receipt.at("original_tensor_count").uint() == schema.size() + omitted_count &&
                receipt.at("tensors").rows().size() == schema.size() &&
                receipt.at("omitted_inference_inert_buffers").rows().size() == omitted_count,
            "conversion tensor census differs");
    const Json metadata = expected_metadata(model);
    std::ostringstream wanted_metadata, received_metadata;
    serialize(wanted_metadata, metadata);
    serialize(received_metadata, receipt.at("metadata"));
    require(wanted_metadata.str() == received_metadata.str(), "conversion metadata differs from pinned profile");
    std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> gguf(
        core_gguf::open_metadata(file.string().c_str()), &core_gguf::free_metadata);
    require(bool(gguf) && gguf_get_n_tensors(gguf.get()) == int64_t(schema.size()) &&
                gguf_get_n_kv(gguf.get()) == int64_t(metadata.o.size()),
            "GGUF metadata/tensor count differs");
    for (const auto & item : metadata.o) {
        const int64_t index = gguf_find_key(gguf.get(), item.first.c_str());
        require(index >= 0, "missing GGUF metadata key");
        if (item.second.kind == Json::string)
            require(gguf_get_kv_type(gguf.get(), index) == GGUF_TYPE_STRING &&
                        item.second.s == gguf_get_val_str(gguf.get(), index),
                    "GGUF metadata string differs");
        else
            require(gguf_get_kv_type(gguf.get(), index) == GGUF_TYPE_UINT32 &&
                        item.second.uint() == gguf_get_val_u32(gguf.get(), index),
                    "GGUF metadata integer differs");
    }
    std::ifstream input(file, std::ios::binary);
    const uint64_t data_offset = gguf_get_data_offset(gguf.get());
    uint64_t offset = 0;
    size_t ordinal = 0;
    std::set<std::string> omitted_names;
    for (const auto & entry : schema) {
        const Json & row = receipt.at("tensors").a[ordinal];
        keys(row, { "name", "shape", "source_shape", "source_dtype", "dtype", "bytes", "sha256", "transformation" });
        const std::string & name = row.at("name").str();
        require(name == entry.first && shape(row.at("shape"), entry.second.size()) == entry.second,
                "conversion tensor order/name/shape differs");
        Shape source_shape = entry.second;
        const bool positional = name.size() >= 22 && name.compare(name.size() - 22, 22, ".attn1.pos_conv.weight") == 0;
        if (positional) source_shape.insert(source_shape.begin() + 2, 1);
        require(shape(row.at("source_shape"), source_shape.size()) == source_shape &&
                    row.at("source_dtype").str() == (model == "sdxl_vae" ? "float16" : "float32") &&
                    row.at("dtype").str() == "float32" &&
                    row.at("transformation").str() == (positional ? "squeeze-singleton-depth-f32" : "f32"),
                "unapproved conversion layout or precision change");
        const uint64_t bytes = elements(entry.second) * 4;
        require(row.at("bytes").uint() == bytes && digest_valid(row.at("sha256").str()) &&
                    gguf_get_tensor_type(gguf.get(), int64_t(ordinal)) == GGML_TYPE_F32 &&
                    gguf_get_tensor_size(gguf.get(), int64_t(ordinal)) == bytes &&
                    gguf_get_tensor_offset(gguf.get(), int64_t(ordinal)) == offset &&
                    name == gguf_get_tensor_name(gguf.get(), int64_t(ordinal)),
                "GGUF tensor index differs from receipt");
        const int64_t * ne = gguf_get_tensor_ne(gguf.get(), int64_t(ordinal));
        for (size_t axis = 0; axis < 4; ++axis)
            require(ne[axis] == (axis < entry.second.size() ? entry.second[entry.second.size() - 1 - axis] : 1),
                    "GGUF dimension ordering differs");
        require(data_offset <= fs::file_size(file) && offset <= fs::file_size(file) - data_offset &&
                    bytes <= fs::file_size(file) - data_offset - offset &&
                    hash_region(input, data_offset + offset, bytes) == row.at("sha256").str(),
                "GGUF tensor bytes differ from exact-readback receipt: " + name);
        offset += (bytes + 31) & ~UINT64_C(31);
        ++ordinal;
        const std::string suffix = ".running_mean";
        if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            omitted_names.insert(name.substr(0, name.size() - suffix.size()) + ".num_batches_tracked");
    }
    require(fs::file_size(file) == data_offset + offset, "GGUF trailing data or padding layout differs");
    require(omitted_names.size() == omitted_count, "native BatchNorm counter inventory differs");
    for (const Json & row : receipt.at("omitted_inference_inert_buffers").rows()) {
        keys(row, { "name", "dtype", "shape", "sha256", "reason" });
        require(omitted_names.erase(row.at("name").str()) == 1 && row.at("dtype").str() == "int64" &&
                    row.at("shape").rows().empty() && digest_valid(row.at("sha256").str()) &&
                    row.at("reason").str() ==
                        "BatchNorm evaluation uses running_mean and running_var, not this training counter",
                "unapproved omitted checkpoint tensor");
    }
    require(omitted_names.empty(), "conversion omitted-buffer census is incomplete");
    Json evidence = Json::dict();
    evidence["model"] = model;
    evidence["gguf_sha256"] = digest;
    evidence["receipt_sha256"] = core_sha256::file(receipt_path.string());
    evidence["gguf_bytes"] = size_t(fs::file_size(file));
    evidence["tensor_payload_hashes_checked"] = schema.size();
    evidence["checkpoint_sha256"] = latent_inpaint::checkpoint_sha256(model);
    evidence["source_commit"] = latent_inpaint::source_commit(model);
    return evidence;
}

Json read_original_report(const fs::path & directory, const std::string & model) {
    require(!fs::is_symlink(directory) && fs::is_directory(directory), "reference root must be a real directory");
    const Json doc = read_json(regular_child(directory, "report.json", max_json_bytes));
    keys(doc, { "schema", "model", "source_commit", "source_contract_sha256", "versions", "python", "torch_config",
                "device", "threads", "cases", "dumper_sha256", "helper_sha256", "producer", "complete",
                "native_qualified", "representative_quality_qualified" });
    require(doc.at("schema").str() == "crispembed-latent-inpaint-original-v1" && doc.at("model").str() == model &&
                doc.at("source_commit").str() == latent_inpaint::source_commit(model) &&
                doc.at("source_contract_sha256").str() == latent_inpaint::source_contract_sha256() &&
                doc.at("complete").flag() && !doc.at("native_qualified").flag() &&
                !doc.at("representative_quality_qualified").flag(),
            "original run identity/status differs");
    require(digest_valid(doc.at("dumper_sha256").str()) && digest_valid(doc.at("helper_sha256").str()) &&
                !doc.at("python").str().empty() && !doc.at("torch_config").str().empty() &&
                !doc.at("device").str().empty() && doc.at("threads").uint() >= 1 && doc.at("threads").uint() <= 8,
            "original environment provenance is incomplete");
    require(model != "pixelhacker" || doc.at("device").str().rfind("cuda", 0) == 0,
            "PixelHacker original requires actual CUDA FLA");
    const Json & versions = doc.at("versions");
    keys(versions, { "torch", "diffusers", "flash-linear-attention", "numpy", "Pillow", "timm", "einops",
                     "transformers", "accelerate" });
    auto base_version = [&](const char * package) {
        const std::string & value = versions.at(package).str();
        return value.substr(0, value.find('+'));
    };
    require(base_version("torch") == (model == "moebius" ? "2.7.1" : "2.3.0") &&
                base_version("diffusers") == (model == "moebius" ? "0.38.0" : "0.30.2") &&
                base_version("flash-linear-attention") == "0.3.2",
            "original dependency version differs");
    keys(doc.at("producer"), { "github_sha", "github_run_id" });
    for (const char * key : { "github_sha", "github_run_id" })
        require(doc.at("producer").at(key).kind == Json::null || doc.at("producer").at(key).kind == Json::string,
                "invalid original producer identity");
    std::set<std::string> seen;
    require(!doc.at("cases").rows().empty() && doc.at("cases").rows().size() <= cases.size(),
            "original case inventory is empty or excessive");
    for (const Json & row : doc.at("cases").rows()) {
        keys(row, { "id", "file", "sha256" });
        (void)fixture(row.at("id").str());
        require(seen.insert(row.at("id").str()).second, "duplicate original case");
    }
    return doc;
}

struct Observer {
    const Case & reference;
    fs::path output;
    std::map<std::string, size_t> required;
    Inventory all_modules;
    std::set<std::string> observed;
    std::map<std::string, Json> comparisons;
    std::ofstream stream;
    size_t first_difference = std::numeric_limits<size_t>::max(), diagnostics = 0;

    Observer(const Case & reference, const fs::path & output, const std::string & model)
        : reference(reference), output(output), all_modules(stage_inventory(model, true)),
          stream(output / "stages.jsonl", std::ios::binary) {
        require(bool(stream), "cannot open streamed stage evidence");
        stream.imbue(std::locale::classic());
        for (size_t i = 0; i < reference.stages.size(); ++i) required.emplace(reference.stages[i].name, i);
        if (model == "pixelhacker") {
            Inventory recurrent;
            const std::string suffix = ".attn1";
            for (const auto & module : all_modules)
                if (module.first.size() >= suffix.size() &&
                    module.first.compare(module.first.size() - suffix.size(), suffix.size(), suffix) == 0)
                    recurrent[module.first + ".recurrent"] = module.second;
            all_modules.insert(recurrent.begin(), recurrent.end());
        }
    }
    void observe(const std::string & name, const Tensor & tensor) {
        require(observed.insert(name).second, "duplicate native observer stage: " + name);
        const auto definition = all_modules.find(name);
        require(definition != all_modules.end() && definition->second == Shape{ tensor.c, tensor.h, tensor.w } &&
                    tensor.v.size() == elements(definition->second),
                "unexpected or misshaped native observer stage: " + name);
        Json row = Json::dict();
        row["name"] = name;
        row["shape"] = shape_json(definition->second);
        const auto found = required.find(name);
        if (found == required.end()) {
            // Only explicit native-only GLA state or first-step modules omitted
            // by the reference's declared case profile arrive here. Still check
            // every float, and never count an unreferenced diagnostic as parity.
            require(std::all_of(tensor.v.begin(), tensor.v.end(), [](float v) { return std::isfinite(v); }),
                    "nonfinite unreferenced native diagnostic: " + name);
            row["reference_available"] = false;
            row["finite"] = true;
            ++diagnostics;
        } else {
            const Record & expected = reference.stages[found->second];
            const Tensor original = read_tensor(reference.directory, expected);
            const Metrics metrics = compare(tensor.v, original.v);
            row["reference_available"] = true;
            row["source_ordinal"] = found->second;
            row["metrics"] = metrics.json();
            row["reference_sha256"] = expected.sha256;
            comparisons.emplace(name, row);
            if (!metrics.pass() && found->second < first_difference) {
                first_difference = found->second;
                save_tensor(output / "first-divergence.native.f32", tensor);
                row["native_file"] = "first-divergence.native.f32";
                row["native_sha256"] = core_sha256::file((output / "first-divergence.native.f32").string());
                row["reference_file"] = expected.filename;
                save_json(output / "first-divergence.json", row);
            }
        }
        serialize(stream, row);
        stream << '\n';
        stream.flush();
        require(bool(stream), "cannot retain stage evidence");
    }
    static void callback(void * opaque, const std::string & name, const Tensor & value) {
        static_cast<Observer *>(opaque)->observe(name, value);
    }
};

Json compare_rgb(const std::vector<uint8_t> & actual, const std::vector<uint8_t> & reference) {
    require(actual.size() == reference.size() && !actual.empty(), "RGB comparison size differs");
    int maximum = 0;
    uint64_t total = 0;
    size_t index = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const int error = std::abs(int(actual[i]) - int(reference[i]));
        total += uint64_t(error);
        if (error > maximum) {
            maximum = error;
            index = i;
        }
    }
    const double mean = double(total) / double(actual.size());
    Json result = Json::dict();
    result["max_abs"] = maximum;
    result["mean_abs"] = mean;
    result["max_abs_index"] = index;
    result["pass"] = maximum <= rgb_max_abs && mean <= rgb_mean_abs;
    return result;
}

void save_bytes(const fs::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
    stream.close();
    require(bool(stream), "cannot save native RGB evidence");
}

Json replay_case(latent_inpaint::context * context, const Case & reference, const fs::path & output,
                 const std::string & model) {
    require(fs::create_directory(output), "native case output already exists");
    Json report = Json::dict();
    report["id"] = reference.id;
    report["reference_manifest_sha256"] = reference.manifest_sha256;
    report["complete"] = false;
    report["pass"] = false;
    report["errors"] = Json::list();
    report["stages"] = Json::list();
    save_json(output / "report.json", report);
    Observer observer(reference, output, model);
    latent_inpaint::set_observer(context, &Observer::callback, &observer);
    Tensor decoded;
    std::string error;
    const auto start = std::chrono::steady_clock::now();
    bool ran = false;
    try {
        ran = latent_inpaint::run_reference(context, reference.image, reference.mask, reference.noise, decoded, error);
    } catch (const std::exception & e) {
        error = e.what();
    }
    latent_inpaint::set_observer(context, nullptr, nullptr);
    report["native_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (!ran) report["errors"].a.emplace_back("native execution failed: " + error);
    bool stages_pass = true;
    for (const Record & stage : reference.stages) {
        const auto found = observer.comparisons.find(stage.name);
        if (found == observer.comparisons.end()) {
            Json missing = Json::dict();
            missing["name"] = stage.name;
            missing["pass"] = false;
            missing["error"] = "native observer did not produce required stage";
            report["stages"].a.push_back(std::move(missing));
            report["errors"].a.emplace_back("missing native stage: " + stage.name);
            stages_pass = false;
        } else {
            report["stages"].a.push_back(found->second);
            if (!found->second.at("metrics").at("pass").flag()) {
                stages_pass = false;
                report["errors"].a.emplace_back("component budget exceeded: " + stage.name);
            }
        }
    }
    report["unreferenced_finite_diagnostics"] = observer.diagnostics;
    report["earliest_source_divergence"] = observer.first_difference == std::numeric_limits<size_t>::max()
                                               ? Json()
                                               : Json(reference.stages[observer.first_difference].name);
    bool rgb_pass = false;
    if (ran) {
        try {
            std::vector<uint8_t> composed(reference.rgb.size()), generated(reference.rgb.size()),
                full_mask(reference.fill_mask.size(), 255);
            latent_inpaint::detail::composite_rgb8(decoded, reference.rgb.data(), reference.rgb.size(),
                                                   reference.fill_mask.data(), reference.fill_mask.size(),
                                                   composed.data(), composed.size());
            latent_inpaint::detail::composite_rgb8(decoded, reference.rgb.data(), reference.rgb.size(),
                                                   full_mask.data(), full_mask.size(), generated.data(),
                                                   generated.size());
            report["final_generated_rgb"] = compare_rgb(generated, reference.generated);
            report["final_composited_rgb"] = compare_rgb(composed, reference.composited);
            bool known_exact = true;
            for (size_t p = 0; p < reference.fill_mask.size(); ++p)
                if (!reference.fill_mask[p])
                    for (int ch = 0; ch < 3; ++ch)
                        if (composed[p * 3 + ch] != reference.rgb[p * 3 + ch]) known_exact = false;
            report["known_rgb_exact"] = known_exact;
            save_bytes(output / "generated.rgb", generated);
            save_bytes(output / "composited.rgb", composed);
            report["final_generated_rgb"]["native_file"] = "generated.rgb";
            report["final_generated_rgb"]["native_sha256"] = core_sha256::bytes(generated.data(), generated.size());
            report["final_composited_rgb"]["native_file"] = "composited.rgb";
            report["final_composited_rgb"]["native_sha256"] = core_sha256::bytes(composed.data(), composed.size());
            if (!known_exact) report["errors"].a.emplace_back("known source RGB pixels changed");
            if (!report["final_generated_rgb"].at("pass").flag())
                report["errors"].a.emplace_back("generated RGB budget exceeded");
            if (!report["final_composited_rgb"].at("pass").flag())
                report["errors"].a.emplace_back("composited RGB budget exceeded");
            rgb_pass = known_exact && report["final_generated_rgb"].at("pass").flag() &&
                       report["final_composited_rgb"].at("pass").flag();
        } catch (const std::exception & e) {
            report["errors"].a.emplace_back(e.what());
        }
    }
    report["complete"] = ran && observer.comparisons.size() == reference.stages.size();
    report["pass"] = ran && stages_pass && rgb_pass && report["errors"].a.empty();
    save_json(output / "report.json", report);
    return report;
}

template <typename Function> void rejects(Function fn) {
    try {
        fn();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error("malformed control was accepted");
}

void contract_controls() {
    require(Parser("{\"name\":\"x\\u03bb\",\"shape\":[4,64,64],\"ok\":true,\"none\":null}")
                    .parse()
                    .at("shape")
                    .a[2]
                    .uint() == 64,
            "valid strict JSON failed");
    for (const std::string invalid :
         { "{\"x\":1,\"x\":2}", "{\"x\":1,\"\\u0078\":2}", "[1,]", "{\"x\":1,}", "true false", "01", "1e999", "NaN",
           "Infinity", "1.", "1e", "\"\\ud800\"", "\"\\udc00\"", "\"\\q\"" })
        rejects([&] { Parser(invalid).parse(); });
    rejects([] { Parser(std::string(70, '[') + "0" + std::string(70, ']')).parse(); });
    rejects([] { Parser(std::string("\"") + char(0xc0) + char(0xaf) + "\"").parse(); });
    rejects([] { Json(-1).uint(); });
    rejects([] { Json(1.5).uint(); });
    for (const std::string path : { "../x", "/tmp/x", "a/../b", "a//b", "./x", "a/", "C:/x", "a\\b", "a b" })
        rejects([&] { relative_name(path); });
    relative_name("subtitle_bar/reference.json");
    require(compare({ 0, 0 }, { 0, 0 }).pass(), "zero-vector comparison failed");
    require(!compare({ 0, 0 }, { 1, 0 }).pass(), "cosine zero-vector bypass");
    require(!compare({ 10.01f, 10.01f }, { 10, 10 }).pass(), "cosine hid absolute drift");
    require(!compare({ 0.0009f, 0.0009f }, { 0.001f, 0.001f }).pass(), "absolute error hid relative drift");
    require(!compare({ -0.00001f }, { 0.00001f }).pass(), "absolute error hid directional drift");
    require(!compare({ std::numeric_limits<float>::quiet_NaN() }, { 0 }).pass(), "nonfinite comparison passed");
    for (const auto & model : { std::string("moebius"), std::string("pixelhacker") }) {
        const Inventory small = stage_inventory(model, false), full = stage_inventory(model, true);
        require(small.size() == 83 && full.size() > small.size() &&
                    full.at("unet.step.0.conditional.embedding_layer") == Shape{ 3072, 1, 10 },
                "incomplete reference stage inventory");
        require(full.at("unet.step.0.unconditional.diff_model.up_blocks." + std::to_string(model == "moebius" ? 2 : 3) +
                        ".resnets.2.conv2") == Shape{ 320, 64, 64 },
                "final decoder module inventory differs");
    }
    Json roundtrip = Json::dict();
    roundtrip["escaped"] = std::string("quote\"\\\n\t");
    roundtrip["numeric"] = -3.25;
    std::ostringstream json;
    serialize(json, roundtrip);
    require(Parser(json.str()).parse().at("escaped").str() == roundtrip.at("escaped").str(),
            "JSON evidence escaping differs");
    std::cout << "LATENT_INPAINT_CONTRACT {\"passed\":true,\"model_inference_performed\":false}\n";
}

struct Arguments {
    std::string model, model_sha, vae_sha;
    fs::path denoiser, vae, reference, output;
    int threads = 2;
};

Arguments arguments(int argc, char ** argv) {
    std::map<std::string, std::string> values;
    const std::set<std::string> allowed = { "--model",      "--denoiser",      "--vae", "--model-sha256",
                                            "--vae-sha256", "--reference-dir", "--out", "--threads" };
    for (int i = 1; i < argc; i += 2) {
        require(i + 1 < argc && allowed.count(argv[i]) == 1 && values.emplace(argv[i], argv[i + 1]).second,
                "unknown, missing or duplicate command argument");
    }
    for (const char * key :
         { "--model", "--denoiser", "--vae", "--model-sha256", "--vae-sha256", "--reference-dir", "--out" })
        require(values.count(key) && !values.at(key).empty(), std::string("required argument: ") + key);
    Arguments a;
    a.model = values.at("--model");
    a.denoiser = values.at("--denoiser");
    a.vae = values.at("--vae");
    a.model_sha = values.at("--model-sha256");
    a.vae_sha = values.at("--vae-sha256");
    a.reference = values.at("--reference-dir");
    a.output = values.at("--out");
    require((a.model == "moebius" || a.model == "pixelhacker") && digest_valid(a.model_sha) && digest_valid(a.vae_sha),
            "model name or GGUF SHA-256 argument differs");
    if (values.count("--threads")) {
        const std::string value = values.at("--threads");
        require(!value.empty() && value.size() <= 2 &&
                    std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }),
                "threads must be an integer 1..8");
        a.threads = std::stoi(value);
    }
    require(a.threads >= 1 && a.threads <= 8, "threads must be 1..8");
    require(!fs::exists(a.output) && !fs::is_symlink(a.output), "choose a new output directory");
    require(!a.output.filename().empty() && a.output.filename() != "." && a.output.filename() != "..",
            "invalid output directory");
    return a;
}

int run(const Arguments & a) {
    require(fs::create_directories(a.output), "cannot create new evidence directory");
    Json report = Json::dict();
    report["schema"] = "crispembed-latent-inpaint-native-v1";
    report["model"] = a.model;
    report["source_contract_sha256"] = latent_inpaint::source_contract_sha256();
    report["complete"] = false;
    report["synthetic_parity_pass"] = false;
    report["full_synthetic_panel_pass"] = false;
    report["native_qualified"] = false;
    report["representative_quality_qualified"] = false;
    report["qualification_note"] = "This bounded synthetic replay does not establish representative image quality, "
                                   "throughput or a release qualification.";
    report["errors"] = Json::list();
    report["cases"] = Json::list();
    report["threads"] = a.threads;
    report["budgets"] = Json::dict();
    report["budgets"]["component_max_abs"] = component_atol;
    report["budgets"]["component_relative_l2"] = component_relative_l2;
    report["budgets"]["component_min_cosine"] = component_min_cosine;
    report["budgets"]["final_rgb_max_abs"] = rgb_max_abs;
    report["budgets"]["final_rgb_mean_abs"] = rgb_mean_abs;
    report["budgets"]["known_rgb_exact"] = true;
    save_json(a.output / "report.json", report);
    try {
        const Json original = read_original_report(a.reference, a.model);
        report["original_report_sha256"] = core_sha256::file((a.reference / "report.json").string());
        report["original_producer"] = original.at("producer");
        report["original_versions"] = original.at("versions");
        // Validate every file before loading multi-GB models. An incomplete
        // reference cannot become a pass because a missing stage was skipped.
        std::vector<Case> prepared;
        for (const Json & row : original.at("cases").rows()) {
            try {
                prepared.push_back(read_case(a.reference, row, a.model));
            } catch (const std::exception & e) {
                report["errors"].a.emplace_back(row.at("id").str() + ": reference validation: " + e.what());
            }
        }
        require(report["errors"].a.empty(), "one or more original case manifests failed validation");
        report["denoiser_conversion"] = verify_receipt(a.denoiser, a.model_sha, a.model);
        report["vae_conversion"] = verify_receipt(a.vae, a.vae_sha, "sdxl_vae");
        save_json(a.output / "report.json", report);
        latent_inpaint::Options options;
        options.model = a.model;
        options.denoiser_path = a.denoiser.string();
        options.vae_path = a.vae.string();
        options.denoiser_sha256 = a.model_sha;
        options.vae_sha256 = a.vae_sha;
        options.threads = a.threads;
        options.guidance = a.model == "moebius" ? 2.0f : 4.5f;
        std::string error;
        std::unique_ptr<latent_inpaint::context, decltype(&latent_inpaint::free)> context(
            latent_inpaint::init(options, error), &latent_inpaint::free);
        require(bool(context), "native initialization failed: " + error);
        bool passed = true;
        for (const Case & reference : prepared) {
            std::cout << "Replaying " << a.model << '/' << reference.id << " with " << reference.stages.size()
                      << " original stages\n"
                      << std::flush;
            Json result = replay_case(context.get(), reference, a.output / reference.id, a.model);
            if (!result.at("pass").flag()) {
                passed = false;
                report["errors"].a.emplace_back("native parity failed: " + reference.id);
            }
            report["cases"].a.push_back(std::move(result));
            save_json(a.output / "report.json", report);
        }
        report["complete"] = true;
        report["synthetic_parity_pass"] = passed;
        report["full_synthetic_panel_pass"] = passed && prepared.size() == cases.size();
    } catch (const std::exception & e) {
        report["errors"].a.emplace_back(e.what());
    }
    save_json(a.output / "report.json", report);
    const bool passed =
        report.at("complete").flag() && report.at("synthetic_parity_pass").flag() && report.at("errors").a.empty();
    std::cout << "LATENT_INPAINT_PARITY {\"passed\":" << (passed ? "true" : "false")
              << ",\"report\":" << escape((a.output / "report.json").string()) << "}\n";
    return passed ? 0 : 1;
}

} // namespace

static int crispembed_test_main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--contracts") {
            contract_controls();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout
                << "test-latent-inpaint --contracts\n"
                   "test-latent-inpaint --model moebius|pixelhacker --denoiser MODEL.gguf --vae VAE.gguf\n"
                   "  --model-sha256 SHA --vae-sha256 SHA --reference-dir ORIGINAL_DIR --out NEW_DIR [--threads 1..8]\n"
                   "Requires matching .gguf.json receipts and the model's explicit default-off environment gate.\n";
            return 0;
        }
        return run(arguments(argc, argv));
    } catch (const std::exception & e) {
        std::cerr << "latent-inpaint parity: " << e.what() << '\n';
        return 1;
    }
}

int main(int argc, char ** argv) {
    const int result = crispembed_test_main(argc, argv);
    std::cout.flush();
    std::cerr.flush();
    core_util::clean_exit(result);
}
