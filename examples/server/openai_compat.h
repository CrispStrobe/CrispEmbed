// openai_compat.h — request parsing for crispembed-server's opt-in
// OpenAI-shaped OCR route, POST /v1/chat/completions.
//
// OpenAI-compatible clients (the official SDKs, LiteLLM, vLLM-style OCR
// tooling) send page images as chat messages:
//
//   {"model": "...", "max_tokens": 1024,
//    "messages": [{"role": "user", "content": [
//        {"type": "text", "text": "Convert this page to markdown."},
//        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}]}]}
//
// Only what an OCR model can honour is accepted: the LAST user message, its
// text parts (joined with "\n", used as the per-request prompt) and exactly
// one image, given as a base64 data URL. Remote http(s) URLs and file:// are
// rejected — the server must not become a fetch proxy, and server-side paths
// stay on the confined native routes (--image-root). Multi-turn history,
// tools, streaming and n > 1 are rejected explicitly instead of being ignored.
//
// Header-only, no httplib/ggml, so tests/test_server_openai_compat.cpp runs
// host-only.
#pragma once

#include "core/json.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace server_compat {

// Strict RFC 4648 base64 (standard alphabet). ASCII whitespace is skipped
// (line-wrapped payloads); padding is optional but, when present, must be
// correct and final. Returns false on any other character or a bad length.
inline bool base64_decode(const std::string & in, size_t begin, std::string & out, size_t max_out = SIZE_MAX) {
    auto val = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    out.clear();
    uint32_t acc = 0;
    int n = 0;   // sextets in acc
    int pad = 0; // '=' seen
    for (size_t i = begin; i < in.size(); i++) {
        const unsigned char c = (unsigned char)in[i];
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        if (c == '=') {
            pad++;
            continue;
        }
        if (pad) return false; // data after padding
        const int v = val(c);
        if (v < 0) return false;
        acc = (acc << 6) | (uint32_t)v;
        if (++n == 4) {
            if (out.size() + 3 > max_out) return false;
            out.push_back((char)((acc >> 16) & 0xFF));
            out.push_back((char)((acc >> 8) & 0xFF));
            out.push_back((char)(acc & 0xFF));
            acc = 0;
            n = 0;
        }
    }
    if (n == 1) return false; // 6 bits cannot encode a byte
    if (pad && (n == 0 || n + pad != 4)) return false;
    if (n == 2) {
        if (out.size() + 1 > max_out) return false;
        out.push_back((char)((acc >> 4) & 0xFF));
    } else if (n == 3) {
        if (out.size() + 2 > max_out) return false;
        out.push_back((char)((acc >> 10) & 0xFF));
        out.push_back((char)((acc >> 2) & 0xFF));
    }
    return true;
}

// data:[<mediatype>][;param...];base64,<payload> → raw bytes.
inline bool decode_data_url(const std::string & url, std::string & bytes, std::string & err,
                            size_t max_bytes = SIZE_MAX) {
    if (url.compare(0, 5, "data:") != 0) {
        err = "image_url must be a base64 data URL (data:image/...;base64,...); remote and file URLs are not fetched";
        return false;
    }
    const size_t comma = url.find(',');
    if (comma == std::string::npos) {
        err = "malformed data URL";
        return false;
    }
    const std::string meta = url.substr(5, comma - 5);
    const std::string tag = ";base64";
    if (meta.size() < tag.size() || meta.compare(meta.size() - tag.size(), tag.size(), tag) != 0) {
        err = "data URL must be base64-encoded";
        return false;
    }
    // Upper bound before decoding (whitespace only shrinks it), so an oversize
    // payload is refused without allocating its decoded copy.
    const size_t payload = url.size() - comma - 1;
    if (max_bytes != SIZE_MAX && payload / 4 * 3 > max_bytes + 2) {
        err = "image exceeds the server's size limit";
        return false;
    }
    if (!base64_decode(url, comma + 1, bytes, max_bytes)) {
        err = "invalid base64 in data URL";
        return false;
    }
    if (bytes.empty()) {
        err = "empty image";
        return false;
    }
    return true;
}

struct chat_ocr_request {
    std::string model;
    std::string prompt;    // joined text parts of the last user message ("" = model default)
    std::string image_url; // the one image part's URL, undecoded
    int max_tokens = 0;    // 0 = server default
};

namespace detail {

inline size_t skip_ws(const std::string & s, size_t i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
    return i;
}

// Split a JSON array value starting at body[i] == '[' into its element texts.
inline bool array_elements(const std::string & body, size_t i, std::vector<std::string> & out) {
    out.clear();
    if (i >= body.size() || body[i] != '[') return false;
    i = skip_ws(body, i + 1);
    if (i < body.size() && body[i] == ']') return true;
    while (i < body.size()) {
        const size_t end = core_json::json_skip_value(body, i);
        if (end == std::string::npos) return false;
        out.emplace_back(body.substr(i, end - i));
        i = skip_ws(body, end);
        if (i < body.size() && body[i] == ',') {
            i = skip_ws(body, i + 1);
            continue;
        }
        return i < body.size() && body[i] == ']';
    }
    return false;
}

// Top-level string field of an object text; false when absent or not a string.
inline bool string_field(const std::string & obj, const char * key, std::string & out) {
    const size_t v = core_json::json_find_key_value(obj, key);
    if (v == std::string::npos || obj[v] != '"') return false;
    size_t end = 0;
    return core_json::json_decode_string(obj, v, out, end);
}

inline bool has_field(const std::string & obj, const char * key) {
    return core_json::json_find_key_value(obj, key) != std::string::npos;
}

// The literal JSON token of a top-level field ("true", "1", ...), or "".
inline std::string raw_field(const std::string & obj, const char * key) {
    const size_t v = core_json::json_find_key_value(obj, key);
    if (v == std::string::npos) return "";
    const size_t end = core_json::json_skip_value(obj, v);
    return end == std::string::npos ? "" : obj.substr(v, end - v);
}

} // namespace detail

inline bool parse_chat_ocr_request(const std::string & body, chat_ocr_request & out, std::string & err) {
    using namespace detail;
    out = chat_ocr_request{};
    detail::string_field(body, "model", out.model);

    // Explicit rejections: features an OCR model cannot honour must not be
    // silently dropped.
    if (raw_field(body, "stream") == "true") {
        err = "stream is not supported";
        return false;
    }
    {
        const std::string n = raw_field(body, "n");
        if (!n.empty() && n != "1" && n != "null") {
            err = "n must be 1";
            return false;
        }
    }
    if (has_field(body, "tools") && raw_field(body, "tools") != "null" && raw_field(body, "tools") != "[]") {
        err = "tools are not supported";
        return false;
    }

    for (const char * key : { "max_completion_tokens", "max_tokens" }) {
        const std::string raw = raw_field(body, key);
        if (raw.empty() || raw == "null") continue;
        const double v = core_json::json_extract_number(body, key, -1.0);
        if (!(v >= 1 && v <= 1e6) || v != (double)(int)v) {
            err = std::string(key) + " must be a positive integer";
            return false;
        }
        out.max_tokens = (int)v;
        break;
    }

    const size_t mv = core_json::json_find_key_value(body, "messages");
    std::vector<std::string> messages;
    if (mv == std::string::npos || !array_elements(body, mv, messages) || messages.empty()) {
        err = "messages must be a non-empty array";
        return false;
    }
    const std::string * user = nullptr;
    for (const auto & m : messages) {
        std::string role;
        if (!string_field(m, "role", role)) {
            err = "every message needs a string role";
            return false;
        }
        if (role == "user") user = &m;
    }
    if (!user) {
        err = "no user message";
        return false;
    }

    const size_t cv = core_json::json_find_key_value(*user, "content");
    if (cv == std::string::npos) {
        err = "user message has no content";
        return false;
    }
    if ((*user)[cv] == '"') {
        err = "user message has no image (send content parts with an image_url)";
        return false;
    }
    std::vector<std::string> parts;
    if (!array_elements(*user, cv, parts)) {
        err = "content must be a string or an array of parts";
        return false;
    }
    int images = 0;
    for (const auto & p : parts) {
        std::string type;
        if (!string_field(p, "type", type)) {
            err = "every content part needs a type";
            return false;
        }
        if (type == "text") {
            std::string text;
            if (!string_field(p, "text", text)) {
                err = "text part without text";
                return false;
            }
            if (!out.prompt.empty() && !text.empty()) out.prompt += '\n';
            out.prompt += text;
        } else if (type == "image_url") {
            images++;
            const size_t iv = core_json::json_find_key_value(p, "image_url");
            if (iv == std::string::npos) {
                err = "image_url part without image_url";
                return false;
            }
            if (p[iv] == '"') { // legacy flat string form
                size_t end = 0;
                core_json::json_decode_string(p, iv, out.image_url, end);
            } else {
                const size_t end = core_json::json_skip_value(p, iv);
                if (end == std::string::npos || !string_field(p.substr(iv, end - iv), "url", out.image_url)) {
                    err = "image_url needs a url";
                    return false;
                }
            }
        } else {
            err = "unsupported content part type: " + type;
            return false;
        }
    }
    if (images != 1) {
        err = images == 0 ? "user message has no image" : "exactly one image per request is supported";
        return false;
    }
    return true;
}

} // namespace server_compat
