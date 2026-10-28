// Unlimited-OCR-2 output text: the original's decode, plus an explicitly named clean rendering.
//
// Original (baidu/Unlimited-OCR-2 @04647c7a, modeling_unlimitedocr.py:1201):
//     text = tokenizer.decode(gen, skip_special_tokens=False)
// with `gen` excluding EOS (the loop breaks before appending it, :1180-1182). The tokenizer is
// LlamaTokenizerFast (tokenizer_config.json: clean_up_tokenization_spaces=false) over a
// tokenizer.json whose decoder is ByteLevel. So every generated id -- including the added
// layout/table tokens <PAGE> <BLOCK> <FORK> <td> </td> <tr> </tr> -- is emitted, and HF
// `tokenizers` ByteLevel decode is:
//   * per token, map every char through the GPT-2 byte table; if ANY char is not in the table
//     the WHOLE token contributes its raw UTF-8 bytes instead (not per char),
//   * concatenate, then String::from_utf8_lossy (each maximal invalid subpart -> U+FFFD).
// decode_raw() reproduces that byte-exactly (tests/test_unlimited_ocr2_text.cpp replays the
// Kaggle trial's original IDs and texts).
//
// clean() is a separate rendering, never the default: drops <PAGE>, the per-block layout header
// `<BLOCK>label[<LEVELn>][x1, y1, x2, y2]<FORK>`, and stray <BLOCK>/<FORK>/<CHILD>, keeping the
// block text, its newlines and table HTML (<table><tr><td>...) unchanged.
#pragma once

#include "core/bpe.h"

#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

namespace uocr2_text {

// Rust String::from_utf8_lossy / Unicode "maximal subpart" substitution.
inline std::string utf8_lossy(const std::string & s) {
    static constexpr const char kRepl[] = "\xEF\xBF\xBD";
    std::string out;
    out.reserve(s.size());
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char b = (unsigned char)s[i];
        if (b < 0x80) {
            out.push_back((char)b);
            i++;
            continue;
        }
        int len = 0;
        unsigned char lo = 0x80, hi = 0xBF; // valid range of the first continuation byte
        if (b >= 0xC2 && b <= 0xDF)
            len = 2;
        else if (b >= 0xE0 && b <= 0xEF) {
            len = 3;
            if (b == 0xE0) lo = 0xA0;
            if (b == 0xED) hi = 0x9F;
        } else if (b >= 0xF0 && b <= 0xF4) {
            len = 4;
            if (b == 0xF0) lo = 0x90;
            if (b == 0xF4) hi = 0x8F;
        }
        if (len == 0) {
            out += kRepl;
            i++;
            continue;
        }
        int k = 1;
        for (; k < len && i + k < n; k++) {
            const unsigned char c = (unsigned char)s[i + k];
            const unsigned char l = (k == 1) ? lo : 0x80, h = (k == 1) ? hi : 0xBF;
            if (c < l || c > h) break;
        }
        if (k == len) {
            out.append(s, i, len);
        } else {
            out += kRepl;
        }
        i += k;
    }
    return out;
}

// One token's ByteLevel bytes: all chars mapped, or (any unmapped) the raw piece.
inline void append_token_bytes(const std::string & piece, std::string & out) {
    const auto & dec = core_bpe::byte_decoder();
    std::string mapped;
    mapped.reserve(piece.size());
    size_t i = 0;
    while (i < piece.size()) {
        const unsigned char c = (unsigned char)piece[i];
        size_t len = core_bpe::utf8_len(c);
        if (i + len > piece.size()) {
            out += piece;
            return;
        }
        uint32_t cp = 0;
        if (len == 1)
            cp = c;
        else if (len == 2)
            cp = ((c & 0x1F) << 6) | (piece[i + 1] & 0x3F);
        else if (len == 3)
            cp = ((c & 0x0F) << 12) | ((piece[i + 1] & 0x3F) << 6) | (piece[i + 2] & 0x3F);
        else
            cp = ((c & 0x07) << 18) | ((piece[i + 1] & 0x3F) << 12) | ((piece[i + 2] & 0x3F) << 6) |
                 (piece[i + 3] & 0x3F);
        auto it = dec.find(cp);
        if (it == dec.end()) {
            out += piece; // whole-token fallback, as tokenizers' ByteLevel decoder
            return;
        }
        mapped.push_back((char)it->second);
        i += len;
    }
    out += mapped;
}

// tokenizer.decode(ids, skip_special_tokens=False) for generated ids. `eos` (never part of the
// original's `gen`) and out-of-vocabulary ids are skipped.
inline std::string decode_raw(const std::vector<std::string> & pieces, const int32_t * ids, int n, int eos) {
    std::string bytes;
    for (int i = 0; i < n; i++) {
        const int id = ids[i];
        if (id == eos || id < 0 || id >= (int)pieces.size()) continue;
        append_token_bytes(pieces[(size_t)id], bytes);
    }
    return utf8_lossy(bytes);
}

// `label[<LEVELn>][x1, y1, x2, y2]` (box optional) -- the header between <BLOCK> and <FORK>.
inline bool is_layout_header(const std::string & h) {
    size_t i = 0;
    const size_t n = h.size();
    while (i < n && (isalnum((unsigned char)h[i]) || h[i] == '_' || h[i] == '-' || h[i] == ' ')) i++;
    if (i == 0) return false;
    if (h.compare(i, 6, "<LEVEL") == 0) {
        i += 6;
        const size_t d0 = i;
        while (i < n && isdigit((unsigned char)h[i])) i++;
        if (i == d0 || i >= n || h[i] != '>') return false;
        i++;
    }
    if (i == n) return true;
    if (h[i] != '[') return false;
    i++;
    while (i < n && (isdigit((unsigned char)h[i]) || h[i] == ',' || h[i] == ' ' || h[i] == '.' || h[i] == '-')) i++;
    return i + 1 == n && h[i] == ']';
}

inline std::string clean(const std::string & raw) {
    static constexpr const char kPage[] = "<PAGE>", kBlock[] = "<BLOCK>", kFork[] = "<FORK>", kChild[] = "<CHILD>";
    std::string out;
    out.reserve(raw.size());
    size_t i = 0;
    const size_t n = raw.size();
    while (i < n) {
        if (raw.compare(i, 6, kPage) == 0) {
            if (!out.empty()) {
                while (!out.empty() && out.back() == '\n') out.pop_back();
                out += "\n\n";
            }
            i += 6;
        } else if (raw.compare(i, 7, kBlock) == 0) {
            i += 7;
            const size_t f = raw.find(kFork, i);
            if (f != std::string::npos && is_layout_header(raw.substr(i, f - i))) i = f + 6;
        } else if (raw.compare(i, 6, kFork) == 0) {
            i += 6;
        } else if (raw.compare(i, 7, kChild) == 0) {
            i += 7;
        } else {
            out.push_back(raw[i++]);
        }
    }
    return out;
}

enum class output_mode { raw, clean, legacy };

// CRISPEMBED_UNLIMITED_OCR2_OUTPUT: unset/"raw" = original decode (default); "clean" = clean();
// "legacy" = the pre-2026-10-28 native text (all <...> pieces dropped, kept for A/B).
inline output_mode parse_output_mode(const char * v) {
    if (!v || !*v) return output_mode::raw;
    const std::string s(v);
    if (s == "clean" || s == "markdown") return output_mode::clean;
    if (s == "legacy") return output_mode::legacy;
    return output_mode::raw;
}

} // namespace uocr2_text
