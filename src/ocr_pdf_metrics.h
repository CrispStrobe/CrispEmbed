// Original geometric TrueType glyphs for invisible PDF run metrics.
// No external font program, runtime font package or shaping dependency.
#pragma once

#include "ocr_pdf_font.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace ocr_pdf_metrics {
static constexpr uint16_t units_per_em = 64;
static constexpr uint16_t maximum_width = 32767;

inline uint16_t read16(const std::string & data, size_t offset) {
    return (uint16_t(uint8_t(data.at(offset))) << 8) | uint8_t(data.at(offset + 1));
}
inline uint32_t read32(const std::string & data, size_t offset) {
    return (uint32_t(read16(data, offset)) << 16) | read16(data, offset + 2);
}
inline void write16(std::string & data, size_t offset, uint16_t value) {
    data.at(offset) = char(value >> 8);
    data.at(offset + 1) = char(value);
}
inline void write32(std::string & data, size_t offset, uint32_t value) {
    write16(data, offset, uint16_t(value >> 16));
    write16(data, offset + 2, uint16_t(value));
}
inline void append32(std::string & data, uint32_t value) {
    const size_t offset = data.size();
    data.resize(offset + 4);
    write32(data, offset, value);
}
inline void pad4(std::string & data) {
    while (data.size() & 3) data += '\0';
}
inline uint32_t checksum(std::string data) {
    pad4(data);
    uint32_t result = 0;
    for (size_t i = 0; i < data.size(); i += 4) result += read32(data, i);
    return result;
}

struct font {
    std::string bytes;
    std::map<uint16_t, uint16_t> glyph_ids;
    uint16_t widest = units_per_em;
};

inline font build(const std::set<uint16_t> & requested_widths) {
    font result;
    std::vector<uint16_t> widths{ units_per_em };
    result.glyph_ids.emplace(units_per_em, 1);
    for (const uint16_t width : requested_widths) {
        if (width == units_per_em) continue;
        widths.push_back(width);
        result.glyph_ids.emplace(width, uint16_t(widths.size()));
        result.widest = std::max(result.widest, width);
    }
    const uint16_t count = uint16_t(widths.size() + 1); // empty .notdef first
    const std::string source(reinterpret_cast<const char *>(ocr_pdf_font), sizeof(ocr_pdf_font));
    std::map<std::string, std::string> tables;
    for (size_t i = 0; i < read16(source, 4); i++) {
        const size_t entry = 12 + i * 16;
        tables.emplace(source.substr(entry, 4), source.substr(read32(source, entry + 8), read32(source, entry + 12)));
    }

    auto & head = tables.at("head");
    write32(head, 8, 0); // checksum adjustment is filled after all tables
    write16(head, 18, units_per_em);
    write16(head, 40, result.widest);
    write16(head, 42, units_per_em);
    write16(head, 50, 1); // long loca offsets
    auto & hhea = tables.at("hhea");
    write16(hhea, 4, units_per_em);
    write16(hhea, 10, result.widest);
    write16(hhea, 16, result.widest);
    write16(hhea, 34, count);
    write16(tables.at("maxp"), 4, count);
    auto & os2 = tables.at("OS/2");
    for (const size_t offset : { 2u, 68u, 74u, 86u, 88u }) write16(os2, offset, units_per_em);
    auto & post = tables.at("post");
    post.resize(32);
    write32(post, 0, 0x00030000); // no glyph-name array
    write32(post, 12, 0);         // proportional metrics

    auto & hmtx = tables.at("hmtx");
    hmtx.assign(size_t(count) * 4, '\0');
    write16(hmtx, 0, units_per_em);
    auto & glyf = tables.at("glyf");
    glyf.clear();
    auto & loca = tables.at("loca");
    loca.clear();
    append32(loca, 0); // empty .notdef
    append32(loca, 0);
    for (size_t i = 0; i < widths.size(); i++) {
        const uint16_t width = widths[i];
        write16(hmtx, (i + 1) * 4, width);
        // One four-point rectangle, with explicit signed 16-bit deltas.
        std::string glyph(34, '\0');
        write16(glyph, 0, 1);
        write16(glyph, 6, width);
        write16(glyph, 8, units_per_em);
        write16(glyph, 10, 3);
        for (size_t flag = 14; flag < 18; flag++) glyph[flag] = 1;
        write16(glyph, 20, width);
        write16(glyph, 24, uint16_t(-int32_t(width)));
        write16(glyph, 30, units_per_em);
        pad4(glyph);
        glyf += glyph;
        append32(loca, uint32_t(glyf.size()));
    }

    const uint16_t table_count = uint16_t(tables.size());
    uint16_t power = 1, selector = 0;
    while (power * 2 <= table_count) {
        power *= 2;
        selector++;
    }
    result.bytes.assign(12 + size_t(table_count) * 16, '\0');
    write32(result.bytes, 0, 0x00010000);
    write16(result.bytes, 4, table_count);
    write16(result.bytes, 6, power * 16);
    write16(result.bytes, 8, selector);
    write16(result.bytes, 10, table_count * 16 - power * 16);
    size_t entry = 12, head_offset = 0;
    for (const auto & table : tables) {
        pad4(result.bytes);
        const size_t offset = result.bytes.size();
        result.bytes.replace(entry, 4, table.first);
        write32(result.bytes, entry + 4, checksum(table.second));
        write32(result.bytes, entry + 8, uint32_t(offset));
        write32(result.bytes, entry + 12, uint32_t(table.second.size()));
        result.bytes += table.second;
        if (table.first == "head") head_offset = offset;
        entry += 16;
    }
    pad4(result.bytes);
    write32(result.bytes, head_offset + 8, 0xb1b0afba - checksum(result.bytes));
    return result;
}
} // namespace ocr_pdf_metrics
