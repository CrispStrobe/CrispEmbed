// ocr_render.cpp — OCR result renderers.
//
// Implements plain text, hOCR, ALTO, and searchable PDF output.
// PDF uses the existing image decoder and an original embedded metric font.
// Page images are visible; Unicode OCR text is searchable but never painted.

#include "ocr_render.h"
#include "ocr_pdf_font.h"
#include "ocr_pdf_srgb.h"
#include <SheenBidi/SheenBidi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Implemented once in image_preprocess.cpp and core/image_out.cpp.
extern "C" {
int stbi_info(const char * filename, int * x, int * y, int * comp);
unsigned char * stbi_load(const char * filename, int * x, int * y, int * comp, int req_comp);
void stbi_image_free(void * data);
unsigned char * stbi_zlib_compress(unsigned char * data, int data_len, int * out_len, int quality);
}

// ---------------------------------------------------------------------------
// Renderer state
// ---------------------------------------------------------------------------

// Stored page data (for PDF which needs all pages at end)
struct stored_word {
    std::string text;
    int x, y, w, h;
    float confidence;
};
struct stored_line {
    std::vector<stored_word> words;
    int x, y, w, h;
};
struct stored_page {
    std::vector<stored_line> lines;
    int width, height;
    double width_points, height_points;
    std::string image_path; // for PDF image embedding
};

struct ocr_renderer {
    ocr_render_format format;
    std::string separator; // page separator for text mode
    std::string output;    // accumulated output
    int page_count;
    bool begun = false, ended = false;
    std::vector<stored_page> pages; // for PDF deferred rendering
    bool pdfa = false;              // explicit PDF/A-2b request
    bool pdf_bidi = false;          // explicit Unicode layout opt-in
    std::string error;
    double next_width_points = 0, next_height_points = 0;

    ocr_renderer() : format(OCR_RENDER_TEXT), separator("\f"), page_count(0) {}
};

// ---------------------------------------------------------------------------
// XML helpers
// ---------------------------------------------------------------------------

// Consume Unicode scalars, replacing malformed UTF-8. Never copy an invalid
// byte sequence into XML or encode a surrogate as an independent PDF glyph.
static std::vector<uint32_t> unicode_scalars(const char * text) {
    std::vector<uint32_t> out;
    if (!text) return out;
    const auto * s = reinterpret_cast<const unsigned char *>(text);
    const size_t size = std::strlen(text);
    for (size_t i = 0; i < size;) {
        const unsigned char c = s[i];
        size_t n = c < 0x80                 ? 1
                   : c >= 0xc2 && c <= 0xdf ? 2
                   : c >= 0xe0 && c <= 0xef ? 3
                   : c >= 0xf0 && c <= 0xf4 ? 4
                                            : 0;
        uint32_t cp = n == 1 ? c : n == 2 ? c & 0x1f : n == 3 ? c & 0x0f : c & 0x07;
        bool valid = n != 0 && n <= size - i;
        for (size_t j = 1; valid && j < n; j++) {
            valid = (s[i + j] & 0xc0) == 0x80;
            cp = (cp << 6) | (s[i + j] & 0x3f);
        }
        valid = valid && !(n == 2 && cp < 0x80) && !(n == 3 && cp < 0x800) && !(n == 4 && cp < 0x10000) &&
                cp <= 0x10ffff && !(cp >= 0xd800 && cp <= 0xdfff);
        out.push_back(valid ? cp : 0xfffd);
        i += valid ? n : 1;
    }
    return out;
}

static void append_utf8(std::string & out, uint32_t cp) {
    if (cp < 0x80) {
        out += char(cp);
    } else if (cp < 0x800) {
        out += char(0xc0 | (cp >> 6));
        out += char(0x80 | (cp & 0x3f));
    } else if (cp < 0x10000) {
        out += char(0xe0 | (cp >> 12));
        out += char(0x80 | ((cp >> 6) & 0x3f));
        out += char(0x80 | (cp & 0x3f));
    } else {
        out += char(0xf0 | (cp >> 18));
        out += char(0x80 | ((cp >> 12) & 0x3f));
        out += char(0x80 | ((cp >> 6) & 0x3f));
        out += char(0x80 | (cp & 0x3f));
    }
}

static std::string xml_escape(const char * s) {
    std::string out;
    for (uint32_t cp : unicode_scalars(s)) {
        switch (cp) {
        case '&':
            out += "&amp;";
            break;
        case '<':
            out += "&lt;";
            break;
        case '>':
            out += "&gt;";
            break;
        case '"':
            out += "&quot;";
            break;
        case '\n':
            out += "&#xA;";
            break;
        case '\r':
            out += "&#xD;";
            break;
        case '\t':
            out += "&#x9;";
            break;
        default:
            // XML 1.0 forbids other C0 controls, U+FFFE and U+FFFF.
            append_utf8(out, cp < 0x20 || cp == 0xfffe || cp == 0xffff ? 0xfffd : cp);
            break;
        }
    }
    return out;
}

static double bounded_confidence(float value) {
    return std::isfinite(value) ? std::max(0.0, std::min(1.0, double(value))) : 0.0;
}

static std::string decimal(double value, int precision = 16) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

// ---------------------------------------------------------------------------
// Plain text renderer
// ---------------------------------------------------------------------------

static void text_begin(ocr_renderer *) {}

static void text_add_page(ocr_renderer * r, const ocr_render_page * page) {
    if (r->page_count > 0) r->output += r->separator;

    for (int i = 0; i < page->n_lines; i++) {
        const ocr_render_line * line = &page->lines[i];
        for (int j = 0; j < line->n_words; j++) {
            if (j > 0) r->output += ' ';
            if (line->words[j].text) r->output += line->words[j].text;
        }
        r->output += '\n';
    }
    r->page_count++;
}

static void text_end(ocr_renderer *) {}

// ---------------------------------------------------------------------------
// hOCR renderer
// ---------------------------------------------------------------------------

// Producer identity for the output formats. An archived hOCR or ALTO should be
// traceable to a build: naming the software without a version is not that, and
// Tesseract's own convention is "tesseract 4.1.1". One definition, so the three
// formats cannot drift apart.
//
// NOTE: this identifies the SOFTWARE, not the recognition engine. POLICY.md §6
// distinguishes CTC/attention recognisers, which transcribe, from VLM engines,
// which confabulate through a smudge rather than leave a gap — and a reader of
// the archived file cannot tell them apart from the text. Recording the engine
// needs it threaded through the orchestrator; see PLAN.md.
#ifndef CRISPEMBED_VERSION_STR
#define CRISPEMBED_VERSION_STR "unknown"
#endif
static const char * producer_name() {
    return "CrispEmbed " CRISPEMBED_VERSION_STR;
}

static void hocr_begin(ocr_renderer * r) {
    r->output += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                 "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\"\n"
                 "  \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n"
                 "<html xmlns=\"http://www.w3.org/1999/xhtml\">\n"
                 "<head>\n"
                 "  <meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" />\n"
                 "  <meta name=\"ocr-capabilities\" content=\"ocr_page ocr_carea ocr_line ocrx_word\" />\n";
    r->output += std::string("  <meta name=\"ocr-system\" content=\"") + producer_name() + "\" />\n";
    r->output += "</head>\n"
                 "<body>\n";
}

static void hocr_add_page(ocr_renderer * r, const ocr_render_page * page) {
    char buf[512];
    int pid = r->page_count + 1;

    snprintf(buf, sizeof(buf),
             "  <div class=\"ocr_page\" id=\"page_%d\" "
             "title=\"bbox 0 0 %d %d; ppageno %d\">\n",
             pid, page->page_width, page->page_height, r->page_count);
    r->output += buf;

    for (int i = 0; i < page->n_lines; i++) {
        const ocr_render_line * line = &page->lines[i];
        int lid = i + 1;
        snprintf(buf, sizeof(buf),
                 "    <span class=\"ocr_line\" id=\"line_%d_%d\" "
                 "title=\"bbox %d %d %d %d\">\n",
                 pid, lid, line->x, line->y, line->x + line->w, line->y + line->h);
        r->output += buf;

        for (int j = 0; j < line->n_words; j++) {
            const ocr_render_word * w = &line->words[j];
            int wid = j + 1;
            const std::string confidence =
                std::isfinite(w->confidence) && w->confidence >= 0
                    ? "; x_wconf " + std::to_string(int(bounded_confidence(w->confidence) * 100))
                    : "";
            snprintf(buf, sizeof(buf),
                     "      <span class=\"ocrx_word\" id=\"word_%d_%d_%d\" "
                     "title=\"bbox %d %d %d %d%s\">",
                     pid, lid, wid, w->x, w->y, w->x + w->w, w->y + w->h, confidence.c_str());
            r->output += buf;
            r->output += xml_escape(w->text);
            r->output += "</span>\n";
        }
        r->output += "    </span>\n";
    }
    r->output += "  </div>\n";
    r->page_count++;
}

static void hocr_end(ocr_renderer * r) {
    r->output += "</body>\n</html>\n";
}

// ---------------------------------------------------------------------------
// ALTO 3.1 renderer
// ---------------------------------------------------------------------------

static void alto_begin(ocr_renderer * r) {
    r->output += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                 "<alto xmlns=\"http://www.loc.gov/standards/alto/ns-v3#\"\n"
                 "      xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n"
                 "      xsi:schemaLocation=\"http://www.loc.gov/standards/alto/ns-v3# "
                 "http://www.loc.gov/alto/v3/alto-3-1.xsd\">\n"
                 "  <Description>\n"
                 "    <MeasurementUnit>pixel</MeasurementUnit>\n"
                 "    <OCRProcessing ID=\"ocr_0\">\n"
                 "      <ocrProcessingStep>\n"
                 "        <processingSoftware>\n"
                 "          <softwareName>CrispEmbed</softwareName>\n"
                 "          <softwareVersion>" CRISPEMBED_VERSION_STR "</softwareVersion>\n"
                 "        </processingSoftware>\n"
                 "      </ocrProcessingStep>\n"
                 "    </OCRProcessing>\n"
                 "  </Description>\n"
                 "  <Layout>\n";
}

static void alto_add_page(ocr_renderer * r, const ocr_render_page * page) {
    char buf[512];
    int pid = r->page_count;
    snprintf(buf, sizeof(buf),
             "    <Page ID=\"page_%d\" WIDTH=\"%d\" HEIGHT=\"%d\">\n"
             "      <PrintSpace>\n",
             pid, page->page_width, page->page_height);
    r->output += buf;

    // Each line becomes a TextBlock with a single TextLine
    for (int i = 0; i < page->n_lines; i++) {
        const ocr_render_line * line = &page->lines[i];
        snprintf(buf, sizeof(buf),
                 "        <TextBlock ID=\"block_%d_%d\" "
                 "HPOS=\"%d\" VPOS=\"%d\" WIDTH=\"%d\" HEIGHT=\"%d\">\n"
                 "          <TextLine HPOS=\"%d\" VPOS=\"%d\" WIDTH=\"%d\" HEIGHT=\"%d\">\n",
                 pid, i, line->x, line->y, line->w, line->h, line->x, line->y, line->w, line->h);
        r->output += buf;

        for (int j = 0; j < line->n_words; j++) {
            const ocr_render_word * w = &line->words[j];
            r->output += "            <String CONTENT=\"" + xml_escape(w->text) + "\" HPOS=\"" + std::to_string(w->x) +
                         "\" VPOS=\"" + std::to_string(w->y) + "\" WIDTH=\"" + std::to_string(w->w) + "\" HEIGHT=\"" +
                         std::to_string(w->h) + "\"";
            if (std::isfinite(w->confidence) && w->confidence >= 0)
                r->output += " WC=\"" + decimal(bounded_confidence(w->confidence), 2) + "\"";
            r->output += " />\n";

            // Add space after word (except last)
            if (j < line->n_words - 1) {
                int sp_x = w->x + w->w;
                int next_x = line->words[j + 1].x;
                int sp_w = next_x > sp_x ? next_x - sp_x : 4;
                snprintf(buf, sizeof(buf), "            <SP WIDTH=\"%d\" HPOS=\"%d\" VPOS=\"%d\" />\n", sp_w, sp_x,
                         w->y);
                r->output += buf;
            }
        }
        r->output += "          </TextLine>\n"
                     "        </TextBlock>\n";
    }
    r->output += "      </PrintSpace>\n"
                 "    </Page>\n";
    r->page_count++;
}

static void alto_end(ocr_renderer * r) {
    r->output += "  </Layout>\n</alto>\n";
}

// ---------------------------------------------------------------------------
// Searchable PDF renderer
// ---------------------------------------------------------------------------
// Every used Unicode scalar gets a CID and a ToUnicode mapping. An original
// embedded metric font gives selection predictable bounds without relying on
// installed fonts. Text rendering mode 3 hides its geometric glyphs. Images
// are decoded and flattened to RGB before being compressed as PDF image data.
// Object bodies are assembled before offsets: page counts have no fixed cap.

struct pdf_document {
    std::vector<std::string> objects{ "" }; // object zero is the free xref entry
    size_t body_bytes = 0;

    int reserve() {
        if (objects.size() >= size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("PDF has too many objects");
        objects.emplace_back();
        return int(objects.size() - 1);
    }
    void set(int id, std::string body) {
        body_bytes -= objects.at(size_t(id)).size();
        if (body.size() > size_t(std::numeric_limits<int>::max()) - body_bytes)
            throw std::runtime_error("PDF exceeds the renderer byte-count limit");
        body_bytes += body.size();
        objects[size_t(id)] = std::move(body);
    }
    int add(std::string body) {
        const int id = reserve();
        set(id, std::move(body));
        return id;
    }
    int stream(const std::string & bytes, const std::string & properties = "") {
        return add("<< " + properties + " /Length " + std::to_string(bytes.size()) + " >>\nstream\n" + bytes +
                   "\nendstream");
    }
    int hex_stream(const std::string & bytes, const std::string & properties = "") {
        constexpr char digits[] = "0123456789ABCDEF";
        std::string encoded;
        encoded.reserve(bytes.size() * 2 + 1);
        for (unsigned char byte : bytes) {
            encoded += digits[byte >> 4];
            encoded += digits[byte & 15];
        }
        encoded += '>';
        return stream(encoded, properties + " /Filter /ASCIIHexDecode");
    }
};

static std::string pdf_ref(int id) {
    return std::to_string(id) + " 0 R";
}

static void hex16(std::string & out, uint32_t value) {
    constexpr char digits[] = "0123456789ABCDEF";
    for (int shift = 12; shift >= 0; shift -= 4) out += digits[(value >> shift) & 15];
}

static void utf16_hex(std::string & out, uint32_t cp) {
    if (cp <= 0xffff) {
        hex16(out, cp);
    } else {
        cp -= 0x10000;
        hex16(out, 0xd800 + (cp >> 10));
        hex16(out, 0xdc00 + (cp & 0x3ff));
    }
}

// CID numbers are not Unicode, but some independent readers first decode
// Identity-H bytes as UTF-16. Avoid surrogate-valued CID slots so adjacent
// glyphs cannot accidentally collapse into one supplementary character.
static constexpr size_t pdf_cids_per_font = 65535 - 2048;
static uint32_t pdf_cid(size_t index) {
    const uint32_t code = uint32_t(index + 1);
    return code >= 0xd800 ? code + 0x800 : code;
}

static bool needs_bidirectional_layout(uint32_t cp) {
    // Conservative script-block guard, including reserved RTL ranges from
    // Unicode 17 DerivedBidiClass. A correct full UAX #9 implementation is
    // required before these can be enabled; reversing whole strings is unsafe
    // for mixed scripts, numbers, combining marks and directional isolates.
    // https://www.unicode.org/Public/17.0.0/ucd/extracted/DerivedBidiClass.txt
    return (cp >= 0x0590 && cp <= 0x08ff) || (cp >= 0xfb1d && cp <= 0xfdff) || (cp >= 0xfe70 && cp <= 0xfeff) ||
           (cp >= 0x10800 && cp <= 0x10fff) || (cp >= 0x1e800 && cp <= 0x1efff) || cp == 0x200e || cp == 0x200f ||
           (cp >= 0x202a && cp <= 0x202e) || (cp >= 0x2066 && cp <= 0x2069);
}

static void validate_physical_size(double width, double height) {
    if (!std::isfinite(width) || !std::isfinite(height) || width <= 0 || height <= 0)
        throw std::runtime_error("Physical PDF page dimensions must be finite and positive");
    if (width < 0.01 || height < 0.01 || width > 14400 || height > 14400)
        throw std::runtime_error(
            "Physical PDF page dimensions must be between 0.01 and 14400 points; set a page resolution");
}

static std::string pdf_unicode_string(const std::vector<uint32_t> & text) {
    std::string out = "<FEFF";
    for (uint32_t cp : text) utf16_hex(out, cp);
    return out + ">";
}

static void pdf_begin(ocr_renderer * r) {
    r->pages.clear();
}

static void pdf_add_page(ocr_renderer * r, const ocr_render_page * page) {
    stored_page sp;
    sp.width = page->page_width;
    sp.height = page->page_height;
    sp.width_points = r->next_width_points > 0 ? r->next_width_points : sp.width;
    sp.height_points = r->next_height_points > 0 ? r->next_height_points : sp.height;
    validate_physical_size(sp.width_points, sp.height_points);
    if (page->image_path) sp.image_path = page->image_path;
    for (int i = 0; i < page->n_lines; i++) {
        const ocr_render_line * line = &page->lines[i];
        stored_line sl;
        sl.x = line->x;
        sl.y = line->y;
        sl.w = line->w;
        sl.h = line->h;
        for (int j = 0; j < line->n_words; j++) {
            const ocr_render_word * w = &line->words[j];
            sl.words.push_back({ w->text ? w->text : "", w->x, w->y, w->w, w->h, w->confidence });
        }
        sp.lines.push_back(std::move(sl));
    }
    r->pages.push_back(std::move(sp));
    r->page_count++;
}

static std::vector<int> pdf_fonts(pdf_document & doc, const std::map<uint32_t, size_t> & characters) {
    if (characters.empty()) return {};
    const int font_file =
        doc.hex_stream(std::string(reinterpret_cast<const char *>(ocr_pdf_font), sizeof(ocr_pdf_font)),
                       "/Length1 " + std::to_string(sizeof(ocr_pdf_font)));
    const int descriptor = doc.add("<< /Type /FontDescriptor /FontName /CrispOcrInvisible /Flags 5 "
                                   "/FontBBox [0 0 1000 1000] /ItalicAngle 0 /Ascent 1000 /Descent 0 /CapHeight 1000 "
                                   "/StemV 80 /MissingWidth 1000 /FontFile2 " +
                                   pdf_ref(font_file) + " >>");
    std::vector<std::vector<uint32_t>> subsets((characters.size() + pdf_cids_per_font - 1) / pdf_cids_per_font);
    for (const auto & entry : characters) subsets[(entry.second - 1) / pdf_cids_per_font].push_back(entry.first);
    std::vector<int> fonts;
    for (size_t f = 0; f < subsets.size(); f++) {
        const auto & subset = subsets[f];
        std::string cmap = "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
                           "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
                           "/CMapName /CrispOcrUnicode" +
                           std::to_string(f + 1) +
                           " def\n/CMapType 2 def\n"
                           "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n";
        for (size_t start = 0; start < subset.size(); start += 100) {
            const size_t end = std::min(start + 100, subset.size());
            cmap += std::to_string(end - start) + " beginbfchar\n";
            for (size_t i = start; i < end; i++) {
                cmap += '<';
                hex16(cmap, pdf_cid(i));
                cmap += "> <";
                utf16_hex(cmap, subset[i]);
                cmap += ">\n";
            }
            cmap += "endbfchar\n";
        }
        cmap += "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n";
        const int unicode = doc.stream(cmap);
        std::string gids((pdf_cid(subset.size() - 1) + 1) * 2, '\0');
        for (size_t i = 0; i < subset.size(); i++) gids[pdf_cid(i) * 2 + 1] = 1;
        const int gid_map = doc.hex_stream(gids);
        const int descendant = doc.add("<< /Type /Font /Subtype /CIDFontType2 /BaseFont /CrispOcrInvisible "
                                       "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
                                       "/FontDescriptor " +
                                       pdf_ref(descriptor) + " /DW 1000 /CIDToGIDMap " + pdf_ref(gid_map) + " >>");
        fonts.push_back(doc.add("<< /Type /Font /Subtype /Type0 /BaseFont /CrispOcrInvisible "
                                "/Encoding /Identity-H /DescendantFonts [" +
                                pdf_ref(descendant) + "] /ToUnicode " + pdf_ref(unicode) + " >>"));
    }
    return fonts;
}

static int pdf_image(pdf_document & doc, const stored_page & page) {
    if (page.image_path.empty()) return 0; // callers may intentionally export a text-only layer
    int width = 0, height = 0, channels = 0;
    if (!stbi_info(page.image_path.c_str(), &width, &height, &channels) || width <= 0 || height <= 0)
        throw std::runtime_error("Could not read PDF page image: " + page.image_path);
    if (width != page.width || height != page.height)
        throw std::runtime_error("PDF page image dimensions do not match the OCR coordinate space");
    constexpr size_t max_pixels = 256 * 1024 * 1024 / 4; // at most 256 MiB decoded RGBA per page
    if (size_t(width) > max_pixels / size_t(height))
        throw std::runtime_error("PDF page image exceeds the 256 MiB decoded-image limit");
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> rgba(
        stbi_load(page.image_path.c_str(), &width, &height, &channels, 4), &stbi_image_free);
    if (!rgba) throw std::runtime_error("Could not decode PDF page image: " + page.image_path);
    if (width != page.width || height != page.height)
        throw std::runtime_error("PDF page image changed dimensions while being read");
    const size_t pixels = size_t(width) * size_t(height);
    std::string rgb(pixels * 3, '\0');
    for (size_t i = 0; i < pixels; i++) {
        const unsigned int alpha = rgba.get()[i * 4 + 3];
        for (size_t c = 0; c < 3; c++)
            rgb[i * 3 + c] = char((rgba.get()[i * 4 + c] * alpha + 255 * (255 - alpha) + 127) / 255);
    }
    rgba.reset();
    int compressed_size = 0;
    std::unique_ptr<unsigned char, decltype(&std::free)> compressed(
        stbi_zlib_compress(reinterpret_cast<unsigned char *>(&rgb[0]), int(rgb.size()), &compressed_size, 8),
        &std::free);
    if (!compressed || compressed_size <= 0) throw std::runtime_error("Could not compress PDF page image");
    return doc.stream(std::string(reinterpret_cast<char *>(compressed.get()), size_t(compressed_size)),
                      "/Type /XObject /Subtype /Image /Width " + std::to_string(width) + " /Height " +
                          std::to_string(height) +
                          " /ColorSpace /DeviceRGB /BitsPerComponent 8 /Interpolate false /Filter /FlateDecode");
}

static void pdf_word(std::string & content, const stored_word & word, int page_height,
                     const std::map<uint32_t, size_t> & characters, bool trailing_space) {
    const auto text = unicode_scalars(word.text.c_str());
    if (text.empty()) return;
    std::vector<std::vector<uint32_t>> rows(1);
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == '\n' || text[i] == '\r') {
            if (text[i] == '\r' && i + 1 < text.size() && text[i + 1] == '\n') i++;
            rows.emplace_back();
        } else {
            rows.back().push_back(text[i]);
        }
    }
    // A corrected paragraph can occupy one region. Extremely narrow glyph
    // advances make readers discard repeated letters as duplicate overprints.
    // Wrap only these overfull rows, preserving all scalars and preferring
    // existing spaces. Explicit line breaks remain hard boundaries. This is
    // an estimated text layer, not recovered character/typographic geometry.
    const double original_height = double(word.h) / double(rows.size());
    std::vector<std::vector<uint32_t>> wrapped;
    for (const auto & row : rows) {
        if (row.size() * original_height <= word.w * 6.0) {
            wrapped.push_back(row);
            continue;
        }
        const size_t count = size_t(std::ceil(std::sqrt(row.size() * original_height / (2.0 * word.w))));
        const size_t columns = std::max(size_t(1), (row.size() + count - 1) / count);
        for (size_t start = 0; start < row.size();) {
            size_t end = std::min(start + columns, row.size());
            if (end < row.size()) {
                for (size_t boundary = end; boundary > start + columns / 2; boundary--) {
                    if (row[boundary - 1] == ' ') {
                        end = boundary;
                        break;
                    }
                }
            }
            wrapped.emplace_back(row.begin() + start, row.begin() + end);
            start = end;
        }
    }
    rows = std::move(wrapped);
    const double height = double(word.h) / double(rows.size());
    for (size_t row = 0; row < rows.size(); row++) {
        auto & scalars = rows[row];
        if (scalars.empty()) continue;
        if (trailing_space && row + 1 == rows.size() && scalars.back() != ' ') scalars.push_back(' ');
        const double advance = double(word.w) / double(scalars.size());
        const double y = double(page_height) - word.y - (row + 1) * height;
        content += "/Span << /ActualText " + pdf_unicode_string(scalars) + " >> BDC\nBT\n3 Tr\n";
        for (size_t start = 0; start < scalars.size();) {
            const size_t font = (characters.at(scalars[start]) - 1) / pdf_cids_per_font;
            size_t end = start;
            std::string encoded = "<";
            while (end < scalars.size() && (characters.at(scalars[end]) - 1) / pdf_cids_per_font == font) {
                hex16(encoded, pdf_cid((characters.at(scalars[end]) - 1) % pdf_cids_per_font));
                end++;
            }
            encoded += "> Tj\n";
            content += "/F" + std::to_string(font + 1) + " 1 Tf\n" + decimal(advance) + " 0 0 " + decimal(height) +
                       " " + decimal(word.x + start * advance) + " " + decimal(y) + " Tm\n" + encoded;
            start = end;
        }
        content += "ET\nEMC\n";
    }
}

// A glyph is never painted. These clusters give combining marks the same
// selection cell as their base and leave direction controls in ActualText
// without allocating visible width to them. UAX #9 resolves levels for the
// whole paragraph before any estimated line wrapping (rules P1 through L3).
static bool pdf_format_control(uint32_t cp) {
    const auto bidi = SBCodepointGetBidiType(cp);
    return SBBidiTypeIsFormat(bidi) || bidi == SBBidiTypeBN || cp == 0x200e || cp == 0x200f || cp == 0x061c;
}

static bool pdf_combining_mark(uint32_t cp) {
    return SBGeneralCategoryIsMark(SBCodepointGetGeneralCategory(cp));
}

static void pdf_bidi_word(std::string & content, const stored_word & word, int page_height,
                          const std::map<uint32_t, size_t> & characters, bool trailing_space) {
    auto text = unicode_scalars(word.text.c_str());
    if (text.empty()) return;
    if (trailing_space && text.back() != ' ') text.push_back(' ');
    const SBCodepointSequence sequence{ SBStringEncodingUTF32, text.data(), text.size() };
    std::unique_ptr<const struct _SBAlgorithm, decltype(&SBAlgorithmRelease)> algorithm(SBAlgorithmCreate(&sequence),
                                                                                        &SBAlgorithmRelease);
    if (!algorithm) throw std::runtime_error("Could not allocate Unicode bidirectional analysis");
    using paragraph_ptr = std::unique_ptr<const struct _SBParagraph, decltype(&SBParagraphRelease)>;
    std::vector<paragraph_ptr> paragraphs;
    struct row_range {
        size_t paragraph, start, end;
    };
    std::vector<row_range> hard_rows;
    for (size_t start = 0; start < text.size();) {
        SBUInteger length = 0, separator = 0;
        SBAlgorithmGetParagraphBoundary(algorithm.get(), start, text.size() - start, &length, &separator);
        if (!length) throw std::runtime_error("Invalid Unicode paragraph boundary");
        paragraph_ptr paragraph(SBAlgorithmCreateParagraph(algorithm.get(), start, length, SBLevelDefaultLTR),
                                &SBParagraphRelease);
        if (!paragraph) throw std::runtime_error("Could not resolve Unicode paragraph direction");
        const size_t index = paragraphs.size();
        paragraphs.push_back(std::move(paragraph));
        const size_t end = start + length - separator;
        size_t line_start = start;
        for (size_t i = start; i < end; i++) {
            if (text[i] == 0x2028) {
                hard_rows.push_back({ index, line_start, i });
                line_start = i + 1;
            }
        }
        hard_rows.push_back({ index, line_start, end });
        start += length;
    }
    const double original_height = double(word.h) / double(hard_rows.size());
    std::vector<row_range> rows;
    for (const auto & hard : hard_rows) {
        const size_t size = hard.end - hard.start;
        if (size * original_height <= word.w * 6.0) {
            rows.push_back(hard);
            continue;
        }
        const size_t count = size_t(std::ceil(std::sqrt(size * original_height / (2.0 * word.w))));
        const size_t columns = std::max(size_t(1), (size + count - 1) / count);
        for (size_t start = hard.start; start < hard.end;) {
            size_t end = std::min(start + columns, hard.end);
            if (end < hard.end) {
                for (size_t boundary = end; boundary > start + columns / 2; boundary--) {
                    if (text[boundary - 1] == ' ') {
                        end = boundary;
                        break;
                    }
                }
                // Never split the base from its combining marks at an estimated wrap.
                while (end < hard.end && pdf_combining_mark(text[end])) end++;
            }
            rows.push_back({ hard.paragraph, start, end });
            start = end;
        }
    }
    const double height = double(word.h) / double(rows.size());
    for (size_t row_index = 0; row_index < rows.size(); row_index++) {
        const auto & row = rows[row_index];
        if (row.start == row.end) continue;
        std::unique_ptr<const struct _SBLine, decltype(&SBLineRelease)> line(
            SBParagraphCreateLine(paragraphs[row.paragraph].get(), row.start, row.end - row.start), &SBLineRelease);
        if (!line) throw std::runtime_error("Could not resolve Unicode line direction");
        struct visual_run {
            size_t start, end, cell;
            std::vector<std::vector<size_t>> clusters;
        };
        std::vector<visual_run> runs;
        const SBRun * resolved = SBLineGetRunsPtr(line.get());
        size_t cells = 0;
        for (size_t i = 0; i < SBLineGetRunCount(line.get()); i++) {
            const auto & run = resolved[i];
            visual_run visual{ run.offset, run.offset + run.length, cells, {} };
            for (size_t j = visual.start; j < visual.end; j++) {
                if (pdf_format_control(text[j])) continue;
                if (pdf_combining_mark(text[j]) && !visual.clusters.empty())
                    visual.clusters.back().push_back(j);
                else
                    visual.clusters.push_back({ j });
            }
            if (run.level & 1) std::reverse(visual.clusters.begin(), visual.clusters.end());
            cells += visual.clusters.size();
            runs.push_back(std::move(visual));
        }
        if (!cells) {
            // Controls-only corrections still retain their logical Unicode.
            runs = { { row.start, row.end, 0, { { row.start } } } };
            cells = 1;
        }
        const double advance = double(word.w) / double(cells);
        const double y = double(page_height) - word.y - (row_index + 1) * height;
        // Store runs in logical reading order, with independently resolved
        // visual positions. Per-run ActualText supports accurate run selection;
        // the glyph stream is in visual order for older untagged PDF readers.
        std::sort(runs.begin(), runs.end(),
                  [](const visual_run & a, const visual_run & b) { return a.start < b.start; });
        for (const auto & run : runs) {
            const std::vector<uint32_t> logical(text.begin() + run.start, text.begin() + run.end);
            content += "/Span << /ActualText " + pdf_unicode_string(logical) + " >> BDC\nBT\n3 Tr\n";
            size_t current_font = std::numeric_limits<size_t>::max();
            for (size_t cell = 0; cell < run.clusters.size(); cell++) {
                for (const auto index : run.clusters[cell]) {
                    const size_t character = characters.at(text[index]) - 1;
                    const size_t font = character / pdf_cids_per_font;
                    // Readers can flush their directional text buffer on Tf,
                    // even when the selected font does not change. Preserve a
                    // buffered run instead of selecting its font per glyph.
                    if (font != current_font) {
                        content += "/F" + std::to_string(font + 1) + " 1 Tf\n";
                        current_font = font;
                    }
                    std::string encoded = "<";
                    hex16(encoded, pdf_cid(character % pdf_cids_per_font));
                    content += decimal(advance) + " 0 0 " + decimal(height) + " " +
                               decimal(word.x + (run.cell + cell) * advance) + " " + decimal(y) + " Tm\n" + encoded +
                               "> Tj\n";
                }
            }
            content += "ET\nEMC\n";
        }
    }
}

static std::string pdf_archival_catalog(pdf_document & doc) {
    const int profile =
        doc.hex_stream(std::string(reinterpret_cast<const char *>(ocr_pdf_srgb), sizeof(ocr_pdf_srgb)), "/N 3");
    const int intent = doc.add("<< /Type /OutputIntent /S /GTS_PDFA1 "
                               "/OutputConditionIdentifier (sRGB IEC61966-2.1) "
                               "/Info (sRGB IEC61966-2.1) /RegistryName (http://www.color.org) "
                               "/DestOutputProfile " +
                               pdf_ref(profile) + " >>");
    const std::string xmp = "<?xpacket begin=\"\xef\xbb\xbf\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n"
                            "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n"
                            "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n"
                            "<rdf:Description rdf:about=\"\" xmlns:pdfaid=\"http://www.aiim.org/pdfa/ns/id/\" "
                            "pdfaid:part=\"2\" pdfaid:conformance=\"B\"/>\n"
                            "<rdf:Description rdf:about=\"\" xmlns:pdf=\"http://ns.adobe.com/pdf/1.3/\" "
                            "pdf:Producer=\"" +
                            xml_escape(producer_name()) +
                            "\"/>\n"
                            "<rdf:Description rdf:about=\"\" xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\" "
                            "xmp:CreatorTool=\"CrispEmbed OCR\"/>\n"
                            "<rdf:Description rdf:about=\"\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
                            "dc:format=\"application/pdf\"/>\n"
                            "</rdf:RDF></x:xmpmeta>\n<?xpacket end=\"w\"?>";
    const int metadata = doc.stream(xmp, "/Type /Metadata /Subtype /XML");
    return " /OutputIntents [" + pdf_ref(intent) + "] /Metadata " + pdf_ref(metadata);
}

static std::string pdf_identifier() {
    // The two IDs match for the first revision; a new document gets a fresh
    // 128-bit identifier. No customer text or image hashes are exposed here.
    std::random_device random;
    std::string id = "<";
    for (int i = 0; i < 8; i++) hex16(id, random() & 0xffff);
    return id + ">";
}

static void pdf_end(ocr_renderer * r) {
    if (r->pages.empty()) throw std::runtime_error("A PDF document must contain at least one page");
    pdf_document doc;
    const int catalog = doc.reserve();
    const int pages = doc.reserve();
    std::map<uint32_t, size_t> characters;
    characters.emplace(' ', 0);
    for (const auto & page : r->pages)
        for (const auto & line : page.lines)
            for (const auto & word : line.words)
                for (uint32_t cp : unicode_scalars(word.text.c_str())) {
                    if (!r->pdf_bidi && needs_bidirectional_layout(cp))
                        throw std::runtime_error("Searchable PDF for right-to-left scripts or directional controls "
                                                 "requires a verified bidirectional layout engine; export hOCR, ALTO "
                                                 "or text to preserve the reviewed Unicode text");
                    characters.emplace(cp, 0);
                }
    size_t next_character = 1;
    for (auto & entry : characters) entry.second = next_character++;
    const auto fonts = pdf_fonts(doc, characters);
    std::string font_resources = " /Font <<";
    for (size_t f = 0; f < fonts.size(); f++) font_resources += " /F" + std::to_string(f + 1) + " " + pdf_ref(fonts[f]);
    font_resources += " >>";
    std::vector<int> page_ids;
    for (const auto & page : r->pages) {
        const int image = pdf_image(doc, page);
        std::string content = "q\n" + decimal(page.width_points / page.width) + " 0 0 " +
                              decimal(page.height_points / page.height) + " 0 0 cm\n";
        if (image)
            content +=
                "q\n" + std::to_string(page.width) + " 0 0 " + std::to_string(page.height) + " 0 0 cm\n/Im1 Do\nQ\n";
        for (const auto & line : page.lines)
            for (size_t i = 0; i < line.words.size(); i++) {
                const auto scalars = unicode_scalars(line.words[i].text.c_str());
                if (r->pdf_bidi && std::any_of(scalars.begin(), scalars.end(), needs_bidirectional_layout))
                    pdf_bidi_word(content, line.words[i], page.height, characters, i + 1 < line.words.size());
                else
                    pdf_word(content, line.words[i], page.height, characters, i + 1 < line.words.size());
            }
        content += "Q\n";
        const int stream = doc.stream(content);
        std::string resources = "<<" + font_resources;
        if (image) resources += " /XObject << /Im1 " + pdf_ref(image) + " >>";
        resources += " >>";
        page_ids.push_back(doc.add("<< /Type /Page /Parent " + pdf_ref(pages) + " /MediaBox [0 0 " +
                                   decimal(page.width_points) + " " + decimal(page.height_points) + "] /Contents " +
                                   pdf_ref(stream) + " /Resources " + resources + " >>"));
    }
    std::string kids = "<< /Type /Pages /Kids [";
    for (int page : page_ids) kids += pdf_ref(page) + " ";
    doc.set(pages, kids + "] /Count " + std::to_string(page_ids.size()) + " >>");
    doc.set(catalog, "<< /Type /Catalog /Pages " + pdf_ref(pages) + (r->pdfa ? pdf_archival_catalog(doc) : "") + " >>");
    const int info = doc.add(std::string("<< /Producer (") + producer_name() + ") /Creator (CrispEmbed OCR) >>");

    // Keep text-only PDFs safe for legacy NUL-terminated UTF-8 callers. The
    // small binary font/map streams above use ASCIIHex. Image-bearing PDFs
    // require the binary-safe output-size/checked APIs.
    const bool binary_images = std::any_of(r->pages.begin(), r->pages.end(),
                                           [](const stored_page & page) { return !page.image_path.empty(); });
    std::string out = r->pdfa         ? "%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
                      : binary_images ? "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
                                      : "%PDF-1.4\n%CrispEmbed\n";
    std::vector<size_t> offsets(doc.objects.size());
    for (size_t id = 1; id < doc.objects.size(); id++) {
        offsets[id] = out.size();
        if (doc.objects[id].size() + 64 > size_t(std::numeric_limits<int>::max()) - out.size())
            throw std::runtime_error("PDF exceeds the renderer byte-count limit");
        out += std::to_string(id) + " 0 obj\n" + doc.objects[id] + "\nendobj\n";
    }
    const size_t xref = out.size();
    std::ostringstream table;
    table.imbue(std::locale::classic());
    table << "xref\n0 " << doc.objects.size() << "\n0000000000 65535 f \n";
    for (size_t id = 1; id < offsets.size(); id++)
        table << std::setfill('0') << std::setw(10) << offsets[id] << " 00000 n \n";
    const std::string identifier = r->pdfa ? pdf_identifier() : "";
    table << "trailer\n<< /Size " << doc.objects.size() << " /Root " << pdf_ref(catalog) << " /Info " << pdf_ref(info)
          << (r->pdfa ? " /ID [" + identifier + " " + identifier + "]" : "") << " >>\nstartxref\n"
          << xref << "\n%%EOF\n";
    const std::string trailer = table.str();
    if (trailer.size() > size_t(std::numeric_limits<int>::max()) - out.size())
        throw std::runtime_error("PDF exceeds the renderer byte-count limit");
    r->output = std::move(out);
    r->output += trailer;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

template <typename Function> static void render_guard(ocr_renderer * r, Function function) {
    if (!r || !r->error.empty()) return;
    try {
        function();
        if (r->output.size() > size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("Document exceeds the renderer byte-count limit");
    } catch (const std::exception & error) {
        r->output.clear();
        r->error = error.what();
    } catch (...) {
        r->output.clear();
        r->error = "Unknown document rendering failure";
    }
}

ocr_renderer * ocr_render_create(ocr_render_format format) {
    if (format < OCR_RENDER_TEXT || format > OCR_RENDER_PDF) return nullptr;
    try {
        auto * r = new ocr_renderer();
        r->format = format;
        return r;
    } catch (...) {
        return nullptr;
    }
}

uint32_t crispembed_ocr_render_capabilities(void) {
    return OCR_RENDER_CAP_PDF | OCR_RENDER_CAP_UNICODE | OCR_RENDER_CAP_RASTER_IMAGES | OCR_RENDER_CAP_ARCHIVAL |
           OCR_RENDER_CAP_BIDI;
}

void ocr_render_set_separator(ocr_renderer * r, const char * sep) {
    render_guard(r, [&] {
        if (sep) r->separator = sep;
    });
}

void ocr_render_set_pdfa(ocr_renderer * r, int enabled) {
    if (r) r->pdfa = (enabled != 0);
}

void ocr_render_set_pdf_bidi(ocr_renderer * r, int enabled) {
    if (r) r->pdf_bidi = (enabled != 0);
}

void ocr_render_set_page_size_points(ocr_renderer * r, double width, double height) {
    render_guard(r, [&] {
        if (!r->begun || r->ended) throw std::runtime_error("Set page size after begin and before add_page");
        validate_physical_size(width, height);
        r->next_width_points = width;
        r->next_height_points = height;
    });
}

void ocr_render_begin(ocr_renderer * r) {
    if (!r) return;
    r->output.clear();
    r->error.clear();
    r->pages.clear();
    r->page_count = 0;
    r->next_width_points = r->next_height_points = 0;
    r->begun = true;
    r->ended = false;
    render_guard(r, [&] {
        switch (r->format) {
        case OCR_RENDER_TEXT:
            text_begin(r);
            break;
        case OCR_RENDER_HOCR:
            hocr_begin(r);
            break;
        case OCR_RENDER_ALTO:
            alto_begin(r);
            break;
        case OCR_RENDER_PDF:
            pdf_begin(r);
            break;
        }
    });
}

static void validate_page(const ocr_render_page * page, bool pdf) {
    if (!page) throw std::runtime_error("Page is null");
    if (page->n_lines < 0 || (page->n_lines > 0 && !page->lines))
        throw std::runtime_error("Page line array is invalid");
    if (pdf && (page->page_width <= 0 || page->page_height <= 0))
        throw std::runtime_error("PDF page dimensions must be positive");
    for (int i = 0; i < page->n_lines; i++) {
        const auto & line = page->lines[i];
        if (line.n_words < 0 || (line.n_words > 0 && !line.words))
            throw std::runtime_error("Page word array is invalid");
        if (int64_t(line.x) + line.w > std::numeric_limits<int>::max() ||
            int64_t(line.y) + line.h > std::numeric_limits<int>::max() || line.x < 0 || line.y < 0 || line.w < 0 ||
            line.h < 0)
            throw std::runtime_error("Line bounds are invalid");
        for (int j = 0; j < line.n_words; j++) {
            const auto & word = line.words[j];
            const int64_t right = int64_t(word.x) + word.w, bottom = int64_t(word.y) + word.h;
            if (word.x < 0 || word.y < 0 || word.w < 0 || word.h < 0 || right > std::numeric_limits<int>::max() ||
                bottom > std::numeric_limits<int>::max())
                throw std::runtime_error("Word bounds are invalid");
            if (pdf && word.text && *word.text &&
                (word.w <= 0 || word.h <= 0 || right > page->page_width || bottom > page->page_height))
                throw std::runtime_error("PDF text bounds must fit the page image coordinate space");
        }
    }
}

void ocr_render_add_page(ocr_renderer * r, const ocr_render_page * page) {
    render_guard(r, [&] {
        if (!r->begun || r->ended) throw std::runtime_error("Add pages between begin and end");
        validate_page(page, r->format == OCR_RENDER_PDF);
        switch (r->format) {
        case OCR_RENDER_TEXT:
            text_add_page(r, page);
            break;
        case OCR_RENDER_HOCR:
            hocr_add_page(r, page);
            break;
        case OCR_RENDER_ALTO:
            alto_add_page(r, page);
            break;
        case OCR_RENDER_PDF:
            pdf_add_page(r, page);
            break;
        }
        r->next_width_points = r->next_height_points = 0;
    });
}

void ocr_render_end(ocr_renderer * r) {
    render_guard(r, [&] {
        if (!r->begun || r->ended) throw std::runtime_error("End requires an unfinished document");
        switch (r->format) {
        case OCR_RENDER_TEXT:
            text_end(r);
            break;
        case OCR_RENDER_HOCR:
            hocr_end(r);
            break;
        case OCR_RENDER_ALTO:
            alto_end(r);
            break;
        case OCR_RENDER_PDF:
            pdf_end(r);
            break;
        }
        r->ended = true;
    });
}

const char * ocr_render_error(const ocr_renderer * r) {
    return !r ? "Renderer is null" : r->error.empty() ? nullptr : r->error.c_str();
}

const char * ocr_render_output(const ocr_renderer * r) {
    return r && r->error.empty() ? r->output.c_str() : "";
}

int ocr_render_output_size(const ocr_renderer * r) {
    return r && r->error.empty() && r->output.size() <= size_t(std::numeric_limits<int>::max()) ? int(r->output.size())
                                                                                                : 0;
}

void ocr_render_free(ocr_renderer * r) {
    delete r;
}
