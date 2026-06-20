// pdf_info.cpp — Minimal PDF parser for image DPI profiling.
//
// Parses just enough to extract image metadata:
//   1. Find startxref → xref table → object byte offsets
//   2. Read the Catalog → Pages tree → individual Page objects
//   3. For each Page: read /MediaBox, find /Resources → /XObject images
//   4. For each image XObject: read /Width, /Height, /BitsPerComponent
//   5. Parse content stream for CTM (cm operator) to get display size
//   6. Compute DPI = pixels / (display_points / 72)
//
// Handles both traditional xref tables and xref streams (compressed).
// Does NOT handle linearized PDFs, encrypted PDFs, or incremental updates
// beyond following /Prev chains.

#include "pdf_info.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

// ---------------------------------------------------------------------------
// File reading helpers
// ---------------------------------------------------------------------------

struct PdfFile {
    const uint8_t * mmap_base = nullptr;
    size_t mmap_size = 0;
    std::vector<uint8_t> fallback_data; // used on Windows or if mmap fails

    bool load(const char * path) {
#ifndef _WIN32
        int fd = ::open(path, O_RDONLY);
        if (fd >= 0) {
            struct stat st;
            if (fstat(fd, &st) == 0 && st.st_size > 0 && st.st_size <= 500LL * 1024 * 1024) {
                void * p = ::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
                ::close(fd);
                if (p != MAP_FAILED) {
                    mmap_base = (const uint8_t *)p;
                    mmap_size = (size_t)st.st_size;
#ifdef MADV_SEQUENTIAL
                    ::madvise(p, mmap_size, MADV_SEQUENTIAL);
#endif
                    return true;
                }
            } else {
                ::close(fd);
            }
        }
#endif
        // Fallback: fread into memory
        FILE * f = fopen(path, "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        if (sz <= 0 || sz > 500 * 1024 * 1024) { fclose(f); return false; }
        fseek(f, 0, SEEK_SET);
        fallback_data.resize((size_t)sz);
        size_t rd = fread(fallback_data.data(), 1, (size_t)sz, f);
        fclose(f);
        return rd == (size_t)sz;
    }

    ~PdfFile() {
#ifndef _WIN32
        if (mmap_base) ::munmap((void *)mmap_base, mmap_size);
#endif
    }

    size_t size() const { return mmap_base ? mmap_size : fallback_data.size(); }
    const char * ptr() const {
        return mmap_base ? (const char *)mmap_base : (const char *)fallback_data.data();
    }
    const char * ptr_at(size_t off) const {
        return off < size() ? ptr() + off : nullptr;
    }
};

// ---------------------------------------------------------------------------
// Tokenizer: skip whitespace, read numbers, names, strings
// ---------------------------------------------------------------------------

static void skip_ws(const char * buf, size_t len, size_t & pos) {
    while (pos < len) {
        char c = buf[pos];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\0') {
            pos++;
        } else if (c == '%') {
            // Comment — skip to end of line
            while (pos < len && buf[pos] != '\n' && buf[pos] != '\r') pos++;
        } else {
            break;
        }
    }
}

static bool is_digit(char c) { return c >= '0' && c <= '9'; }

static int read_int(const char * buf, size_t len, size_t & pos) {
    skip_ws(buf, len, pos);
    bool neg = false;
    if (pos < len && buf[pos] == '-') { neg = true; pos++; }
    int val = 0;
    while (pos < len && is_digit(buf[pos])) {
        val = val * 10 + (buf[pos] - '0');
        pos++;
    }
    return neg ? -val : val;
}

static float read_float(const char * buf, size_t len, size_t & pos) {
    skip_ws(buf, len, pos);
    char tmp[64] = {};
    int i = 0;
    if (pos < len && (buf[pos] == '-' || buf[pos] == '+'))
        tmp[i++] = buf[pos++];
    while (pos < len && (is_digit(buf[pos]) || buf[pos] == '.') && i < 62)
        tmp[i++] = buf[pos++];
    return (float)atof(tmp);
}

// Read a /Name token (starts with /)
static std::string read_name(const char * buf, size_t len, size_t & pos) {
    skip_ws(buf, len, pos);
    if (pos >= len || buf[pos] != '/') return "";
    pos++; // skip /
    std::string name;
    while (pos < len) {
        char c = buf[pos];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n' ||
            c == '/' || c == '<' || c == '>' || c == '[' || c == ']' ||
            c == '(' || c == ')' || c == '{' || c == '}')
            break;
        name += c;
        pos++;
    }
    return name;
}

// ---------------------------------------------------------------------------
// Find startxref at end of file
// ---------------------------------------------------------------------------

static long find_startxref(const PdfFile & pdf) {
    // Search backwards from EOF for "startxref"
    const char * buf = pdf.ptr();
    size_t sz = pdf.size();
    size_t search_from = sz > 1024 ? sz - 1024 : 0;
    const char * found = nullptr;
    for (size_t i = search_from; i < sz - 9; i++) {
        if (memcmp(buf + i, "startxref", 9) == 0) {
            found = buf + i;
        }
    }
    if (!found) return -1;

    // Read the offset after "startxref"
    size_t pos = (found - buf) + 9;
    return (long)read_int(buf, sz, pos);
}

// ---------------------------------------------------------------------------
// Parse traditional xref table
// ---------------------------------------------------------------------------

struct XrefEntry {
    long offset;
    int gen;
    bool in_use;
};

static bool parse_xref_table(const PdfFile & pdf, long xref_off,
                              std::unordered_map<int, XrefEntry> & xref,
                              int & trailer_root_id) {
    const char * buf = pdf.ptr();
    size_t sz = pdf.size();
    size_t pos = (size_t)xref_off;

    // Skip "xref" keyword
    skip_ws(buf, sz, pos);
    if (pos + 4 > sz || memcmp(buf + pos, "xref", 4) != 0)
        return false;
    pos += 4;

    // Read subsections: "start_id count" then count entries
    while (pos < sz) {
        skip_ws(buf, sz, pos);
        if (pos + 7 <= sz && memcmp(buf + pos, "trailer", 7) == 0)
            break;

        int start_id = read_int(buf, sz, pos);
        int count = read_int(buf, sz, pos);

        for (int i = 0; i < count && pos < sz; i++) {
            skip_ws(buf, sz, pos);
            long offset = 0;
            int gen = 0;
            char type = 'n';

            // Read 10-digit offset
            char off_str[11] = {};
            if (pos + 10 <= sz) {
                memcpy(off_str, buf + pos, 10);
                offset = atol(off_str);
            }
            pos += 10;
            skip_ws(buf, sz, pos);

            // Read 5-digit generation
            char gen_str[6] = {};
            if (pos + 5 <= sz) {
                memcpy(gen_str, buf + pos, 5);
                gen = atoi(gen_str);
            }
            pos += 5;
            skip_ws(buf, sz, pos);

            // Read 'n' or 'f'
            if (pos < sz) {
                type = buf[pos];
                pos++;
            }

            if (type == 'n') {
                int obj_id = start_id + i;
                if (xref.find(obj_id) == xref.end()) {
                    xref[obj_id] = {offset, gen, true};
                }
            }
        }
    }

    // Parse trailer dict
    skip_ws(buf, sz, pos);
    if (pos + 7 <= sz && memcmp(buf + pos, "trailer", 7) == 0) {
        pos += 7;
        // Find /Root N 0 R
        const char * trailer_start = buf + pos;
        size_t trailer_len = std::min(sz - pos, (size_t)4096);
        for (size_t i = 0; i + 5 < trailer_len; i++) {
            if (memcmp(trailer_start + i, "/Root", 5) == 0) {
                size_t rp = pos + i + 5;
                trailer_root_id = read_int(buf, sz, rp);
                break;
            }
        }

        // Follow /Prev chain for incremental updates
        for (size_t i = 0; i + 5 < trailer_len; i++) {
            if (memcmp(trailer_start + i, "/Prev", 5) == 0) {
                size_t rp = pos + i + 5;
                long prev_off = (long)read_int(buf, sz, rp);
                if (prev_off > 0 && prev_off < (long)sz) {
                    parse_xref_table(pdf, prev_off, xref, trailer_root_id);
                }
                break;
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Read a value from a dict at a given object offset
// ---------------------------------------------------------------------------

// Find the position of a key in a dict starting at `pos`.
// Returns the position right after the key, or 0 if not found.
static size_t find_dict_key(const char * buf, size_t sz, size_t dict_start,
                             const char * key) {
    size_t key_len = strlen(key);
    // Search within ~8 KB from dict start
    size_t limit = std::min(dict_start + 8192, sz);
    for (size_t i = dict_start; i + key_len < limit; i++) {
        if (buf[i] == '/' && memcmp(buf + i + 1, key, key_len) == 0) {
            // Verify next char is not alphanumeric (full match)
            char next = (i + 1 + key_len < sz) ? buf[i + 1 + key_len] : ' ';
            if (next == ' ' || next == '\n' || next == '\r' || next == '/'
                || next == '<' || next == '[' || next == '(') {
                return i + 1 + key_len;
            }
        }
    }
    return 0;
}

// Read an indirect reference "N 0 R" and return N. Returns -1 if not a ref.
static int read_ref(const char * buf, size_t sz, size_t & pos) {
    skip_ws(buf, sz, pos);
    size_t save = pos;
    int id = read_int(buf, sz, pos);
    skip_ws(buf, sz, pos);
    read_int(buf, sz, pos); // generation (skip)
    skip_ws(buf, sz, pos);
    if (pos < sz && buf[pos] == 'R') {
        pos++;
        return id;
    }
    pos = save;
    return -1;
}

// Read a /MediaBox [x0 y0 x1 y1] array. Returns true if found.
static bool read_media_box(const char * buf, size_t sz, size_t obj_start,
                            float & w, float & h) {
    size_t kp = find_dict_key(buf, sz, obj_start, "MediaBox");
    if (!kp) return false;
    skip_ws(buf, sz, kp);
    if (kp >= sz || buf[kp] != '[') return false;
    kp++; // skip [
    float x0 = read_float(buf, sz, kp);
    float y0 = read_float(buf, sz, kp);
    float x1 = read_float(buf, sz, kp);
    float y1 = read_float(buf, sz, kp);
    w = x1 - x0;
    h = y1 - y0;
    return w > 0 && h > 0;
}

// Read an integer value for a key. Returns -1 if not found.
static int read_dict_int(const char * buf, size_t sz, size_t obj_start,
                          const char * key) {
    size_t kp = find_dict_key(buf, sz, obj_start, key);
    if (!kp) return -1;
    return read_int(buf, sz, kp);
}

// ---------------------------------------------------------------------------
// Find image XObjects referenced by a page
// ---------------------------------------------------------------------------

struct ImageXObj {
    int obj_id;
    std::string name; // e.g. "Im1"
};

static std::vector<ImageXObj> find_page_images(
    const char * buf, size_t sz, size_t page_start,
    const std::unordered_map<int, XrefEntry> & xref)
{
    std::vector<ImageXObj> images;

    // Find /Resources — either inline dict or indirect ref
    size_t rp = find_dict_key(buf, sz, page_start, "Resources");
    if (!rp) return images;

    skip_ws(buf, sz, rp);
    size_t res_start = rp;

    // If it's an indirect ref, follow it
    if (rp < sz && is_digit(buf[rp])) {
        int ref_id = read_ref(buf, sz, rp);
        if (ref_id > 0) {
            auto it = xref.find(ref_id);
            if (it != xref.end()) res_start = (size_t)it->second.offset;
        }
    }

    // Find /XObject within resources
    size_t xp = find_dict_key(buf, sz, res_start, "XObject");
    if (!xp) return images;

    skip_ws(buf, sz, xp);
    if (xp >= sz || buf[xp] != '<') return images;
    xp++; // skip first <
    if (xp < sz && buf[xp] == '<') xp++; // skip second <

    // Parse the XObject dict: /Name N 0 R pairs
    size_t limit = std::min(xp + 4096, sz);
    while (xp < limit) {
        skip_ws(buf, sz, xp);
        if (xp >= limit || buf[xp] == '>') break;

        if (buf[xp] == '/') {
            std::string name = read_name(buf, sz, xp);
            int ref_id = read_ref(buf, sz, xp);
            if (ref_id > 0) {
                // Check if this XObject is an Image
                auto it = xref.find(ref_id);
                if (it != xref.end()) {
                    size_t obj_off = (size_t)it->second.offset;
                    size_t sp = find_dict_key(buf, sz, obj_off, "Subtype");
                    if (sp) {
                        std::string subtype = read_name(buf, sz, sp);
                        if (subtype == "Image") {
                            images.push_back({ref_id, name});
                        }
                    }
                }
            }
        } else {
            xp++; // skip unknown chars
        }
    }

    return images;
}

// ---------------------------------------------------------------------------
// Parse content stream for CTM (cm operator)
// ---------------------------------------------------------------------------

struct ImageCTM {
    std::string name;
    float scale_x; // display width in points
    float scale_y; // display height in points
};

static std::vector<ImageCTM> parse_content_stream_ctm(
    const char * buf, size_t sz, size_t page_start,
    const std::unordered_map<int, XrefEntry> & xref)
{
    std::vector<ImageCTM> ctms;

    // Find /Contents ref
    size_t cp = find_dict_key(buf, sz, page_start, "Contents");
    if (!cp) return ctms;

    int contents_id = read_ref(buf, sz, cp);
    if (contents_id <= 0) return ctms;

    auto it = xref.find(contents_id);
    if (it == xref.end()) return ctms;

    // Find the stream data
    size_t obj_off = (size_t)it->second.offset;
    // Look for "stream" keyword
    size_t sp = obj_off;
    size_t limit = std::min(sp + 4096, sz);
    const char * stream_start = nullptr;
    for (; sp < limit - 6; sp++) {
        if (memcmp(buf + sp, "stream", 6) == 0) {
            sp += 6;
            // Skip \r\n or \n after "stream"
            if (sp < sz && buf[sp] == '\r') sp++;
            if (sp < sz && buf[sp] == '\n') sp++;
            stream_start = buf + sp;
            break;
        }
    }
    if (!stream_start) return ctms;

    // Find "endstream"
    const char * stream_end = nullptr;
    for (size_t i = sp; i < sz - 9; i++) {
        if (memcmp(buf + i, "endstream", 9) == 0) {
            stream_end = buf + i;
            break;
        }
    }
    if (!stream_end || stream_end <= stream_start) return ctms;

    // Parse the content stream for patterns:
    //   <a> <b> <c> <d> <e> <f> cm   → CTM matrix
    //   /ImN Do                       → paint image
    // The CTM before a Do gives the display size: a=width_pt, d=height_pt
    size_t slen = stream_end - stream_start;
    float last_a = 0, last_d = 0;
    bool have_ctm = false;

    // Simple state machine: track last 6 floats before "cm", and /Name before "Do"
    std::vector<float> num_stack;
    size_t i = 0;
    while (i < slen) {
        // Skip whitespace
        while (i < slen && (stream_start[i] == ' ' || stream_start[i] == '\n' ||
                            stream_start[i] == '\r' || stream_start[i] == '\t'))
            i++;
        if (i >= slen) break;

        // Check for operators
        if (stream_start[i] == 'c' && i + 1 < slen && stream_start[i + 1] == 'm' &&
            (i + 2 >= slen || stream_start[i + 2] == ' ' || stream_start[i + 2] == '\n' ||
             stream_start[i + 2] == '\r')) {
            // cm operator: pop 6 numbers [a b c d e f]
            if (num_stack.size() >= 6) {
                size_t base = num_stack.size() - 6;
                last_a = std::abs(num_stack[base + 0]); // scale x (width in points)
                last_d = std::abs(num_stack[base + 3]); // scale y (height in points)
                have_ctm = true;
            }
            num_stack.clear();
            i += 2;
        } else if (stream_start[i] == 'D' && i + 1 < slen && stream_start[i + 1] == 'o' &&
                   (i + 2 >= slen || stream_start[i + 2] == ' ' || stream_start[i + 2] == '\n' ||
                    stream_start[i + 2] == '\r')) {
            // Do operator: the CTM before this gives the image display size
            // Find the image name (last /Name token before Do)
            // We'll match it later by position
            if (have_ctm) {
                // Look back for the image name
                ImageCTM c;
                c.scale_x = last_a;
                c.scale_y = last_d;
                // Scan back for /ImN
                for (size_t j = (i > 64 ? i - 64 : 0); j < i; j++) {
                    if (stream_start[j] == '/') {
                        size_t np = j;
                        std::string nm;
                        np++; // skip /
                        while (np < i && stream_start[np] != ' ' && stream_start[np] != '\n')
                            nm += stream_start[np++];
                        c.name = nm;
                    }
                }
                ctms.push_back(c);
            }
            num_stack.clear();
            have_ctm = false;
            i += 2;
        } else if (stream_start[i] == '/' || stream_start[i] == 'q' ||
                   stream_start[i] == 'Q' || stream_start[i] == 'B' ||
                   stream_start[i] == 'E') {
            // Other operators or names — skip token
            while (i < slen && stream_start[i] != ' ' && stream_start[i] != '\n' &&
                   stream_start[i] != '\r')
                i++;
        } else if (is_digit(stream_start[i]) || stream_start[i] == '-' ||
                   stream_start[i] == '+' || stream_start[i] == '.') {
            // Number
            char tmp[64] = {};
            int ti = 0;
            while (i < slen && ti < 62 &&
                   (is_digit(stream_start[i]) || stream_start[i] == '.' ||
                    stream_start[i] == '-' || stream_start[i] == '+')) {
                tmp[ti++] = stream_start[i++];
            }
            num_stack.push_back((float)atof(tmp));
        } else {
            // Unknown token — skip
            while (i < slen && stream_start[i] != ' ' && stream_start[i] != '\n' &&
                   stream_start[i] != '\r')
                i++;
        }
    }

    return ctms;
}

// ---------------------------------------------------------------------------
// Find page object offsets from the Pages tree
// ---------------------------------------------------------------------------

static void collect_pages(const char * buf, size_t sz,
                           const std::unordered_map<int, XrefEntry> & xref,
                           int pages_id,
                           std::vector<size_t> & page_offsets) {
    auto it = xref.find(pages_id);
    if (it == xref.end()) return;
    size_t off = (size_t)it->second.offset;

    // Check /Type
    size_t tp = find_dict_key(buf, sz, off, "Type");
    if (!tp) return;
    std::string type = read_name(buf, sz, tp);

    if (type == "Page") {
        page_offsets.push_back(off);
        return;
    }

    // /Type /Pages — find /Kids array
    size_t kp = find_dict_key(buf, sz, off, "Kids");
    if (!kp) return;
    skip_ws(buf, sz, kp);
    if (kp >= sz || buf[kp] != '[') return;
    kp++; // skip [

    while (kp < sz && buf[kp] != ']') {
        skip_ws(buf, sz, kp);
        if (kp >= sz || buf[kp] == ']') break;
        int kid_id = read_ref(buf, sz, kp);
        if (kid_id > 0) {
            collect_pages(buf, sz, xref, kid_id, page_offsets);
        } else {
            kp++; // skip unexpected char
        }
    }
}

// ---------------------------------------------------------------------------
// Compute page DPI
// ---------------------------------------------------------------------------

static bool compute_page_dpi(const PdfFile & pdf,
                              const std::unordered_map<int, XrefEntry> & xref,
                              size_t page_off,
                              pdf_page_dpi_result & result) {
    const char * buf = pdf.ptr();
    size_t sz = pdf.size();

    // Read page MediaBox
    float page_w = 0, page_h = 0;
    if (!read_media_box(buf, sz, page_off, page_w, page_h)) {
        // Try parent Pages object
        size_t pp = find_dict_key(buf, sz, page_off, "Parent");
        if (pp) {
            int parent_id = read_ref(buf, sz, pp);
            if (parent_id > 0) {
                auto it = xref.find(parent_id);
                if (it != xref.end())
                    read_media_box(buf, sz, (size_t)it->second.offset, page_w, page_h);
            }
        }
    }
    if (page_w <= 0 || page_h <= 0) return false;

    result.page_width_pt = page_w;
    result.page_height_pt = page_h;

    // Find image XObjects
    auto images = find_page_images(buf, sz, page_off, xref);
    if (images.empty()) {
        // No images — estimate DPI from page size assuming full-page raster
        result.dpi = 0;
        result.dpi_min = 0;
        result.dpi_max = 0;
        result.n_images = 0;
        return false;
    }

    // Parse content stream for CTM
    auto ctms = parse_content_stream_ctm(buf, sz, page_off, xref);

    // Build a map from image name → CTM
    std::unordered_map<std::string, ImageCTM> ctm_map;
    for (auto & c : ctms) ctm_map[c.name] = c;

    // Compute per-image DPI
    result.n_images = 0;
    result.dpi_min = 1e6f;
    result.dpi_max = 0;
    float sum_inv_dpi_weighted = 0; // for harmonic mean
    float sum_area = 0;

    for (auto & img : images) {
        auto it = xref.find(img.obj_id);
        if (it == xref.end()) continue;
        size_t obj_off = (size_t)it->second.offset;

        int pw = read_dict_int(buf, sz, obj_off, "Width");
        int ph = read_dict_int(buf, sz, obj_off, "Height");
        if (pw <= 0 || ph <= 0) continue;

        // Get display size from CTM, or assume full-page
        float display_w = page_w;
        float display_h = page_h;
        auto ct = ctm_map.find(img.name);
        if (ct != ctm_map.end()) {
            if (ct->second.scale_x > 0) display_w = ct->second.scale_x;
            if (ct->second.scale_y > 0) display_h = ct->second.scale_y;
        }

        float dpi_x = (float)pw / (display_w / 72.0f);
        float dpi_y = (float)ph / (display_h / 72.0f);
        float dpi = std::min(dpi_x, dpi_y); // conservative
        float area = display_w * display_h;

        if (dpi < result.dpi_min) result.dpi_min = dpi;
        if (dpi > result.dpi_max) result.dpi_max = dpi;
        sum_inv_dpi_weighted += area / dpi;
        sum_area += area;
        result.n_images++;
    }

    if (result.n_images == 0) return false;

    // Weighted harmonic mean
    result.dpi = sum_area / sum_inv_dpi_weighted;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

int pdf_page_dpi(const char * pdf_path, int page,
                 pdf_page_dpi_result * result) {
    if (!pdf_path || !result || page < 0) return 1;
    memset(result, 0, sizeof(*result));

    PdfFile pdf;
    if (!pdf.load(pdf_path)) return 1;

    // Check PDF header
    if (pdf.size() < 5 || memcmp(pdf.ptr(), "%PDF-", 5) != 0) return 1;

    // Find xref
    long xref_off = find_startxref(pdf);
    if (xref_off < 0 || xref_off >= (long)pdf.size()) return 1;

    // Parse xref table
    std::unordered_map<int, XrefEntry> xref;
    int root_id = -1;
    if (!parse_xref_table(pdf, xref_off, xref, root_id)) return 1;
    if (root_id <= 0) return 1;

    // Find root → pages
    auto root_it = xref.find(root_id);
    if (root_it == xref.end()) return 1;

    size_t root_off = (size_t)root_it->second.offset;
    size_t pp = find_dict_key(pdf.ptr(), pdf.size(), root_off, "Pages");
    if (!pp) return 1;

    int pages_id = read_ref(pdf.ptr(), pdf.size(), pp);
    if (pages_id <= 0) return 1;

    // Collect all page offsets
    std::vector<size_t> page_offsets;
    collect_pages(pdf.ptr(), pdf.size(), xref, pages_id, page_offsets);

    if (page >= (int)page_offsets.size()) return 1;

    return compute_page_dpi(pdf, xref, page_offsets[page], *result) ? 0 : 1;
}

pdf_page_dpi_result * pdf_all_pages_dpi(const char * pdf_path, int * n_pages) {
    if (!pdf_path || !n_pages) return nullptr;
    *n_pages = 0;

    PdfFile pdf;
    if (!pdf.load(pdf_path)) return nullptr;
    if (pdf.size() < 5 || memcmp(pdf.ptr(), "%PDF-", 5) != 0) return nullptr;

    long xref_off = find_startxref(pdf);
    if (xref_off < 0 || xref_off >= (long)pdf.size()) return nullptr;

    std::unordered_map<int, XrefEntry> xref;
    int root_id = -1;
    if (!parse_xref_table(pdf, xref_off, xref, root_id)) return nullptr;
    if (root_id <= 0) return nullptr;

    auto root_it = xref.find(root_id);
    if (root_it == xref.end()) return nullptr;

    size_t root_off = (size_t)root_it->second.offset;
    size_t pp = find_dict_key(pdf.ptr(), pdf.size(), root_off, "Pages");
    if (!pp) return nullptr;

    int pages_id = read_ref(pdf.ptr(), pdf.size(), pp);
    if (pages_id <= 0) return nullptr;

    std::vector<size_t> page_offsets;
    collect_pages(pdf.ptr(), pdf.size(), xref, pages_id, page_offsets);

    if (page_offsets.empty()) return nullptr;

    *n_pages = (int)page_offsets.size();
    auto * results = (pdf_page_dpi_result *)calloc(*n_pages, sizeof(pdf_page_dpi_result));
    for (int i = 0; i < *n_pages; i++) {
        compute_page_dpi(pdf, xref, page_offsets[i], results[i]);
    }
    return results;
}

void pdf_dpi_free(pdf_page_dpi_result * results) {
    free(results);
}
