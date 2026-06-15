// ocr_render.cpp — OCR result renderers.
//
// Implements plain text, hOCR, ALTO, and searchable PDF output.
// No external dependencies (PDF is a minimal subset: one image + text layer
// per page, no fonts beyond a base Type1 invisible text font).

#include "ocr_render.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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
    std::string image_path; // for PDF image embedding
};

struct ocr_renderer {
    ocr_render_format format;
    std::string separator;     // page separator for text mode
    std::string output;        // accumulated output
    int page_count;
    std::vector<stored_page> pages;  // for PDF deferred rendering
    bool pdfa = false;         // PDF/A-2b compliance

    ocr_renderer() : format(OCR_RENDER_TEXT), separator("\f"), page_count(0) {}
};

// ---------------------------------------------------------------------------
// XML helpers
// ---------------------------------------------------------------------------

static std::string xml_escape(const char * s) {
    std::string out;
    if (!s) return out;
    for (; *s; s++) {
        switch (*s) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default:  out += *s; break;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Plain text renderer
// ---------------------------------------------------------------------------

static void text_begin(ocr_renderer *) {}

static void text_add_page(ocr_renderer * r, const ocr_render_page * page) {
    if (r->page_count > 0)
        r->output += r->separator;

    for (int i = 0; i < page->n_lines; i++) {
        const ocr_render_line * line = &page->lines[i];
        for (int j = 0; j < line->n_words; j++) {
            if (j > 0) r->output += ' ';
            if (line->words[j].text)
                r->output += line->words[j].text;
        }
        r->output += '\n';
    }
    r->page_count++;
}

static void text_end(ocr_renderer *) {}

// ---------------------------------------------------------------------------
// hOCR renderer
// ---------------------------------------------------------------------------

static void hocr_begin(ocr_renderer * r) {
    r->output += "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                 "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\"\n"
                 "  \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n"
                 "<html xmlns=\"http://www.w3.org/1999/xhtml\">\n"
                 "<head>\n"
                 "  <meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" />\n"
                 "  <meta name=\"ocr-system\" content=\"CrispEmbed\" />\n"
                 "  <meta name=\"ocr-capabilities\" content=\"ocr_page ocr_carea ocr_line ocrx_word\" />\n"
                 "</head>\n"
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
            pid, lid, line->x, line->y,
            line->x + line->w, line->y + line->h);
        r->output += buf;

        for (int j = 0; j < line->n_words; j++) {
            const ocr_render_word * w = &line->words[j];
            int wid = j + 1;
            snprintf(buf, sizeof(buf),
                "      <span class=\"ocrx_word\" id=\"word_%d_%d_%d\" "
                "title=\"bbox %d %d %d %d; x_wconf %d\">",
                pid, lid, wid, w->x, w->y, w->x + w->w, w->y + w->h,
                (int)(w->confidence * 100));
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
            pid, i, line->x, line->y, line->w, line->h,
            line->x, line->y, line->w, line->h);
        r->output += buf;

        for (int j = 0; j < line->n_words; j++) {
            const ocr_render_word * w = &line->words[j];
            snprintf(buf, sizeof(buf),
                "            <String CONTENT=\"%s\" HPOS=\"%d\" VPOS=\"%d\" "
                "WIDTH=\"%d\" HEIGHT=\"%d\" WC=\"%.2f\" />\n",
                xml_escape(w->text).c_str(),
                w->x, w->y, w->w, w->h, w->confidence);
            r->output += buf;

            // Add space after word (except last)
            if (j < line->n_words - 1) {
                int sp_x = w->x + w->w;
                int next_x = line->words[j + 1].x;
                int sp_w = next_x > sp_x ? next_x - sp_x : 4;
                snprintf(buf, sizeof(buf),
                    "            <SP WIDTH=\"%d\" HPOS=\"%d\" VPOS=\"%d\" />\n",
                    sp_w, sp_x, w->y);
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
// Searchable PDF renderer (minimal PDF 1.4 subset)
// ---------------------------------------------------------------------------
// Approach: for each page, embed the original image as a full-page XObject,
// then overlay invisible text using a base font with zero rendering mode.
// This creates a PDF where the image is visible but the text is selectable.
//
// We use PDF's text rendering mode 3 (invisible) for the text layer.
// Font: built-in Helvetica (no embedding needed).
//
// This is a minimal PDF writer — no external library required.

struct pdf_object {
    int id;
    int offset;
};

static void pdf_begin(ocr_renderer * r) {
    r->pages.clear();
}

static void pdf_add_page(ocr_renderer * r, const ocr_render_page * page) {
    // Store page data for deferred rendering in pdf_end
    stored_page sp;
    sp.width = page->page_width;
    sp.height = page->page_height;
    if (page->image_path) sp.image_path = page->image_path;
    for (int i = 0; i < page->n_lines; i++) {
        const ocr_render_line * line = &page->lines[i];
        stored_line sl;
        sl.x = line->x; sl.y = line->y; sl.w = line->w; sl.h = line->h;
        for (int j = 0; j < line->n_words; j++) {
            const ocr_render_word * w = &line->words[j];
            sl.words.push_back({w->text ? w->text : "", w->x, w->y, w->w, w->h, w->confidence});
        }
        sp.lines.push_back(sl);
    }
    r->pages.push_back(std::move(sp));
    r->page_count++;
}

// PDF string escaping: escape parens and backslashes
static std::string pdf_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '(' || c == ')' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

static void pdf_end(ocr_renderer * r) {
    // Build a complete PDF with invisible text layer per page.
    // Text rendering mode 3 = invisible (for searchable PDF).
    // Font: Helvetica (built-in, no embedding needed).
    // Coordinate system: PDF y-axis is bottom-up; OCR y-axis is top-down.

    std::string & out = r->output;
    out = "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n";

    std::vector<pdf_object> objects;
    int next_id = 1;
    char buf[1024];

    auto obj_start = [&](int id) {
        objects.push_back({id, (int)out.size()});
        snprintf(buf, sizeof(buf), "%d 0 obj\n", id);
        out += buf;
    };

    // Object 1: Catalog (placeholder — patched at end for PDF/A refs)
    int cat_id = next_id++;
    obj_start(cat_id);
    int cat_body_offset = (int)out.size();
    out += std::string(512, ' ');
    out += "\nendobj\n";

    // Object 2: Pages (placeholder — we'll patch Kids/Count below)
    int pages_id = next_id++;
    obj_start(pages_id);
    int pages_body_offset = (int)out.size();
    // Reserve space — will be overwritten
    out += std::string(256, ' ');
    out += "\nendobj\n";

    // Object 3: Font
    int font_id = next_id++;
    obj_start(font_id);
    out += "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n";

    // Per-page objects: each page needs a Page dict + content stream
    // and optionally an image XObject.
    //
    // Text rendering approach (from Tesseract pdfrenderer.cpp, Apache-2.0):
    // - Use text matrix `Tm` with affine coefficients for rotation-aware
    //   positioning (not simple `Td`).
    // - Scale font size so rendered text width matches bounding box width.
    //   Helvetica average glyph width ≈ 0.6 * font_size.
    // - Rendering mode 3 = invisible text (selectable but not drawn).
    std::vector<int> page_ids;
    for (auto & sp : r->pages) {
        // Build content stream
        std::string content;

        // If image path is provided, embed as JPEG XObject
        // (image drawn first, text overlaid on top)
        int img_obj_id = 0;
        std::vector<uint8_t> jpeg_data;
        if (!sp.image_path.empty()) {
            // Read the image file as raw bytes (assume JPEG or re-encode)
            FILE * imgf = fopen(sp.image_path.c_str(), "rb");
            if (imgf) {
                fseek(imgf, 0, SEEK_END);
                long sz = ftell(imgf);
                fseek(imgf, 0, SEEK_SET);
                jpeg_data.resize(sz);
                fread(jpeg_data.data(), 1, sz, imgf);
                fclose(imgf);
            }
        }

        if (!jpeg_data.empty()) {
            // Draw image as full-page background
            // cm = Coordinate Transform Matrix: scale to page size
            snprintf(buf, sizeof(buf), "q\n%d 0 0 %d 0 0 cm\n/Im1 Do\nQ\n",
                     sp.width, sp.height);
            content += buf;
        }

        // Invisible text layer
        content += "BT\n";
        content += "3 Tr\n"; // Rendering mode 3 = invisible

        for (auto & sl : sp.lines) {
            for (auto & sw : sl.words) {
                if (sw.text.empty()) continue;

                // PDF coordinates: y is bottom-up
                double pdf_x = (double)sw.x;
                double pdf_y = (double)(sp.height - sw.y - sw.h);

                // Compute font size to match word width.
                // Helvetica average glyph width ≈ 0.55 * font_size.
                // We want: text_length ≈ word_width_in_points
                int n_chars = (int)sw.text.size();
                double desired_width = (double)sw.w;
                double font_size = (n_chars > 0)
                    ? desired_width / (n_chars * 0.55)
                    : (double)sw.h;
                font_size = std::max(4.0, std::min(200.0, font_size));

                // Use Tm (text matrix) for positioning.
                // For horizontal text: a=1, b=0, c=0, d=1, e=x, f=y
                // This allows easy extension to rotated text later.
                snprintf(buf, sizeof(buf),
                    "/F1 %.1f Tf\n"
                    "1 0 0 1 %.1f %.1f Tm\n"
                    "(%s) Tj\n",
                    font_size, pdf_x, pdf_y,
                    pdf_escape(sw.text).c_str());
                content += buf;
            }
        }
        content += "ET\n";

        // Image XObject (if image data available)
        if (!jpeg_data.empty()) {
            img_obj_id = next_id++;
            obj_start(img_obj_id);

            // Detect if JPEG (starts with FF D8) or PNG
            bool is_jpeg = jpeg_data.size() >= 2 &&
                           jpeg_data[0] == 0xFF && jpeg_data[1] == 0xD8;
            const char * filter = is_jpeg ? "/DCTDecode" : "/FlateDecode";
            // For simplicity, only JPEG is truly embedded; PNG would need
            // decompression + re-encoding. Just embed raw bytes with DCTDecode.

            snprintf(buf, sizeof(buf),
                "<< /Type /XObject /Subtype /Image\n"
                "   /Width %d /Height %d\n"
                "   /ColorSpace /DeviceRGB\n"
                "   /BitsPerComponent 8\n"
                "   /Filter %s\n"
                "   /Length %d >>\nstream\n",
                sp.width, sp.height, filter, (int)jpeg_data.size());
            out += buf;
            out.append((const char *)jpeg_data.data(), jpeg_data.size());
            out += "\nendstream\nendobj\n";
        }

        // Content stream object
        int stream_id = next_id++;
        obj_start(stream_id);
        snprintf(buf, sizeof(buf), "<< /Length %d >>\nstream\n", (int)content.size());
        out += buf;
        out += content;
        out += "endstream\nendobj\n";

        // Page object
        int page_id = next_id++;
        page_ids.push_back(page_id);
        obj_start(page_id);

        // Resources: font + optional image
        std::string resources = "<< /Font << /F1 " + std::to_string(font_id) + " 0 R >>";
        if (img_obj_id > 0)
            resources += " /XObject << /Im1 " + std::to_string(img_obj_id) + " 0 R >>";
        resources += " >>";

        snprintf(buf, sizeof(buf),
            "<< /Type /Page /Parent %d 0 R "
            "/MediaBox [0 0 %d %d] "
            "/Contents %d 0 R "
            "/Resources %s "
            ">>\nendobj\n",
            pages_id, sp.width, sp.height, stream_id, resources.c_str());
        out += buf;
    }

    // Patch the Pages object with actual Kids and Count
    {
        std::string pages_body = "<< /Type /Pages /Kids [";
        for (size_t i = 0; i < page_ids.size(); i++) {
            if (i > 0) pages_body += " ";
            snprintf(buf, sizeof(buf), "%d 0 R", page_ids[i]);
            pages_body += buf;
        }
        snprintf(buf, sizeof(buf), "] /Count %d >>", (int)page_ids.size());
        pages_body += buf;
        // Overwrite the placeholder
        if ((int)pages_body.size() <= 256) {
            pages_body.resize(256, ' ');
            memcpy(&out[pages_body_offset], pages_body.data(), 256);
        }
    }

    // PDF/A-2b: add XMP metadata stream + OutputIntent with sRGB profile
    int metadata_id = 0, output_intent_id = 0;
    if (r->pdfa) {
        // XMP metadata stream (declares PDF/A-2b conformance)
        std::string xmp =
            "<?xpacket begin=\"\xef\xbb\xbf\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n"
            "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n"
            "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n"
            "  <rdf:Description rdf:about=\"\"\n"
            "    xmlns:dc=\"http://purl.org/dc/elements/1.1/\"\n"
            "    xmlns:pdfaid=\"http://www.aiim.org/pdfa/ns/id/\"\n"
            "    xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\">\n"
            "    <pdfaid:part>2</pdfaid:part>\n"
            "    <pdfaid:conformance>B</pdfaid:conformance>\n"
            "    <dc:title><rdf:Alt><rdf:li xml:lang=\"x-default\">OCR Document</rdf:li></rdf:Alt></dc:title>\n"
            "    <dc:creator><rdf:Seq><rdf:li>CrispEmbed</rdf:li></rdf:Seq></dc:creator>\n"
            "    <xmp:CreatorTool>CrispEmbed OCR</xmp:CreatorTool>\n"
            "    <xmp:ProducerTool>CrispEmbed</xmp:ProducerTool>\n"
            "  </rdf:Description>\n"
            "</rdf:RDF>\n"
            "</x:xmpmeta>\n"
            "<?xpacket end=\"w\"?>";

        metadata_id = next_id++;
        obj_start(metadata_id);
        snprintf(buf, sizeof(buf),
            "<< /Type /Metadata /Subtype /XML /Length %d >>\nstream\n",
            (int)xmp.size());
        out += buf;
        out += xmp;
        out += "\nendstream\nendobj\n";

        // OutputIntent with sRGB IEC61966-2.1 (minimal — just the intent dict,
        // no embedded ICC profile binary for size reasons; validators may flag
        // this but it's the common minimal-compliance approach)
        output_intent_id = next_id++;
        obj_start(output_intent_id);
        out += "<< /Type /OutputIntent\n"
               "   /S /GTS_PDFA1\n"
               "   /OutputConditionIdentifier (sRGB IEC61966-2.1)\n"
               "   /RegistryName (http://www.color.org)\n"
               "   /Info (sRGB IEC61966-2.1)\n"
               ">>\nendobj\n";
    }

    // Patch the Catalog object
    {
        std::string cat_body = "<< /Type /Catalog /Pages "
            + std::to_string(pages_id) + " 0 R";
        if (metadata_id > 0)
            cat_body += " /Metadata " + std::to_string(metadata_id) + " 0 R";
        if (output_intent_id > 0)
            cat_body += " /OutputIntents [" + std::to_string(output_intent_id) + " 0 R]";
        cat_body += " >>";
        if ((int)cat_body.size() <= 512) {
            cat_body.resize(512, ' ');
            memcpy(&out[cat_body_offset], cat_body.data(), 512);
        }
    }

    // Xref table
    int xref_offset = (int)out.size();
    out += "xref\n";
    snprintf(buf, sizeof(buf), "0 %d\n", next_id);
    out += buf;
    out += "0000000000 65535 f \n";
    for (auto & obj : objects) {
        snprintf(buf, sizeof(buf), "%010d 00000 n \n", obj.offset);
        out += buf;
    }

    // Trailer
    out += "trailer\n";
    snprintf(buf, sizeof(buf), "<< /Size %d /Root %d 0 R >>\n", next_id, cat_id);
    out += buf;
    out += "startxref\n";
    snprintf(buf, sizeof(buf), "%d\n", xref_offset);
    out += buf;
    out += "%%EOF\n";
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

ocr_renderer * ocr_render_create(ocr_render_format format) {
    auto * r = new ocr_renderer();
    r->format = format;
    return r;
}

void ocr_render_set_separator(ocr_renderer * r, const char * sep) {
    if (r && sep) r->separator = sep;
}

void ocr_render_set_pdfa(ocr_renderer * r, int enabled) {
    if (r) r->pdfa = (enabled != 0);
}

void ocr_render_begin(ocr_renderer * r) {
    if (!r) return;
    r->output.clear();
    r->page_count = 0;
    switch (r->format) {
        case OCR_RENDER_TEXT: text_begin(r); break;
        case OCR_RENDER_HOCR: hocr_begin(r); break;
        case OCR_RENDER_ALTO: alto_begin(r); break;
        case OCR_RENDER_PDF:  pdf_begin(r); break;
    }
}

void ocr_render_add_page(ocr_renderer * r, const ocr_render_page * page) {
    if (!r || !page) return;
    switch (r->format) {
        case OCR_RENDER_TEXT: text_add_page(r, page); break;
        case OCR_RENDER_HOCR: hocr_add_page(r, page); break;
        case OCR_RENDER_ALTO: alto_add_page(r, page); break;
        case OCR_RENDER_PDF:  pdf_add_page(r, page); break;
    }
}

void ocr_render_end(ocr_renderer * r) {
    if (!r) return;
    switch (r->format) {
        case OCR_RENDER_TEXT: text_end(r); break;
        case OCR_RENDER_HOCR: hocr_end(r); break;
        case OCR_RENDER_ALTO: alto_end(r); break;
        case OCR_RENDER_PDF:  pdf_end(r); break;
    }
}

const char * ocr_render_output(const ocr_renderer * r) {
    return r ? r->output.c_str() : "";
}

int ocr_render_output_size(const ocr_renderer * r) {
    return r ? (int)r->output.size() : 0;
}

void ocr_render_free(ocr_renderer * r) {
    delete r;
}
