// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "marker_document.h"
#include <memory>

namespace crisp_marker {
// PDFium is an explicit, optional, dynamically loaded native dependency. No
// downloading, Python interpreter or executable model code in this path.
class pdf_provider {
public:
    pdf_provider(const std::string & library, const std::string & file, const std::string & password);
    ~pdf_provider();
    pdf_provider(const pdf_provider &) = delete;
    pdf_provider & operator=(const pdf_provider &) = delete;
    int page_count() const;
    page read_page(int index, bool extract_text = true);
    image render_page(int index, int dpi, size_t max_pixels);

private:
    struct impl;
    std::unique_ptr<impl> state;
};

image load_image(const std::string & file, size_t max_pixels);
image crop_image(const image & input, const block & region, double page_width, double page_height);
asset encode_image_asset(const image & input, const std::string & name);

class teleocr_backend {
public:
    explicit teleocr_backend(const options & opts);
    ~teleocr_backend();
    teleocr_backend(const teleocr_backend &) = delete;
    teleocr_backend & operator=(const teleocr_backend &) = delete;
    std::vector<block> layout(const image & input, double page_width, double page_height,
                              std::vector<std::string> & warnings, std::string * raw = nullptr,
                              int * generated_tokens = nullptr, bool * token_cap_reached = nullptr);
    void recognize(const image & input, block & region, double page_width, double page_height);

private:
    struct impl;
    std::unique_ptr<impl> state;
};
} // namespace crisp_marker
