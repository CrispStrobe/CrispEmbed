# PDF DPI Profiling

Analyse embedded raster images in PDF files to determine effective DPI per page. Used by the OCR pipeline to decide whether text super-resolution is needed before recognition.

## Architecture

Minimal PDF parser (`src/pdf_info.{h,cpp}`) -- zero external dependencies. Walks the PDF cross-reference table, finds image XObjects on each page, reads their pixel dimensions (`/Width`, `/Height`) and the page MediaBox, then computes DPI from the ratio.

- No rendering, no decompression -- just metadata extraction
- Handles linearized and non-linearized PDFs
- Reports min/max/mean DPI across all images on a page

## API

### C ABI (`crispembed.h`)

```c
int crispembed_pdf_page_dpi(
    const char * pdf_path, int page,
    float * out_dpi, int * out_n_images);
```

### Low-level C API (`pdf_info.h`)

```c
typedef struct pdf_page_dpi_result {
    float dpi;
    float dpi_min;
    float dpi_max;
    int n_images;
    float page_width_pt;
    float page_height_pt;
} pdf_page_dpi_result;

int pdf_page_dpi(const char * pdf_path, int page, pdf_page_dpi_result * result);
pdf_page_dpi_result * pdf_all_pages_dpi(const char * pdf_path, int * n_pages);
void pdf_dpi_free(pdf_page_dpi_result * results);
```

### CLI

```bash
# Per-page DPI report as JSON
crispembed --pdf-dpi document.pdf
# Output: {"pages":[{"page":0,"dpi":300.0,"dpi_min":300.0,"dpi_max":300.0,"n_images":1,...}]}
```

### HTTP Server

```bash
curl -X POST http://localhost:8080/pdf/dpi \
  -H 'Content-Type: application/json' \
  -d '{"file": "/path/to/document.pdf"}'
```

### Python

```python
from crispembed import CrispPreprocess
pp = CrispPreprocess()
dpi, n_images = pp.pdf_page_dpi("document.pdf", page=0)
print(f"DPI={dpi}, images={n_images}")
```

### Rust

```rust
use crispembed::pdf_page_dpi;
let (dpi, n_images) = pdf_page_dpi("document.pdf", 0).unwrap();
println!("DPI={dpi}, images={n_images}");
```

### Dart/Flutter

```dart
final pp = CrispPreprocess();
final result = pp.pdfPageDpi("document.pdf", page: 0);
if (result != null) {
  print("DPI=${result.dpi}, images=${result.nImages}");
}
```

## Integration Matrix

| Layer | Status |
|-------|--------|
| C API (crispembed.h) | Y |
| CLI (--pdf-dpi) | Y |
| Server (POST /pdf/dpi) | Y |
| Python (CrispPreprocess.pdf_page_dpi) | Y |
| Rust (crispembed::pdf_page_dpi) | Y |
| Dart/Flutter (CrispPreprocess.pdfPageDpi) | Y |
| Rust FFI (crispembed-sys) | Y |

## Test Results

6 DPI test cases with synthetic PDFs, all exact:

| Test | Expected DPI | Result |
|------|-------------|--------|
| 300 DPI single image | 300.0 | PASS |
| 150 DPI single image | 150.0 | PASS |
| 72 DPI (screen) | 72.0 | PASS |
| Mixed 150+300 DPI | min=150, max=300 | PASS |
| No images (vector-only) | n_images=0 | PASS |
| Multi-page PDF | per-page correct | PASS |

## Files

| File | Purpose |
|------|---------|
| `src/pdf_info.h` | Public C header |
| `src/pdf_info.cpp` | Minimal PDF parser + DPI calculator |
| `src/crispembed.cpp` | C ABI wrapper (`crispembed_pdf_page_dpi`) |
| `examples/cli/main.cpp` | `--pdf-dpi` flag handler |
| `examples/server/server.cpp` | `POST /pdf/dpi` endpoint |
| `python/crispembed/_binding.py` | ctypes binding (`CrispPreprocess.pdf_page_dpi`) |
| `crispembed-sys/src/lib.rs` | Raw Rust FFI declaration |
| `crispembed/src/lib.rs` | Safe Rust wrapper (`pdf_page_dpi()`) |
| `flutter/crispembed/lib/src/crispembed_bindings.dart` | Dart FFI typedef |
| `flutter/crispembed/lib/src/crispembed.dart` | Dart high-level wrapper |
