# Native Marker-style document conversion

CrispEmbed converts PDFs and page images into Markdown, HTML, nested JSON and
document chunks through a native C++ pipeline. It combines PDFium text and
geometry, optional TeleOCR GGUF/GGML layout and recognition, document structure
processing, and image extraction. No Python interpreter, PyTorch process or
external service is required by the production pipeline.

This integration uses Apache-licensed processing concepts from
[Marker 2.1.0 at the audited revision](https://github.com/datalab-to/marker/tree/71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4)
with an alternative inference backend. It does not implement the original
Surya checkpoints or every upstream provider/processor. Read the
[feature coverage](marker-feature-coverage.md), [license audit](marker-license.md)
and [executed validation](marker-validation.md) before choosing it for a workload.
The inherited TeleOCR strict numerical qualification remains false; this
experimental path is disabled by default.

## Build and dependencies

Use the included sources and pinned ggml dependency. In a Git checkout, run
`git submodule update --init --recursive`; the complete source ZIP already
contains ggml and needs no Git command.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF \
  -DCRISPEMBED_BUILD_SHARED=ON
cmake --build build --target crispembed-marker crispembed-shared crispembed-server \
  --parallel 2
```

For PDF input, supply the absolute path to a compatible native PDFium library:
`libpdfium.so`, `libpdfium.dylib`, or `pdfium.dll`. PDFium is loaded only when a
PDF is opened. The pipeline does not search for, download or install it.
Image-only conversion needs no PDFium. The tested PDFium build is documented
in the validation record; other builds and platforms require qualification.
When redistributing a PDFium binary, include the notices for that exact build.
No PDFium binary is bundled with this source distribution.

Auto and OCR modes use an explicit local TeleOCR GGUF. Follow the existing
[TeleOCR conversion instructions](teleocr.md#convert-the-checkpoint) for F16,
Q8_0 and protected Q4_K files. Marker is a document pipeline, so there is no
separate `marker.gguf` or Marker weight converter. Datalab/Surya GGUF files do
not satisfy the TeleOCR architecture contract.

Enable the paths explicitly:

```sh
export CRISPEMBED_MARKER=1
export CRISPEMBED_TELEOCR=1
export CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1  # TeleOCR weights: Qwen RESEARCH LICENSE, non-commercial
```

Both gates use `core_env::on`: unset, empty and literal `0` mean disabled;
every other nonempty value, including `false`, means enabled. Use `0` to
disable a gate. Text mode needs only `CRISPEMBED_MARKER`.

## Modes

| Mode | PDF processing | Image processing | Model required |
| --- | --- | --- | --- |
| `auto` | TeleOCR layout; reuse usable PDF text; recognize regions needing OCR | TeleOCR layout and recognition | TeleOCR |
| `text` | PDF text, font/geometry grouping and deterministic structure processing | Rejected | None |
| `ocr` | Render pages; bypass embedded text extraction; TeleOCR layout and recognition | TeleOCR layout and recognition | TeleOCR |

Text mode is useful for searchable digital PDFs. It cannot recover scans or
guarantee that a syntactically plausible but incorrect text layer is detected.
Auto mode uses conservative Unicode/glyph/repetition checks, not Surya's
OCR-error model. Select OCR mode when the embedded text must be ignored.

Page indices are zero-based and inclusive. `last_page=-1` selects the final
page. A range outside the document fails rather than silently truncating.
An image contains page zero only. PDF coordinates are in visible page units,
including CropBox and intrinsic rotation; image coordinates are pixels.

## Command line

```sh
# Native layout and recognition, reusing usable PDF text.
./build/crispembed-marker invoice.pdf \
  --model teleocr-f16.gguf --pdfium /path/to/libpdfium.so \
  --mode auto --threads 2 --format markdown --output output/invoice/document.md

# Searchable PDF with no neural model.
./build/crispembed-marker report.pdf \
  --pdfium /path/to/libpdfium.so --mode text \
  --format json --output output/report/document.json

# Recognize an image; PDFium is unnecessary.
./build/crispembed-marker page.png --model teleocr-f16.gguf \
  --mode ocr --format html --output output/page/document.html
```

`--output` writes the selected format, a sibling `document_meta.json`, and
any referenced PNG assets. Use a separate directory for each document.
The writer preflights assets and rejects a conflicting existing image before
writing the bundle; identical image bytes may be reused. These collision
checks are not an all-or-nothing filesystem transaction. Without `--output`,
the CLI writes to stdout only when the result has no extracted assets.

Other options are `--first-page`, `--last-page`, `--dpi`, `--max-tokens`,
`--max-layout-tokens`, `--max-blocks`, `--max-pages`, `--max-pixels`,
`--max-asset-bytes`, `--include-headers` and `--no-images`.
PDF passwords are supported by the C, Python, Rust and Dart configuration;
the CLI and HTTP route do not expose a password argument.

## Python

Use the Python package from this source tree with its matching native library.
The Python wrapper copies returned strings and assets and serializes operations
on each context. `render` changes format without running inference again.

```python
import json
from crispembed import CrispMarker

with CrispMarker(
    "teleocr-f16.gguf",
    pdfium_path="/path/to/libpdfium.so",
    mode="auto",
    n_threads=2,
    lib_path="/path/to/libcrispembed.so",
) as marker:
    markdown = marker.convert("invoice.pdf")
    document = json.loads(marker.render("json"))
    chunks = json.loads(marker.render("chunks"))
    metadata = marker.metadata
    image_bytes_by_name = marker.assets
    marker.save("output/invoice/document.md")
```

For text-only extraction, omit the model and pass `mode="text"`. Use
`password="..."` for an encrypted PDF. `convert_image` accepts Pillow images
or contiguous/noncontiguous NumPy `uint8` gray, RGB and RGBA arrays. The wrapper
preserves Pillow palette transparency; the native path composites alpha onto
white. Shapes, channel counts and byte limits are checked before native use.

## C ABI

`src/crispembed.h` exposes the versioned `crispembed_marker_config` and opaque
`crispembed_marker_context`. Initialize every configuration with
`crispembed_marker_default_config()` so `struct_size` and defaults agree with
the linked library. The configuration strings are copied by `init`.

```c
#include "crispembed.h"
#include <stdio.h>

int convert_pdf(const char *pdfium, const char *input) {
    crispembed_marker_config cfg = crispembed_marker_default_config();
    cfg.mode = 1; /* digital text */
    cfg.pdfium_path = pdfium;
    crispembed_marker_context *ctx = crispembed_marker_init(&cfg);
    if (!ctx) {
        fprintf(stderr, "%s\n", crispembed_marker_error(NULL));
        return 1;
    }
    int ok = crispembed_marker_convert(ctx, input);
    const char *text = ok ? crispembed_marker_render(ctx, "markdown") : NULL;
    if (text) fputs(text, stdout);
    else fprintf(stderr, "%s\n", crispembed_marker_error(ctx));
    int status = text ? 0 : 1;
    crispembed_marker_free(ctx);
    return status;
}
```

Conversion returns one on success and zero on failure. Failure clears the
previous document and assets. Render and metadata results are borrowed C
strings; copy them before the next mutation or `free`. Asset names/data are
borrowed until the next conversion or `free`. Individual mutating calls have
an internal lock, but a C client must synchronize the entire sequence when
reading borrowed pointers. Do not concurrently free a live context.

`crispembed_marker_convert_raw` accepts tightly packed gray/RGB/RGBA pixels;
the C caller is responsible for supplying the full `width*height*channels`
buffer. `asset_count`, `asset_name` and `asset_data` expose extracted PNGs.
The language wrappers return independent copies instead of borrowed pointers.

## Rust and Dart

The Rust crate exports `Marker`, `MarkerMode` and `MarkerOptions`. Use the
crate from this source checkout and its matching library, not an older
published package without the new symbols.

```rust
use crispembed::{Marker, MarkerMode, MarkerOptions};

fn main() -> Result<(), String> {
    let mut marker = Marker::new(MarkerOptions {
        pdfium_path: "/path/to/libpdfium.so".into(),
        mode: MarkerMode::Text,
        ..MarkerOptions::default()
    })?;
    println!("{}", marker.convert("report.pdf", "markdown")?);
    let _metadata = marker.metadata_json()?;
    let _assets = marker.assets()?;
    Ok(())
}
```

The native Dart wrapper exports `CrispMarker` and `MarkerMode` from the Flutter
package. It is a native FFI API; no browser/WASM Marker wrapper is included.
Calls are synchronous, so perform document conversion in a worker isolate
when integrating with an interactive application.

```dart
final marker = CrispMarker(
  mode: MarkerMode.text,
  pdfiumPath: '/path/to/libpdfium.so',
  libPath: '/path/to/libcrispembed.so',
);
try {
  final markdown = marker.convert('report.pdf');
  final metadata = marker.metadata;
  final images = marker.assets;
} finally {
  marker.dispose();
}
```

## HTTP

```sh
./build/crispembed-server --marker-text \
  --marker-pdfium /path/to/libpdfium.so \
  --image-root /path/to/documents --host 127.0.0.1 --port 8080

curl -H 'Content-Type: application/json' \
  -d '{"file":"/path/to/documents/report.pdf","format":"json"}' \
  http://127.0.0.1:8080/ocr/marker
```

Use `--marker-model teleocr-f16.gguf` for auto mode, and additionally
`--marker-force-ocr` for OCR mode. `-t` sets native threads and
`--ocr-max-tokens` sets recognition length. Request file paths must satisfy
the existing canonical `--image-root` confinement. Relative paths resolve
against the server's working directory, not against `--image-root`. The endpoint returns
`format`, the selected output as a string, a metadata object, and an `images`
map of PNG data URLs. It serializes the complete conversion/result-copy
sequence for its context. A bad format/path returns 400, an unconfigured
Marker endpoint 503, and a conversion failure 422.

## Output and resource contracts

| Option | Default | Accepted range / meaning |
| --- | ---: | --- |
| `n_threads` | 4 | 1–64 |
| `dpi` | 144 | 36–600; PDF rasterization only |
| `max_tokens` | 1024 | 1–8192, per recognized region |
| `max_layout_tokens` | 2048 | 1–8192, per page layout |
| `max_blocks` | 512 | 1–4096 per page |
| `max_pages` | 1000 | 1–10000 selected PDF pages |
| `max_pixels` | 16000000 | Up to 64000000 per raster; neural mode also needs at least 1036×1036 |
| `max_asset_bytes` | 134217728 | 0–1073741824 total encoded assets |
| `include_page_headers` | false | Include running page headers/footers in rendered text |
| `extract_images` | true | Extract regions identified as figures/pictures/diagrams |

These are input/output bounds, not a promise that the largest accepted
configuration fits available RAM or finishes within a deadline. There is no
per-request cancellation or wall-clock timeout. PDF text mode does not render
pages or infer figure regions, so it cannot extract figures merely because
`extract_images` is enabled. Figure extraction in neural modes depends on
actual layout predictions.

Additional fixed parser/input limits are a 256 MiB encoded image file,
32768 pixels per raster side, one million extracted PDF characters per page,
4 MiB per block/raw-layout text field, nesting depth 24 and 16384 table cells.
Exceeding a limit produces an error or the documented bounded-parser diagnostic;
it does not increase the configured memory budget automatically.

JSON includes page/block IDs, geometry, original layout labels, text source,
model outputs, token counts, confidence values, caps and warnings. Metadata
identifies the actual backend and `crispembed.marker/1` schema. Confidence
validity is checked; probability calibration is not claimed. Repeated
headers/footers remain inspectable in JSON even when suppressed in prose.

Malformed UTF-8 and unsafe control bytes are made safe in rendered output;
raw model strings remain recoverable through JSON and conditional byte-hex
fields. Literal text is escaped. Tables use generated HTML to preserve cell
content and validated row/column spans. Raw OCR markup is not accepted as
arbitrary executable HTML.

See [coverage and bounded heuristics](marker-feature-coverage.md) for reading
order, table discovery, fixed-pitch spacing and the exact differences from
upstream Marker. See [validation](marker-validation.md) for what actually ran,
including retained initial failures and the separate numerical limits.
