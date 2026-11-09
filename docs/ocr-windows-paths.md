# Windows Unicode document paths

The public image and OCR APIs accept UTF-8 filenames. On Windows, a document or
user profile containing accented letters, CJK text or emoji previously reached
the ANSI `fopen` path inside the shared image loader. The same boundary existed
when the image writer created crop files. These calls could fail even though the
file was readable through the operating system's Unicode API.

The production loader now defines `STBI_WINDOWS_UTF8` and the production PNG
writer defines `STBIW_WINDOWS_UTF8` on Windows. Their upstream implementation
converts the UTF-8 filename for the wide Windows file API. The standalone OCR
renderer harness uses the same settings. Image preprocessing, inference gates,
output pixels and non-Windows behavior are unchanged.

## Executable qualification

The dedicated `OCR Windows Unicode file paths` workflow builds and executes
three programs on Windows Server 2022 with MSVC:

- The actual `image_preprocess.cpp`, `core/image_out.cpp` and `ocr_render.cpp`
  production translation units load and write a known 4 × 3 RGB image through
  ASCII and nested Unicode filenames. Their preprocessing patch vectors must
  be identical. Optional deskew stays disabled; its test stubs abort if called.
- The actual standalone renderer image implementation repeats file I/O and PDF
  embedding through both filenames.
- A generated historical control removes only the two Windows UTF-8 defines
  from the production implementation. On a non-UTF-8 ANSI-code-page host, ASCII
  access must work and all three Unicode filename operations must fail. A host
  that cannot reproduce this condition fails the control explicitly.

Pillow independently checks decoded PNG pixels. pypdf independently checks all
four PDFs: one page, exact RGB image bytes, 288 × 216-point physical dimensions,
and the complete reviewed text. Source bytes must remain unchanged. Receipts
include the source commit, dependency gitlink, source and binary hashes, reader
versions and observed Windows ANSI code page.

Native execution is required before calling this path qualified. The workflow
does not claim full Windows application installation, OCR model quality, long
path support beyond the image library's limits, or physical-device results.
