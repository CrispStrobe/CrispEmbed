# SheenBidi source provenance

This directory contains the unchanged `Headers` and `Source` directories and
license from [SheenBidi 3.0.0](https://github.com/Tehreer/SheenBidi/tree/cfe430e7375a7845b679adae9d51dac6deaa8858),
commit `cfe430e7375a7845b679adae9d51dac6deaa8858`, under Apache-2.0.
It implements the Unicode 17.0 bidirectional algorithm. The optional text
editing API is not enabled.

CrispEmbed compiles its unity C source directly into the existing native
library; applications do not need another shared library or mobile archive.
The OCR PDF renderer uses resolved embedding levels and visual runs only for
invisible text geometry. Original page images remain the visible content.

To update, replace the two upstream directories together, retain the upstream
license, update this immutable revision, and rerun the native OCR export
artifact workflow (including bidirectional extraction and PDF/A validation).
