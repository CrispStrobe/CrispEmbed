# Retained form RGB input boundary

The form-only [run40783738840](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40783738840)
succeeded at`535c10fceac28ec04401b7f6a57a027e1be9bbc9`. Its
[independent receipt](../tests/results/ocr-real-scans/rgb-input-form.json)
records the authenticated source, service, all23 small-artifact member hashes,
compiler profile and pixel metrics. No original or model forward was repeated.

The missing form uses the retained709225-byte JPEG, SHA256
`4d6e0d71fe7efdbce4947ccadc4dac2b15743026ea3bac9d84a45ef959a7051a`.
Exact-version Get, Status and every output-list page authenticate the frozen
producer/script before admitting the image. The
[initial audit](../tests/results/ocr-real-scans/rgb-input-initial.json) rejected a
different694795-byte download before either decoder ran. That rejection remains;
none of its nineteen valid comparisons was repeated.

Both Pillow and production STB decode the retained grayscale JPEG to1920×2985,
three channels and17193600 bytes. The frozen manifest's1600×2487 declaration
remains separately reported as different. No image, ground truth, dimensions,
prompt, cap or normalization was changed.

| RGB comparison | Result |
| --- | --- |
| Maximum absolute8-bit channel difference | 1 |
| Mean absolute channel difference | 0.0029569723618090454 |
| RMS channel difference | 0.054378050367855646 |
| Changed channel values / pixels | 50841 /16947 |
| First differing pixel `(x,y)` | `(1445,290)`: Pillow`[178,178,178]`, STB`[177,177,177]` |
| Pillow RGB SHA256 | `b07e7bca0d04f6fcda74b4ee3fbf5527d8ad8032f7da9f1c0195c7755394c7e4` |
| STB RGB SHA256 | `5370c212cda8e174dfa189820e277e3ebd9260a451a877c9fb38c53b9e80f6b7` |

Pillow11.3.0 uses libjpeg-turbo3.1.1 and`Image.open().convert('RGB')` with no EXIF
transpose, ICC transform, resize or normalization. The STB probe uses the exact
production implementation prefix/header2.28 from a06. Actual portable native
flags (`-O3 -DNDEBUG -fPIC -mavx2 -mfma -mf16c -std=gnu++17`), prefix versus full
production translation-unit decoder macros and the nonstatic`stbi_load` export
were independently checked from the small records. Compiler commands were
reserialized into the uploaded JSON; their semantic fields/flags were rechecked,
without claiming the uploaded file reproduces the original command-file byte
hash. Historical original wheel/libjpeg binary identity remains unproven.

Together, the two immutable audits cover all twenty admitted image identities:
five PNGs are byte-exact and fifteen JPEGs differ, with maximum channel error3
across the full set. This is a pairing of nineteen earlier comparisons plus one
retained form comparison. It establishes a measured input boundary, not the
cause of an OCR token difference or a qualified codec/model correction. Existing
[image and ground-truth credits](ocr-real-scans-v1.json) remain unchanged.

Only the70595-byte review ZIP was downloaded for independent review, SHA256
`b3620f75f8587ea383057d0892e56f326d740010a069be56e7460a95710f2d0b`.
Raw artifact12861682669 remains remote:4357147 bytes, SHA256
`47a8361d6720e13fe4933f01870eac8e5c48e475d4e81a03ae16185ed56d853b`,
expiring26 January2027. Independent review checked report/source/member hashes,
metric/count identities and compile/process records; it did not download or
recompute raw pixels, decode an image or run a model locally.
