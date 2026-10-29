# Q8 packed-profile controls

Source preparation adds standard Q8_0 header7 admission to UltraBERT and a named
`crispembed-josev-packed-mixed-v2` Josev profile. Header7 requires actual Q8_0 and
no Q4_K tensors. Existing Q4 headers require genuine Q4_K. Critical vocabulary,
normalization, router/head and nonprojection tensors retain their original
validation, and all existing default-off quantization gates remain mandatory.
The F32 and legacy Q4 paths remain available.

The C quantizer's optional `--metadata-string KEY=VALUE` replaces an existing
STRING key before writing the output header. Header lengths/offsets are rebuilt
by GGUF; matrix quantization policy is unchanged. Unknown keys, numeric fields,
empty values and duplicate overrides reject before creating an output. Example:

```sh
crispembed-quantize source-f32.gguf candidate-q8.gguf q8_0 \
  --metadata-string josev2b.packed.format=crispembed-josev-packed-mixed-v2
```

A model-specific converter must also protect critical tensors and authenticate
all source/output bytes and actual stored tensor types. This command alone does
not establish that recipe or qualify a model.

The standalone `portfolio-q8-profile-controls-v1.yml` workflow compiles actual
metadata-only Josev/UltraBERT tests and the C quantizer. It tests13 standard/legacy
methods, an actual variable-length string stamp, unchanged tensor payloads and
invalid-option output preservation. Josev raw-bound fixtures use sparse files;
they allocate no model weights and run no inference. Full original-bound Q8
intermediate/decoded/retrieval and public-consumer tests remain mandatory before
publication or a release. No quality gate is relaxed by these compiler fixtures.
