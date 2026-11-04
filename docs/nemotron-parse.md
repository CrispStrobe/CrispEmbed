# Nemotron 3.5 Parse: pinned source and bounded native bring-up

## Scope and observed evidence

The source audit targets
[`nvidia/Nemotron-3.5-Parse@4035ffb1`](https://huggingface.co/nvidia/Nemotron-3.5-Parse/tree/4035ffb1d8f8bafcdb867dfaf6f7514f101464f9).
The full revision and small-file hashes are in
`models/nemotron-parse-source.json`. The observed audit is
`tests/results/nemotron-parse/source-audit.json`.

All 13 pinned blueprint/config/license files were read and hashed. A bounded
HTTP 206 range read inspected the safetensors header, not the weight payload.
The actual archive has 510 tensors: 630,800,496 parameter elements and eight
buffer elements. Of these, 2,099,200 parameters belong to the optional MTP heads;
ordinary HF generation uses 628,701,296 parameters. The file is 2,523,266,640
bytes, containing F32 parameters and I64 summary indices. The reported LFS hash
is upstream metadata, not an independently verified full-file hash.

The executable census derives every tensor name and shape from the pinned
blueprint, accounts for all stored offsets/bytes, and detects omitted MTP heads,
GQA shape substitutions and an invented cross-attention decoder. Run:

```sh
python tools/nemotron_parse_contract.py
python tools/nemotron_parse_contract.py --source-directory source-snapshot
python tests/test_nemotron_parse_contract.py -v
```

These are source/metadata controls. They execute no model and establish no
native numerical parity, recognition accuracy, throughput, registry artifact or
publication qualification.

## Frontend prototype

`src/nemotron_parse_geometry.h` implements the pinned size decision and
normalized 16×16 RGB patch packing behind
`CRISPEMBED_NEMOTRON_PARSE_GEOMETRY=1`. It retains separate content/padded
sizes, pads bottom/right to height 16 and width 64, and accepts tightly packed
RGB8/grayscale8 pixels already resized by the original processor.
`tools/nemotron_parse_frontend.py` captures the original processor's resized
pixels and packed outputs on eleven frozen positive cases plus negative
aspect-ratio controls. Hosted native compilation and the model-free `self`
controls pass at the reviewed PR138 source. The
[manual processor job41810180035](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41810180035)
at merge source `2f1bbdb1` now passes all11 cases exactly: zero changed values
among16,665,600 packed values, max absolute error0, and all9 negative controls
pass. The [preserved comparison](../tests/results/nemotron-parse/frontend-parity-41810180035.json)
retains each input/output hash and numerical result; the original receipt hash
was independently checked. Both portable CPU flags were OFF. This completes
the bounded size/packing milestone and does not require a duplicate run.
It does not establish native Pillow-resize equivalence, learned
positional interpolation, neural model parity, an OCR engine ID, or publication
qualification. Applicable NVIDIA copyright, OpenMDW-1.1 license and adaptation
notices are retained under `licenses/`.

The hosted processor-only job is separately bounded and downloads no model
weights:

```sh
gh workflow run heavy-cpu.yml --ref feat/ocr-benchmark-integration \
  -f script=tools/ci-heavy/nemotron_parse_frontend.sh \
  -f build_mode=portable -f isolated_lane=nemotron-parse-frontend
```

Its model-free target is `test-nemotron-parse-geometry`; `self` exercises native
positive/negative cases without Python dependencies or weights. Python receipt
controls run with `python tests/test_nemotron_parse_frontend.py -v`.

## Actual architecture to port

Read the exact
[HF model implementation](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/4035ffb1d8f8bafcdb867dfaf6f7514f101464f9/hf_nemotron_parse_modeling.py),
[processor](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/4035ffb1d8f8bafcdb867dfaf6f7514f101464f9/hf_nemotron_parse_processor.py),
and [position helper](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/4035ffb1d8f8bafcdb867dfaf6f7514f101464f9/hf_prefix_geometry.py).

| Stage | Pinned behavior |
|---|---|
| Image preparation | RGB; downscale within height 2048/width 1664 without upscaling; PIL bilinear; white padding at bottom/right to height multiple 16 and width multiple 64. Preserve separate content and padded sizes. |
| Patches | CLIP mean/std normalization; 16×16 RGB patches flattened channel/row/column; 768 values per patch. |
| Position field | Learned 128×128×1152 field. Content-size interpolation with antialiasing, a separate one-patch-axis adaptive-average path, then bilinear border grid sampling. Plain resized positions are not equivalent. |
| C-RADIO vision | 27 global ViT blocks, hidden 1152, 16 heads, FFN 4304; ten class/register tokens, summary indices 0 and 1. Patch projection has no bias. Vision norm epsilon is 1e-6. |
| Compression neck | 1D projection 1152→1024 + norm; horizontal 1×4 stride-4 convolution + norm; 2304→1024 summary projection + norm. Append summary after compressed spatial features. |
| Decoder | Ten pre-norm mBART-style layers, hidden 1024, FFN 4096, 16 query heads / 8 KV heads, exact GELU, causal attention over the visual prefix and text. No cross-attention or decoder position-embedding tensor appears in this blueprint. |
| Vocabulary | 72,512 embedding/output rows, tied output head; embedding scale √1024. Tokenizer has a separately declared 72,505 entries, so preserve reserved/unused vocabulary rows. |
| Generation | Visual prefix enters once; subsequent token calls reuse KV and the visual mask. Context limit 16,384 includes vision. Start/EOS ID 2; ordinary greedy generation does not use the optional MTP tensors. |

The project has useful ViT, GQA, normalization and mBART operators, but no
existing engine implements this composition. PP-FormulaNet's cross-attention
mBART and the Qwen3.5 OCR backend are not compatible aliases.

## Bounded implementation and qualification order

1. Native input-geometry/packed-patch frontend, checked against the exact
   processor on portrait, landscape, nonmultiple, narrow, RGB/grayscale and
   downscaled images. Reject unsupported inputs rather than silently repairing
   geometry. Keep the operator test separate from complete OCR support.
2. Strict GGUF converter covering the entire 510-tensor census. Retain the MTP
   tensors or explicitly represent their omission in a distinct converter
   profile. Verify the actual full weight hash on the hosted runner.
3. Original-first captures for conditioner/patches/positions, all 27 vision
   layers, summaries/compression, all ten decoder layers, caches and full
   vocabulary logits. Freeze actual original arrays and decoded output before
   native generation. Do not reconstruct gold using the new native math.
4. Default-off native graph and one bounded full-page comparison. Compare
   earliest-stage absolute errors/magnitudes, final logits, IDs/text, EOS/caps,
   structured output and portable CPU memory/time. Batch and MTP remain separate
   protocols after ordinary single-page greedy correctness.
5. Public API/orchestrator/CLI/server/Python/Rust/Dart, representative document
   quality, then separate Q8/Q4 qualification and immutable publication/readback.

The 9000-token upstream generation default and repetition penalty 1.1 are part
of a source recipe, not permission to change an existing CrispEmbed quality
contract's cap. Freeze the new protocol before seeing output. OmniParseBench's
single/greedy/triple modes are distinct pipelines; do not transfer their score
to this native implementation or a different source revision.

## Rights and distribution

The pinned [LICENSE](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/4035ffb1d8f8bafcdb867dfaf6f7514f101464f9/LICENSE)
states OpenMDW-1.1 for the weights and NVIDIA-provided source/documentation;
the [tokenizer notice](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/4035ffb1d8f8bafcdb867dfaf6f7514f101464f9/TOKENIZER_ATTRIBUTION.md)
states MIT and traces the modified Nougat tokenizer. Retain applicable licenses,
copyright/origin notices and the pinned third-party notice inventory in any
distribution. Older indexed tokenizer-license descriptions are not the authority
for this revision. Source eligibility does not qualify a GGUF or downstream app.
