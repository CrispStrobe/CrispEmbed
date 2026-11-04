# MinerU2.5-Pro-2605: exact checkpoint lane

This is an **unqualified, opt-in source implementation and hosted qualification
path** for `opendatalab/MinerU2.5-Pro-2605-1.2B`. It does not change the published
2604 aliases, their receipts, the 2604 converter default, or the 2604 helper guard.
There is no 2605 registry alias and no benchmark score claimed by this lane.

## Audited sources

| Component | Exact identity |
|---|---|
| Checkpoint | [`opendatalab/MinerU2.5-Pro-2605-1.2B@08aaea840498d49ce16247b6263196cf02814885`](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B/tree/08aaea840498d49ce16247b6263196cf02814885) |
| Safetensors | 2,312,126,640 bytes; SHA256 `abf8681ca63b8dec7b67de257af47b821f179442f72998d0696ae2ed9232a5f0` |
| Original helper | [`mineru-vl-utils==2.0.7`](https://pypi.org/project/mineru-vl-utils/2.0.7/); wheel SHA256 `3bf184c359d69e78d0509f58324c58ad323037cf97bd4086cacb9caf53db0db7` |
| Reference model implementation | [`transformers==5.10.1`](https://github.com/huggingface/transformers/tree/v5.10.1/src/transformers/models/qwen2_vl) with CPU PyTorch 2.9.0 and torchvision 0.24.0 |
| Existing native document translation | `mineru-vl-utils@c9797cde8f54cc48306196fc6aebe73ea31dd4a3`, unchanged; comparison with 2.0.7 is pending |

[The source manifest](../models/mineru25-pro-2605-source.json) freezes all seven
conversion input hashes, the helper wheel, its entire 41-file Python source tree,
and the original 73,384-byte safetensors header. The manifest records the LFS
weight hash returned by the pinned Hugging Face API; the converter verifies the
complete downloaded weight file before writing any GGUF. Initial source audit
read metadata and the bounded tensor header only, without downloading weights.
The checkpoint model card and helper wheel declare Apache-2.0. This lane does
not install the separate full MinerU application or its additional dependencies.

The actual 2605 header has the same 681 tensor names, BF16 dtypes, shapes and
offsets as 2604, but the weight payload hash differs. Its Qwen2-VL decoder remains
24 layers, width 896, 14 query heads, 2 KV heads, 151,936 embedding rows, and an
8,192-token **nested text configuration** limit. The top-level 32,768 declaration
does not override that nested limit. The 32-layer vision tower remains width
1,280 with 14-pixel patches, temporal duplication by 2, and merge size 2.

The actual config change adds top-level `tie_word_embeddings: true`; nested tying
was already present. Processor values are unchanged (only JSON formatting
changed), and generation metadata names Transformers 4.57.2 instead of 4.57.3.
That saved metadata is not the reference environment: helper 2.0.7's Transformers
extra requires Transformers >=5.10.1. The exact reference uses its original
`TransformersVlmClient`, torchvision processor, chat template, greedy generation,
no-repeat-ngram size 100, EOS IDs 151645/151643, and the complete structural-token
vocabulary. Sampling flags from generation_config are not substituted for the
helper's generation policy.

## Conversion and runtime

```bash
python models/convert-mineru25-pro-to-gguf.py --checkpoint 2605 \
  --model /path/to/exact-snapshot --dtype f32 --output mineru2605-f32.gguf
```

`--checkpoint 2604` remains the default. Selecting 2605 changes the exact allowed
repository, revision, byte manifest and GGUF family, while reusing the audited
Qwen2-VL tensor mapping. F16 retains norms, biases and tied embeddings in F32.
2605 `--check-only` also verifies full file identities; matching shape alone is
insufficient. F32/F16 conversion receipts explicitly leave quality and speed
qualification false.

Native loading requires both `CRISPEMBED_MINERU25_PRO=1` and the new default-off,
value-parsed `CRISPEMBED_MINERU25_PRO_2605=1`. The latter is an experimental profile
admission gate. The family must be `mineru2.5-pro-2605`, processor
`qwen2vl-fast`, and repository/revision/expected-revision metadata must match the
exact pin. `=0`, unset, an unknown family, or missing source identity does not
admit a 2605 model. The existing MinerU C ABI and native full-document route are
reused. No new enum or language binding is necessary.

`CrispMinerU.native_two_step_extract` uses the existing native document helper.
`CrispMinerU.two_step_extract` continues to require the old exact helper source
hash. The new comparison tool independently imports verified 2.0.7 sources;
it does not weaken that guard or relabel the native helper as a 2.0.7 port.
Cross-page table merging is not qualified by this single-page lane.

## Hosted measurements

Use the existing `heavy-cpu.yml` workflow, on the implementation branch:

```bash
gh workflow run heavy-cpu.yml --ref <branch> \
  -f script=tools/ci-heavy/mineru2605_qualify.sh \
  -f args="receipt f32,f16,q8_0,q4_k 2048 3 1" \
  -f build_mode=portable -f isolated_lane=mineru2605-receipt
```

Arguments are one case ID, precision tiers, token budget, repeat count,
and whether to run the document-helper comparison. The frozen
[protocol](mineru25-pro-2605-protocol.json) covers formula, table, receipt, German,
Japanese, Arabic, full-page layout and rotated layout. It verifies source image
hashes and decodes each image once into a shared RGB PNG. This is an internal
parity panel, not OmniDocBench or OmniParseBench evaluation.

Dispatch each of the eight frozen cases as its own isolated lane. `all` and
comma-separated case lists are rejected before downloads or builds. A disk and
available-memory preflight records conservative bounds for the chosen unchanged
token limit; scratch and artifact-output requirements are added when they share
a filesystem. Use a hosted Linux runner with at least 12 GiB of available RAM.
The runner retains one F32 GGUF and at most one additional candidate GGUF at a
time. Conversion/quantizer logs, exact model hashes and tensor censuses remain
in the artifact when each temporary candidate file is removed.

The script builds with `GGML_NATIVE=OFF` and `CRISPEMBED_NATIVE=OFF`, records the
CMake cache, source commit, package freeze, image/model hashes, actual tensor
precision census, and four-thread CPU timings. It captures original raw logits
at **every generated position**, then forces those original token IDs through
the native model to compare identical prefixes. A native F32 model is not used
as a substitute for the original Python model.

`mineru2605_qualify.py` applies the current precision policy: minimum final-logit
cosine >0.99 F32, >0.98 F16, >0.97 Q8_0, >0.94 Q4_K. Missing positions, altered
original sources, incomplete stage diagnostics or nonportable builds fail
closed. Every required stage name must appear exactly once, with one observation
and a nonempty finite array. It records all 64 prefill stages, their original strict 0.002/0.99999
budgets, worst intermediate cosine, and magnitude-aware diagnostics. A strict
stage failure or decoded-token disagreement remains visible but is not silently
substituted for the final-output precision decision.

A separate free-running arm records input/token IDs, EOS/caps, text, CER/WER
against the original text, resident repeat stability, cold/warm timings and
machine comparability. The document probe compares verified 2.0.7 helpers with
the existing native route using the same native neural inference in both arms;
it compares ordered request counts, prompts, crop dimensions, token IDs, text,
EOS and cap flags, as well as full JSON/Markdown. Both arms must reach EOS on
every request and the native aggregate cap flag must be false. Equal rendered
output cannot hide an incomplete or different request. This isolates helper
semantics and does not independently establish original neural parity. Its
ordered first/second timings are explicitly not a warm speed comparison.

The shared RGB PNG, original stage archive, every original logit row and every
native dump are written directly into the uploaded output, including partial
or failed captures. At 2048 tokens and four precision tiers, the five complete
logit streams have a 6,223,298,560-byte upper bound for one case; stage arrays,
dependencies and scratch model/build files are budgeted separately. These are
capacity bounds, not observed model performance. Do not download the large
artifact locally; review its small receipts and selected fields remotely.

Artifacts retain failures. `summary.json` distinguishes selected cases from the
complete eight-case panel and never grants an alias, publication, full benchmark
score, cross-page claim or default change. Native compilation and the model-free
profile controls pass at the reviewed PR138 source. Original/native hosted model
execution and full-panel qualification are still pending.

## Lightweight controls

```bash
python -m unittest discover -s tests -p 'test_mineru2605_contract.py' -v
python -m unittest discover -s tests -p 'test_mineru25_conversion.py' -v
bash -n tools/ci-heavy/mineru2605_qualify.sh
```

These controls test source/profile separation, actual metadata hashes, nested
context limits, portable build checks, missing logit positions, changed prefixes,
source relabeling and dequantized-reference substitution. They do not load model
weights or claim native compilation/inference success. The existing C++ profile
control adds exact 2605 admission-source positives and negatives for hosted CI.
