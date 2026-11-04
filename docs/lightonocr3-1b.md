# LightOnOCR-3-1B candidate profile

This lane adds explicit source support for **LightOnOCR-3-1B**, reusing the
Pixtral/Qwen3 LightOn backend. It is unqualified until a hosted original/native
comparison completes. It creates no model-manager alias and makes no benchmark
or publication claim. LightOnOCR-3-0.8B/4B use the separate Qwen3.5 architecture.

## Pinned sources and inference contract

The checkpoint is
[`lightonai/LightOnOCR-3-1B@23d544085c886bbdace647bcfab91203299417e2`](https://huggingface.co/lightonai/LightOnOCR-3-1B/tree/23d544085c886bbdace647bcfab91203299417e2),
licensed Apache-2.0. The complete SHA-256 manifest is
[`models/lightonocr3-1b-source.json`](../models/lightonocr3-1b-source.json).
The weight object is 2,011,367,464 bytes, SHA-256
`32879861d5cdc3b2c3909d3f9a47dc8f7338685d13113c8c98c144f68b358dbb`.
No large weights are checked into this repository.

The source audit uses the exact Transformers commit
[`3568df44fff52979f1f9cecc7f729713e940d277`](https://github.com/huggingface/transformers/tree/3568df44fff52979f1f9cecc7f729713e940d277/src/transformers/models/lighton_ocr).
The manifest pins the LightOn model/config/processor, Pixtral model/image processor,
Qwen3 model and shared image backend files. The reference tool verifies those
installed source files before inference; a different Transformers checkout is an
error. Package versions are in
[`models/requirements-lightonocr3-reference.txt`](../models/requirements-lightonocr3-reference.txt),
and each run retains its environment freeze.

| Boundary | Pinned original behavior | Candidate implementation |
|---|---|---|
| Architecture | Pixtral 24 layers, width 1024, patch 14; Qwen3 28 layers, width 1024, 16 query/8 KV heads, QK norm, tied embeddings | Existing `lightonocr` graph and tensor mapping |
| Projector | RMS norm, channel-major 2×2 `unfold`, merging linear, linear, exact GELU, linear | Existing LightOn projector; odd trailing patch rows/columns are dropped by the merge |
| Resize | CPU uint8 bicubic with antialias; fit longest edge 1540 using floor; ceil each dimension to a 14-pixel grid | Shared `core/torch_bicubic_u8.h`; a distinct profile branch; no forced even patch count |
| Normalize | CLIP mean/std from the pinned processor | Per-channel F32 transform after uint8 resize |
| Prompt | Checkpoint Jinja includes an empty system message, user image, then assistant prefix | Converter renders/verifies actual tokenizer/Jinja and stores both prompt ID arrays |
| Modes | Empty prompt for transcription; literal `grounding` for layout | Transcription by default inside the opt-in profile; grounding has a second gate |
| Stops | IDs 151645 and 151643 | Both stop generation only for this profile |
| Decode | `skip_special_tokens=True`, no cleanup | Exclude the pinned special-ID list and unassigned vocabulary padding; preserve ordinary markdown and grounding markers |
| Output | Grounding labels + normalized 0–1000 xyxy + following content | Raw markdown returned by the existing string API; pure parser exposes document blocks |

The original LightOnOCR-2 path keeps its existing prompt, resize, stop and decode
behavior. A candidate profile refuses to load unless
`CRISPEMBED_LIGHTONOCR3=1`. `CRISPEMBED_LIGHTONOCR3_GROUNDING=1` selects grounding
for that profile. Both use `core_env::on`; `=0` disables the corresponding gate.
Unset `CRISPEMBED_MAX_PIXELS` for this lane: an override would change the frozen
processor contract and is rejected rather than silently changing the reference.

The current upstream image processor aligns to 14 pixels, not the effective
28-pixel merge. For example a 452×317 image becomes 462×322, producing an odd
33×23 patch grid. This distinction is covered by a model-free native control and
must remain visible in the hosted pixel/projector comparisons.

## Grounding adapter

[`tools/lightonocr3_output.py`](../tools/lightonocr3_output.py) exposes
`parse_grounding(raw) -> {blocks, unlocated_text, errors}` without model imports.
Each block has `id`, `label`, `bbox` normalized into `[0,1]`, `source_bbox` in
0–1000, `text`, and `continuation`. It supports inline markers, whitespace inside
coordinates and continuation suffixes. It preserves tables/formulas/code as text;
the shared document evaluator owns quality metrics. Unknown labels are retained
with a diagnostic. Invalid coordinates remain unlocated and are not clipped.

The accepted marker grammar comes from the upstream
[`lightonocr/grounding.py@36755d461be079737860a5f03ae0c803501269e9`](https://github.com/lightonai/LightOnOCR/blob/36755d461be079737860a5f03ae0c803501269e9/lightonocr/grounding.py)
and its benchmark provider, both hash-pinned in the manifest. The candidate
adapter additionally refuses invalid boxes and does not interpret marker literals
inside fenced code. Continuations remain separate for evaluation. Calling this
parser does not upgrade the existing C ABI to structured region output.

## Converter

On hosted compute, install the pinned environment and convert the verified source:

```bash
python -m pip install -r models/requirements-lightonocr3-reference.txt
python models/convert-lightonocr-to-gguf.py \
  --profile lightonocr3-1b --model lightonai/LightOnOCR-3-1B \
  --dtype f32 --output candidate-lightonocr3-1b-f32.gguf
```

The profile pins the revision and verifies all checkpoint files including weights.
A wrong revision, substituted config, incomplete vocabulary or template mismatch
fails conversion. The profile uses the same `general.architecture=lightonocr`
tensor layout, with `lightonocr.profile=lightonocr3-1b` and a versioned processor,
prompt, stop, special-token and provenance contract. A converted candidate stays
unqualified. No permissive alias or floating download is added.
The native loader also requires the exact source repository, revision, both
prompt ID arrays and the complete special-token set; merely in-range IDs do not
satisfy this profile.

## Hosted qualification

The existing heavy-CPU workflow is the entrypoint; no extra workflow is required.
CMake builds the separate contract and trace targets:

```cmake
add_executable(test-lightonocr3-contract tests/test_lightonocr3_contract.cpp)
target_link_libraries(test-lightonocr3-contract PRIVATE crispembed)
add_executable(lightonocr3-trace tests/lightonocr3_trace.cpp src/lightonocr.cpp)
target_compile_definitions(lightonocr3-trace PRIVATE CRISPEMBED_LIGHTONOCR3_TESTING=1)
crispembed_apply_arch_flags(lightonocr3-trace)
target_link_libraries(lightonocr3-trace PRIVATE crispembed)
```

The trace target compiles a separate backend translation unit. Observer hooks and
reference-token forcing are absent from production archives. The ordinary arm
selects its own tokens; the forced-prefix arm compares vocabulary logits under
the original prefix at every position, even after greedy choices diverge. A
third arm with observation disabled must return identical text and token IDs.

Dispatch the shell entrypoint through the existing workflow at the reviewed
source ref:

```bash
gh workflow run heavy-cpu.yml --ref "$REVIEWED_REF" \
  -f script=tools/ci-heavy/lightonocr3_1b.sh -f build_mode=portable
```

`lightonocr3_1b.sh` installs the pinned requirements in a fresh virtual environment,
verifies the actual installed original source and package versions, configures
`GGML_NATIVE=OFF` and `CRISPEMBED_NATIVE=OFF`, builds the two targets, and runs the
native/Python controls before calling the Python driver. A receipt binds the
source commit, binary and CMake cache; a mismatching prebuilt executable is rejected.
Bootstrap logs and failures are retained in `HEAVY_OUT/lightonocr3-1b-bootstrap`.

The driver requires hosted Linux, records load/RAM/disk and build mode, and
requires 12 GiB available RAM. Pre-build storage checks reserve 16 GiB for
models/builds/packages and 8 GiB for retained evidence (24 GiB when sharing one
filesystem); the driver rechecks remaining capacity after the build. Its frozen manifest
contains fox, receipt and form images, both prompt modes, and 128 new tokens.
Default arms are F32 and F16. Optional Q8_0/Q4_K arms require explicit candidate
GGUF paths via `--formats`/`--model-format`; no mixed recipe is assumed to qualify.
The actual tensor-type census must match the declared precision before a cosine
bar is assigned. GGUFs, HF cache, packages and builds stay in `HEAVY_SCRATCH`;
`--scratch` is also required for direct driver use outside the wrapper. The
workflow retains raw original/native arrays, tensor censuses, digests, logs and
comparisons under `HEAVY_OUT/lightonocr3-1b`. No model publication occurs. Each
inference subprocess is bounded to one hour with four CPU threads and per-child
RSS/time/exit/timeout receipts. Failures update the summary immediately. Output
directories must be empty, protecting existing receipts.

The original producer calls actual pinned Transformers modules with F32, eager
attention, cache enabled and deterministic greedy generation. It observes input
pixels/prompt IDs, patch convolution, pre-vision norm, all 24 vision layers,
projector, decoder input, all 28 prefill text layers, final norm and full
vocabulary logits at every generated position. Token IDs include stops; decoded
text, grounding, generation cap and time are retained. Source identity, installed
package versions, image digest, shape, prompt and all 151,936 vocabulary entries
are checked; shortened logits cannot masquerade as full-vocabulary evidence.
Decoded and forced-ID files are also hashed. Every native arm records its actual
applied and argmax IDs, image hash, mode, cap and observation setting, alongside
the runner's binary/model/process receipt. This is a deterministic
precision reference; the model card's sampled quality recipe is a separate task.

`tools/compare_lightonocr3.py` enforces complete finite captures and the owner
policy in `docs/precision-acceptance.md`: minimum decisive-logit cosine strictly
above 0.99 F32, 0.98 F16, 0.97 Q8_0 or 0.94 Q4_K. Strict 0.002 absolute/0.99999
cosine failures remain diagnostics alongside magnitudes, RMS, argmax, decoded
IDs/text, stops and cap status. Missing layers/logits and observer changes fail.
Prepared-pixel differences remain reported diagnostics under the same owner
policy; the original/native source-image hash and prompt-prefix contract must
match before logits are compared.
A capped page is a capped page; a passing bounded precision receipt cannot claim
complete document extraction, structured quality, or an OmniParseBench score.

## Model-free controls and remaining gates

```bash
python tests/test_lightonocr3_profile.py
./build-portable/test-lightonocr3-contract
python tools/list_env_vars.py --check
```

The Python controls cover source/fixture pins, incompatible architectures,
processor substitutions, dual stops, prompts, grounding edge cases, numerical
nonfinites, missing/short logits, source/input/provenance substitutions, forcing,
observer token changes and actual GGUF precision-label mismatches. The native
control covers official resize dimensions, odd grids, invalid geometry,
normalization and rejected repository/prompt/special-ID substitutions. Both are
included in the existing hosted model-free workflows and pass at the reviewed
PR138 source.

Outstanding evidence: hosted original/native F32/F16 results;
Q8_0/Q4_K recipe search and original-logit results; actual multilingual/layout
quality with licensed annotations; structured C ABI/orchestrator regions; artifact
publication/readback and registry aliases only after applicable gates. Existing
legacy LightOnOCR-2 frozen receipts remain intact. There is no new qualified model
alias or default change in this lane.
