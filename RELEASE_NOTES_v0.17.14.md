# CrispEmbed v0.17.14

Since v0.17.12: the remaining default flips that were measured against the
original implementations are shipped, the app-facing packaging bugs are fixed,
CI now runs only the jobs a change needs (and reports its own status honestly),
and three new document-OCR checkpoints arrive as **default-off, unqualified
profiles** with a shared structured-document evaluation harness.

Nothing in the new OCR work publishes an alias, changes a runtime default or
claims model quality; it is source integration plus reproducible qualification
jobs. See "Verification and limits" before relying on it.

## Default changes since v0.17.12

Every flip below was A/B-measured against the original implementation and keeps
an explicit opt-out. The authoritative list with defaults and side effects is
`docs/environment-variables.md`.

| Area | New default | Opt-out |
|---|---|---|
| EasyOCR LSTM state | per-character state resets between steps instead of the legacy first-match reset | `CRISPEMBED_EASYOCR_LSTM_STATE_RESET=0` |
| EasyOCR line recovery | contrast retry and floor-width retry on low-confidence lines | `CRISPEMBED_EASYOCR_CONTRAST_RETRY=0`, `CRISPEMBED_EASYOCR_FLOOR_WIDTH=0` |
| GOT-OCR2 pipeline | original-pipeline parity switches (system turn, no-repeat n-gram, PIL bicubic preprocessing) | the `CRISPEMBED_GOT_OCR_ORIG_*` / `_PIL_BICUBIC` gates |
| SPM tokenization | space collapse now matches the HF normaliser for SentencePiece models | `CRISPEMBED_SPM_COLLAPSE_SPACES=0` (and `CRISPEMBED_SPM_HF_NORM=0`) |
| XLM-R / RoBERTa pairs | HF's `<s> a </s></s> b </s>` layout for cross-encoder pairs | `CRISPEMBED_PAIR_ROBERTA_SEP=0` |
| Position table | encoder input truncates to the learned position table (512 for XLM-R) instead of aborting | `CRISPEMBED_POS_EMBD_TRUNCATE=0` |
| CLIP text embedding | keep the EOS token | `CRISPEMBED_CLIP_TEXT_KEEP_EOS=0` |
| GPU offload (CUDA) | qualified GPU defaults for ESRGAN, InstructIR, PAN, Restormer and LightOnOCR | the engine's `CRISPEMBED_<ENGINE>_GPU=0`, or `CUDA=0` |
| GPU offload (Metal) | qualified Metal defaults for TeleOCR, MinerU, OXR, D-FINE and heron | the engine's `CRISPEMBED_<ENGINE>_GPU=0` |
| MiniCPM-OCR speed | F32 vision/projector copies, fused vision flash attention, batched vision slices and the graph decoder are on; a low-memory path replaces the F32 copies below the measured RAM headroom | `CRISPEMBED_MINICPM_OCR_VISION_F32=0`, `_VISION_FLASH=0`, `_VISION_BATCH=0`, `CRISPEMBED_QWEN35_GRAPH_DECODER=0`, `CRISPEMBED_MINICPM_OCR_LOW_MEMORY=0` |
| Unlimited-OCR-2 speed | F32 vision copies (`4/4` trial pages exact, `1.43x` on real scans, `17/19` token ids identical to the F16 path, original-exact pages `12 -> 13/20`) and the vision cache | `CRISPEMBED_UNLIMITED_OCR2_VISION_F32=0`, `_VISION_CACHE=0`, `_LOW_MEMORY=0` |

The low-memory paths are automatic: they skip the higher-footprint copies when
available RAM would not leave the measured headroom, and print one stderr
notice. Setting `=0` disables the check itself.

## New document-OCR profiles (default-off, not yet qualified)

`feat/ocr-benchmark-integration` (PR #138) adds exact checkpoint/task profiles
through the existing backends. None of them is reachable without its admission
gate, and none has a registry alias or a published model:

- **MinerU2.5-Pro-2605** — exact 2605 conversion profile with source/tensor
  pins and an original-first portable qualification job.
- **LightOnOCR-3-1B** — transcription and grounding tasks, exact processor
  profile, full-vocabulary comparison job.
- **Xiaomi-OCR-0** — format-4 profile, public task/cap admission, strict OTSL
  table checks against upstream conversions, and bounded document assembly
  around a **pinned external PP-DocLayoutV3 producer** (the detector is not
  bundled).
- **Nemotron Parse frontend** — bounded size/padding/patch-packing geometry and
  an original-processor comparator. There is no neural backend or native resize
  yet.
- **Structured document evaluation** — versioned text, ordered typed blocks and
  boxes, tables, formulas, per-generation-call provenance and resource
  accounting. Forced EOS, tokens after EOS and missing calls cannot pass.

The LightOnOCR-3-0.8B precision review under the current policy also lands here:
F16 and Q8 clear the numerical bars on the old three-page panel, Q4 does not.
A separate current-source portable replay is prepared but has not run.

## CI and release engineering

- **Selective CI** (PR #134): push triggers for evidence-only workflows are
  limited to `main`; a path selector classifies docs/evidence and Python tooling
  as "not built"; desktop build legs are a matrix chosen from the change. Pure
  documentation pushes no longer start a build.
- **Honest main health** (PR #135): the scheduled report no longer treats
  cancelled, pending, skipped, missing or API-failed checks as green. It reports
  the newest owned-`main` run per watched workflow with exact run IDs and source
  SHAs, keeps historical successes labelled `HISTORICAL_SUCCESS`, and fails
  `UNVERIFIED` instead of hiding gaps.

## Packaging and platforms

- macOS release artifacts install `CrispEmbed`/`ggml` dylibs under unversioned
  install names, so bundled binaries resolve without DYLD tricks (PR #128).
- Flutter podspecs accept an opt-in `CRISPEMBED_PREBUILT_VERSION` override in
  addition to the release default (PR #132).
- Server additions from the 3 November fold: usage/metrics accounting, request
  limits (429), and an OpenAI-shaped `POST /v1/chat/completions` +
  `GET /v1/models` surface for the OCR server (default-off).

## Verification and limits

- v0.17.14 is tagged on a `main` head where every workflow passed: all 16
  workflows green, including all 11 native build jobs (Linux, macOS arm64,
  Windows, Android ×3, iOS, Flutter bundle, sibling shared libs), both WASM
  builds and the browser end-to-end smoke test.
- The 162 focused OCR controls for the new profiles pass locally and in the
  shared hosted job; the source pin, environment documentation, clean-exit and
  formatting checks pass.
- **No inference has been run for the new OCR profiles.** Their hosted
  qualification jobs are prepared but not executed, so no benchmark, fidelity or
  precision claim is made for MinerU2605, LightOnOCR-3-1B or Xiaomi-OCR-0.
- OvisOCR3's frozen capped-page failure and every old LightOnOCR-3-0.8B verdict
  remain recorded; the Markdown task audit does not replace them.
- The heavy manual diagnostics (`MoEdern real padded Q4_K`, the Layer0 attention
  replay) remain recorded as non-release-blocking in HISTORY; the MoEdern job
  stops at authenticated input preparation rather than reducing its protocol.

## Upgrading

- Downstream apps should drop `CRISPEMBED_PREBUILT_VERSION`; Android builds
  from `main` fetch matching native libraries again.
- If a model regressed on your data, start with the opt-out column above — each
  switch restores the pre-0.17.14 behaviour without a rebuild.
- The new OCR engines remain opt-in: set the engine's admission variable (for
  example `CRISPEMBED_MINERU25_PRO=1` plus `CRISPEMBED_MINERU25_PRO_2605=1`) and
  expect unqualified output.
