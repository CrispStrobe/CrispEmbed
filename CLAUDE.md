# Agent handoff notes

## Read first: public task handoff

Read the [current CrispMath/CrispEmbed state and executable lanes](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/current-state-and-next-steps.md)
and [contribution checklist](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/contributing.md).
The handoff gives exact source/workflow evidence, bounded tasks, dependencies and
completion checks. Private machine setup and access references belong outside
public repositories; do not make task execution depend on a published host path.

For runtime work, read the real Python inference blueprint before changing C++.
Compare the earliest intermediate divergence, absolute errors/magnitudes and final
decoded output. Component cosine alone is not quality parity. Cover vocabulary,
EOS and output processing separately. Keep probes out of production artifacts.
Run large workloads and batches on hosted CI/authorized GPU resources; never
occupy a constrained shared host with builds or model inference.

Two runtime requirements apply to every new path:

- **Env-gate every new path**, and use the shared helpers — `core_env::on()` for
  default-off gates, `core_env::explicitly_off()` for default-on ones. A bare
  `getenv(X) != nullptr` makes `X=0` mean ON; that defect class has already required a
  repo-wide audit once.
- **A/B against ground truth before flipping any default**, judge by decoded output and
  measured numbers rather than pass/fail, keep both paths behind a gate, and never delete
  a working path.

## Working location and repository policy

- Work in an isolated git worktree and preserve unrelated work. Keep large GGUF
  artifacts outside Git as remote artifacts; never place host paths in Markdown.
- Before changing code, fetch and merge `origin/main` into the feature worktree.
  After a coherent change, commit and push the feature branch so the work is
  recoverable. Coordinate all implementation and benchmark work through
  `PLAN.md`.
- Preserve functioning experimental paths behind environment-variable gates.
  Do not delete an opt-in path merely because it is slower or does not improve
  the current fixture; record its measured result and keep the default stable.

## Tesseract-LSTM findings

- The production default is the greedy/single-code path for single-code
  (Latin) models; since 2026-08-08 composed recoding AUTO-enables when the
  loaded recoder is multi-code (CJK traineddata) — `b61f22ae`, value-parsed
  tri-state, `=0`/`=1` keep absolute precedence. Wider recode beams,
  page-segmentation prototypes, scratch reuse, projection segmentation, and
  DAWG scoring remain opt-in gates.
- The relevant gates include `CRISPEMBED_TESSERACT_RECODE_COMPOSE`,
  `CRISPEMBED_TESSERACT_RECODE_BEAM_WIDTH`,
  `CRISPEMBED_TESSERACT_DAWG_LOAD`,
  `CRISPEMBED_TESSERACT_DAWG_SCORE`,
  `CRISPEMBED_TESSERACT_FORCE_CPU`, and the page-segmentation gates listed in
  `PLAN.md`.
- `convert-tesseract-to-gguf.py --embed-dawgs` preserves Tesseract's optional
  LSTM DAWG components as GGUF UINT8 arrays. The runtime has a bounded parser,
  load diagnostics, and C ABI complete/prefix queries, including UTF-8 input.
  Keep the converter's explicit GGUF array type; implicit list inference is
  incorrect for this metadata.
- DAWG bonus scoring currently adds a small system-dictionary bonus to
  completed words in the opt-in recode beam. It is diagnostic only: the
  available English smoke fixture produced the same `Se` output with scoring
  enabled and disabled, and no quality promotion is justified yet.

## Benchmark and quality requirements

- Always compare actual text, region/line counts, character counts, CER, WER,
  confidence, and timing—not only a pass/fail or cosine score.
- Use the page comparator/benchmark tools for paired official-Tesseract versus
  native runs. Native stage timings cover detector, grouping, crop, and
  recognition; official CLI timing is end-to-end, so do not call those
  per-stage timings apples-to-apples without adding matching reference stages.
- A full-page Fraktur result is still below official quality: the known
  `scan_strip` comparison has fewer native regions and non-zero CER/WER. The
  remaining high-value work is crop/line-image geometry, exact recoder and
  decoder semantics, dictionary candidate reranking, and a transcribed
  multi-code fixture. Wider beams are currently much slower and must remain
  gated until they demonstrably improve word choice.
- Open benchmark follow-ups belong in `PLAN.md`; completed milestones and
  evidence belong in `HISTORY.md`, with stable timing/quality data in `PERFORMANCE.md`. Include whether output is on par, worse, or simply
  unverified, plus a concrete TODO when it is worse or materially slower.

## Next actions

Use [the executable next-agent handoff](docs/next-agent-handoff.md) and
[PLAN](PLAN.md). Completed runs and initial failures belong in HISTORY, measured
limits in PERFORMANCE. Do not repeat dated historical commands as current tasks.

The wheel matrix/23 installed native-load checks/aggregation and native tokenizer
component audit pass. UltraBERT default-off CPU/F32 text API is merged viaPR74;
PR75 merges the actual Rust/Dart wrappers after [39979622930](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39979622930)
passes all42 pooled checks. C API/Python/CLI/HTTP and owned retrieval smoke pass.
Public pairs, representative quality and release gates, numerical errors and
strict reference-replay drift remain unresolved.
Corrected local audit39009703362 completed with retained numerical failures. Preserve its
scoped targets separately from the original frozen full-model references.

Read exact contracts before runtime work, use original-module intermediate dumps
and crispembed-diff, keep gates/old alternatives, and require actual output and
quality evidence before promotion. OCR quantized publication remains conditional;
no current candidate qualifies. OvisOCR2 metadata/export/frontend and controlled
original T4 GPU execution pass. Reviewed [native40049244654](tests/results/qwen35-ocr/ovisocr2-model-audit.json)
passes all14 F32/F16 arms on943 frozen fields with the explicit `both` preset;
four raw strict0.002 vision exceedances remain under unchanged calibrated
budgets. [Public40053283383](tests/results/qwen35-ocr/ovisocr2-public-surfaces.json)
passes27 C/Python/CLI/HTTP/Rust/Dart arms, and the
[original HF eleven-case reference40054043629](tests/results/qwen35-ocr/ovisocr2-original-quality.json)
is frozen. Independently reviewed
[first native F16 quality40056641800](tests/results/qwen35-ocr/ovisocr2-f16-quality.json)
passes all eleven cases with1,841 IDs and exact original HF raw/processed text,
all EOS/no cap, eight synthetic WER0 and15/15 inventory cells. It is the frozen
F16 baseline for quantized comparisons; per-case CPU timings across different
hosts are diagnostic only. The quant workflow defaults to one `dev-hello` F16
host control after exact source/runtime checks, while direct driver calls and
an explicit workflow option retain the full eleven-case control. Each candidate
still runs all eleven OCR cases and complete numerical/tensor checks. Protected
[Q8 40061433357](tests/results/qwen35-ocr/ovisocr2-q8-quality.json) is fully
reviewed and rejected: first decoder-layer error0.00376427, cosine0.999920726,
and a two-column `Lorem Ipsum`→`Lorem Ispum` regression with697→700 IDs and
worse raw/content CER/WER. All eleven cases reach EOS; the frozen F16 baseline
remains qualified. Both actual Q4 recipes are now complete and rejected:
[plain40066573198](tests/results/qwen35-ocr/ovisocr2-q4-quality.json) and
[calibrated40066574754](tests/results/qwen35-ocr/ovisocr2-q4-calibrated-quality.json).
Each retains11 EOS-completed cases, passing18/18 same-run F16 local controls,
473 tensor records and261 raw manifest entries; candidate numerical and document
screens fail. The development-only72-vector calibration proof is positive
separately from candidate qualification. The separate
[Q8 raw replay40066576197](tests/results/qwen35-ocr/ovisocr2-q8-raw-diff.json)
retains243 vectors and first source-fed QKV max0.0564889908; the initial Q8
wrapper remains unchanged. [Selection](tests/results/qwen35-ocr/ovisocr2-quant-attempts.json)
chooses qualified F16. [Publication40074370335](tests/results/qwen35-ocr/ovisocr2-publication.json)
passes actual upload and immutable empty-cache full readback at Hub b7229e46;
anonymous small-file/tree review is positive. The initial namespace denial is
preserved separately and the temporary publisher binding removed. The real
`ovisocr2-f16` alias and CLI/server resolution pass actual published
consumer40080458492: two fresh-cache hashes, offline reuse and CLI/HTTP text/
confidence. PR79 and final-source/merged-source checks are integrated; continue
the ordered follow-ups in PLAN. Bounded latency/RSS attribution is complete in
[the profile](tests/results/qwen35-ocr/ovisocr2-profiling.json). At the published
consumer run, all25 native source files and qualified `src`/ggml/CMake objects
matched the original qualification; later runtime edits require a new replay.
Ten consumer controls and
three adapter controls pass; the consumer rejects native-source drift before
build. Retain initial failures and raw strict flags. No runtime release, app
upgrade or default flip follows from model publication. Development
CPU-baseline PR76 is merged; successor release assets remain separate. The
[issue71 feasibility audit](docs/inpainting-feasibility.md) is source-only:
MI-GAN is the first bounded still-image experiment; LaMa checkpoint rights and
FFT implementation remain unresolved. The fourth OvisOCR2
[attention-F16/FFN-Q4 run40172527229](tests/results/qwen35-ocr/ovisocr2-q4-attention-f16-quality.json)
is rejected despite72 FFN-Q4/401 byte-exact tensors and passing first-linear
probes: `prefill.decoder.layer.0` fails numerical gates and two realistic OCR
cases regress. [Selection v2](tests/results/qwen35-ocr/ovisocr2-quant-attempts-v2.json)
keeps published F16 and the frozen first F16 quality receipt. The expanded
runtime F16 quality replay40178330437 matches all11 frozen decoded cases,
and final-source frozen14-arm40183491593 plus F16 eleven-case40183493328 are
independently positive, preserving all1,841 first-F16 outputs and confidences.
[The combined larger-input audit](tests/results/qwen35-ocr/ovisocr2-larger-inputs.json)
is complete: frozen/1024/1200 pass14/14;2048 passes8/14, with six composed final-
norm numerical failures despite exact text/EOS. That boundary remains unqualified. No new large-image default or qualified precision follows yet.

OvisOCR3 metadata and its original943-stage CPU reference are reviewed. Its
[original plain OCR eleven-case capture40178328864](tests/results/qwen35-ocr/ovisocr3-original-quality.json)
is source-quality negative: two-column-page hits the frozen1024-token cap
without EOS (CER0.52365/WER0.83993). The earlier native14 retry exposed two
test-harness defects. [Final-source native1440183489780](tests/results/qwen35-ocr/ovisocr3-native-model.json)
is independently positive on all14 arms. [Paired F16 diagnostic40187405735](tests/results/qwen35-ocr/ovisocr3-f16-quality.json)
faithfully matches all11 original cases/1,679 IDs/raw text/stops/caps, including
the unchanged capped page. [Interface-only public diagnostic40196755934](tests/results/qwen35-ocr/ovisocr3-public-diagnostic.json)
is independently positive on all29 recorded checks across six clients, ABI,
cache and negative gate/profile/prompt controls. Its model/publication flags
remain false; preserve initial omitted-artifact-name preflight failure separately.
Completed audits are in HISTORY; start scoped reuse/attribution/permissive lanes
from PLAN. No interface result upgrades negative source quality. The
source completion failure blocks model qualification, publication and alias;
do not change its prompt, cap, gold or scorer to pass a gate.
The permissive queue, encoder follow-ups and portfolio tasks remain scoped in
PLAN and model documents. SAM work stays with
its existing owner. The app remains pinned to immutable0.17.12.

This file is tracked public documentation even though an ignore pattern matches
its name. Never append private environment/access details here. Optional private
continuity is stored in ignored `.claude/local-handoff/AGENTS.md` when available.
