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

Bounded prompt/registry CI repair is complete in [PR61](https://github.com/CrispStrobe/CrispEmbed/pull/61):
all six affected artifacts byte-verified, old digests retained, strict304/304
freshness replay and all PR platform/regression/packaging checks pass.

Follow the current execution order in [PLAN.md](PLAN.md): retain the completed
book vision diagnostic and revisit independent-document quantization/publication,
UltraBERT and queued
permissive models. The [independent OCR quality audit](tests/results/qwen35-ocr/quality.json)
completes all3 EOS outputs with exact1,404 tokens/stops and15/15 inventory cells,
but retains6 prefill and13 book-logit failures (worst0.0060377121 versus0.002).
Receipt/book recognition errors remain shared with Python. The contract retains raw
CER/WER separately from ordered content/table cells; unannotated structure stays
unmeasured. x86/ARM preparation matches exactly and all three images are reviewed;
all22 hosted controls and frozen-hash replay pass. The guarded held-out F16 audit
correctly concludes failure for those numerical records; exact decoded results
and cosine rounding to1 do not waive them. The [completed book isolation](tests/results/qwen35-ocr/book-isolation.json) eliminates
all book prefill/score/probe failures with original prefill embeddings, narrowing
the upstream boundary. This is diagnostic, not a production fix. The [completed book vision trace](tests/results/qwen35-ocr/book-vision.json)
passes24 exact native controls and reproduces434 original CPU/77 baseline prefill
records. Four existing settings retain5/6/5/8 prefill failures; all280 conditional
attention vectors pass raw0.002, while original-input block11 down error is
0.00268554688. A single causal defect remains unproved; preserve the remaining
projection/reduction task in PLAN while moving to independent quantized evaluation.
Smaller merger errors are not decoded-quality or default-promotion evidence.
The [completed receipt audit](tests/results/qwen35-ocr/receipt.json)
reaches EOS at455/434 tokens with exact Python output and889 passing teacher-score
vectors; both prompts still emit HTML. Lower raw CER does not prove better OCR.
The [parallel-F16 trial](tests/results/qwen35-ocr/f16-materialization.json)
preserves every converted F32 bit and frozen inference record, with 1.37–1.67×
paired ARM64 speedups; its gate remains default-off. Further cache/memory and
other-host/default-adoption work remains pending. The fixed-cap four-arm decoded
panel matches all 16
ID/text/stop comparisons; preserve its 56 prefill failures and receipt cap controls.
Use the validated attention trace for any divergence.
UltraBERT feasibility is recorded in [docs/ultrabert.md](docs/ultrabert.md); it is
not native support. Keep its dependency environment separate from pinned OCR references.

Tesseract page/crop/recoder/dictionary work remains in the plan's portfolio backlog.
Read the archived technical briefs for evidence, then verify current code before
reviving a historical failure. Move completed lane narratives into HISTORY rather
than growing PLAN with dated work logs. Do not treat archived machine commands,
agent assignments or release snapshots as current instructions.
