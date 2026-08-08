# Agent handoff notes

## READ FIRST — the dev guide governs everything here

Before any work in this repo, read **`crispasr-crispembed-dev.md`** in full and follow
it. It holds the HARD RULES (read the Python blueprint line by line, use the diff harness
at every boundary, decoded-output roundtrip is the only acceptance test), the mandatory
A/B protocol, the env-gating convention, and the new-backend checklist. It outranks
convenience and it outranks your own judgement about what is "obviously fine".

It sits NEXT TO this repo, not inside it:

| box | path |
|---|---|
| Mac dev box | `/Users/christianstrobele/code/crispasr-crispembed-dev.md` |
| VPS | `/mnt/volume1/crispasr-crispembed-dev.md` |

⚠ **`../crispasr-crispembed-dev.md` only resolves from the REPO ROOT.** If you are in a
worktree (`.claude/worktrees/<name>/`, which is where you should be) it is four levels
deeper and that relative path silently does not exist — no error, just a missing file and
an agent that never read the rules. This has already happened. Use the absolute path.

Two rules from it that are violated most often, repeated here so there is no excuse:

- **Env-gate every new path**, and use the shared helpers — `core_env::on()` for
  default-off gates, `core_env::explicitly_off()` for default-on ones. A bare
  `getenv(X) != nullptr` makes `X=0` mean ON; that defect class has already required a
  repo-wide audit once.
- **A/B against ground truth before flipping any default**, judge by decoded output and
  measured numbers rather than pass/fail, keep both paths behind a gate, and never delete
  a working path.

## Working location and repository policy

- Work in the isolated backup worktree, not the original checkout that was
  explicitly retired. Keep large GGUF artifacts outside Git under the shared
  backup storage; never place machine-specific absolute paths in Markdown.
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
- Every benchmark finding belongs in `PLAN.md` and, for stable timing/quality
  data, `PERFORMANCE.md`. Include whether output is on par, worse, or simply
  unverified, plus a concrete TODO when it is worse or materially slower.

## Next actions

1. Re-sync with remote `main` and run the comparator on the canonical page
   fixture with DAWG scoring off/on and controlled recode beam widths.
2. Extend the benchmark JSON/table to expose the exact paired outputs and
   region/line counts for each mode; do not hide output differences in a
   summary metric.
3. Build a real multi-code/word-choice fixture and validate DAWG candidate
   ranking against official Tesseract before considering any default change.
4. Profile the largest native stage gaps, then optimize one gated path at a
   time and re-run the same quality and performance gates.

