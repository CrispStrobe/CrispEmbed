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
[issue71 feasibility audit](docs/inpainting-feasibility.md) is complete.
MI-GAN's bounded fixed256 CPU/F32 prototype is integrated throughPR85/`1b74b12a`:
[corrected40344396533](tests/results/migan/native-parity.json) at`2526fcac`
passes494 unchanged comparisons and all8 image gates against original40330674649.
The initial256 stale observer fields remain preserved; the test-only owner
retention repair leaves all8 ordinary final hashes unchanged. Public API/registry,
representative subtitle/perceptual/video quality and publication remain separate;
issue71's product scope stays open. LaMa checkpoint rights and FFT implementation
remain unresolved. The fourth OvisOCR2
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
The bounded per-context decoder weight cache is also audited in
[the preserved-vector receipt](tests/results/qwen35-ocr/ovisocr2-weight-cache.json)
and [eleven-case quality](tests/results/qwen35-ocr/ovisocr2-weight-cache-quality.json).
All14 arms per gate/1,662 vector pairs and1,841 first-F16 outputs/confidences
match. The initial manifest adapter failure remains red and archived; actual
hosted reanalysis rehashes3,324 preserved vectors without repeating inference.
Three pairs show only2.734% cold/0.797% warm reductions and increased RSS;
keep the256MiB cache default-off. Components pass all three native platforms;
full model evidence is bounded Linux/OvisOCR2 CPU only. The bounded
[first-FFN attribution40231922668](tests/results/qwen35-ocr/ovisocr2-first-ffn-attribution.json)
passes unchanged same-weight gates: all independent dequantization values and
original/native anchors are bit exact; local errors stay below1.669e-6. Original
Python with Q4 reproduces the0.032889 layer error, supporting precision loss at
this boundary rather than a local operator mismatch. Keep every quality rejection
and published F16 unchanged. Completed norm/position operators are in HISTORY;
corrected composed [40376882009](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40376882009)
at`8bb9f074d1ca6fa2cb2be6a892fc2fd116202912` still needs full numerical review.
Both eleven-case quality jobs pass all1,841 IDs/text/stops; gate1's maximum
confidence difference is2.4796e-5. Initial40356657327's raw/public report path
redaction failure remains retained; its comparison-only correction changes no
arrays, model, gold or budget. The interpolation gate stays default-off.
No interface result upgrades negative source quality. OvisOCR3's
source completion failure blocks its model qualification, publication and alias;
do not change its prompt, cap, gold or scorer to pass a gate.
The permissive queue, encoder follow-ups and portfolio tasks remain scoped in
PLAN and model documents. SAM work stays with
its existing owner. The app remains pinned to immutable0.17.12.

The bounded [UltraBERT RoPE audit40336758177](tests/results/ultrabert/rope-attribution.json)
at`b689f871fffe33b741061269561fc2ebb276ff97` is complete:1,344 comparisons/
336 bit controls pass, and source-table-fed native Q/K rotations are exact.
Empty position0 cannot explain earliest pre-RoPE Q or strict replay drift;
no production RoPE gate is justified. Historical183 F32/1,294 F16 failures,
strict replay failures and original0.002/0.99999 budgets remain unchanged.

Full canonical SciFact [40375626336](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40375626336)
at`e35cd1a0ba3c2dcac1fdfbded739f5663d0b8902` is pending after independently
reviewed [prepare40372423343](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40372423343).
All5,483 actual wrapper ID/mask rows match before forward;460 documents truncate
at512 tokens, none of300 queries truncate and no rows drop. Thirteen original
preflight rows are hook-neutral;2,840.06 seconds is a conservative original
shard projection, not measured full-panel runtime. Strict author-column v1 stays
rejected; canonical v2 is a distinct protocol chosen before model outputs after
exact legacy/Parquet equality and an author-difference journal. Keep all vector
and ranking gates, original-first8→freeze→native8 barrier and public-pair exclusions.

Initial asset consumer [40354251257](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40354251257)
at`49161f60`, producer`40a82158`, is negative: Windows header decoding failed
after fresh CLI/model checks and before the standalone ABI panel (explicit UTF8/MSVC repair
prepared); Apple PNG identity failed
before library/model initialization; browser execution failed unchanged parity
at0.00837269/cosine0.99864984. Initial Linux image-only proof40376735641
failed member coverage before PNG comparison; its original NOTICE is now retained.
Corrected [40386968045](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40386968045)
at`f284b1f6` now passes all50 encoded PNG hashes/manifest/dimensions/notice;
[the reviewed input proof](tests/results/packaging/posformer-corpus-proof.json)
is positive. [Mac-only40399358753](tests/results/packaging/successor-mac-only.json)
now verifies actual artifacts/RGB and all50 released17.12/default/1 IDs, text,
EOS and confidences exactly. Legacy0 and historical7/50 quality remain unchanged;
this closes initial40 Mac compatibility, while final-source archives still need
fresh full consumer qualification.
The [actual WASM trace](tests/results/packaging/successor-wasm-source-trace.json)
is independently reviewed: embedding norm exact, first captured attention0
difference, same-input activation-rounding discrepancy and unchanged final failure.
The default-off per-backend candidate is under actual
[40391544739](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40391544739)
at`580e3ae3`: [actual weight-free operators](tests/results/packaging/wasm-rounding-operators.json)
now pass all five records,158/189 outputs, backend isolation and cached plans.
The [completed model A/B](tests/results/packaging/wasm-rounding-ab-v1.json)
rejects8/10 finals despite preserved legacy/observer/context controls. Attention0
improves to1.19e-7 and layers0–2 remain below9.54e-7; first remaining divergence
is layer3 max0.00291893, then encoder_out0.02850568. Keep default off and use
same-input earliest-operator replay before another candidate. No gate is widened.
The [prepared final-integrated-v1 validator](docs/ovisocr2-final-integrated-v1.md)
pins the exact assembled runtime, references, helpers and six binaries without
weakening historical guards. Its13 small controls and actual retained quality
report reanalysis pass; pending independent-completion pins still block model
access. Source preparation is not qualification.
Final integrated source requires fresh seven native/two WASM archive producers
and all consumers. No tag, model publication, app-pin update or overall loop
closure follows from these preparations.

[Ovis3 preservation40287253703](tests/results/qwen35-ocr/ovisocr3-vector-preservation.json)
is complete and integrated: copy12635202371 retains all1,662 original payloads
and expires24 January2027. The corrected MI-GAN artifacts and RoPE raw artifact
also expire24 January2027; renew exact bytes remotely before expiry. Ovis3's
negative source quality/publication flags and all other archive deadlines remain.

This file is tracked public documentation even though an ignore pattern matches
its name. Never append private environment/access details here. Optional private
continuity is stored in ignored `.claude/local-handoff/AGENTS.md` when available.
