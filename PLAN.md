# CrispEmbed — Architecture & Roadmap

Lightweight, dependency-free text/image/audio embedding inference via ggml.
Same philosophy as CrispASR: pure C/C++, GGUF models, quantisation,
GPU-ready via ggml backends (CUDA/Metal/Vulkan), no Python at runtime.

## 🚧 Active work in flight (update + push to `main` at EVERY checkpoint)

Multiple sessions/worktrees run in parallel and push to `main` concurrently.
Before starting a task, add a row; at every checkpoint update it and push this
file to `main` so others see what's claimed (avoids duplicate work + CI-cancel
races). Remove the row when the branch lands.

| Since | Branch / worktree | Task | Status |
|-------|-------------------|------|--------|
| 2026-08-02 | `perf/ocr-h-items` / `.claude/worktrees/perf-ocr-h` | **Picked:** the OCR performance backlog H-items (H1 1x1 conv, H2 detector scalar path, H3 EasyOCR stage split, H7 wasted GPU inits). **H2's per-conv profile reframes the backlog and is the main result:** `CRISPEMBED_PPOCRV6_DET_PROFILE=1` on `german_official_print.jpg` shows **1x1 pointwise convolutions are 51.6% of all detector convolution time**, depthwise 20.4%, deconv 6.4%, everything else 21.6% (proportions stable across two runs at different box load; absolute totals are not, so read the shares). So H1 *is* the lever for H2 rather than a separate item. The single most expensive layer in the network is one 7x7 **depthwise** at 240x184 — 13.7% of all conv time at **0.17 GF/s** against ~1.2 GF/s for the pointwise layers; depthwise as a class runs 0.02-0.19 GF/s. **That is a second lever the backlog did not list.** H1: retiled the 1x1 fast path (8192-element pixel tiles so the input slab stays L2-resident, plus a 4-wide output-channel unroll so each loaded element feeds four FMAs) — the old form won only ~6% because it streamed the whole output plane once per (oc, ic) pair. **Both new kernels turned out NOT to be wins, and the measurement protocol was the real bug.** A single median-of-3 A/B said 9.1% on the M1; re-run as interleaved off/on pairs it does not replicate — Mac deltas +15.7/-1.5/-1.2/+1.9 (mean carried entirely by one outlier *baseline*; -0.3% without it, CI spans zero) and x86 -4.8% (5/6 pairs negative, CI excludes zero). So the 1x1 kernel is neutral on ARM and a ~5% **regression** on x86, and the earlier "sign flips with the instruction set" reading was one noisy Mac number against one real x86 regression. Measured noise floor: sd 8.1% (Mac) and 5.2% (VPS) per paired delta, so resolving a 5% effect needs **41** and **16** interleaved pairs respectively — median-of-3 with a control bracket cannot see effects of this size, and both controls agreed within 30% across the very pair that read +15.7%. Everything at the 3-10% level in PLAN/PERFORMANCE resting on one median-of-3 is now suspect. New `conv2d_depthwise_cpu` behind `CRISPEMBED_CONVDW_FAST` inverts the loop nest so each kernel tap is a contiguous axpy over an output row, removing the per-pixel gather that has nothing to amortise against when there is one channel per group; **not yet A/B'd**. H7: found **three more engines with the P2 bug** — `text_sr`, `tps_locnet` and `bert_ner` each build a GPU backend, use it only to pull the GGUF through `core_gguf::load_weights`, copy every weight out to host vectors and free it, without ever running a graph on it; all three now load through `ggml_backend_cpu_init()` with the old path gated (`TEXT_SR_GPU_LOAD`, `TPS_LOCNET_GPU_LOAD`, `BERT_NER_GPU_LOAD`) — **load-time win not yet measured**. H3: `CRISPEMBED_EASYOCR_STAGE_BENCH=1` splits the recognizer loop into detect/crop/set_width/recognize with width-rebuild counts — **not yet run on a page**. Both new kernels have equivalence guards in `test-core-cpu-ops` (117 passed / 0 failed) that are tolerance-based on purpose: `dot_product` accumulates through eight FMA lanes plus a horizontal add on aarch64 while the axpy forms accumulate in channel order, so an exact assertion would fail for a correct implementation. The 1x1 guard was **proven to bite** by injecting a `w1`-for-`w2` index swap — 7 of 9 shapes fail, and the 2 that do not are exactly the grouped cases where `ch_per_group_out < 4` so the unroll never runs | **IN PROGRESS** |
| 2026-08-02 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** third AI Act audit pass. Verified the round-1/2 controls hold in code (gate at `crispembed_face_init` keyed on declared model type; no 1:N primitive; no prohibited-category model in the registry; both CI-enforced), then closed six gaps the earlier passes left. (1) POLICY.md never reached PyPI/pub.dev users — `setup.py` now stages it into the wheel (verified present in the built wheel) and both package READMEs carry the Art. 5 prohibitions + GDPR Art. 9 note; (2) `/face`+`/detect` read arbitrary server-side paths — new `--image-root` confines all 30 `{"image":…}` endpoints via `weakly_canonical` + component-wise prefix, new `tests/test_image_root.py` covers absolute/traversal/symlink/sibling-prefix escapes; (3) Art. 4 AI-literacy duty was absent → new POLICY §8, and §1 now admits the project *deploys* two systems (Space, WASM demo), not only ships a component; (4) Art. 50 restated as in force (2 Aug 2026 has passed) with the no-watermark absence stated as unresolved; (5) model downloads were unverified → SHA-256 pins for all 232 pinnable registry URLs (`examples/cli/model_hashes.h`, generated by `tools/fetch_model_hashes.py` from HF LFS oids), fail-closed on mismatch/unpinned/non-HTTPS, wired into `main-health.yml`; (6) uploaded pages moved off predictable `/tmp` names to `mkstemp` 0600. **Side finding: 8 registry URLs are 404** — `pix2struct-base-q8_0.gguf` and `lid-glotlid-f16.gguf` are filename typos (repos hold `-f32`/`glotlid-f16`), and `InstructIR`/`AdaIR`/4 `*-crispembed-GGUF` repos have no GGUF uploaded. **Follow-up in the same branch: 4 of the 8 now fixed.** `instructir-f16.gguf` (quantized from the published f32; output cos 1.0, max 1 LSB vs f32) and `pix2struct-base-q8_0.gguf` (byte-identical greedy decode vs f32) were built and uploaded to `cstr/instructir-GGUF` / `cstr/pix2struct-GGUF`; pix2struct's registry size said 300 MB but the real q8_0 is 467 MB. `glotlid` had two bugs — the `lid-` prefix is CrispASR's naming convention and never existed in this repo, and "3.3 MB" was wrong by ~250x (GlotLID-V3 f16 is 848 MB) — now points at `glotlid-f16.gguf`, with `glotlid-q8`/`glotlid-q4k` added; FUNCTIONALLY UNVERIFIED because this repo has no LID engine (`text_lid_dispatch.h` is an optional CrispASR header behind `__has_include`). Repo ids normalized to canonical lowercase `cstr/instructir-GGUF` / `cstr/adair-GGUF`, which previously only resolved via HF's case redirect. Unpinned URLs 8 -> 4. The remaining 4 are the VLM repos that do not exist, with no artifact anywhere on the backup volume, so they need real reconversion (`convert-qwen2vl-to-gguf.py` for german-ocr-3.1 and nanonets-ocr2-1.5b, `convert-internvl2-to-gguf.py` for both H2OVL). **~~TODO~~ RESOLVED 2026-08-02 (`feat/ocr-followups`) — AdaIR F16 was an engine bug, not a bad artifact, and this row called it correctly:** `adair-5d-f16.gguf` quantizes cleanly but aborts at run time on the default ggml_conv path with `GGML_ASSERT(buf != NULL && "tensor buffer not set")` in ggml_backend_tensor_set, from `adair_kernel` (src/adair.cpp:692) via `adair_conv`. f32 runs fine and InstructIR f16 from the same quantizer runs fine, so it is specific to src/adair.cpp's F16 path. Root cause NOT established: the scalar fallback (ADAIR_SCALAR=1) was too slow to finish, so whether the bug is confined to the ggml path is unknown. No f16 was uploaded; the registry points at the published f32 (115 MB) instead. **VLM reconversion follow-up: 3 of the 4 missing VLMs now built, verified and uploaded; unpinned URLs 8 -> 1.** Sources found: `keyvan-ai/german-ocr-3.1` (llama.cpp split GGUFs; byte-identical mirror of `Keyven/german-ocr-3.1`), `nanonets/Nanonets-OCR2-1.5B-exp` (the `-exp` suffix is why a plain `Nanonets-OCR2-1.5B` probe 404s), `h2oai/h2ovl-mississippi-{2b,800m}`. (1) **german-ocr-3.1** — merged upstream F16 LLM + F16 mmproj via `merge-llamacpp-qwen2vl-gguf.py` (4.12 GB, matching the original b58d7805 commit note), quantized to q4_k = 1684 MB (registry said 1301; corrected). Verified: near-perfect transcription of `scan_page_pd.png`. NB merging the *pre-quantized* Q4_K_M LLM instead is wrong — it yields a 2.3 GB hybrid, not the recorded recipe. (2) **h2ovl-800m** — 644 MB q4_k (registry said 398; corrected), verified legible full-page OCR with edge-model artifacts. (3) **nanonets-ocr2-1.5b** — 1346 MB q4_k, *exactly* the size the registry already claimed, verified near-perfect full-page transcription (421 tokens). **Two converter bugs found and fixed, both silent-failure classes:** (a) `convert-internvl2-to-gguf.py` routed the LLM attention layout off `config.model_type`, so H2OVL's Danube LLMs (model_type `llama` for 800m, `mistral` for 2b) went down the InternLM2 fused-`wqkv` branch, every lookup missed, `lw()` returned False silently, and the writer emitted **only the per-layer norms** — a GGUF with no LLM weight matrices that loads fine then segfaults in `ggml_mul_mat` on a null tensor (381 tensors instead of 493). Now routed on tensor-name presence, plus a post-export guard that aborts naming the missing tensors. This means PLAN's earlier "H2OVL-Mississippi-2B **Ported**" claim was stale for BOTH H2OVL models. (b) `convert-qwen2vl-to-gguf.py` wrote `qwen2vl.tie_word_embeddings` twice when a checkpoint ships no `lm_head` (gguf raises on duplicate keys), killing conversion after the vision tower was written; now derived once from whether `lm_head.weight` is actually in the checkpoint. **TODO — h2ovl-mississippi-2b is the one still unshipped.** It converts (565 tensors, 4.42 GB) and loads, but emits degenerate output at BOTH f16 and q4_k — f16 repeats one token then EOS, q4_k emits "." then EOS — so it is not a quantization artifact. Suspects not yet separated: chat/prompt template for Danube-1.8B, Mistral sliding-window attention, or the 32H/8KV GQA ratio (the working 800m is 16H/8KV). Its registry URL still 404s and is commented as such. **Measurement warning:** a parallel session drove load average to 75-316 during this work; the first full-page nanonets run produced zero tokens in 900 s purely from CPU starvation and would have been misread as a hang. Re-run VLM timings on a quiet machine. **h2ovl-mississippi-2b root cause: `use_msac`, now IMPLEMENTED.** The 2b sets `use_msac: true`, the 800m false — the only material difference between the working and broken model (same `template: h2ogpt2`, vocab, downsample/ps_version; rope_theta differs 10000 vs 100000 but is read correctly and the engine is arch-agnostic). H2OVL's Multi-Scale Adaptive Cropping tiles the page twice: coarse grid, then a fine grid keeping only ratios where `prior_cols%c!=0 && prior_rows%r!=0` (so it is not a sub-grid), concatenated `fine[:-1] + coarse[:-1] + fine[-1:]`, thumbnail last. Single-scale tiles give a model trained on that layout fluent nonsense. Implemented in `image_preprocess::preprocess_internvl_msac_rgb` + `internvl2.use_msac` dispatch. **Two parity bugs in the existing tiler had to be fixed for the fine grid to come out right:** (a) aspect ties were broken toward FEWER tiles; upstream breaks them toward more when `area > 0.5*size*size*blocks` — for 800x800 that is 1x1 vs 2x2; (b) a pass producing one block gets no thumbnail upstream, so `[:-1]` drops the tile itself and it contributes nothing — we gave 6 tiles where upstream gives 5. `tests/test_msac_tiling.cpp` pins all five cases against values transcribed from H2OVL `image_process.py`, including 800x800 where no admissible fine grid exists and we must DECLINE rather than fall back to single-scale. Model-free. The tie-break touches every InternVL model, so h2ovl-800m was re-run end to end: same 7 tiles (3x2), same 1122-byte transcription. **The 2b GGUF is being built by `tools/kaggle/h2ovl-convert`, not locally** — 4.3 GB safetensors + 4.4 GB f16 + 1.4 GB q4_k exceeds this machine's free space and a local attempt ran the disk out mid-write. The kernel asserts `use_msac` survived into the GGUF and the LLM attention/FFN tensors exist, OCR-smoke-tests the page fixture, and refuses to upload unless MSAC tiling ran and >=200 chars came back. **Also this pass:** adair-5d back on f16 (115 -> 59 MB) now that 67ec560c fixed the runtime shape bug — f16 vs f32 cos 0.99999994, max 1 LSB; and `--adair-model`/`--instructir-model` now resolve registry names, which they never did, so those two entries were unreachable from the CLI. Touches `examples/cli/model_mgr.cpp`, `examples/server/server.cpp`, packaging, POLICY/README, CI — **no OCR/model/graph code**. Verified: SHA-256 against 4 NIST vectors incl. the 56-byte padding edge; pinned download verifies, tampered pin rejected + cache left clean, unpinned refused, override works; biometric gate 11/11 still passes; image-root 5/5 + rejection logging; `tools/format.sh` clean.  **COORDINATION (h2ovl-2b, 3 sessions):** this branch owns the CONVERT half only — `tools/kaggle/h2ovl-convert` + the MSAC runtime. `feat/ocr-followups` owns the PARITY half (`tools/kaggle/h2ovl-parity`) and `tools/kaggle/h2ovl-publish`. I nearly broke that twice and both are worth recording: (1) my kernel called `create_repo` WITHOUT `private=True` on the same repo their publish kernel creates private — `exist_ok=True` does not flip visibility, so whichever ran first decided it, and mine would have made a model that emits 29 chars of `.assistant.assist` PUBLIC. Pre-created the repo private out of band, then removed my upload path. (2) Their row says they were *waiting on convert to publish the f16*; h2ovl-publish has since landed f16+q8_0+q4_k in that private repo, so that dependency is satisfied. **MSAC is implemented and the 2b is still broken, but the suspect list is now small.** Their parity run measured 27 stages at cos_min 0.999972 vs the Python blueprint, so the ported compute is right — it is NOT the graph and NOT the MSAC tile math. Combined with my fix that the call site hardcoded min/max_dynamic_patch to 1/12 and threw away the GGUF's declared 6 (19 tiles where the reference gives 13; the 800m is unaffected, same 3x2 grid either way), the remaining suspects are prompt construction, sampling, and detokenisation. **Next, locally, not on Kaggle:** `cstr/crispembed-regression-fixtures` already has `internvl2/h2ovl-mississippi-2b/ref.gguf` (112 MB) and the private model repo has the q4_k (1.46 GB) — that pair is enough for `tests/test_internvl2_diff.cpp` on this Mac, so my kernel's own 4-layer ref-gen is redundant and should be dropped. **CAUTION on the parity claim everyone is now reasoning from.** "27 stages at cos_min 0.999972" is almost certainly VISION-ONLY: `tools/dump_internvl2_reference.py` emits exactly 27 vision-side stages (`vis_patch_embed` + `vis_layer_0..23` + `vis_pixel_unshuffle` + `vis_proj_output`), and the parity kernel runs the dumper with `--max-llm-layers 4`, which would add `llm_embed` + 4 layers = 32 if the LLM side were counted. So what is established is that the InternViT tower and the projector are right. The 24-layer Danube-2 decoder, the LM head and the logits are NOT covered — and that is exactly where a mistral-vs-llama porting bug would live, the 2b being the `mistral` one while the working 800m is `llama`. Confirm the stage list in the parity log before concluding "the compute is right, the fault is downstream of the logits"; on this reading the decoder is the prime suspect, not detokenisation. **Decoder coverage now exists** (shared tooling, not their kernel): the diff harness ran `run_llm_forward`, printed the output SHAPE, freed the buffers and compared NOTHING — and returned 0 unconditionally, so even the vision stages were advisory. Added `llm_output_norm` + `llm_logits` to `dump_internvl2_reference.py` (emitted only on a full-stack dump; tied head read from the weights, not a config flag), and the harness now compares both, counts failures, returns non-zero, and checks **argmax of the last position separately** — cosine stays high while the argmax moves, and the argmax is what generation acts on, which is exactly this failure's shape. To get decoder coverage, regenerate the fixture WITHOUT `--max-llm-layers 4`. **h2ovl-mississippi-2b WORKS — unpinned URLs 8 -> 0.** The remaining bug was the hardcoded patch limit, not MSAC and not the decoder: with the model's declared `max_dynamic_patch=6` honoured, the run tiles to **13** (the reference geometry, was 19) and returns rc=0 with 1109 chars instead of 29 chars of `.assistant.assist`. Published, validated, repo flipped private->public now that it earns it (it was private by `h2ovl-publish`'s correct call while broken), registry entry activated at the measured 1459 MB, ref.gguf alongside it. **RESOLVED — it was the invocation, and it took BOTH halves.** Compared against upstream `conversation.py`/`modeling_h2ovl_chat.py`/`tokenizer_config.json` rather than guessing. (a) `add_bos_token: false` for BOTH h2ovl checkpoints, and upstream `chat()` just calls `tokenizer(query)` — the blueprint prompt has no `<s>`; we prepended one unconditionally. (b) `"OCR this image."` is too terse for a general-purpose VLM; upstream's own examples are explicit imperatives, and `qwen2vl_ocr` had already been forced into the same change. **Neither alone works** — the 2x2 matrix is: BOS+terse -> describes; BOS+explicit -> describes; noBOS+terse -> describes; noBOS+explicit -> TRANSCRIBES. That is why one-at-a-time attempts kept failing. Full page, defaults only: 2b 1806 chars verbatim (keeps curly quotes and `dis- played` hyphenation); 800m 1749 chars, also improved — it used to emit `NEW PAGE / <- / 36 / PRIDE AND PREJUDICE.` layout artifacts. Converter emits `internvl2.tokenizer.add_bos_token`; for older GGUFs the h2ogpt2 template defaults it off. `CRISPEMBED_INTERNVL2_PROMPT` / `CRISPEMBED_INTERNVL2_ADD_BOS` added for bisecting. **Trap worth remembering:** reading `tok.h2ogpt2` before the GGUF template key was parsed made the BOS default a silent no-op; only the A/B caught it. Superseded TODO was:  it *describes* the page ("The image presents a page from a book, specifically page 36...") rather than transcribing it, while quoting the content accurately. The internvl2 engine is handing H2OVL a captioning-style prompt; it needs an OCR/transcribe instruction, as the qwen2vl engine already does. **Also landed this pass:** `--check-sizes` in `tools/fetch_model_hashes.py` (found 4 more wrong registry sizes on first run: bidirlm-omni 2.6 GB->1834 MB, ppformulanet-l 180->252, transcoda 120->69, bttr-hw 5->11; all fixed, 241 clean, wired into main-health); adair-5d back on f16 (115->59 MB) after 67ec560c; and `--adair-model`/`--instructir-model` now resolve registry names, which they never did.| **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** validate the opt-in Tesseract DAWG scorer, model-owned runtime lookup, and diagnostic beam-confidence contract after the remote recoder merge; fix prefix ranking/token boundaries, wire runtime tests, and keep production dictionary scoring/calibration disabled | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** close the remaining beam-confidence comparator gap so `--require-beam-sequence-only` rejects fabricated word certainty as well as character certainty; add model-free coverage | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** make the documented Tesseract row-blob-bounds geometry A/B reproducible through the page comparator and repeated benchmark manifests, while keeping it diagnostic-only | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** align the standalone Tesseract geometry comparator with the row-blob-bounds benchmark switch and record the policy in its JSON output | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** reconcile the stale EasyOCR-plan int-mode status with the detailed parity evidence, keeping recoder/DAWG and full-page decoded parity explicitly open | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** preserve unmapped Tesseract recoder classes as explicit `<class>` diagnostics instead of silently dropping or exposing numeric class labels; keep full composed-script parity open | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** preserve valid composed recoder segments around unmapped classes with a diagnostic partial composer; leave the default decoder and full composed-script parity gate unchanged | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** consolidate repeated CRAFT/DBNet warm-graph probes into a versioned JSON manifest with explicit reference/native timing ratios and box-count quality status; keep device mismatch and page-text parity visible. Live scan-strip manifest: CRAFT native/reference `29,511.835/11,480.765 ms` (`2.57x`) with `106=106` boxes; DBNet `44,647.873/16,153.006 ms` (`2.76x`) with native `98` boxes, reference count unavailable in the timing-only probe. | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** produce an independent EasyOCR Python page manifest for `lines` mode and compare ordering, line grouping, crop geometry, decoded text, and confidence against the native DBNet→EasyOCR handoff; keep page parity separate from detector-only timing. Live `scan_strip.png`: Python CRAFT produced 11 lines; native DBNet produced 12. The first mismatch is line 0 (`"They are going to be , encamped near   Brighton"` vs `& They are going to be, encamped near   Brighton`), with geometry `[62,0,412,25]` vs `[46.97,0,423.54,21.76]`; all subsequent records shift, so page quality parity is **not** passed. | **COMPLETED — parity failed; quality TODO** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** replay independent Python EasyOCR line boxes through the native CRNN to separate recognizer/crop parity from DBNet detector geometry; preserve the failed page gate if identical boxes still diverge. External replay now uses exact caller-supplied boxes (no native 2-pixel margin), returns 11/11 regions, and still diverges in native text/confidence (line 0 Python `"They are going to be , encamped near   Brighton"` vs native `They are going to be, encamped near   Brighton`; confidence `0.8541` vs `0.5483`; line 4 is a severe recognition failure). | **COMPLETED — recognizer/crop quality TODO** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** run fresh `crispembed-diff` on exact Python line crops before changing recognizer math. English Gen2 line 0 and the worst line pass input, features, sequence input, both BiLSTM outputs, and logits; line 0 decodes identically, while the worst line reproduces Python's own poor decode. Input cosine is `0.99981`; recurrent/logit cosines are at least `0.99972`; feature global cosine is `0.99993` (sparse per-row feature cosine is not a valid promotion gate). The remaining page discrepancy is therefore detector geometry/crop selection, recognizer asset/preprocessing identity, and Python/native postprocess confidence—not an unexplained GGML LSTM divergence. | **COMPLETED — page quality still open; no recognizer math change justified** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** make the EasyOCR manifest boundary distinguish the padded postprocess `crop` from the actual recognizer input. Python and native manifests now emit `recognizer_crop`; `compare_easyocr_manifests.py --recognizer-crop-only` checks exact model-input geometry while preserving legacy crop comparisons by default. Contract tests pass, and the rebuilt `test-easyocr-pipeline` links at `[88/88]`. A real external replay confirmed 11/11 caller regions and showed the remaining text/confidence mismatch is genuine output quality, not a mislabeled crop field. | **COMPLETED — page quality TODO remains** |
| 2026-08-01 | `feat/ocr-engine-parity` / `.claude/worktrees/feat-ocr-engine-parity` | **Picked:** end-to-end head-to-head parity (CER/WER **and** latency) of the CrispEmbed OCR lanes against system Tesseract 5.5.2, Python EasyOCR 1.7.2, and Python PaddleOCR 2.10.0. See "OCR external head-to-head" below for the harness, the reachability fixes, and the first measured gaps. Touches `examples/cli/main.cpp`, `examples/cli/model_mgr.cpp`, `src/crispembed.{h,cpp}` engine-id mapping, `src/ocr_orchestrator.{h,cpp}` (new `engine::easyocr` case only), and new `tests/` scripts — **no OCR graph/runtime math** | **IN PROGRESS** |
| 2026-07-31 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** unify CRAFT/DBNet/Tesseract-style segmentation with EasyOCR lines and LayoutLM/Tesseract words; then validate downstream OCR handoffs. Latest checkpoint: fresh Latin Gen1/Gen2 and English fixed-width references pass; only English’s actual width-128 scan retains the documented dynamic-width row-wise logits residual | **IN PROGRESS** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** add a dependency-free EasyOCR interoperability contract test covering Python `lines`/`words` ordering, crop/normalized geometry, and LayoutLM `apply_ocr=False` serialization; keep real-page reference parity as the separate live gate. `tests/test_easyocr_interop_contract.py` passes with 3 words, 2 grouped lines, and ordered LayoutLM sidecar metadata | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** retain PP-OCRv6 detector/crop/orientation/recognizer per-stage timings in the reproducible benchmark JSON; parser and stderr-capture slice. `tests/ppocrv6_pipeline_benchmark.py` now sets the bench switch, parses native stderr, preserves partial timeout telemetry, and labels unavailable stage rows. A live tiny German fixture produced detector/crop/orientation/recognizer timings and 34 detector boxes → 30 recognized results; full 10-fixture/medium quality sweep remains pending | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** add dependency-free PP-OCRv6 benchmark-parser, backend-capability, and OCR interoperability contract tests to the mandatory OCR regression smoke job; leave model/gold execution artifact-gated. Workflow YAML and all four smoke/contract checks pass locally; the gold step skips unless an artifact-equipped runner supplies `CRISPEMBED_GGUF_DIR` | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** generalize the PP-OCRv6 graph-gold harness from hard-coded small-only artifacts to explicit tiny/small/medium tier selections with tier-specific reference fixtures. The harness now supports all three tiers; tiny remains explicitly blocked until its legacy 16-tensor Arabic reference is regenerated as a full graph gold archive, while small remains the default accepted lane | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** add opt-in PP-OCRv6 detector graph-vs-CPU box geometry diagnostics without changing the production CPU accept-gate; report count, greedy matches, mean IoU, and minimum IoU for each diagnostic run. Implemented and compiled; the available tiny fox fixture reports graph=0 vs CPU=2, so detector graph geometry remains a quality/performance TODO and is not accepted by default | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** repair the manifest-driven O9 engine benchmark so structured detector specs (`{repo,file,revision}`) are normalized like recognizer specs; add a contract test and rerun Tesseract/PARSeq/PP-OCRv6 rows. Fixed structured detector normalization; Tesseract-LSTM `175.7 s` CER `0.040`, German Tesseract `101.8 s` unscored, PARSeq `1.206 s` unscored. The PP-OCRv6 artifacts load-fail (`missing stem conv`) and are now correctly marked errors instead of false `ok` rows | **COMPLETED** |
| 2026-08-01 | `chore/ai-act-hardening` / `.codex/worktrees/chore-ai-act-hardening` | **Picked:** third AI Act audit, run against `origin/main` after `f7f89032` landed. Re-verified the code-backed claims and found them all true: no 1:N/gallery primitive in the C ABI, no emotion/age/gender/ethnicity code anywhere, no scraping tooling, server gate fails closed before `crispembed_face_init()`, gate keyed on declared model type so a renamed `.gguf` is still caught, `/doc` temp uploads tracked and unlinked. Two gaps remained, both closed here: (1) the deployed GitHub Pages demo (`examples/wasm-ocr/index.html`) carried no notice at all while the HF Space had one — added an AI-output/data-locality/no-biometrics footer linking POLICY.md; (2) POLICY §3 and README asserted the absence of biometric-categorisation *models* in terms a reader could mistake for a guarantee about capability — CLIP/SigLIP zero-shot means the caller supplies the classifier, now stated. Also strengthened §7: Art. 53 does not engage for task-specific models at all (the quantization argument is the fallback), and Art. 53(2) does not waive the copyright policy or training-data summary. Docs/HTML only — no C/C++, no rebuild needed | **COMPLETED** |
| 2026-08-01 | `chore/ai-act-audit-followups` / `.codex/worktrees/chore-ai-act-audit-followups` | **Picked:** close the five gaps a second AI Act audit found in the `chore/ai-act-policy` work. (1) biometric gate moved into `crispembed_face_init()` so the Python/Rust/Dart bindings are covered, not just CLI+server — new ABI `crispembed_accept_biometric_use()`; (2) `check_registry_licenses.py` read only HF's `license` tag and missed `license_name`, so the 4 correct lfm2 rows failed — fixed, now exit 0, and wired into `main-health.yml`; (3)(4)(5) POLICY.md: Art. 50(2) reframed as reasoned-position-not-settled-exemption, OCR-VLM text addressed, and the regulatory dates corrected — the Omnibus is **Reg (EU) 2026/1744, OJ 24 Jul 2026**, not "adopted June 2026". Touches `src/crispembed.{h,cpp}`, `examples/cli/model_mgr.*`, bindings, POLICY/README/PLAN, `tests/check_registry_licenses.py` — **no OCR/model/graph code**. Verified: CLI + Python both refuse a recognition model without acknowledgement and load it with one, byte-identical embeddings either way; licence check exits 0; `tools/format.sh` clean. | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** fix O9 pipeline benchmark routing to use each manifest entry’s engine family (`ppocrv6`) instead of the tiered display name (`ppocrv6-tiny`), which had sent PP-OCRv6 fixtures through generic DB postprocessing and produced false `missing stem conv` load failures. Rebuilt CLI and verified tiny `4.98 s`/2 regions, small `20.82 s`/2 regions; medium exceeded the `120 s` guard and is recorded as a timeout, not a quality pass | **COMPLETED** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** resolve the remaining official PP-OCRv6 quality discrepancy by testing the HF/PaddleX preprocessing contract (RGB/BGR, resize, normalization) and CTC decode on known-text crops; promote a runtime change only if native output diverges from the official source under the same input. Result: preprocessing is aligned; remaining issue is checkpoint/vocabulary/line-crop quality | **COMPLETED** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** validate PP-OCRv6 checkpoint provenance, CTC vocabulary selection, and line-crop suitability against known-text crops before claiming quality parity; preserve native/reference decoded strings and timing evidence. Root cause was confirmed upstream: 320 is a minimum width, not a cap; the native/reference path now preserves dynamic CTC width and the 18,710-class space vocabulary | **COMPLETED** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** make the static PP-OCRv6 recognizer graph safe with dynamic-width line crops; bypass the fixed 320-wide graph for wider crops and retain the CPU reference path until a dynamic-shape graph is implemented and benchmarked. Live graph-debug test on an 800×100 fox line reports `input elements=55296` and cleanly bypasses the 320-wide graph; CPU decodes `The quick brown fox jumps` | **COMPLETED** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** implement and benchmark a truly dynamic-width PP-OCRv6 GGML graph, with width-keyed graph/cache ownership and CPU/Metal parity; keep the current explicit CPU fallback as the acceptance baseline. Implemented width-keyed graph rebuilds that retain the loaded GGUF source weights; a single process now runs 320-wide and 384-wide crops with graph outputs `80x3x384` and `96x3x384`, and graph-accepted text matches CPU (`De t 4 dg 14` / `The quick brown fox jumps`) | **COMPLETED — CPU graph validated; Metal parity pending** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** run the width-keyed recognizer graph on Metal for 320/384-wide crops, compare graph-accepted output and stage logits to CPU, and retain CPU fallback for any backend-specific divergence. Metal `MTL0` builds both widths and graph-accepted text matches CPU (`De t 4 dg 14` / `The quick brown fox jumps`); no dynamic gold archive exists yet for stage-logit comparison. Two-crop cold process timing was `28.32 s` Metal versus `2.94 s` CPU, so Metal is currently slower due pipeline compilation and remains diagnostic-only | **COMPLETED — parity passes; performance TODO** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** add reusable Metal pipeline/cache timing and dynamic-width gold logits, then benchmark warm 320/384-width recognizer graphs; do not promote Metal acceptance until cold/warm cost and numerical parity are recorded. Fixed repeated Metal scheduler reuse: re-plan Metal buffers per invocation while CPU retains allocation reuse. Same-width repeated Metal now exits 0 with identical text (`19.78 s`/2 crops); alternating 320/384/320/384 also exits 0 with identical text (`21.57 s`/4 crops). Dynamic stage-logit gold remains pending | **COMPLETED — stability fixed; performance/logit TODO** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** reduce Metal dynamic-width overhead by caching compiled graph plans per width or batching same-width crops; add width-specific gold logits before any Metal acceptance promotion. Added `tests/ppocrv6_width_benchmark.py`, which preserves decoded strings and graph shapes for grouped/alternating runs. Current `n=1` Metal timings: short `2582.1 ms`, wide `2250.0 ms`, alternating pair `2780.6 ms`; all return 0 with exact CPU-matching text and MTL0 shapes `80x3x384`/`96x3x384`. CPU controls are `468.3`/`403.8`/`463.8 ms`. | **COMPLETED — benchmark harness; cache/logit TODO** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** generate width-specific official-source activation golds and add CPU/Metal stage-logit comparison to the width benchmark; quantify whether current Metal numerical drift is acceptable before enabling any production graph gate. `tests/ppocrv6_width_benchmark.py` now accepts separate short/wide references and reports per-stage cosines. Fresh golds pass CPU logits cosine `0.999892` at 320 width and `0.999993` at 384; Metal passes `0.999861` and `0.999993`; decoded text is identical in every case. The older 320 archive was stale and was regenerated from the corrected official mirror | **COMPLETED — numerical parity passes; Metal remains opt-in for cost** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** benchmark Metal graph acceptance on the full PP-OCRv6 line/page route with dynamic-width gold coverage, then decide whether the recognizer graph can leave diagnostic-only mode. Added `--recognizer-graph` to `tests/ppocrv6_pipeline_benchmark.py`. The isolated 384-wide gold lane passes (`logits cos=0.999993`), but the German CC0 full route with 33 regions exceeded the 120 s guard under Metal graph acceptance; the prior CPU-accepted route completes in about 20.8 s. Keep recognizer graph acceptance diagnostic-only; full-route batching/residency is required before promotion | **COMPLETED — promotion rejected on measured cost** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** reduce full-page Metal graph cost by batching same-width line crops or reusing width-keyed graph residency across the detector→crop→recognizer loop; compare full-route text and stage timings against the CPU-accepted baseline. Before batching, added a safe per-page graph budget: with 33 detected regions the explicit graph request now selects CPU fallback and completes instead of timing out; measured German CC0 route `38.55 s`, 33/33 results, 1,146 chars, with recognize `32.65 s`. | **COMPLETED — safe fallback; batching still required for speed** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** batch same-width PP-OCRv6 line crops in the full route, preserving original order and per-line dynamic widths; compare batched CPU/Metal logits and decoded text against the current scalar fallback. Added native width-distribution telemetry and JSON capture. The current orchestrator still calls the recognizer one crop at a time; a German 33-region live run remained over the 180 s graph-debug guard, so no batching claim is made and graph acceptance stays budgeted/diagnostic-only | **COMPLETED — instrumentation and safety decision; no batch API yet** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** add a real PP-OCRv6 recognizer batch API grouped by identical dynamic model width, retain original result ordering, and require CPU-vs-Metal per-stage logits plus decoded-text parity before enabling it in the full route. Added the C ABI batch contract and wired the detector→crop→orientation→recognizer route through it. Live small-rec two-crop contract (fox + receipt) completed both items with byte-identical scalar/batch text; CPU sample was scalar `9.564 s`, grouped batch `6.070 s` (`1.58x`, warm-cache/small-sample evidence only) | **COMPLETED — safe grouped API; fused graph still required** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** implement fused GGML batch dimensions for same-width PP-OCRv6 crops, with bounded batch size, per-item error isolation, CPU/Metal logits cosine gates, and full-route German CC0 benchmark comparison. Added a bounded batch dimension to the tiny logits graph and kept it behind `CRISPEMBED_PPOCRV6_BATCH_GRAPH`. Important correction: the first `52.5 ms`/`35.1 ms` CPU smoke had `CRISPEMBED_PPOCRV6_FORCE_CPU`, which intentionally disables graphs, so it proved grouped scalar parity, not fused graph execution. A real Metal fused probe exposed a GGML pooling shape assertion; Metal is explicitly forced back to grouped scalar execution and no GPU promotion is claimed | **COMPLETED — safe gate; fused CPU proof still pending** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** make fused batching Metal-safe by preserving per-item spatial dimensions through pooling/reshape and adding CPU-vs-Metal logits cosine checks; then extend the fused path to large-stem SVTR only after the tiny lane is stable. Added an explicit Metal capability gate and fallback telemetry. German CC0 full route remains complete and text-bearing (`33/33`, `1,146` chars); current run was `68.75 s` total (`46.45 s` recognition), with `22` unique dynamic widths, so no same-width page batch gain is claimed | **COMPLETED — safe gate and baseline; shape rework remains** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** rework the tiny fused graph around an explicit per-item branch/sequence dimension that survives pooling, permutation, and CTC flattening on Metal; add a two-crop gold-logit cosine contract before considering any Metal batch execution. Keep `CRISPEMBED_PPOCRV6_BATCH_GRAPH` CPU-only until that contract passes | **IN PROGRESS** |
| 2026-08-02 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Picked:** the orphaned AdaIR F16 TODO from `feat/tesseract-kernel-opt` (that branch merged into `main` at `ee099eb0` and is gone; the item was left `IN PROGRESS`). Root cause identified before any edit: `tools/quantize.cpp` (~line 167) flattens every 4-D F32 conv weight to 2-D `[IC*KH*KW, OC]` in the output header, and `src/adair.cpp` infers three hidden dims from `->ne[3]`, which is `1` on a flattened tensor. Confirmed against the artifacts — `net.decoder_level1.0.ffn.project_in.weight` is `[1,1,96,510]` in `adair-5d-f32.gguf` and `[96,510]` in both `adair-5d-f16.gguf` and the rebuild. `hidden=1` ⇒ `half = hidden/2 = 0` ⇒ a conv with `ic=0` ⇒ the zero-size kernel descriptor the earlier audit saw. **No OCR/perf overlap — does not touch H1–H8 or any paused codex branch.** **Fixed:** `conv1x1_out_channels()` derives OC from `ggml_nelements(t)/ic`, correct under both layouts, with fail-loud guards at all three sites; `ADAIR_LEGACY_NE3_DIMS=1` keeps the old read so both arms are in one binary. Measured on `adair-ref.gguf` (64×64), same binary: f32 `cos 0.999382 / max_abs 0.027892` (reproduces the audit exactly ⇒ regression control), `adair-5d-f16.gguf` **`0.999383 / 0.027871`**, and the `adair-5d-f16-rebuilt.gguf` quantizer rebuild the audit also blamed gives the **identical** `0.999383 / 0.027871` — so neither artifact was ever bad. Independent artifact check: 60 sampled tensors f16-vs-f32 worst cosine `0.999998`, worst max_abs `1.22e-4`, i.e. pure F16 rounding. End-to-end through the real CLI (not just the diff harness): a 96×96 restore returns rc=0 on both models with the outputs agreeing at cosine `1.0`, max_abs `1/255`. **No timings claimed** — load average was 55–127 from parallel agents all session and the 64×64 fixture took `312 s` at f32 against a `2.65 s` quiet-box reference. Registry still ships f32 on purpose: repointing needs the f16 uploaded to `cstr/adair-GGUF` + a SHA-256 pin in `model_hashes.h`, which is an outward-facing step left for the owner. Exposure is narrower than "f16": only `tools/quantize.cpp` output flattens — a converter-emitted f16 keeps 4-D shapes (`surya-det-f16.gguf` has 79 genuinely 4-D F16 tensors), so the *producer* predicts the layout, not the precision. Follow-up recorded, not blind-fixed: `src/surya_det.cpp:700` and `src/tps_locnet.cpp:219` read conv OC off `ne[3]` the same way and would misread a quantizer-produced artifact (`src/cnn_embed.cpp:148` is the both-layouts precedent); neither ships one today and neither is verifiable on this box. | **COMPLETED — runtime fixed; f16 upload/registry pending owner** |
| 2026-08-02 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Picked:** settle the `ne[3]` conv-output-channel follow-up left by the AdaIR F16 fix instead of leaving it as speculation. `surya-det-f16.gguf` is only 77 MB, so the claim IS testable here: run it through `crispembed-quantize` to produce the flattened layout and A/B surya detection against the converter-made 4-D artifact. Fix `src/surya_det.cpp:700` only if the test actually breaks. **Not** h2ovl-mississippi-2b — that needs ~9–10 GB for source + conversion and both volumes are full (internal 6.7 GB free, backups 7.8 GB), so it is disk-blocked, not skipped. **Result — the two suspects split apart.** (1) The flatten fires only on 4-D **F32**: quantizing `surya-det-f16.gguf` (4-D F16) to q8_0 leaves all 79 4-D tensors intact, so precision alone does not predict exposure — source dtype + producer does. (2) **`src/surya_det.cpp:700` is NOT a bug** — `g_conv` reshapes a 2-D weight to 4-D *before* that read, so my earlier 'same bug class' note was wrong. No change. (3) **`src/tps_locnet.cpp:219` WAS real and is fixed** — it reads `ne[3]` at load with no normalisation and `convert-tps-loc-to-gguf.py` **defaults to F32**, so a quantized tps-loc GGUF hits it; instrumented pre-fix the four layers loaded `ndims=2, channels=1` instead of `16/32/64/128`, and `channels` feeds the fc1 input width and per-layer output channels. Fixed with the `cnn_embed.cpp:148` convention. New hermetic guard in `tests/test_tps_locnet.cpp` (no model file) compares 4-D vs flattened builds of the same fixed-seed weights: worst control-point deviation `0.026871 px` → `0.000000 px`, suite 14/15 → 15/15. **The guard's first version passed against the broken code** — the synthetic `fc2.weight` was all zeros so the output was `fc2.bias` alone and never touched the conv stack; `fc2.weight` now carries small non-zero values. Recorded because a green new guard means nothing until it has been seen to fail. **Sweep extension — two quantized SR/denoise models turned out to have never been run at all, and both aborted.** `esrgan` (`GGML_ASSERT(cgraph->n_nodes < cgraph->size)`): not a layout bug — `esrgan_prep_conv` reshapes correctly — but the graph budget. Measured 18-conv x4 at 64x32: f32 builds `283` nodes vs the `n_convs*12+100 = 316` budget, quantized builds `335` (dequant cast + `ggml_cont` add ~3 nodes/conv) and overflows by 19; budget now `n_convs*16+128`. q8_0 vs f32 cosine `0.999998`/PSNR `51.89 dB`; **q4_k runs but degrades hard (`29.55 dB`, max_abs `91/255`) — q8_0 is the usable quant.** `scunet` (`GGML_ASSERT(a->ne[2] == b->ne[2])`): this one IS the flatten — the persistent kernel cache copies source `ne` verbatim, so a flattened weight caches as `[K*K*IC, OC, 1, 1]`; `scunet_run_conv` now restores the shape from call-site dims, with the conventions **measured** on the working f32 path (plain `[kw,kh,ic,oc]`, transpose `[kw,kh,oc,ic]`) rather than assumed. q8_0 vs f32 cosine `0.999999`/PSNR `60.54 dB`. `pan`/`swinir`/`tbsrn` clean both ways. Regression control: f32 output **byte-identical** before/after both patches. Suites after: tps-locnet 15/15, tps-warp 19/19, core-cpu-ops 118/118. | **COMPLETED** |
| 2026-08-02 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **h2ovl-2b ROOT-CAUSED — prompt template, not the graph.** ⚠️ **TO THE SESSION OWNING `h2ovl-2b-convert`: two versions (#4 and #5) are RUNNING SIMULTANEOUSLY.** kaggle_usage #25 — a re-push stacks a session, it does not cancel; `yes | kaggle kernels delete <slug>` is the only stop. Both are burning the shared 30 h/week chr1str quota. **Good news: you do not need them.** `main` already carries the fix, so your v4/v5 clones picked it up — their log shows `chat_template: h2ogpt2`, which is my converter change. **Evidence chain:** `h2ovl-2b-parity` baked a `-ref.gguf` from the Python blueprint and `test-internvl2-diff` gave **27 stages at cos_min `0.999972`** on the f16 — the ported compute is correct — while the same build read a full page as **29 chars**. Correct compute + wrong output = the harness-blind zone (HARD RULE #3b). **Cause:** h2ovl declares `template: h2ogpt2` (`<|prompt|>…<|end|><|answer|>`, eos `<|end|>`=32009, `generation_config.eos_token_id=[2,32009]`), but `build_prompt()` emitted InternVL2 ChatML unconditionally — and this vocab has **no `<|im_start|>`/`<|im_end|>` at all**, so `add_special(-1)` dropped every role marker and the model got an unmarked prompt it never saw. Fixed in `5f617351`: template dispatch (explicit `internvl2.chat_template` from the converter, with vocab inference so pre-existing artifacts still work) + a real multi-token stop set. **All three original suspects are now eliminated** — MSAC (convert logs `19 tiles (3x2 grid, 448px)` and output was still degenerate), Mistral SWA (`sliding_window: null` in config) and the 32H/8KV GQA ratio (inside the 27 passing stages). **Artifacts:** `cstr/crispembed-regression-fixtures` (new) holds `internvl2/h2ovl-mississippi-2b/ref.gguf`; `cstr/h2ovl-mississippi-2b-crispembed-GGUF` is **PRIVATE + carded UNVALIDATED** with f16/q8_0/q4_k for the local debug loop — **do not add a registry entry or SHA pin until the decoded-output gate passes.** **Verified so far:** the InternVL2 ChatML path does not regress (internvl2-1b-q4_k still OCRs the fox line, inference correctly does not fire). **NOT yet verified:** the h2ovl half end-to-end — downloading q4_k + ref now to run both gates locally. | **IN PROGRESS — fix pushed, local verification running** |
| 2026-08-02 | `chore/ai-act-audit-fixes` / `.codex/worktrees/chore-ai-act-audit-fixes` | **Picked:** fourth AI Act audit, run against `main` at `52172c10`. Re-verified the code-backed claims and this time **executed** the gate against real GGUFs (yunet + sface pulled from `cstr/*`) rather than trusting that the test exists: all 8 prior cases pass, including the renamed-model case. Also verified the registry is clean of emotion/age/gender/ethnicity models (555 entries), that no training code exists (so §7's "quantization only" argument holds), and — since Reg (EU) 2026/1744 postdates the assistant's knowledge cutoff — checked POLICY's whole date table against the EUR-Lex text: **accurate**, including 2 Dec 2026 for the new Art. 5(1)(ba)/(bb) NCII/CSAM prohibitions. Three gaps found and closed here: (1) `--dim` returned *before* the gate on both CLI paths that reach it, making the CLI laxer than `crispembed_face_init()` — all three CLI sites now share one `cnn_biometric_ok()` helper keyed on declared type (which defaults to `recognition`, so it fails closed), and the gate test grew 3 cases; (2) POLICY §4 claimed "both gates key off declared type" while the `--face-pipeline` gate was unconditional-before-load — that path is now type-keyed too, so the sentence is true as written; (3) neither POLICY nor README told deployers that the server acknowledgement is **once per process** with **no authentication** and server-side-path input — documented in both, plus a startup warning when a recognition model is loaded on a non-loopback bind. Touches `examples/cli/main.cpp`, `examples/server/server.cpp`, `tests/test_biometric_gate.py`, POLICY/README/PLAN — **no OCR/model/graph code**. Gap (1) was mis-placed from the start, not a regression: `git log -L` shows the gate was added *below* the pre-existing `--dim` early-return in `6d87d6bd`. Verified before/after with the same toolchain — a CLI built from `HEAD:examples/cli/main.cpp` prints `128` (sface's template width) unacknowledged on **both** paths; the patched CLI refuses both. Gate test now 11/11 with real yunet+sface GGUFs, server warning fires on `--host 0.0.0.0` and stays silent on loopback, `--face-pipeline` still refuses without ack and runs with it, text embed + `--dim` unaffected, `format.sh --check` and `check_registry_licenses.py` clean | **COMPLETED** |

EasyOCR cross-check benchmark checkpoint (10 repeated recognitions, identical image/
width; native Metal versus Miniconda PyTorch CPU reference): Latin Gen2 formula
200 `16.523/12.460 ms`, scan 128 `10.885/7.137 ms`, Latin Gen1 scan 128
`154.082/78.648 ms`, English Gen2 scan 200 `16.536/10.035 ms`, and scan 128
`10.697/7.287 ms` (native/reference totals). Outputs match in every case:
`x=0442`, `82`, `==#`, `032`, and `@32`; the English width-128 strict
row-wise logits gate is still open despite decoded parity. Native is slower in
all measured graph/total paths, so graph/kernel and width optimization are
performance TODOs. CRAFT, DBNet page modes, and Tesseract still need equivalent
timing/output manifests; their existing parity checks are not performance
acceptance evidence.

Tesseract confidence checkpoint (2026-08-01): after rebuilding
`test-confidence`, the explicit `/opt/homebrew/share/tessdata` Fraktur PSM 7
comparison on `scan_strip.png` produced official `iE` (mean word confidence
`0.043433`), native greedy `BEEES` (word confidence `0.884625`), and native
beam-8 `BEEES` (sequence confidence `0.644788`, no fabricated character
confidences). Official-word validity passed, but decoded text and confidence
calibration failed; recoder/DAWG and page/line confidence aggregation remain
quality TODOs. Timings were official `5881 ms`, native greedy `305 ms`, and
beam `984 ms`; these whole-process values are diagnostic only.

**PP-OCRv6 checkpoint/line-crop follow-up (2026-08-02).** The official-source
reference and native f16 agree on the decoded garbage for the CC0 line/page
fixtures: Arabic printed line `¿づE₆¿づLyi` and receipt `上批业/|` (the German
document is likewise not a valid line-crop quality gate). The 18,710-entry CTC
vocabulary is present and ends with the expected space token, so this is not a
missing-vocabulary load failure. Keep quality blocked until the exact official
checkpoint provenance and a known-good PP-OCRv6 line-crop sample are verified;
do not improve CER by silently switching to a different model family.

**PP-OCRv6 official preprocessing A/B (2026-08-02).** Audited the official
PaddleX/HF `preprocessor_config.json` and PaddleOCR `inference.yml` against the
native path. Both use 48-pixel height, pixel-center bilinear resize, right
padding to width 320, and `(pixel/255 - 0.5) / 0.5`; PaddleOCR's shipped
inference contract is BGR while the HF processor advertises RGB conversion.
Fresh official-source runs on `fox.png` decoded `上ai` in both channel modes;
the native f16 recognizer also decoded `上ai`. On the German uneven-illumination
fixture, the official source decoded `澳臻肉企NM` (BGR) and `澳臻肉門企NM`
(RGB), while native BGR decoded `澳臻肉企NM`; this confirms the runtime is
aligned to the BGR official inference path, not suffering a hidden channel or
normalization mismatch. Added `CRISPEMBED_PPOCRV6_RGB=1` as a diagnostic-only
native A/B switch; production remains BGR. The actual text is still nonsense
in both native and official outputs, so this is a checkpoint/model-quality
problem, not a runtime-quality fix. **IN PROGRESS:** validate checkpoint
provenance/vocabulary and line-crop suitability before any model promotion.

Tesseract decoder groundwork (2026-08-01): the converter now preserves every
present DAWG component from `.traineddata` as base64 GGUF metadata with a
component SHA-256 digest. This supplies the missing serialized source for a
future native dictionary scorer; it does not claim DAWG traversal, recoder
beam, or output parity, and existing GGUFs must be regenerated before they can
use the metadata.

The native loader now reads the DAWG manifest, validates that every listed
component has a nonempty payload, and reports the loaded count in its model
diagnostic. Older GGUFs remain compatible with zero DAWG entries. This closes
metadata integrity only; DAWG traversal/scoring and decoded-output parity are
still open.

The loader now also structurally validates each base64 SquishedDawg payload:
magic/header, dimensions, edge-array bounds, next-node bounds, and forward-edge
run termination. `test-tesseract-dawg` passes and the regenerated English
smoke model loads with `dawg=3`; no dictionary score is applied and no
production beam behavior changed.

The DAWG diagnostic layer now also supports exact-word membership lookup over
unichar-ID sequences. The minimal fixture covers a positive and negative
lookup; this is traversal infrastructure only, with no score or production
decoder behavior change.

DAWG metadata decoding is now strict about quartet length, padding placement,
and unused padding bits, with malformed-input coverage in
`test-tesseract-dawg`; corrupt payloads are rejected before traversal.

The DAWG layer now exposes a reusable parsed context that caches the decoded
edge array and bit masks for repeated exact/prefix queries. This is preparation
for beam integration only; production OCR still does not consult dictionary
state.

The native Tesseract API now exposes exact-word and legal-prefix queries against
the model-owned cached DAWG contexts by component name. Missing components and
null contexts fail closed; production decoding still does not invoke them.

The API also provides a tri-state result: invalid prefix, legal non-terminal
prefix, or complete word. This is a scorer-facing contract only; no dictionary
state changes OCR output yet.

An opt-in `CRISPEMBED_TESSERACT_DAWG_PREFIX` filter now applies cached system
DAWG prefix legality inside the recoder beam after a prefix fully composes to
unichar IDs. Incomplete recoder codes remain open; default decoding is
unchanged and official parity is still pending.

The same filter A/B on the English smoke model passed 37/37 in both modes,
decoded `Se` in both, and produced sequence confidence `0.562293` in both.
This confirms safe operation only; dictionary quality parity remains open.

Native Tesseract model loading now constructs one cached parsed DAWG context per
manifest component and frees them with the recognizer. The cached dictionaries
remain diagnostic-only; production OCR does not apply their state.

The diagnostic DAWG layer now distinguishes exact-word membership from legal
prefix membership. Its fixture covers `1` as a legal non-terminal prefix and
`1,2` as the complete word; neither path is used by production OCR yet.

CRAFT's old folded-F16 diff printed error statistics: the earliest divergent
stage was `basenet_0` (`max_abs=1.52823`, RMS `0.195515`, global cosine
`0.995623`), which propagated to score-map `max_abs=0.06910`, RMS `0.008026`,
global cosine `0.999716` and changed the threshold-sensitive decoded box count
from Python's 106 to native's 107. Re-converting with runtime BN (raw
convolution weights plus explicit BN scale/shift) makes F32 match to
floating-point noise; runtime-BN F16 reaches score-map global cosine
`0.9999999` and 106 boxes. The CPU-forced and Metal outputs are byte-identical.
The folded artifact is stale; threshold tuning is not the fix.

Detector benchmark audit: the fresh CRAFT reference for `scan_strip.png` uses a
288x544 canvas and decodes 106 boxes; runtime-BN F32/F16 native runs decode 106,
so CRAFT box parity passes. Native diff runtime was ~2.34 s;
the Python dump was ~9.13 s including model load and serializing 84 tensors, not
an inference-only comparison. DBNet native page smoke measured 6.63 s in line
mode (12 units, 1.34 s summed recognizer work) and 6.67 s in word mode (98
units, 2.50 s summed recognizer work). The restored official DBNet checkpoint
now has tensor parity evidence below; native's 98-word segmentation remains
not comparable to Tesseract's 106-word segmentation. These are explicit
quality/performance TODOs.

Fresh DBNet reference checkpoint parity is now available: the official MMOCR
IC15 ResNet-18 checkpoint was restored and dumped with Miniconda Python on the
same 736x1472 preprocessed `scan_strip.png`. Native F16 passes the final
probability-map boundary (`max_abs=0.00154233`, RMS `0.00008044`, cosine
`0.9999974`, global `1.0000000`) and decodes 96 regions. Q4_K decodes the same
96 regions but fails tensor parity (`cosine=0.9311001`, global `0.9986384`),
so its prior README parity claim is stale for this reference. The DBNet diff
harness now retains every backbone, lateral, smooth, fused, head, and final
probability-map tap; F16 passes all of them. Q4_K's earliest divergence is
already at
`backbone_stage_0` (global cosine `0.9960006`, RMS `0.07697`), and it worsens
through the neck to final-map cosine `0.9311001`; this is a quantization
quality TODO, not a postprocessing issue. The detector now uses a shape-keyed
persistent GGML graph, with diagnostic tap retention enabled only by
`OCR_DETECT_CAPTURE_TAPS=1`. The fresh native CPU-forced page benchmark reports
detector graph `4178.6 ms`, postprocess `8.3 ms`, total `4186.9 ms`, and 12 line
units. The Miniconda reference uses 4 compute threads and 8 interop threads.
Corrected rapid-mode repeated benchmarking gives native CPU `5661.1 ms` warm
with 4 threads (`4.67x` slower), `2907.2 ms` warm with 8 threads (`2.40x`),
versus `1213.450 ms` reference. The same Python blueprint on MPS averages
`577.342 ms`; native Metal `3499.4 ms` is therefore `6.06x` slower on the
same device. Both native backends pass all taps in diff mode and retain
readable output; native CPU/Metal convolution and deconvolution kernels remain
mandatory optimization TODOs.

An opt-in `OCR_DETECT_DIRECT_CONV=1` experiment was investigated against the
GGML direct-convolution op. CPU requires F32 kernels, and the resulting direct
graph did not finish a diff run within roughly two minutes; it remains
disabled and is not parity/performance evidence. The default persistent
im2col path is unchanged. A vectorized direct kernel is a future performance
item. A later 8-thread baseline run was resource-contended (44.1 s cold,
66.7 s warm) and is excluded from the stable benchmark ratios.
An initial per-tap prefix-graph profiler was discarded because shared arena
allocation changed the restored output to zero boxes. No stage timing from that
experiment is evidence; profiling needs an isolated allocator before use.

CRAFT repeated inference benchmark: after one warm-up, 10 runs on the captured
288x544 `scan_strip.png` input averaged `396.027 ms` for Miniconda PyTorch CPU
and `850.018 ms` for native runtime-BN F16 Metal graph compute, with 106 boxes
from both (`2.15x` native/reference directional slowdown). CRAFT quality is
on par on this fixture; its graph/kernel path remains a performance TODO.

The old folded-F16 CRAFT taps showed error accumulation through the VGG. The
runtime-BN conversion removes that divergence: F32 captured taps match to
floating-point noise, and F16 remains within the accepted global gate. The
CPU-forced and Metal runtime-BN runs are byte-identical, including taps, score
map, and 106-box decoded result. Remaining CRAFT work is repeated inference
benchmarking, not postprocessing threshold tuning.
| 2026-07-31 | `main` | External document-parser-informed OCR pipeline: structured routing, in-memory handoffs, service contracts, batching, and benchmark gates | **IN PROGRESS** |
| 2026-07-31 | `main` | Real-world public-domain OCR corpus and manifest-driven multi-engine live benchmarks | **IN PROGRESS** |
| 2026-08-01 | `feat/tesseract-fraktur` / `CrispEmbed-tesseract-fraktur` worktree | **Picked:** validate Tesseract beam/sequence confidence against official line/page outputs; improve gated blob→row segmentation while preserving DBNet as default; optimize the recognizer precision frontier with reproducible mixed-precision GGUF candidates | **IN PROGRESS** |
| 2026-08-02 | `feat/tesseract-kernel-opt` / `.codex/worktrees/feat-tesseract-kernel-opt` | **Picked:** optimize the cached Tesseract int-mode LSTM kernel and immutable-weight reuse; preserve the exact seeded-output contract, benchmark warm recognition against official/native baselines, and keep the precision fallback gated until parity holds | **COMPLETED** |
| 2026-08-02 | `feat/tesseract-kernel-opt` / `.codex/worktrees/feat-tesseract-kernel-opt` | **Picked:** reuse per-LSTM temporary vectors across sequential line recognitions; retain isolated per-context ownership, exact cached/uncached output parity, and the existing diagnostic gates | **COMPLETED** |
| 2026-08-02 | `feat/tesseract-kernel-opt` / `.codex/worktrees/feat-tesseract-kernel-opt` | **Picked:** reproduce the AdaIR F16 ggml buffer assertion from the registry audit, repair only the F16 backend path if the root cause is local, and retain F32/scalar fallback coverage. Added allocation guards (no backend assert) and fixed `DequantCache` identity for backend-resident tensors. F32 remains cos 0.999382 / 2.65 s; scalar F32 remains cos 0.999379. F16 and a fresh F32→F16 rebuild now exit cleanly but remain cos 0.729509 / max_abs 0.707725, so F16 is not shippable. Diagnostics showed the CPU buffer allocator reports an allocation size of zero for the F16 kernel descriptor; explicitly selecting or manually allocating the CPU buffer did not change the result and was reverted. F16 metadata also flattens representative 4-D weights (`[3,3,3,48]` → `[27,48]`, `[1,1,48,144]` → `[48,144]`) while mixing F32/F16 tensors. ~~**TODO:** fix or reject this degenerate F16 descriptor/loader path, then rerun tensor-level and end-to-end parity before any upload.~~ **Closed 2026-08-02 on `feat/ocr-followups`** — the "allocation size of zero" was a downstream symptom, not an allocator fault. The flattened metadata this row already noted *is* the cause: `src/adair.cpp` read three hidden widths off `ne[3]`, which is `1` once flattened, collapsing the GDFN width to 1 and building the next conv with `ic=0`. f16 is now cos `0.999383` and the rebuild is identical, so neither artifact was degenerate. | **COMPLETED (superseded by `feat/ocr-followups`)** |

Mixed-precision checkpoint: the old Q8 artifact lacked `sample_iteration`.
Fresh F32 conversion reaches 9/9 stages with logits cosine `0.993819`; a
seed-preserving mixed Q8/F32 candidate reaches 9/9 and `0.994876`. Both decoded
outputs still differ from Python, so mixed precision is not promoted. The
blueprint now explicitly models native row-wise int8 FC arithmetic.

Backup audit found 46 Tesseract model GGUFs, but 45 lack
`tesseract_lstm.sample_iteration`; only the explicitly named Homebrew English
artifact has the seed. Missing-seed language and quantized variants require
regeneration or verified metadata reconstruction before parity acceptance.
Fresh F32 Fraktur conversion now passes all 9 stages exactly (logits max error
`2.09e-7`) and decoded text matches Python after the blueprint adopted int8
LSTM row arithmetic. The seed-preserving mixed Q8/F32 candidate still fails at
logits cosine `0.989655` and decoded output, so quantization—not preprocessing
or graph topology—is the remaining quality blocker.
| 2026-07-31 | `feat/ppocr-next-20260731` | O10.1 live preprocessor benchmark harness: raw/cleanup/binarize outcome rows on CC0/German fixtures | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O9/O10 reproducible PP-OCRv6 tiny/small/medium benchmark JSON wrapper for the 10-fixture detector/orientation/recognizer sweep; tiny/small live sweeps validated, medium first fixture passes in 125.34 s (full sweep still exceeds the 900 s guard and remains pending). Fresh routed fox benchmark after engine-name fix: CPU tiny `4.98 s`, small `20.82 s`; Metal tiny `24.52 s`, small `34.28 s`; medium timed out at `120.6 s` on both routes because the medium detector remains CPU-only. | **IN PROGRESS** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** measure repeated warm p50/p95 timings for runnable PP-OCRv6 tiny/small lanes and preserve cold-vs-warm evidence in the O9 survey; medium remains excluded from the warm lane until its CPU detector path is optimized. Metal repeated fox runs: tiny cold `49.14 s`, warm p50 `42.91 s`; small cold `70.66 s`, warm p50 `84.87 s` (the harness now persists per-run timings and inclusive p95; the small run’s p95 is not claimed from the pre-field artifact) | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** audit the PP-OCRv6 recognizer quality gate after the external head-to-head invalidated cosine-only parity. Local PaddleX exposes PP-OCRv6 detector code but no recognizer blueprint; the current torch mirror still guesses head count/activation/pooling and reproduces unreadable text. Keep the lane unvalidated and block default promotion until the official recognizer implementation/config is recovered and text-gold parity passes | **COMPLETED — quality gate remains blocked** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** align PP-OCRv6 large-recognizer activation semantics with the recovered PaddleX source: StemBlock Conv-BN uses ReLU; LightSVTR conv branches use configured SiLU. Updated native CPU/graph and `dump_ppocrv6_reference.py`; small fox taps pass through stage4 (`0.999958`), head input (`0.999992`), logits (`0.999996`), and full graph logits (`0.999995`). Regenerated official-source-backed Arabic/receipt/German gold archives; the required small graph lane passes `0.999992–0.999997`, while native and reference still decode nonsensical `¿づE₆¿づLyi`, `上批业/|`, and `澳臻肉し企M`. | **COMPLETED — parity fixed, quality blocked** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O11 backend/graph capability audit: record CPU-only, partial-graph, and full-GGML-backend paths per OCR engine and prevent unsupported GPU claims. The capability matrix covers all 12 required OCR families, concrete CPU seams, PP-OCRv6/PP-LCNet partial claims, and explicit Metal/CUDA build boundaries; `tests/test_ocr_backend_matrix.py` is a mandatory smoke guard. CUDA execution and per-engine performance remain separate follow-ups | **COMPLETED** |
| 2026-07-31 | `main` | O11.1 PP-OCRv6 detector/recognizer graph port: replace CPU conv/linear forward with persistent ggml graphs on CPU/Metal/CUDA; preserve Q8 head policy and parity taps | **PENDING** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** O11.1 full-graph implementation contract: persistent static-shape graphs, scheduler-selected backend, backend-resident/dequantized weights, reusable input/output staging, batched line crops, and CPU cosine/logit parity fallback; PP-OCRv6 tiny recognizer now runs one persistent graph through logits with CPU/Metal accepted-output parity on two fixtures, while the detector constructs an opt-in full stem/backbone/neck/head graph via `CRISPEMBED_PPOCRV6_DET_GRAPH=1`; corrected detector neck channel order brings graph-vs-CPU probability cosine to 0.99113 and head pre-sigmoid to 0.99898, but graph box count is still 31 vs CPU 30, so detector CPU accept-gate fallback remains. The regenerated small/medium fox references are valid (the prior small archive had `large_stem2a` length `87,768`); CPU logits parity is `0.999998` small and `0.999992` medium. The small and medium recognizers now build a persistent GGML stem+backbone graph on CPU: all six large-stem taps are `1.000000`, stage taps are `1.000000`/`0.999994` small and `1.000000`/`0.999983` medium, and the CPU SVTR decoder receives the graph backbone with end-to-end logits unchanged (`0.999998`/`0.999992`). The asymmetric stride transition is now supported. Metal reaches stage4 cosine `0.999907` small / `0.999969` medium and logits cosine `0.999982` / `0.999986`, decoding `涨RiI` in both cases. The opt-in `CRISPEMBED_PPOCRV6_SVTR_GRAPH=1` seam graphs SVTR tokenization, and `CRISPEMBED_PPOCRV6_SVTR_DECODER_GRAPH=1` now graphs both SVTR attention/MLP blocks plus final norm; CPU and Metal small/medium runs preserve output parity, with Metal logits cosine `0.999982`/`0.999986` and full graph timing `214`/`512 ms` on the fox crop. Multi-fixture direct recognizer smoke now shows identical CPU fallback/CPU full-graph text on Arabic line, receipt, and German document fixtures, and Metal full-graph text matches CPU on all three. Regenerated gold archives for those fixtures are backed up under `/Volumes/backups/ai/crispembed-gguf/`; the new `tests/test_ppocrv6_graph_gold.py --require` lane passes CPU at `0.999995–0.999996` and Metal at `0.999956–0.999982` (lowest on German), with unchanged decoded text. Remaining work is detector geometry parity and wiring the artifact-backed lane into model-equipped CI; keep both graph gates opt-in until that lane is reproducible there | **IN PROGRESS** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O11.2 PP-LCNet line/page orientation graph port: backend-scheduled depthwise/pointwise/SE blocks with CPU fallback and orientation gates; canonical weight layout and `[1280,2]` linear view are implemented. Metal passes 9/10 German/Arabic/derived parity fixtures; the one uneven-illumination Arabic outlier localizes to Metal accumulation around SE block 4 (`1.0679/3.2166` final logit deltas), while CPU graph passes (`0.0046/0.0139`). Repeated standalone Metal execution passes 31/31 with required per-crop scheduler reallocation, and the explicit pipeline graph smoke remains `141/141` with 30/30 regions. The safe production behavior is complete: graph stays opt-in and automatically falls back to CPU unless explicit debug acceptance is requested. Metal SE/depthwise numerical parity remains a separate optimization TODO | **COMPLETED — safe fallback shipped** |
| 2026-07-31 | `main` | O11.3 GPU preprocessing handoff: benchmark and, where beneficial, graph-accelerate detector resize/normalize, quad warp, crop batching, and postprocessing without changing geometry | **COMPLETED — retain CPU geometry path** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** O11.3 preprocessing/geometry cost split: `CRISPEMBED_PPOCRV6_DET_BENCH=1` reports detector normalize/graph/total timings and `CRISPEMBED_PPOCRV6_BENCH=1` reports routed detector, quad crop, orientation, and recognizer timings. German CC0 Metal measured detector 6.9 s, crop 3.4 ms, CPU orientation 358.6 ms, recognition 455.2 ms; explicit Metal orientation graph is safe but 1.15 s for 30 crops. Crop/warp is not the bottleneck and GPU orientation is slower, so no geometry graph promotion is justified; retain CPU preprocessing while preserving the opt-in graph for diagnostics | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O11.4 OCR portfolio graph audit: capability matrix schema now classifies PP-OCRv6 and PP-LCNet as partial rather than overstating GPU support; tiny/small/medium tier details and acceptance gates are recorded. Source audit confirms FormulaNet, MixTeX, SmolDocling, HMER/BTTR/PosFormer are CPU-scheduled today; VLM audit records concrete CPU seams (window partition, spatial merge, host-side unshuffle/merge, scalar merger, and opt-in neck/MoE fallbacks). PP-OCRv6’s gated full recognizer graphs are now reflected accurately; detector geometry and per-engine residency/performance remain separate follow-ups | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O11.5 backend build/device matrix: Metal macOS, CUDA Linux, CPU reference; graph smoke accepts an explicit backend, reports requested/selected device, and passes on Apple M1 with `GGML_METAL=ON`; CUDA remains pending. Metal PP-OCRv6 orchestrator smoke now passes 141/141; recognizer graph and detector diagnostic graph execute on MTL0 | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O11.6 graph performance/parity gates: backend smoke records warm compute latency after a successful graph run; per-line recognizer timing and non-CPU F16 resident detector weights are measured. Metal detector diagnostic improved from ~3.6 s to ~3.3 s on German CC0 with unchanged probability cosine `0.99114`, so it remains CPU-accepted. Small/medium stem+backbone timing is CPU `10,377`/`10,791 ms` versus Metal `222`/`650 ms` on the fox crop; the full SVTR decoder graph now measures `214`/`512 ms` on Metal and multi-fixture gold logits remain `0.999956–0.999986` with decoded parity. Recognizer performance/parity gates are therefore complete; detector geometry acceptance and CUDA remain separate TODOs | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** O11.7 PP-OCRv6 recognizer weight-cache optimization: immutable convolution dequantization, static-shape scheduler allocation, backend-resident weights, and explicit incompatible-tensor diagnostics are implemented. Corrected asymmetric graph convolution dimensions, tiny activation topology, and the 10-token head shape. CPU/Metal debug taps and small/medium multi-fixture graph gold pass with unchanged decoded output; the tiered gold harness now rejects stale 16-tensor references. Recognizer cache/graph work is complete; detector geometry acceptance and the remaining Metal orientation outlier stay explicitly gated | **COMPLETED** |
| 2026-07-31 | `main` | O11.7 persistent graph and weight-cache optimization: reuse static shapes, scheduler buffers, dequantized critical weights, and batched line crops | **COMPLETED — recognizer path; detector geometry remains gated** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.4 live PP-OCRv6 detector → quad crop → PP-LCNet line orientation → recognizer regression across 10 CC0/derived fixtures using cached Q8/F16 artifacts | **COMPLETED** |
| 2026-07-31 | `diagnose/pp-ocrv6-quality` / `.codex/worktrees/diagnose-pp-ocrv6-quality` | **Picked:** PP-OCRv6 Python/C++ detector geometry and crop parity on 10 CC0 fixtures; DBNet→PP-OCRv6 line/word path comparison added; quad handoff and PP-LCNet PIR→GGUF decoder landed; native classifier wired as optional `model_c` stage with NumPy/native cosine parity and CC0 sweep harness | **IN PROGRESS** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.2 deterministic problematic-input corpus: skew, border, illumination, haze, speckle, low-DPI, JPEG, rotation, perspective, and mixed-orientation variants with parent hashes/recipes | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.6 shared crop preparation contract: aspect/stretch geometry, fixed height/width, max width, padding, and RGB/grayscale output | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.4 structured line-orientation telemetry: detected angle/confidence and applied-rotation metadata through native/C APIs | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.8 benchmark accept-gate classification and preprocessor output provenance | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.5 model-free four-way page-orientation fallback and explicit C API; learned classifier remains separate | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.8 VLM cleanup safeguard: explicit opt-in barrier for destructive classical/learned cleanup on full-page VLM stages | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.8 structured stage metrics through the native/C pipeline API | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.5 orientation API surface: explicit `/preprocess/orientation` advisory endpoint with angle/confidence | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.7 explicit multi-page autorotate option for `/ocr/document`, with confidence gate and temporary rotated page handoff | **COMPLETED** |

### OCR external head-to-head — CrispEmbed vs Tesseract / EasyOCR / PaddleOCR

Every OCR parity artifact in this repo so far compares CrispEmbed against a
*per-stage tensor reference*. That proves a graph is faithful to its blueprint;
it does not answer whether the shipped pipeline reads a page as well or as fast
as the engine it ports. Two new dependency-light scripts close that gap:

- `tests/ocr_synth_corpus.py` — deterministic rendered corpus (20 fixtures: 4
  paragraphs x clean/blur/noise/lowdpi/skew) carrying its own exact ground
  truth, so CER/WER becomes an absolute number. On clean rendered text every
  mature engine scores near zero, which makes a port bug separable from a hard
  input.
- `tests/ocr_external_parity.py` — runs system Tesseract, Python EasyOCR, Python
  PaddleOCR and the native lanes over the same images. Reports absolute CER/WER
  vs ground truth, *port-fidelity* CER vs the upstream engine's own reading of
  the same image, and latency in two deliberately separate columns: `proc_ms`
  (whole invocation, includes model load) and `engine_ms` (load-excluded, from
  the native stage-bench stderr or from the in-process Python engines). The
  columns are not comparable to each other; a non-zero exit is never timed as a
  win.

**Reachability fixes (the lanes were not runnable at all).** `engine::ppocrv6`
is fully implemented in `ocr_orchestrator.cpp` — detector, quad crop, PP-LCNet
orientation, recognizer, stage bench — but had **no `map_engine` C-ABI id and no
`--ocr-engine` name**, so the PaddleOCR-parity lane could not be invoked from
the CLI, the C ABI, or any binding. EasyOCR had a validated
`easyocr_pipeline::{load,run_file}` but no orchestrator engine at all. Both are
now wired: `--ocr-engine ppocrv6` (C-ABI id 16) and `--ocr-engine easyocr`
(id 17), plus `--ocr-cls` for the optional PP-LCNet 0/180 classifier and
registry entries naming the locally-converted artifacts.

**Where the lanes stand (20 synthetic fixtures with exact ground truth).**
Quality is settled; latency is only partly settled — see the caveat below.

| engine | kind | CER | WER | CER vs tesseract-cli |
|---|---|--:|--:|--:|
| `crispembed-ppocrv6` | native | **0.0031** | **0.0178** | 0.0253 |
| `paddleocr-py` | external | 0.0185 | 0.1153 | 0.0368 |
| `tesseract-cli:eng` | external | 0.0256 | 0.0890 | — |
| `crispembed-tesseract` | native | 0.0290 | 0.1623 | 0.0490 |
| `easyocr-py` | external | 0.0769 | 0.2363 | 0.0928 |
| `crispembed-easyocr` | native | 0.0808 | 0.3190 | 0.0974 |

All three native lanes reached or beat their upstreams on character error.
`crispembed-ppocrv6` — which produced pure noise at the start of this work — is
now the most accurate arm in the comparison, 8x below `tesseract-cli` and 6x
below PaddleOCR's own Python pipeline. `crispembed-tesseract` (0.0290 vs
0.0256) and `crispembed-easyocr` (0.0808 vs 0.0769) sit within noise of theirs.
WER remains the weaker column for the native lanes, which is a spacing/grouping
question rather than a recognition one and is the natural next target.

Starting point for comparison: `crispembed-tesseract` 0.0814, `crispembed-easyocr`
0.1412, `crispembed-ppocrv6` unusable.

**Latency: the dominant costs are found and two are fixed.** Three separate
things were eating the time, none of them the recognizer math:

1. **The detector enlarged pages that already resolved their text** — 84% of
   the tesseract lane (`detect=4938 ms` against `recognize=119 ms`). Capping the
   upscale is 4.7x and *also* cut CER 2.8x. Fixed, on by default (below).
2. **A CPU-only recognizer was initialising Metal to load itself.**
   `tesseract_lstm` is entirely host-side, but its loader asked for a GPU
   backend purely to pull the GGUF through `core_gguf::load_weights` — spinning
   up Metal, shader library and all, for a sub-2 MB model, then freeing it.
   Measured 4971 ms cold / 1069 ms warm against **4.8 ms** on the CPU backend.
   The whole invocation went **5.9 s -> 0.47 s (12.5x)**, output byte-identical.
   Fixed; `CRISPEMBED_TESSERACT_GPU_LOAD` restores the old path.
3. **PP-OCRv6's recognizer ran a CPU scalar SVTR.** Its detector already
   follows the correct never-upscale convention (`min(1, 960/max(w,h))`), so the
   remaining cost was compute. The now-correct graph is **~1.9x faster
   end-to-end with byte-identical decoded text on 26/26 fixtures** (20 synthetic
   + 6 CC0 scans, largest 71 regions; `synth_00_clean` 1214 -> 651 ms, the
   1920x2518 `german_official_print` scan 9369 -> 4964 ms). **Promoted to
   default**; `CRISPEMBED_PPOCRV6_NO_GRAPH` reverts. Scope is the recognizer
   only — the detector graph stays diagnostic-only on geometry parity, and the
   tiny variant keeps its own accept gate since the evidence is for small.

Where that leaves a one-shot CLI invocation on `synth_00_clean.png`. Measured
by a self-gating harness that waits for load average < 6, then brackets the run
with the `tesseract-cli` control before *and* after and discards the window if
the two readings differ by more than 30%. This window: control 0.12/0.15 s,
which is the true quiet baseline.

| engine | quiet wall | vs `tesseract-cli` | earlier this round |
|---|--:|--:|--:|
| `tesseract-cli:eng` | 0.135 s | — | — |
| `crispembed-tesseract` | **0.48 s** | 3.6x | 5.9 s at the start |
| `crispembed-ppocrv6` | **1.39 s** | 10.3x | 3.70 s |
| `crispembed-easyocr` | **1.47 s** | 10.9x | 2.06 s |

`tesseract-cli` is the only like-for-like comparison: one subprocess per image,
model load included on both sides. The Python arms (`easyocr-py` 0.75 s,
`paddleocr-py` 1.07 s) are measured in-process with the model already resident,
so quoting our per-invocation CLI against them would flatter or damn us
arbitrarily — that is exactly the `proc_ms`/`engine_ms` asymmetry the harness
refuses to collapse into one column.

**Remaining speed work.**

**The "shared GPU backend" item was wrong and is withdrawn.** It assumed a
pipeline initialises Metal once per engine. Counting the actual inits after the
detector fixes: **tesseract lane 0, EasyOCR 1, PP-OCRv6 1**. DBNet already
defaults to CPU (Metal conv is slower for it — `OCR_DETECT_USE_GPU` is opt-in),
and the two detector loaders now use the CPU backend, so no lane duplicates the
init. A refcounted shared backend would be speculative machinery for a problem
that no longer occurs; the plumbing was written, measured against reality, and
reverted. The cost is a *single* Metal init, not duplication.

**And that single init earns its keep** — checked rather than assumed, since the
tesseract lane's whole 12.5x came from deleting a Metal init. Median-of-3
CPU-seconds, control 0.35-0.39 s:

| lane | with Metal | forced CPU |
|---|--:|--:|
| EasyOCR recognizer | **3.65** | 6.43 |
| PP-OCRv6 recognizer | **3.25** | 3.75 |

So both defaults stay. The same run re-confirms the graph promotion from the
CPU-scalar reference: PP-OCRv6 **3.25 with the graph versus 5.50 with
`NO_GRAPH`**, a 1.7x that matches the 1.9x wall-clock figure measured earlier.

**EasyOCR profiled, and an earlier claim in this file corrected.** On a real
1920x2485 document (`commons_test_ocr_document.jpg`, 31 units) on a quiet box:
`load=2645 ms detect+recognize=12362 ms`. Compute dominates by 5x. The earlier
"load is 94% of the stage" reading came from the tiny synthetic page under heavy
contention, where a single Metal init blocked for tens of seconds — it was a
contention artifact, not a property of the lane, and should not be planned
against.

Same page, same DBNet detector, the two lanes differ almost entirely in their
recognizer: **tesseract lane 8.13 s wall / 7.86 s user, EasyOCR lane 17.98 s /
9.57 s user**. Per-line Tesseract LSTM runs 46-69 ms.

**Negative result — recognizer graph rebuilds are NOT the cost (do not retry).**
`easyocr_ocr_set_width()` tears down and rebuilds the graph whenever the canvas
width changes, and width is per crop (bucketed to a multiple of 64), so reading
order rebuilds every time consecutive lines land in different buckets. Sorting
regions by canvas width makes that O(distinct widths) instead of O(regions).
Implemented and A/B'd: **3.00 vs 3.02 CPU-s on a 47-region receipt and 7.43 vs
7.68 on the 31-unit document** — 0-3%, at the edge of noise. The rebuild is
cheap because it is graph construction plus a gallocr pass, with no weight
reload. Reverted rather than kept gated: ~25 lines of reordering in a
result-ordering-sensitive path is the wrong trade for 3%.

**What is actually left.** (a) PP-OCRv6's detector is CPU scalar convolution and
is that lane's dominant compute; its graph stays diagnostic-only on box-geometry
parity, so closing that parity is the unlock and it is real work, not a knob.
(b) DBNet costs ~400 ms on a capped 572x188 page against `tesseract-cli`'s
0.135 s for the whole job — CPU by deliberate choice (Metal conv measured
slower), so the win is kernel work. (c) The EasyOCR CRNN is ~2.2x the Tesseract
LSTM on the same detections and has no per-stage split below
`detect+recognize` yet.

**Measure with CPU time on this box.** `user+sys` stayed within 12% across runs
where wall clock swung 10x, and an early cross-run wall comparison of the
detector loader reported the *opposite* of what a same-binary A/B later showed.
Contention-robust ratios at the time of writing (median-of-3 CPU-seconds,
`tesseract-cli` control 0.46-0.49 s): tesseract lane 3.4x, PP-OCRv6 6.6x,
EasyOCR 8.9x control.

**Measurement discipline, learned the hard way here.** This box runs 3–6
concurrent agent builds; load average hit 103 mid-sweep and `tesseract-cli`
itself drifted from 150 ms to 10.3 s inside the same harness. Every number
above is quoted with the control that bracketed it, and several runs were
discarded outright. `tests/ocr_external_parity.py` prints the `tesseract-cli`
arm for exactly this reason — if it is far above ~150 ms the run is measuring
contention, not engines.

**The detector, not the recognizers, is the whole latency gap — and its resize
rule is an upstream deviation.** Stage bench on `synth_00_clean.png` via the
tesseract lane: `detect=4938 ms group=1.8 ms crop=0.8 ms recognize=119 ms`. So
84% of the time is DBNet, and the recognizers are already cheap. The cause is
`ocr_detect::detect_rgb_ex`'s `scale = target_short_side / min(w,h)` with no cap
at 1.0: a 572x188 fixture is enlarged ~3.5x to ~2016x672, i.e. **12.6x the
source pixels**. Upstream DBNet/PaddleOCR (`limit_type="max"`) only ever
shrinks.

`detect_options::max_upscale` (env `CRISPEMBED_OCR_DET_MAX_UPSCALE`, 0 = today's
uncapped behaviour) makes this A/B-able without a rebuild. Same binary,
back-to-back, 20 synthetic fixtures, `tesseract-cli` held as a load control at
120–157 ms across all arms:

| lane | uncapped | cap 1.5 | cap 1.0 |
|---|--:|--:|--:|
| `crispembed-tesseract` CER | 0.0814 | 0.0309 | **0.0290** |
| `crispembed-tesseract` proc ms | 6462 | 2145 | **1369** |
| `crispembed-easyocr` CER | 0.1412 | 0.0955 | **0.0808** |
| `crispembed-easyocr` engine ms | 6295 | 2396 | **1565** |

That is 2.8x lower character error at 4.7x the speed for the tesseract lane, and
it lands both lanes at CER parity with their upstreams (`tesseract-cli` 0.0256,
`easyocr-py` 0.0769). Enlarging a page the scan never resolved was costing both
time *and* accuracy.

**It now ships on by default, gated on image size rather than switched off.**
The first cut was left off because `tests/regression/images/cc0/simple_table.jpg`
(200x102 — a thumbnail, not a page) went from one detected region to zero when
capped. The fix is `upscale_floor`: the cap applies only once the short side is
at least 120 px, which is the difference between "this page already resolves
its text" and "this is a thumbnail". 120 is measured, not picked — the 616x149
low-DPI fixtures are *better* capped on both axes (CER 0.025 at 1.25 s versus
0.066 at 58 s), so the exemption must not reach them, while the 102 px
thumbnail needs it. Verified by the resize decisions: 616x149 and 572x188 now
pass through unscaled while 200x102 still goes to 1443x736 and keeps its
region. Both knobs stay overridable
(`CRISPEMBED_OCR_DET_MAX_UPSCALE`, `CRISPEMBED_OCR_DET_UPSCALE_FLOOR`).
A cap=2.0 arm was also run but is discarded: `tesseract-cli` went 157 -> 1936 ms
on the same fixture inside it, so that run was contended, not slow.

**PP-OCRv6 — root-caused against the real PaddleOCR source and FIXED.**

The accepted PP-OCRv6 parity evidence was per-stage cosine against
`tools/dump_ppocrv6_reference.py`, a hand-written torch mirror with guessed
details — and the mirror could not read text either, so the native port that
matched it at cosine 0.9999 could not read text either. `git clone`ing
PaddleOCR and reading `ppocr/modeling/backbones/rec_lcnetv4.py`,
`ppocr/modeling/necks/rnn.py` and `tools/infer/predict_rec.py` settled every
guess. Four were wrong:

1. **Stem activation.** `StemBlock` is built from `ConvBNAct`, whose activation
   is `ReLU()`; the mirror used SiLU. (Landed on `main` in parallel by the
   Tesseract session.)
2. **The neck dropped the local-conv residual.** Upstream is
   `z = z + local_conv(z)`; the mirror had `z = local_conv(z)`.
3. **The neck skip landed in the wrong place.** `skip = skip_conv(x)` is
   computed first but added **after** the SVTR blocks and the final norm; the
   mirror added it before the blocks and never after.
4. **Recognizer input width.** `max_wh_ratio` is seeded with `imgW/imgH` and
   grows to the widest crop, so 320 is a **floor**: a 520x35 line is 713 px.
   The mirror and the runtime capped width at 320, crushing 44 characters into
   40 CTC timesteps — undecodable by any model, correct or not.

A fifth lives in the vocabulary: `use_space_char: true` makes the label list
`blank + 18708 dict entries + ' '`, which is where the head's 18710 outputs
come from. The GGUF carries only the 18708, so class 18709 decoded to nothing
and every space was dropped.

Result on `synth_00_clean.png`, native CPU lane, all three lines exact
including punctuation:

| before | after |
|---|---|
| `iiiiii` | `The quick brown fox jumps over the lazy dog.` |
| `laúieyotiieieioieioni.` | `Pack my box with five dozen liquor jugs.` |
| `íotuinióniiaieiasaieró` | `How vexingly quick daft zebras jump!` |

Setting `CRISPEMBED_PPOCRV6_FIXED_WIDTH=1` reproduces the old
`Te qu c   vr  .` exactly, which is what pins the width cap as the cause rather
than a correlate.

Open follow-ups for this lane: (a) **latency is unmeasured** — the box sat at
load average 101 from concurrent sessions during the A/B, and wider input plus
the O(tokens^2) CPU scalar SVTR attention means a real cost that needs a quiet
back-to-back run; (b) the opt-in `CRISPEMBED_PPOCRV6_{SVTR_,DET_}GRAPH` paths
still fold the skip the old way and were not touched; (c) the converter should
emit the space class itself rather than the runtime appending it; (d) the
recognizer still confuses `e`/`c` on one Courier fixture, which is an ordinary
error class, not a structural one. Re-run
`tests/ocr_external_parity.py` with `crispembed-ppocrv6` enabled once the box is
quiet to put this lane in the head-to-head table.

**Original stage-bench observation (PP-OCRv6 small, Metal, `synth_00_clean.png`, 3 lines).**
Decoded output is not text: `iiiiii` / `laúieyotiieieioieioni.` /
`íotuinióniiaieiasaieró` against a ground truth of "The quick brown fox jumps
over the lazy dog." — with `mean_conf=0.94`, so the confidence signal does not
detect it. Detector geometry is plausible (3 boxes for 3 lines); the recognizer
output is wrong. Stage bench on the same run: `detect=62177.3 ms
recognize=35630.8 ms total=97844.2 ms`, i.e. ~98 s for a 600x200 image that
PaddleOCR reads in a fraction of a second. That run was CPU-contended; uncontended,
`test-ppocrv6-direct` reports 1774 ms total for the same three lines, so the
98 s figure is contention and the standing latency question for this lane is
moot until it produces text at all.

### OCR performance backlog — every idea, with status and handover

Status vocabulary: **DONE** shipped on by default; **GATED** implemented, works,
output-verified, default off with the measurement that kept it off; **OPEN** not
implemented. Every gated path stays in the tree — a path that does not win today
can win under a different engine mix, and re-deriving it costs more than the
gate does.

#### Shipped (default on)

| # | Change | Win | Gate to revert |
|---|---|--:|---|
| P1 | Detector stops enlarging pages that already resolve their text (`max_upscale`, `upscale_floor=120`) | 4.7x **and** CER 0.0814→0.0290 | `CRISPEMBED_OCR_DET_MAX_UPSCALE=0` |
| P2 | Tesseract-LSTM loads via CPU backend instead of spinning up Metal for a host-side engine | lane 5.9 s → 0.47 s (12.5x) | `CRISPEMBED_TESSERACT_GPU_LOAD=1` |
| P3 | PP-OCRv6 small/medium recognizer graph promoted | 1.9x wall / 1.7x CPU, 26/26 identical text | `CRISPEMBED_PPOCRV6_NO_GRAPH=1` |
| P4 | PP-OCRv6 detector loads via CPU backend | 7-14% CPU | `CRISPEMBED_PPOCRV6_DET_GPU_LOAD=1` |

#### Implemented, gated off (working — reuse these before rewriting them)

| # | Path | Env | Why it is off | When it becomes worth turning on |
|---|---|---|---|---|
| P5 | Process-shared GPU backend (refcount-free singleton + `crispasr_free_gpu_backend`) | `CRISPEMBED_SHARED_GPU_BACKEND=1` | After P2/P4 no lane inits Metal more than once (tesseract 0, EasyOCR 1, PP-OCRv6 1), so it saves nothing today | The moment two GPU-resident engines share a process: a VLM stage beside a recognizer, the detector graph being promoted, or server/batch use. **Hazard:** one `ggml_backend_t` driven from several threads (`CRISPEMBED_TESSERACT_WORKERS`) is not promised safe |
| P6 | EasyOCR width-sorted recognition (O(distinct widths) graph rebuilds instead of O(regions)) | `EASYOCR_WIDTH_SORT=1` | 0-3%, edge of noise — the rebuild is graph construction + gallocr, no weight reload | If the recognizer graph gains an expensive build step (weight residency, kernel specialisation, a shape-keyed resident cache). Also the natural companion to P9 |
| P7 | PP-OCRv6 detector full graph | `CRISPEMBED_PPOCRV6_DET_GRAPH=1` | Box geometry not at parity (31 boxes vs CPU 30) | See O1 — this is the single biggest remaining win |
| P9 | Tiled + 4-wide-unrolled 1x1 convolution (`conv2d_1x1_cpu`) | `CRISPEMBED_CONV1X1_FAST=1` | **Not a win.** Interleaved pairs: neutral on M1/NEON (-0.3% excluding one outlier baseline, CI spans zero), **-4.8% regression** on x86/AVX2 (5/6 pairs negative). Decoded text identical on both | Only if a future engine or shape shows a real gain under interleaved-pair measurement with n reported |
| P10 | Loop-inverted depthwise convolution (`conv2d_depthwise_cpu`) | `CRISPEMBED_CONVDW_FAST=1` | Single-shot 3.6% on M1 is **unverified** against a measured 8.1% noise floor; neutral on x86. Decoded text identical on both | After the tap-unrolling rework, re-measured as interleaved pairs |
| P11 | Four-accumulator `dot_product_wide` | `CRISPEMBED_DOT_WIDE=1` | Slower on M1 (8.52 -> 8.90, single shot). Built to test the hypothesis that 2 accumulator chains starve the M1's FMA pipes; the prediction failed, which is why that explanation was withdrawn | If a future profile shows a genuinely FMA-latency-bound dot rather than a memory-bound one |

#### Open, in descending measured value

**O1 — PP-OCRv6 detector graph box-geometry parity is NOT a speed item.**
Corrected 2026-08-02 by measuring it. The premise in earlier notes — that the
CPU scalar detector is slow and promoting its graph would fix that — is wrong.
Quiet box (load 4.2), 1920x2518 page, same fixture and binary:

| detector path | total |
|---|--:|
| **CPU scalar (shipped default)** | **2350 ms** |
| graph on the CPU backend | 6132 ms (graph alone 2829 ms) |
| graph on Metal (`CRISPEMBED_PPOCRV6_DET_GRAPH=1`) | 15933 ms (graph alone 12663 ms) |

The graph is 2.6x slower on CPU and 6.8x slower on Metal than the hand-written
scalar path it falls back to, which matches the DBNet finding already recorded
in `ocr_detect.cpp` (Metal conv2d/conv-transpose measured ~139 s GPU vs ~10 s
CPU on an M1). So closing box-geometry parity is a **correctness and
portability** goal — it is what a CUDA or Vulkan deployment would need, and it
removes a diagnostic-only caveat from the matrix — but nobody should expect a
speedup from it on this hardware, and it should not be prioritised as one.
Anyone picking it up: the geometry comparator already exists
(`report_graph_box_geometry`, `CRISPEMBED_PPOCRV6_DET_GRAPH_COMPARE=1`), though
it did not emit on the run above and needs its call site checked first.

**O2 — the detector's CPU scalar path is the real target, at 2350 ms.** It is
the dominant cost of the PP-OCRv6 lane and DBNet is the shared detector for the
other two, and ggml graphs are demonstrably the wrong tool for it here (O1). So
this is kernel work on the scalar code: per-node-trace one detect call, and
check specifically whether 1x1 convolutions are being routed through an im2col
path, since a 1x1 conv is a pure channel matmul and its im2col is a copy that
materialises a large intermediate for nothing. The `QWEN3_TTS_CODEC_FASTCONV`
precedent in the dev guide is exactly this shape and was worth 3x.

**O2 answered 2026-08-02 — the trace exists now, and it says the cost is
concentrated in two shapes, not spread out.** `CRISPEMBED_PPOCRV6_DET_PROFILE=1`
prints a per-convolution table keyed on shape signature, sorted heaviest first,
with GF/s so a slow layer is distinguishable from a merely large one. On
`german_official_print.jpg`:

| class | share of detector conv time | rate |
|---|--:|--:|
| 1x1 pointwise | **51.6%** | ~1.2 GF/s |
| depthwise | **20.4%** | **0.02-0.19 GF/s** |
| deconv | 6.4% | ~0.6 GF/s |
| other conv | 21.6% | ~1.0 GF/s |

Read the shares, not the totals: the absolute figures move by 1.5x between runs
on this box, the proportions do not. Two consequences. First, **H1 is the lever
for O2/H2** — the pointwise layers are not "one thing to check", they are half
of everything. Second, **depthwise is a lever nobody had listed**: the single
most expensive layer in the whole network is one 7x7 depthwise at 240x184, 13.7%
of all convolution time at 0.17 GF/s. That is the generic path's worst case by
construction — with one input and one output channel per group there is nothing
to amortise the patch gather against, so it gathers a kh*kw window and consumes
it in a single dot_product, per output pixel. `conv2d_depthwise_cpu`
(`CRISPEMBED_CONVDW_FAST=1`) inverts the nest instead: per channel and output
row, walk the taps and accumulate a whole row per tap, so each tap is a
contiguous axpy, the input row stays in L1 across all taps, the gather
disappears, and the boundary test becomes a per-tap column range in closed form.
Implemented and equivalence-guarded; **not yet A/B'd for speed.**

**O3 — EasyOCR CRNN is ~2.2x the Tesseract LSTM on identical detections**
(17.98 s vs 8.13 s wall on the same 31-unit page, same DBNet boxes). No
per-stage split exists below `detect+recognize` yet; add one mirroring the
tesseract/ppocrv6 load-vs-compute benches before targeting anything.

**O4 — Batched crop recognition.** Every lane recognizes line crops one at a
time. Crops sharing a canvas width could go through one graph dispatch as a
batch dimension, which is where P6's width grouping stops being cosmetic. Needs
a batched graph in each recognizer; largest expected win on many-region pages
(71 regions on one CC0 scan).

**O5 — Model load is still ~0.37 s of the tesseract lane's 0.47 s.** Now that
compute is small, load dominates a one-shot CLI invocation. Options: mmap the
GGUF instead of copying weights into host vectors, or keep a warm process /
server for repeated pages. `tesseract-cli` pays load per invocation too and
still totals 0.135 s, so there is headroom here.

**O6 — Detector resize rule keyed on estimated text height rather than image
size.** P1's `upscale_floor=120` is a proxy: what actually matters is whether
glyphs are tall enough for the detector, not whether the page is. A cheap
stroke-width or connected-component estimate would let the cap apply to a
low-DPI thumbnail with big text and stay off for a high-DPI page of tiny text.

**O7 — Per-engine profiling of the remaining VLM/OMR lanes.** This whole round
covered only the three classical lanes. The load-vs-compute split found a wasted
Metal init in two of three engines it was applied to; it has not been applied to
GOT/GLM/Qwen/InternVL/SmolDocling/the OMR engines, and the same class of bug
(GPU backend created for a host-side path) is plausible in any of them. Cheap to
check: `grep -n crispasr_init_gpu_backend src/*.cpp` and ask, per engine,
whether its compute actually runs on that backend.

### OCR performance — self-contained handover prompts

Everything a fresh agent needs is in this section. Read **§0 Setup** and
**§1 How to measure** once, then take one H-item. Nothing here assumes prior
context beyond this file.

#### §0 Setup (do this first, every time)

```bash
# 1. NEVER edit the main checkout. Make a worktree.
cd /Users/christianstrobele/code/CrispEmbed
git worktree add .claude/worktrees/<your-task> -b <your-branch> main
cd .claude/worktrees/<your-task>

# 2. ggml is a gitlink placeholder in a fresh worktree; cmake needs a real tree.
#    Symlink it TO BUILD, restore the gitlink BEFORE any git command.
rm -rf ggml && ln -s /Users/christianstrobele/code/CrispEmbed/ggml ggml

# 3. Configure + build (Metal + embedded shaders; ~15 min cold, ~1 min warm)
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=ON
cmake --build build -j6
```

**The ggml trap, which will bite you.** `git stash`, `git checkout`, `git add`
and `git commit` all refuse to run (or silently reset the link) while `ggml` is
a symlink. The cycle is: symlink → build/measure → `rm -f ggml && git checkout
HEAD -- ggml` → `git add`/`commit`/`push` → symlink again. If you skip the
re-symlink, the next `cmake --build` fails with `ninja: error: rebuilding
'build.ninja'` **and you will unknowingly measure the stale binary**.

**Models** live in `~/crispembed-live-cache/` (also mirrored at
`/Volumes/backups/ai/crispembed-gguf/`, which is often unmounted — prefer the
home path). The five used below are all present:

| lane | detector | recognizer |
|---|---|---|
| tesseract | `dbnet-ic15-q8_0.gguf` | `tesseract-eng-q8_0-seeded.gguf` |
| easyocr | `dbnet-ic15-q8_0.gguf` | `easyocr-english-g2-f16.gguf` |
| ppocrv6 | `PP-OCRv6_small_det-f16.gguf` | `PP-OCRv6_small_rec-q8-head.gguf` |

**Python** is `~/miniconda3/bin/python` — never the system `python3`. Set
`USE_TF=0` for anything importing transformers.

**Test corpus** — 20 fixtures that carry their own exact ground truth, so CER is
absolute rather than cross-engine agreement. It is generated, not checked in:

```bash
~/miniconda3/bin/python tests/ocr_synth_corpus.py --output ~/crispembed-ocr-synth
```

**⚠ Do not put the corpus under `/tmp`.** Verified 2026-08-02: a corpus written
to `/tmp/ocr-synth` is readable by `build/crispembed` but **not** by the Homebrew
`tesseract`, which fails with `Leptonica Error in findFileFormat: image file not
found` — the session's `/tmp` is a private mapping the external binary cannot
see. Since `tesseract` is the load control for every measurement below, a `/tmp`
corpus silently breaks the control while the native lanes appear to work. A home
path works for both. (`/tmp` also gets wiped between sessions.)

Real scans live at `tests/regression/images/cc0/` (no ground truth; use them for
cross-engine agreement and for many-region stress — `commons_test_ocr_document.jpg`
is 1920x2518 and yields 31 units, `receipt_example.png` yields 47).

**Run one lane:**

```bash
C=~/crispembed-live-cache
./build/crispembed --ocr-pipeline ~/crispembed-ocr-synth/synth_00_clean.png \
  --ocr-engine ppocrv6 --ocr-det $C/PP-OCRv6_small_det-f16.gguf \
                       --ocr-rec $C/PP-OCRv6_small_rec-q8-head.gguf
# expected: the three-line pangram, exactly, punctuation included
```

**Run the full head-to-head** (native lanes vs system Tesseract / Python EasyOCR
/ Python PaddleOCR, reporting CER, WER and latency):

```bash
USE_TF=0 ~/miniconda3/bin/python tests/ocr_external_parity.py \
  --images ~/crispembed-ocr-synth --model-dir ~/crispembed-live-cache --repeats 3
```

#### §1 How to measure on this box — read this or your numbers will be wrong

This machine runs **3–6 concurrent agent builds**. Load average routinely sits
between 10 and 100. Two rules follow, both learned by getting them wrong:

**(a) Measure CPU time, not wall clock.** `user+sys` held within 12% across runs
where wall clock swung 10x. A wall-clock A/B of the 1x1 conv path produced
"1.8x slower" and "2x faster" on consecutive rounds; only CPU time was readable.

```bash
# median-of-3 user+sys, in CPU seconds. Verified working 2026-08-02.
cpu() { for i in 1 2 3; do /usr/bin/time -p "$@" 2>&1 >/dev/null \
        | awk '/real|user|sys/{a[$1]=$2}END{printf "%.2f\n", a["user"]+a["sys"]}'; done \
        | sort -n | sed -n 2p; }
```

Note the `2>&1 >/dev/null` order: it sends **stderr** (where `time -p` writes)
to the pipe and discards the command's own stdout. Reversing it measures
nothing. Pass the command as separate words — `cpu tesseract "$I" stdout -l eng`
— never as a single variable, see (c) below.

**(b) A/B both arms in ONE binary behind an env gate, never across two runs.**
A cross-run comparison of the PP-OCRv6 detector loader reported the *exact
opposite* of what a same-binary gated A/B then showed. Always bracket with a
control on the same fixture in the same run:

```bash
I=~/crispembed-ocr-synth/synth_00_clean.png
control() { cpu tesseract "$I" stdout -l eng --psm 6 \
            --tessdata-dir /opt/homebrew/share/tessdata; }
control            # ~0.13 s quiet. Much above that => you are timing contention.
cpu env MYGATE=1 ./build/...   # arm A
cpu           ./build/...      # arm B
control            # must agree with the first reading within ~30%
```

If you want wall clock, `/private/tmp/.../scratchpad/quiet_bench.sh` in the
2026-08-02 session is the pattern: wait for `loadavg < 6`, bracket with the
control before and after, discard the window unless the two agree within 30%.

**(c) A crash mints a fake win.** A non-zero exit or empty output must never be
timed. zsh does **not** word-split unquoted variables, so `cpu $CMD` runs
nothing and reports `0.00`. Always check the decoded text alongside the timing.

**(d) Never claim a win without output equivalence.** The 26-fixture check:

```bash
C=~/crispembed-live-cache; same=0; diff=0
for f in ~/crispembed-ocr-synth/*.png tests/regression/images/cc0/*.png tests/regression/images/cc0/*.jpg; do
  [ -f "$f" ] || continue
  a=$(./build/test-ppocrv6-direct $C/PP-OCRv6_small_det-f16.gguf $C/PP-OCRv6_small_rec-q8-head.gguf "$f" 2>/dev/null | grep -o 'text=.*' | tr '\n' '|')
  b=$(MYGATE=1 ./build/test-ppocrv6-direct $C/PP-OCRv6_small_det-f16.gguf $C/PP-OCRv6_small_rec-q8-head.gguf "$f" 2>/dev/null | grep -o 'text=.*' | tr '\n' '|')
  [ "$a" = "$b" ] && same=$((same+1)) || { diff=$((diff+1)); echo "DIFF $(basename $f)"; }
done; echo "identical=$same differing=$diff"
```

#### §2 Existing instrumentation (use it before adding more)

Splitting **model load** from **compute** has already found a wasted GPU-backend
init worth 12.5x in one engine and 7–14% in another. It is the highest-yield
first move on any lane.

| env | prints |
|---|---|
| `CRISPEMBED_OCR_ORCH_BENCH=1` | per-stage orchestrator totals + accept gate |
| `CRISPEMBED_TESSERACT_BENCH=1` | `[tesseract-load-bench]` detector/recognizer load, `[tesseract-stage-bench]` detect/group/crop/recognize |
| `CRISPEMBED_PPOCRV6_BENCH=1` | `[ppocrv6-load-bench]`, `[ppocrv6-stage-bench]` |
| `CRISPEMBED_PPOCRV6_DET_BENCH=1` | `[ppocrv6-det-bench]` preprocess/graph/total, `accepted=` |
| `CRISPEMBED_EASYOCR_BENCH=1` | `[easyocr-stage-bench]` load / detect+recognize |
| `CRISPEMBED_EASYOCR_STAGE_BENCH=1` | `[easyocr-recognize-bench]` detect, and inside the region loop crop / set_width / recognize, plus `width_calls` and `width_changes` (H3) |
| `CRISPEMBED_PPOCRV6_DET_PROFILE=1` | `[ppocrv6-det-profile]` per-convolution table for the scalar detector: ms, share, calls, GF/s, shape signature, heaviest first (H2) |
| `CRISPEMBED_OCR_DETECT_BENCH=1` | DBNet resize decision (`raw WxH -> WxH`) |

#### §3 Current state — do not re-derive these

Quality is **done**: all three native lanes at or above their upstreams on the
20-fixture ground-truth corpus (`crispembed-ppocrv6` CER 0.0031, `paddleocr-py`
0.0185, `tesseract-cli` 0.0256, `crispembed-tesseract` 0.0290, `easyocr-py`
0.0769, `crispembed-easyocr` 0.0808). **Speed is the open work.** Quiet-window
wall clock, control 0.135 s: tesseract lane 0.48 s, ppocrv6 1.39 s, easyocr
1.47 s.

Already shipped (P1–P4) and already gated-off-with-a-reason (P5, P6, P8, P7) are
tabulated in the backlog section above. **Read that table before starting** —
three of the eight items below interact with an existing gate.

---

#### H1 — Decide the fate of the 1x1 conv fast path

**Goal** Flip `CRISPEMBED_CONV1X1_FAST=1` to default, restrict it to a size
heuristic, or prove it should stay off.

**Why** A 1x1 convolution is a channel matmul, but `conv2d_cpu`
(`src/core/cpu_ops.h`, ~line 348) gathers a "patch" per output pixel — for
kh=kw=1 that is a pure copy of `ch_per_group_in` floats repeated H*W times
before the dot. The axpy form skips it. **Measured: ~6% CPU on the PP-OCRv6
detector (7.59 vs 8.27/7.93 CPU-s median-of-3), output identical on 5 fixtures.**

**Why it is only 6%** The generic path is better than it looks: it reuses one
gathered patch across every output channel in the group, so on a large spatial
plane it is cache-friendlier than streaming the output plane once per (oc, ic)
pair, which is what the fast path does. Expect the fast path to win where the
plane is small and the channel count large, and to lose where the plane blows
L2. **A global flip is probably wrong; a size heuristic is probably right.**

**RESOLVED 2026-08-02 — H1's answer is "keep it off", because it is not a win
on either machine.** Reported here in corrected form; an earlier revision of
this entry claimed 9.1% on the M1 and an architectural sign flip, and both were
measurement artifacts.

Interleaved off/on pairs (which cancel drift; two separately-taken medians do
not):

| host | pair deltas | mean | sd | 95% CI |
|---|---|--:|--:|---|
| M1 Mac (NEON) | +15.7, -1.5, -1.2, +1.9 | +3.7% | 8.1 | [-4.2, +11.7] |
| VPS Xeon (AVX2) | -7.6, -2.3, -3.3, -13.3, +1.9, -4.2 | -4.8% | 5.2 | [-8.9, -0.7] |

Three of the four Mac pairs are inside ±2%; the mean is carried by pair 1, whose
*baseline* was the outlier (10.23 vs ~9.5), not whose gated arm was fast.
Excluding it the Mac mean is -0.3% and the CI spans zero. On x86 the kernel is a
genuine ~5% regression (5/6 pairs negative, CI excludes zero).

So: **neutral on ARM, a regression on x86, and no architectural flip.** The
earlier "the sign flips with the instruction set" reading was one noisy Mac
number set against one real x86 regression. Both gates stay off because neither
earns its place. The depthwise gate's single-shot 3.6% is unverified against the
same noise floor and was neutral on x86 — treat it as unmeasured.

**§1's measurement protocol is not adequate for effects this size, and that is
the durable finding here.** Median-of-3 CPU-seconds with a control bracket
cannot resolve 3-10% on either box:

| host | sd of paired delta | interleaved pairs to resolve 5% at 95% |
|---|--:|--:|
| M1 Mac | 8.1% | **41** |
| VPS Xeon | 5.2% | **16** |

Comparing two separately-taken medians is worse than pairing, because drift
lands entirely in one arm — that is exactly how pair 1 produced "+15.7%" while
both tesseract controls still agreed within 30%. Interleave the arms, report
every pair, publish the spread. Any prior finding in PLAN/PERFORMANCE at the
3-10% level resting on one median-of-3 should be re-run before it is trusted.

**Earlier detail — the 6% was the implementation, not the idea.** The
diagnosis above is right about *why* the old form underperformed and wrong about
the remedy: the fix is not a size heuristic that avoids large planes, it is to
stop traversing the plane badly. The rewritten `conv2d_1x1_cpu` blocks the pixel
axis into 8192-element tiles so a tile's input slab stays L2-resident, and
computes four output channels at once so each loaded input element feeds four
FMAs from registers rather than one. Same-binary A/B on the same fixture,
CPU-seconds median-of-3, bracketed by the tesseract control (0.40 before / 0.42
after): 8.59 gate-off vs 7.81 gate-on. **That reading was later shown not to
replicate** — see the corrected H1 entry above; repeated as interleaved pairs
the kernel is neutral on ARM. The 8.59 baseline was an unlucky-high sample, the
same artifact that produced a "+15.7%" pair later. The gate stays opt-in: this helper is
shared by 15 engines and exactly one of them has been measured, so the "you
cannot flip it globally on one engine's evidence" constraint below is unchanged.

The rewrite also made the path testable. The gate is a read-once static, so the
two implementations could not be compared inside one process through
`conv2d_cpu`; `conv2d_1x1_cpu` is now callable directly and
`test_conv2d_1x1_equivalence` covers 9 shapes (tile-boundary straddle, out_ch
tails of 1 and 3, grouped, depthwise, null bias). It is a magnitude-scaled
tolerance check **on purpose** — `dot_product` accumulates through eight FMA
lanes plus a horizontal add on aarch64 while the axpy form accumulates in
channel order, so an exact assertion would fail for a correct implementation.
Do not tighten it to equality. The guard was verified to bite by injecting a
`w1`-for-`w2` swap in the unroll: 7 of 9 shapes fail, and the 2 that do not are
exactly the grouped cases where `ch_per_group_out < 4` so the unroll never runs.

**Scope** `conv2d_cpu` is shared by 15 files: `ppocrv6_det`, `ppocrv6_ocr`,
`pplcnet_orientation`, `surya_det`, `nafnet_denoise`, `text_sr`, `got_ocr`,
`deepseek_ocr2`, `unlimited_ocr`, `ppformulanet_ocr`, `ppformulanet_l_ocr`,
`bttr_ocr`, `hmer_ocr`, `posformer_ocr`. You cannot flip it globally on one
engine's evidence.

**Do** A/B per engine that has a runnable fixture, on a quiet box, CPU-seconds,
output-equivalence each time. Then either flip globally, or replace the env gate
with `if (plane_bytes < threshold)` and justify the threshold with the data.

**Acceptance** CPU-seconds down (or neutral) on every engine tested, decoded
output byte-identical everywhere, control in range. Keep the env gate either way
— it is the bisection lever.

---

#### H2 — Cut the PP-OCRv6 detector's 2350 ms scalar path

**Goal** Reduce the dominant compute cost of the ppocrv6 lane.

**Baseline** 2350 ms total on `tests/regression/images/cc0/german_official_print.jpg`
(1920x2518), quiet box, via `CRISPEMBED_PPOCRV6_DET_BENCH=1`.

**⚠ Do NOT try to fix this by promoting the ggml graph.** Measured on the same
fixture and binary: CPU scalar **2350 ms**, graph on CPU backend **6132 ms**,
graph on Metal **15933 ms**. The graph is 2.6–6.8x *slower*. This matches the
note already in `ocr_detect.cpp` that Metal conv2d/conv-transpose measured
~139 s GPU vs ~10 s CPU on an M1 for a 1472x736 map. That route is a dead end
for speed on this hardware.

**Do** Per-node-trace one `ppocrv6_det` detect call to get a per-layer cost
table, then attack the top entries. H1's result says the wins are in memory
traffic, not FLOPs. The FPNC-style neck has 1x1 `pointwise_convolution` layers
(`src/ppocrv6_det.cpp` ~line 855) — check what they cost.

**DONE 2026-08-02 — the trace exists (`CRISPEMBED_PPOCRV6_DET_PROFILE=1`) and
the answer changes what to attack.** Full table and shares are under O2 above.
Short version: the 1x1 pointwise layers cost **51.6% of all detector convolution
time**, so H1 is not a separate item from H2 — it *is* H2's main lever, and it
is not improved by the retiled kernel (that 9.1% did not replicate; see the
corrected H1 entry). The second finding was not on
anyone's list: **depthwise convolutions are 20.4% of conv time at 0.02-0.19
GF/s**, an order of magnitude below the pointwise rate, and the single most
expensive layer in the network is one 7x7 depthwise at 240x184 (13.7% of all
conv time by itself). `conv2d_depthwise_cpu` / `CRISPEMBED_CONVDW_FAST=1` is
implemented and equivalence-guarded but **not yet A/B'd for speed** — that is
the next concrete step on this item, and the remaining acceptance work is
unchanged: CPU-seconds down with byte-identical decoded text on the 26-fixture
check, control in range.

**Acceptance** CPU-seconds down with byte-identical decoded text on the
26-fixture check in §1(d), control in range.

---

#### H3 — Split the EasyOCR recognizer's cost

**Goal** Find out why the lane is ~2.2x the Tesseract LSTM on identical
detections (17.98 s vs 8.13 s wall on the same 31-unit page, same DBNet boxes).

**State** `[easyocr-stage-bench]` reports only `load` vs `detect+recognize`. On
a real document, quiet: `load=2645 ms detect+recognize=12362 ms` — compute
dominates 5x. Nothing is profiled below that.

**⚠ An earlier note in this file claimed "load is 94% of the stage".** That was
the tiny synthetic page under heavy contention, where one Metal init blocked for
tens of seconds. It is an artifact of the box, not the lane. Do not plan
against it.

**Do** Add a per-stage bench to `src/easyocr_ocr.cpp` mirroring
`[tesseract-load-bench]` / `[ppocrv6-load-bench]` (see §2 — that split has the
best hit rate of anything tried). Then target what it exposes.

**Instrumentation landed 2026-08-02, not yet run on a page.**
`CRISPEMBED_EASYOCR_STAGE_BENCH=1` splits `recognize_regions_locked` in
`src/easyocr_pipeline.cpp` (not `easyocr_ocr.cpp` — the per-crop loop that
actually spends the time lives in the pipeline) into `detect_ms`, `crop_ms`,
`set_width_ms` and `recognize_ms`, and reports `width_calls` vs `width_changes`.
That last pair is the point: `easyocr_ocr_set_width` tears down and rebuilds the
recognizer graph on every width change, so the ratio says directly how much
`EASYOCR_WIDTH_SORT` could ever be worth on a given page.

**Run on `commons_test_ocr_document.jpg` (289 detector boxes -> 27 grouped
lines), 2026-08-02.** Absolute ms are contended wall clock on a load-30-plus
box; read the ratios.

| stage | ms | share |
|---|--:|--:|
| detect | 24,778 | ~55% of the lane |
| recognize loop total | 20,130 | ~45% of the lane |
| — crop extraction | 41 | 0.2% of the loop |
| — `set_width` (graph rebuilds) | 4,825 | **24% of the loop** |
| — recognize | 15,263 | 76% of the loop |

Three things follow, and the first contradicts how this item was framed.

1. **Detection is the larger half of the EasyOCR lane, not recognition.** H3 was
   written to find out why the recognizer is 2.2x the Tesseract LSTM, and the
   answer is that a big part of what was being attributed to "the lane" is
   DBNet. Whoever picks up H3 should retarget: the recognizer is 76% of 45%, so
   about a third of the lane.
2. **Graph rebuilds are 24% of the recognition loop** — 25 rebuilds for 27
   regions with sorting off. That is the number P6 never had.
3. **`EASYOCR_WIDTH_SORT` is worth much more than its recorded 0-3%, but only
   against the right denominator.** It cuts rebuilds 25 -> 14 and `set_width`
   4,825 -> 2,348 ms, roughly halving it, which is ~12% of the recognition loop
   — but only ~5% of the lane once detection is counted, and P6's 0-3% was
   measured against total lane time. Both numbers are right; the old one just
   hid the mechanism. The ceiling is set by distinct widths, not region count:
   27 regions over 14 distinct widths means sorting can never remove more than
   13 of the 25 rebuilds.

**Bug found and fixed by this instrumentation.** The width-sort pre-pass computed
its sort key with a hardcoded 2-pixel detector margin while the loop applies that
margin only when `add_detector_crop_margin` is set, so on the external-geometry
path (Python EasyOCR / Tesseract / LayoutLM boxes, pad 0) it sorted by widths
that are never requested: 19 rebuilds instead of the 15 distinct widths that path
actually has. The key now derives from the same `pad` the loop uses; verified
19 -> 15 on the same page.

**Related** `EASYOCR_WIDTH_SORT=1` already exists (0–3%; it makes graph rebuilds
O(distinct widths) instead of O(regions)) and becomes worth more if you make the
graph build expensive.

---

#### H4 — Batched crop recognition

**Goal** Recognize line crops in batches instead of one at a time.

**Why** Every lane loops per crop. Crops sharing a canvas width could go through
one graph dispatch as a batch dimension. Largest win on many-region pages — one
CC0 scan yields 71 regions.

**Prerequisite already done** `EASYOCR_WIDTH_SORT=1` groups equal-width crops
adjacently; that is exactly the ordering a batcher wants.

**Do** Add a batch dimension to a recognizer graph (start with PP-OCRv6, whose
graph is already the default and shape-keyed via `pp_graph_build(c, width)`).

**Acceptance** As H2, plus verify the batch dimension does not change CTC
decoding per row — decode each row independently and compare against the
unbatched result for the same crop.

---

#### H5 — Model load now dominates the tesseract lane

**Goal** Cut the ~0.37 s of load in a 0.47 s invocation (compute is only ~0.11 s
now).

**Do** Either mmap the GGUF instead of copying weights into host vectors —
`load_model` in `src/tesseract_lstm.cpp` does `.assign(...)` for conv, every
LSTM layer and the output FC, then drops the dequant cache — or amortise with a
warm server process.

**Reference point** `tesseract-cli` also pays model load on every invocation and
still totals 0.135 s, so the headroom is real.

---

#### H6 — Resize rule keyed on text height, not image size

**Goal** Replace the `upscale_floor=120` proxy in `src/ocr_detect.h` with
something that measures what actually matters.

**Why** The detector must not enlarge a page that already resolves its glyphs
(that cost 4.7x *and* accuracy), but a genuine thumbnail still needs the
enlargement to be detectable at all. Image short-side is a proxy for "are the
glyphs big enough"; glyph height is the real question. A stroke-width or
connected-component estimate would let the cap apply to a low-DPI thumbnail with
large text and stay off for a high-DPI page of tiny text.

**Acceptance, both required** No CER regression on the 20-fixture ground-truth
corpus, **and** `tests/regression/images/cc0/simple_table.jpg` (200x102) still
detects its one region. That fixture is the entire reason the floor exists —
capping it unconditionally takes it from 1 region to 0.

---

#### H7 — Point the load/compute split at the untouched engines

**Goal** Find more wasted GPU-backend inits.

**Why** The split has been applied to exactly three engines and found the bug in
two: `tesseract_lstm` was spending **4971 ms of a 5.9 s invocation** initialising
Metal for a device it never computes on, and `ppocrv6_det` ~7.3 s of a 14.6 s
stage. `grep -rl crispasr_init_gpu_backend src/*.cpp` lists **40 files**.

**Do** For each engine, ask one question: *does its compute actually run on that
backend, or is the backend only pulling the GGUF through
`core_gguf::load_weights`?* If the latter, load through
`ggml_backend_cpu_init()` and gate the old path (the pattern is in
`tesseract_lstm.cpp`: `CRISPEMBED_TESSERACT_GPU_LOAD`).

**Trap** Do not assume — the same check found that the EasyOCR and PP-OCRv6
recognizers genuinely *do* need Metal (3.65 vs 6.43 and 3.25 vs 3.75 CPU-s
forced to CPU), so removing their init would be a regression.

**Swept 2026-08-02 — three more engines had the bug.** Classifying all 40 files
by whether they ever call `ggml_backend_graph_compute` /
`ggml_backend_sched_graph_compute` leaves exactly four with none:
`tesseract_lstm` (already fixed, P2), plus **`text_sr`**, **`tps_locnet`** and
**`bert_ner`**. All three were read line by line to confirm rather than inferred
from the grep, and all three match the P2 shape exactly — the backend is created,
handed to `core_gguf::load_weights`, every weight is copied out to host vectors
via `ggml_backend_tensor_get`, and the backend is freed, with the actual compute
running in CPU-scalar code (`tsr_nafblock_forward`, `fc_forward`) or, in
`text_sr`'s case, on a *separate* `ggml_backend_cpu_init()` sched it builds
right afterwards. All three now load through `ggml_backend_cpu_init()` with the
old behaviour gated behind `TEXT_SR_GPU_LOAD`, `TPS_LOCNET_GPU_LOAD` and
`BERT_NER_GPU_LOAD`. Remaining engines are all genuine graph users; the
grep-classification table is cheap to re-run if new engines land.

**What the removal is worth, measured 2026-08-02.** No GGUF for any of the three
is in `~/crispembed-live-cache`, so instead of guessing from P2's 12.5x the cost
was measured directly at its source — the backend init itself, which is what
these engines were paying and is engine-independent. `test-backend-smoke` builds
a backend and runs one trivial graph on it; median-of-3, box at load 38:

| backend | CPU-s | wall |
|---|--:|--:|
| `metal` | 2.62 | 6.71 |
| `cpu` | 0.03 | 0.03 |

So each of the three was spending roughly **2.6 CPU-seconds / 6.7 s wall per
invocation** standing up Metal — shader library compilation dominates — to pull
a GGUF it then copies straight out to host vectors. Consistent with the 4971 ms
P2 measured inside `tesseract_lstm`. Caveat on reading this: the figure is init
plus a trivial graph, not pure init, and it is a per-process cost, so it matters
for one-shot CLI use and disappears in a warm server. An end-to-end
before/after on a real `text_sr` / `tps_locnet` / `bert_ner` model is still the
thing that would close this properly.

---

#### H8 — PP-OCRv6 detector graph geometry parity (correctness, NOT speed)

**Goal** Make `CRISPEMBED_PPOCRV6_DET_GRAPH=1` decode the same boxes as the CPU
path, so the graph is usable on CUDA/Vulkan and the diagnostic-only caveat can
come out of `docs/ocr_backend_matrix.md`.

**⚠ This is not a performance item and must not be sold as one.** The graph is
2.6–6.8x slower than the scalar path (see H2). Promoting it on this hardware
would be a regression. The value is portability and removing a caveat.

**State** Box count 31 (graph) vs 30 (CPU). Probability-map cosine is already
0.99113 and head pre-sigmoid 0.99898, so expect a DB-postprocessor disagreement
on one borderline contour — threshold, unclip ratio or dilation applied to a map
with slightly different edges — rather than an arithmetic bug.

**Tooling** A comparator exists: `report_graph_box_geometry` in
`src/ppocrv6_det.cpp` (~line 107), enabled by
`CRISPEMBED_PPOCRV6_DET_GRAPH_COMPARE=1`, printing
`graph=N cpu=M matched=K mean_iou= min_iou=`. **It did not emit on a 2026-08-02
run — check its call site (~lines 1012 and 1120) before trusting it.**

**Do** Get the comparator emitting, find the unmatched box, then dump both
probability maps for that region and decide whether the divergence is in the map
or the postprocessor.

### PP-OCRv6 detector-to-recognizer contract (selected follow-up)

The PP-OCRv6 port must follow the official PaddleOCR/RapidOCR handoff rather
than treating a segmentation result as an axis-aligned crop. The canonical
path is:

`PP-OCRv6 DB segmentation → DBPostProcess quadrilateral → ordered perspective
warp → line-orientation classifier → PP-OCRv6 recognizer`

Requirements:

1. Keep the detector's four-point polygon through postprocessing, including
   Paddle's contour score, `thresh=0.2`, `box_thresh=0.45`, `unclip_ratio=1.4`,
   candidate cap, clipping, and reading-order sort. The external
   `$CRISPEMBED_GGUF_DIR/dbnet-ic15-f16.gguf` remains a fallback
   detector, not a replacement for the PP-OCRv6 detector.
2. Add a shared quadrilateral crop helper equivalent to RapidOCR's
   `get_rotate_crop_image`: canonical TL/TR/BR/BL ordering, perspective
   transform, width/height derivation, clipping, and deterministic minimum
   height padding. Do not collapse rotated or skewed regions to an
   axis-aligned rectangle.
3. Port the optional PP-LCNet text-line orientation classifier (0°/180°),
   exposing predicted angle, confidence, and whether a rotation was applied
   through native and C API results. The existing heuristic 180° detector is
   only a fallback when the classifier is unavailable.
4. Apply the official recognizer input contract after warping: RGB, height 48,
   aspect-preserving width, cap/pad to 320, declared normalization, and the
   model dictionary/CTC decode. Preserve sensitive head weights at higher
   precision and keep `-ref.gguf` fixtures for each tier.
5. Provide one comparison harness for four combinations: PP-det→PP-rec,
   DBNet→PP-rec with grouped line boxes, DBNet→PP-rec word boxes, and the
   direct legacy DBNet pipeline. Report detector polygons, crop dimensions,
   orientation decisions, region/text counts, latency, CER/WER where an oracle
   exists, and `crispembed-diff` cosine metrics.
6. Gate tiny/small/medium on the 10 CC0 fixtures plus derived rotation,
   skew, perspective, low-DPI, and mixed-orientation fixtures. A candidate is
   not publishable until Python/PaddleX and native agree on crop geometry and
   the complete pipeline no longer emits full-page recognizer garbage on the
   German fixtures.

Parity note (2026-07-31): `inference.yml` declares `DecodeImage(img_mode=BGR)`;
the public crop handoff remains RGB, with the recognizer swapping to BGR at
its model boundary. Exact native detector crops now reproduce the visible
German title geometry. On the Fraktur crop, Python and native logits agree at
cosine 0.99998--0.99999, but both decode `Rieilhs–刊臂懒s²ł1&tt.`; this is a
source-model recognition-quality limitation, not a native crop/inference
parity failure. Do not publish a quality claim for this fixture until a
Paddle/Python official run confirms whether the model itself has the same
limitation.

Implementation order: shared quad warp and telemetry; PP-LCNet classifier;
PP-det→PP-rec integration; DBNet fallback adapter; per-stage diff fixtures;
benchmark and regression manifest; then publish only to `cstr/` from the
external model volume. This work must be merged from remote `main` before
each landing checkpoint and pushed back to remote `main` after the checkpoint.

### Historical German Fraktur OCR inventory and port plan (2026-07-31)

The directly portable German-Fraktur model is the official Tesseract `frk`
traineddata shipped by `tesseract-lang`. Local inspection with
`combine_tessdata -u` confirms an LSTM network, `lstm-unicharset`, and
`lstm-recoder`; the existing converter successfully produced
`$CRISPEMBED_GGUF_DIR/tesseract-frk-f32.gguf` (3.6 MiB, 933,763
parameters). This is the first Fraktur model to use in the native
`tesseract_lstm` GGUF path. Preserve the output alphabet, including long-s
and historical characters, and keep the sensitive output layer at F32 for
the first quantization gate. The upstream Tesseract language data and
runtime are Apache-2.0; retain the exact source checksum and attribution in
the eventual model README.

| Source | What it contains | License/provenance | GGML/GGUF decision |
|---|---|---|---|
| [Tesseract `frk`](https://github.com/tesseract-ocr/tessdata) | Current LSTM Fraktur recognizer; local `frk.traineddata` is the 4.00-alpha synthetic-trained network with 100 output codes | Apache-2.0 Tesseract language data; verify exact tessdata revision/checksum per artifact | **Port now:** existing `convert-tesseract-to-gguf.py` and `tesseract_lstm.cpp`; test on German Fraktur crops and full pages |
| [`paalberti/tesseract-dan-fraktur`](https://github.com/paalberti/tesseract-dan-fraktur) `deu_frak` | German Fraktur Tesseract 3.02 package containing the compiled `deu_frak.traineddata`, `deu_frak.config`, `unicharambigs`, dictionaries/lists, build script, and many paired `.tif`/`.box` training samples (including font and scanned-page examples) | Repository `COPYING` says Apache-2.0; retain attribution and separately verify provenance of the historical source scans and annotations before redistributing or using them as a new corpus | **Compatibility and training-data candidate:** the legacy package is not an LSTM GGUF input and must not be passed to the current converter as if it were; use it with a legacy Tesseract 3.02 path, or use the paired image/box material for clean-room retraining into a supported LSTM format after provenance review |
| [`jze/ocropus-model_fraktur`](https://github.com/jze/ocropus-model_fraktur) | OCRopus `pyrnn.gz` and CLSTM Fraktur character models; reports 1.089% held-out error on its own test set | No explicit license found in repository; source books/datasets also need provenance review | **Do not redistribute/port yet:** license clarification required; format is not Tesseract LSTM |
| [`chreul/19th-century-fraktur-OCR`](https://github.com/chreul/19th-century-fraktur-OCR) | MIT Calamari five-model voting ensemble, single Calamari model, and OCRopus model for 19th-century German Fraktur; models expect binarized line images; repository includes at most 50 lines per book so users can adapt transcription guidelines, while the complete corpus is obtained from the linked GT publication | Repository MIT; the supplied model/adaptation files are distinct from the complete training corpus, whose source and ground-truth licensing must be checked independently | **Benchmark/retraining candidate:** no current GGUF importer; use the supplied samples for reproducible adaptation tests, then retrain into a supported LSTM graph if the full corpus permissions and target transcription policy are clear |
| [`yingyangle/fraktur`](https://github.com/yingyangle/fraktur) | Research project using GT4HistOCR line data, character segmentation, zoning/black-pixel features, and a k-NN classifier; reports about 92% character accuracy | **No repository license or explicit reuse grant found.** GT4HistOCR itself is CC-BY-4.0, but that does not license this repository's code, pickles, feature data, or derived artifacts | **Reference only:** do not copy, train from its artifacts, or redistribute until the authors clarify licensing; independently reproduce the simple feature baseline from the separately licensed GT4HistOCR data if useful |
| [`UB-Mannheim/AustrianNewspapers`](https://github.com/UB-Mannheim/AustrianNewspapers) | NewsEye/READ Austrian newspaper ground truth, 1864–1911; revised PAGE-XML with Fraktur/Antiqua, baselines, regions, and long-s transcription | Original dataset explicitly CC BY 4.0; preserve attribution to Mühlberger/Hackl, Austrian National Library, and the repository revision | **Eligible training source:** strong Fraktur/Antiqua classifier and Tesseract fine-tuning corpus after source/page split and attribution manifest |
| [`UB-Mannheim/reichsanzeiger-gt`](https://github.com/UB-Mannheim/reichsanzeiger-gt) | 119,429 lines from 197 German newspaper pages, 1820–1939; Fraktur and Latin, with long-s and historical characters | Current GitHub repository advertises CC0-1.0; verify the exact revision and downloaded scan URLs before redistribution | **Highest-priority training/evaluation source:** large, directly relevant Fraktur corpus; use for classifier and Tesseract fine-tuning, preserving its transcription conventions |
| [`ulb-sachsen-anhalt/ulb-groundtruth-eval-odem-ger`](https://github.com/ulb-sachsen-anhalt/ulb-groundtruth-eval-odem-ger) | OCR-D Phase III ULB VD18 German-Fraktur Page-XML ground truth: 39,823 text lines across 1,026 pages and 6,298 text regions, dated 1700–1799, with non-text region annotations | Repository metadata says CC-BY-4.0, while the GitHub license badge says CC-BY-SA-4.0; treat it conservatively as CC-BY-SA-4.0 until ULB resolves the discrepancy, and preserve the repository citation | **Highest-priority training/evaluation source:** convert Page-XML lines/regions into explicit line crops and manifests; useful for 18th-century Fraktur, layout/region evaluation, and clean-room LSTM fine-tuning once the license is settled |
| [`UB-Mannheim/dach-gt`](https://github.com/UB-Mannheim/dach-gt) | Ground truth and full text for selected prints from German libraries; repository ships data plus image-download tooling and supports PAGE/ALTO/escriptorium workflows | CC0-1.0 repository license; retain institution/source URLs and verify any remote scan terms | **Eligible broad training source:** mine Fraktur lines and full-text alignment for detector/recognizer evaluation and clean-room training; keep each institution as a separate provenance split |
| [`jbaiter/archiscribe-corpus`](https://github.com/jbaiter/archiscribe-corpus) | 4,255 German-Fraktur lines from 112 works across 73 years (1800s–1890s), with transcription directories and Archive.org/IIIF source links | CC-BY-4.0; attribution and source-work provenance are required | **Eligible evaluation/adaptation source:** strong chronological/style diversity; import the line images and transcriptions only with an attribution manifest and use work-level splits to avoid leakage |
| [`UB-Mannheim/charlottenburger-amtsschrifttum`](https://github.com/UB-Mannheim/charlottenburger-amtsschrifttum) | 26 pages of German Fraktur ground truth from 1879–1919, including long-s (ſ), German Mark (ℳ), double oblique hyphen (⸗), fractions, and downloadable image URLs | CC0-1.0 | **Small but valuable regression/adaptation fixture:** use for historical-character coverage, line-recognition tests, and domain adaptation; do not treat its 26 pages as a standalone general corpus |
| [`UB-Mannheim/Reichsanzeiger`](https://github.com/UB-Mannheim/Reichsanzeiger) | Apache-licensed software/data support for the digital edition of *Deutscher Reichsanzeiger und Preußischer Staatsanzeiger*, including the SQL image/issue mapping and project metadata | Apache-2.0 repository; the underlying scans, full text, and linked digital-edition assets must retain their own terms | **Pipeline/provenance reference, not a recognizer:** use the issue/image mapping to locate and reproduce evaluation pages; keep this Apache repository separate from the `reichsanzeiger-gt` CC0 ground-truth corpus |
| [`SimoneRebora/OCRFraktur`](https://github.com/SimoneRebora/OCRFraktur) | OCRopus Fraktur training project: 4,287 real lines from *Tiroler Soldaten-Zeitung*, 3,000 synthetically generated lines from three Fraktur fonts, a 410-line test set, and a `TSZ_Fraktur_model.pyrnn.gz` model reporting 0.479% CER on its own test set | **No repository license found.** The README cites the Musil edition and source texts/fonts; those inputs have independent provenance and must not be assumed redistributable | **Reference only until licensing is clarified:** valuable for synthetic-data design and an OCRopus baseline, but no redistribution, training, or GGUF conversion from its artifacts; independently recreate the experiment from cleared sources if needed |
| [`phildiderichsen/MeMo-Fraktur-OCR-code`](https://github.com/phildiderichsen/MeMo-Fraktur-OCR-code) | Rule-based correction/evaluation workflow: Tesseract re-OCR with `frk`, `fraktur`, and `dan`, alternative OCR comparison, regex/context rules, SymSpell, VRT alignment, and staged intermediate outputs | **No repository license found.** Its PDFs, OCR outputs, dictionaries, and external corpora require separate permission review | **Concepts only:** reproduce the error-analysis, multi-model voting, historical spelling, hyphenation, and correction stages in our own code; do not copy source or derived artifacts into a distributable model/data package |
| [`muratyanasoglu/AI-Powered-OCR-Project-for-Ancient-Languages-Ancient-Greek-Fraktur-Old-German-Classical-Latin`](https://github.com/muratyanasoglu/AI-Powered-OCR-Project-for-Ancient-Languages-Ancient-Greek-Fraktur-Old-German-Classical-Latin) | Django/PyTesseract application using Tesseract 5, OpenCV/Pillow preprocessing, configurable PSM/OEM, `deu_latf`/`frk`/other language packs, and optional Gemini translation/analysis; no custom OCR weights | No explicit repository license found in the inspected tree; Tesseract language data and Gemini service have separate terms | **Workflow reference only:** useful ideas for script-specific preprocessing and output export, but it contributes no native recognizer, detector, or GGUF-compatible weights; do not bundle its external language packs or API integration without review |
| [`Nargizi/oppy`](https://github.com/Nargizi/oppy) | Small German-Fraktur PDF OCR wrapper; dependencies show PyMuPDF, OpenCV, `pytesseract`, and `fsspec`, with package modules for PDF/image/text handling | MIT repository license; Tesseract and any downloaded language data remain separately licensed | **Reference only:** it uses the system Tesseract executable through PyTesseract rather than a custom model; compare its PDF rasterization/preprocessing ideas with our native pipeline, but there is nothing to convert to GGUF |
| [`UB-Mannheim/digitue-gt`](https://github.com/UB-Mannheim/digitue-gt) | CC0 transcriptions for digitized Tübingen books/journals, including Fraktur-tagged material; images fetched from UB Tübingen URLs | Repository advertises CC0-1.0; confirm that downloaded image endpoints carry compatible reuse terms and keep source URLs | **Eligible source:** mine Fraktur/Antiqua line crops for clean-room classifier training and evaluation; retain provenance even under CC0 |
| [`samprietoserrano/archival-ocr-transcription`](https://github.com/samprietoserrano/archival-ocr-transcription) | MIT-licensed Python workflow and outputs for a 1719 German Fraktur book by Peter Kolb; extraction used Google Document AI and Transkribus, followed by local cleanup, historical spell-checking, and reading-order assembly | Repository MIT; Internet Archive scans, DTA corpora, CLARIN GeMiCorpus, and Transkribus/Google outputs retain their own terms | **Reference/evaluation source, not a model port:** useful for a historical Fraktur page fixture, reading-order and post-correction tests, but it supplies no portable LSTM/CNN weights. Audit source-image and corpus permissions before using its text/images for training |
| SchriftLotse `party-v4` | Swin-base vision encoder plus 40M-parameter Llama decoder; page-wise recognition with line prompts | Apache-2.0 model release | **Not a near-term port:** custom Kraken multimodal architecture; potentially reusable after a dedicated Swin/Llama graph and tokenizer audit |
| SchriftLotse Kraken BLLA | Trainable neural baseline/line segmenter | Apache-2.0 via Kraken/model release | **Not a near-term port:** PyTorch Kraken BLLA format and baseline geometry are separate from DBNet; evaluate as an external segmentation oracle first |
| SchriftLotse `orli` | ConvNeXtV2-tiny encoder, multi-scale adapter, autoregressive transformer decoder for baselines and reading order | Apache-2.0 model release | **Not a near-term port:** new autoregressive layout graph; consider only after native line/ordering gates |
| SchriftLotse `trocr-kurrent-19`, `trocr-kurrent-early`, `trocr-medieval` | Historical handwriting TrOCR checkpoints | MIT according to the inventory | **Potential later port:** assess against existing TrOCR encoder/decoder implementation; verify exact HF architecture and tokenizer before conversion |
| SchriftLotse `trocr-modern` | German handwritten TrOCR checkpoint | AFL-3.0 according to the inventory | **Potential later port with license review:** same TrOCR compatibility check, but AFL-3.0 obligations must be documented |
| SchriftLotse Microsoft TrOCR processor | Processor/tokenizer only, not a recognizer model | MIT according to the inventory | **Reuse only as preprocessing/tokenizer reference;** no standalone GGUF model |
| SchriftLotse `qwen-embed` | Qwen3 Embedding 0.6B semantic search model | Apache-2.0 | **Already covered by decoder embedding infrastructure;** not an OCR recognizer |
| [Xilinx `LSTM-PYNQ`](https://github.com/Xilinx/LSTM-PYNQ) `Fraktur_OCR.ipynb` | Quantized BiLSTM Fraktur OCR overlay for PYNQ; notebook constructs `PynqFrakturOCR`, downloads FPGA bitstream and Fraktur weights, and recognizes lines from *Wanderungen durch die Mark Brandenburg* | Repository BSD-3-Clause; the notebook cites the FINN-L paper and an Insiders Technologies text dataset, whose data/model provenance must be audited separately | **Reference only for now:** the model is coupled to FINN/PYNQ fixed-point hardware and does not expose a standard Tesseract/PyTorch checkpoint. Porting would require recovering the quantized layer weights, codec, preprocessing, and CTC decode from `lstm/src/network/fraktur`; do not assume it is interchangeable with `tesseract_lstm` GGUF |

OCR-D’s historical-print catalogue narrows the practical Tesseract targets:

| OCR-D/Tesseract resource | Meaning | Port status |
|---|---|---|
| `deu_latf` (formerly `frk`) | Current German Fraktur language model with some Antiqua coverage | **Directly portable:** use the local `frk.traineddata` LSTM conversion already validated; preserve the upstream name/revision mapping in metadata |
| `Fraktur` | Broader Fraktur script model, including non-German characters and some Antiqua | **Candidate:** obtain the exact `.traineddata`, verify its `lstm` component, then run the same converter/parity/Fraktur regression gates |
| GT4HistOCR-derived models (`GT4HistOCR_*`, `frak2021`, UB Mannheim models) | Historical-print models trained from GT4HistOCR and related German/Fraktur corpora; OCR-D recommends these for broad historical coverage | **Highest next priority:** locate the exact `.traineddata` artifact and license/attribution terms, convert if it contains an LSTM component, and compare against `deu_latf` on our corpus |
| `deu_frak` | Older Tesseract 3 German Fraktur model | **Legacy benchmark only:** OCR-D explicitly says it is no longer recommended; it is not a current LSTM conversion input |

OCR-D documents that Tesseract 4.1+ `.traineddata` files contain an
`unicharset` and neural `lstm` weights, and that models can be combined (for
example `deu+deu_latf` or `Fraktur+Latin`) at an accuracy/runtime cost. The
official Tesseract language data is Apache-2.0, but GT4HistOCR training data is
CC-BY-4.0 and individual UB Mannheim derivatives may have additional
attribution or release conditions. Record the exact model URL, checksum,
training corpus, and license before placing any derivative in the shared
model volume or publishing a GGUF.

Required Fraktur implementation sequence:

1. Add `tesseract-frk-f32.gguf` and a sensitive-head `q8_0` derivative only on
   `$CRISPEMBED_GGUF_DIR`; never commit large weights.
   Apply the same policy to every `tesseract_lstm` language variant: all
   quantized Q8/Q4 artifacts must retain `output.weight` and `output.bias` at
   the source precision (F16 or F32), while only recurrent matrices may be
   quantized. Existing multilingual artifacts must be regenerated if their
   output projection was previously quantized.
2. Add an explicit `tesseract-fraktur` stage/profile using DBNet line crops →
   grayscale crop → `tesseract_lstm` `frk` model, with the normal Tesseract
   path remaining available for modern German.
3. Add a Fraktur regression fixture containing the German title crop and
   full-page `german_official_print.jpg`; compare native GGUF, Python/
   Tesseract, and system Tesseract `-l frk` outputs with CER/WER where an
   oracle exists. Preserve `ſ`, `ß`, ligatures, and Unicode normalization in
   the comparison.
4. Run crispasr-diff-style intermediate parity for the converted `frk` model,
   then test F32, head-only Q8, and debug Q4. Do not publish a quant until
   the Fraktur crop remains readable and the output-layer parity gate passes.
5. Keep `deu_frak` and the Calamari/OCRopus models as separately licensed
   external benchmarks; do not silently convert or redistribute them as
   Apache artifacts.
6. Add OCR-D/GT4HistOCR model comparisons (`deu_latf`, `Fraktur`, and the
   best available `frak2021`/GT4HistOCR derivative) before deciding whether
   `tesseract-frk-f32.gguf` is the production default. Include the Xilinx
   LSTM-PYNQ result only as a hardware-reference baseline unless its weights
   and codec can be legally and technically recovered.

### Fraktur line-classifier retraining (clean-room alternative to AGPL model)

`impresso-project/frakturline-classification-cnn` is useful as a routing
component, not as a recognizer: it classifies a grayscale line crop as
`fraktur` or `other`. Its published architecture is small and reproducible:
approximately 2.1M parameters, input `1x60x800`, three convolutional stages
with ReLU/max-pooling and LayerNorm, adaptive max-pooling to `1x8`, then
`1024 -> 128 -> 1` fully connected layers. The model card reports a 99.75%
accuracy result on its held-out test set, but the model and the linked
Impresso datasets are AGPL-3.0. Do not copy its weights, code, or dataset
into a permissively licensed CrispEmbed artifact.

We can train an independent equivalent classifier from the published
architecture and independently licensed data:

1. Use CC-BY-4.0 GT4HistOCR Fraktur lines and the MIT-licensed
   [`chreul/19th-century-fraktur-OCR`](https://github.com/chreul/19th-century-fraktur-OCR)
   repository's included sample/model-training material for the positive
   class, with attribution and source revision/checksum. The chreul
   repository is MIT-licensed and explicitly supplies small training samples,
   Calamari/OCRopus models, and source-book provenance; that MIT grant does
   not automatically relicense the original historical scans or every
   upstream source corpus, so audit each selected file before redistribution.
   Build the `other` class from separately licensed Antiqua/Latin line corpora,
   or annotate our own public-domain CC0 fixtures. Do not derive a replacement
   dataset by relabeling or mirroring the AGPL Impresso dataset.
2. Split by source publication/page, never by individual crop, to prevent
   font/page leakage. Add hard negatives: ornate Antiqua, mixed lines,
   ornaments, headers, low-DPI scans, skew, bleed-through, and short lines.
3. Reimplement the architecture independently in the existing Python
   training/reference tooling, export an ONNX/PyTorch state dict, and write a
   small GGUF converter with F32 classifier head and F16 convolution weights.
   The native runtime should use the existing crop helper and expose
   `fraktur_probability`, `script_class`, and an abstain/uncertain state.
4. Use the classifier only to route line crops: `fraktur` → `frk`/`deu_latf`
   or a GT4HistOCR-derived Tesseract model; `other` → modern German/Latin
   recognizer. It must not replace recognition or silently alter text.
5. Validate on an independently held-out Fraktur/Antiqua corpus and our
   German page fixtures. Report balanced accuracy, precision/recall/F1,
   calibration, abstention rate, and downstream OCR CER/WER against always
   `frk`, always `deu`, and the classifier route. A 99% classifier score is
   not sufficient if routing worsens recognition.
6. Publish only the independently trained weights and a complete data/
   attribution manifest. Keep the AGPL Impresso model as an external
   benchmark/reference, not as a CrispEmbed dependency.

EasyOCR checkpoint: the CRAFT detector graph now passes Python input, VGG taps,
U-Net feature map, NHWC score-map, and decoded box-count parity on CPU and
Metal; DBNet→EasyOCR crop smoke now runs with the existing cstr/dbnet-ic15
artifact. The next slice explicitly separates detector geometry from ordering:
EasyOCR line grouping, Tesseract/LayoutLM word ordering, and one structured
handoff contract. DBNet Python box/text parity and production orchestration
remain pending.

The reusable `easyocr_pipeline` now exposes the selected `lines` and `words`
policies, runs native DBNet detection plus GPU-resident EasyOCR recognition,
and returns text, detector/recognizer confidence, pixel boxes, reading-order
metadata, and normalized LayoutLM boxes. The model-backed regression produces
12 line records and 98 word records on `scan_strip.png`; Python box/text
reference parity remains a separate pending gate.

The harness-blind postprocessing reference is now explicit: Python EasyOCR
`readtext(detail=1)` JSON is converted by
`tools/easyocr_postprocess_reference.py` into a versioned manifest covering
ordering, crop geometry, text, confidence, and LayoutLM normalization. Its
model-free test passes both line and word policies.

The native `test-easyocr-pipeline` can emit the same manifest schema, and
`tools/compare_easyocr_manifests.py` reports the first record-level mismatch.
The Miniconda runner `tools/run_easyocr_reference_page.py` now produces an
independent Python CRAFT+English Gen-2 `readtext(detail=1)` manifest. On
`scan_strip.png`, Python produced 11 lines while native DBNet→CRNN produced
12. The first mismatch is line 0 (`"They are going to be , encamped near
Brighton"` versus `& They are going to be, encamped near   Brighton`), with
geometry `[62,0,412,25]` versus `[46.97,0,423.54,21.76]`; the rest therefore
cannot be zipped as equivalent lines. Python recognition confidence was
`0.8541` versus native `0.4472` on that first record. Detector confidence is
explicitly unavailable from EasyOCR's public tuple and is not fabricated;
the comparator has `--ignore-detector-confidence` for this case. The manifests
are backed up under `/Volumes/backups/ai/crispembed-gguf/`. Page text/geometry
parity is failed and remains a quality TODO; this is evidence, not a claim
that either detector is universally better.

The detector-independent production handoff is now explicit: `run_regions`
accepts caller-supplied detector boxes and applies the configured lines/words
ordering, crop, recognizer, and LayoutLM normalization path. The model-backed
pipeline test replays the DBNet boxes through this API and matches the normal
98-record run; this validates the boundary, not external Tesseract TSV parity.
The public signature now accepts only `easyocr_layout::region`, keeping
DBNet-specific types inside the implementation. The compile/link proof passes;
the post-merge model replay is currently blocked by the unrelated shared
`ggml` submodule checkout difference and is not claimed green.

The first real page comparison confirms why TSV parity cannot be asserted by
zipping records: native DBNet/CRNN `words` mode emits 98 records, while
Tesseract 5.5.2 `--psm 6` emits 106 TSV words. The first geometry already
differs before recognition (`[46.97,0,62.56,20.88]` native versus
`[50,0,58,19]` TSV), and later indices shift with segmentation. Full-page
Tesseract TSV also reads `Drighton;` on this fixture, whereas the instrumented
internal PSM7 crop reads `Brighton`; page segmentation and crop selection remain
the active parity gate.

The harness-blind CTC/vocabulary/confidence gate is now covered by the native
`easyocr_postprocess` module and `test-easyocr-postprocess`. CTC uses blank 0
with repeated-token collapse, vocabulary entries are 1-based and validated,
and confidence follows EasyOCR's nonblank `custom_mean` formula.

The page-pipeline audit found that EasyOCR's `get_image_list` uses OpenCV
`resize(..., interpolation=1)` (bilinear), whereas the standalone recognizer
fixture was generated with PIL bicubic. An experimental native bilinear
substitution failed the existing diff at `sequence_input`, `bilstm_0`, and
`logits` (`Ea` versus `5a`), so it is not retained in production until a
matching bilinear `-ref.gguf` is regenerated.

An experimental strict port of EasyOCR's horizontal gap thresholds was also
rejected: applied directly to the DBNet artifact it split the 98 fragmented
regions into 26 recognition units instead of the existing 12 line units.
DBNet therefore needs a detector-specific line adapter before those thresholds
can replace the current y-band grouping.

The first DBNet adapter is now explicit in `easyocr_layout`: it preserves the
fragment-tolerant y-band aggregation for DBNet line crops, while word mode
continues to use left-to-right y-band ordering. The adapter is covered by the
layout regression and the model-backed page smoke; horizontal-gap splitting is
still a later detector-specific refinement.

### Tesseract parity status — controlled line proven; page/CLI parity pending

The repository contains a `.traineddata` → GGUF converter, a pure-Python
Tesseract LSTM reference dumper, and `test-tesseract-lstm-diff`. That is an
available validation path, not evidence that the shipped models match the
original Tesseract engine. The controlled exact `eng.traineddata` line run is
now complete; backup artifacts must still be regenerated consistently when
model metadata changes, and page/CLI parity remains separate.

The native implementation is also a line recognizer. Tesseract's page
segmentation, word boundaries, spacing, and reading order are separate
postprocessing behavior. A Python forward pass can prove GGUF/runtime math
against parsed weights, but does not by itself prove full Tesseract CLI parity.
Do not mark the full lane green until page segmentation, spacing, reading order,
and decoded page output are compared with the original engine.

> **Board cleared 2026-07-20** — all 18 previously-listed in-flight items had
> landed; the index + preserved specifics are in `HISTORY.md` "July 20, 2026 —
> PLAN.md active-work board cleared". Add a row here when you START a task; remove
> it when the branch lands.
>
> Completed milestones live in `HISTORY.md`; technical deep-dives in
> `LEARNINGS.md`. This file tracks the current architecture and what is
> still **pending**.

## OCR pipeline workstream — actionable items

### EasyOCR / LayoutLM compatibility — IN PROGRESS

- Port all EasyOCR detector assets (CRAFT, DBNet18, DBNet50) and all shipped
  Gen-1/Gen-2 CRNN recognizers through the GGUF → `-ref.gguf` →
  `crispembed-diff` → decoded-output protocol.
- Preserve per-artifact source/license metadata; do not infer a fine-tuned
  checkpoint's license solely from its backbone.
- [x] Add a weight-free LayoutLMv2/v3 handoff contract for externally produced
  words and normalized boxes. `tools/validate_layoutlm_handoff.py` emits the
  exact `apply_ocr=False` processor payload and retains confidence/pixel boxes
  as sidecar metadata; Transformers' `apply_ocr=True` path uses PyTesseract.
  A live model invocation remains unnecessary for this contract gate.
- Acceptance requires reference parity, decoded text, and real pipeline output
  checks before quantization or registry integration.

#### OCR interoperability blueprint — selected item

The engines do not share one detector. Tesseract supplies page segmentation and
LSTM line recognition; EasyOCR uses CRAFT by default and optionally DBNet, then
groups boxes into line crops for its recognizer; LayoutLMv2/v3 processors normally
call PyTesseract and consume words plus normalized boxes, while
`apply_ocr=False` accepts an external OCR result. Transformers as a library has
no universal OCR detector: TrOCR is recognizer-only and OCR-free image-language
models use their own visual encoder.

The native boundary is therefore:

`detector (CRAFT | DBNet | external/Tesseract geometry) → boxes/scores →
ordering/grouping policy → word or line crops → recognizer → structured words`

Every production path must emit the same weight-free record: text, pixel box,
confidence, block, line, reading-order index, and optional `[0,1000]`
normalized box. The selected implementation keeps two explicit policies:

- `lines`: EasyOCR-compatible y-grouping and x-order, followed by dynamic-width
  EasyOCR CRNN recognition.
- `words`: Tesseract/LayoutLM-compatible word ordering, preserving individual
  boxes and confidence for downstream processors.

Do not port Tesseract as a DBNet checkpoint. Port its segmentation/order
semantics where useful, retain native CRAFT/DBNet inference, and compare the
policies on the same page fixtures. Acceptance is decoded page text and
downstream handoff parity, not detector-box similarity alone.

#### Interoperability gates

- [x] Make detector output and ordering policy first-class production adapters;
      `easyocr_pipeline::run_regions` now accepts detector-independent boxes and
      applies the selected `lines`/`words` policy through the production crop /
      recognizer path; the pipeline test exercises the injected-geometry handoff.
- [x] Validate `lines` against EasyOCR grouping and decoded line text on a
      page fixture with a Python reference manifest. The live comparison is
      intentionally failed: Python CRAFT yields 11 lines, native DBNet yields
      12, and the first text/geometry/confidence mismatch is recorded above;
      detector/order/crop quality parity remains open.
- [ ] Validate `words` against Tesseract TSV-style geometry/order and preserve
      confidence, pixel boxes, and normalized LayoutLM boxes.
- [x] Add native handoff invariants for word-mode line/x ordering and
      normalized-box bounds; external Tesseract TSV geometry/text parity is
      still pending.
- [x] Add a standard-library Tesseract TSV geometry/order comparator and
      self-test; a real page comparison remains an evidence gate, not a claim
      of Tesseract text parity.
- [ ] Run the same structured words through LayoutLMv2/v3 with
      `apply_ocr=False`; verify serialization and ordering independently of
      logits.
- [x] Keep Tesseract LSTM as a separately measured recognizer lane; the
      `test-ocr-identical-crops` harness feeds the exact same RGB crop to the
      dynamic-width EasyOCR CRNN and grayscale Tesseract LSTM, then reports both
      confidence conventions. On three official TSV boxes, both lanes preserve
      the text structure but differ in ambiguous characters/punctuation (for
      example `I`/`[` and the final quote), confirming these are recognizer
      outputs rather than detector/order differences.
- [x] Prove the controlled line-recognizer boundary separately: the exact
      Homebrew `eng.traineddata` hash, Python `-ref.gguf`, native captures,
      decoded text, and official instrumented PSM7 internal crop all match;
      logits differ by at most `6.6e-7` with cosine `1.000000`.
      New reference dumps record the source image dimensions, and the native
      diff harness rejects a mismatched fixture before reporting stage cosine
      scores; this prevents stale `-ref.gguf` archives from masquerading as
      model or runtime parity failures.
- [ ] Compare page segmentation, spacing, and CLI crop geometry independently;
      this remains open because direct line fixtures are not the same internal
      crops selected by official PSM7.
      The native classical page-segmentation adapter is now wired behind the
      explicit `--tesseract-pageseg`/stage option and has a model-free synthetic
      geometry regression. It now also bypasses generic scan cleanup so page
      segmentation sees original-image coordinates, and its row threshold
      rejects sparse antialiasing bridges on gray paper; this does not close
      real CLI parity.
      On `scan_strip.png`, the tuned native CLI path improved from 3 to 7
      decoded regions, while official Tesseract `--psm 3/6` emits 12 lines;
      exact RGB-to-gray conversion is now shared with the proven reference.
      Height-based splitting now recovers 12 candidates and 12 decoded regions
      on `scan_strip.png`; crop widths are tightened per split band. Text still
      differs on `Meryton` and punctuation/quotes, so decoded page parity and
      official crop equivalence remain open.
      The env-gated page-segmentation path now also bypasses cleanup (the
      structured `page_segmentation` parameter already did), so it preserves
      original-page coordinates. On `scan_strip.png` the component fallback
      recovers 12 regions, 567 native characters, and CER/WER `0.0179/0.0841`
      against official 12-line output; native mean confidence is `0.895` versus
      official mean word confidence `0.9108`. This improves geometry and text
      parity but does not make the experimental path production-default.
      The `CRISPEMBED_TESSERACT_COMPONENT_PAGESEG` control now reaches the
      documented component prototype through the orchestrator; its current
      `scan_strip.png` result is 12 regions, 569 chars, and confidence `0.878`,
      so the legacy/fallback adapter remains the default classical choice.
      Review of Tesseract `textord/makerow.cpp` confirms its authoritative
      boundary is connected blobs assigned by vertical overlap, line size,
      spacing, and fitted baselines; our projection splitter is only an
      interim adapter. An opt-in component prototype is available behind
      `CRISPEMBED_TESSERACT_COMPONENT_PAGESEG`. After a Tesseract-style
      reassociation pass for short/detached blobs it produces the expected 12
      rows on `scan_strip.png`, but its enlarged first-line crop currently
      worsens recognizer output, so it remains experimental and is not enabled
      in production.
      `tools/compare_tesseract_page_geometry.py` now measures the independent
      geometry boundary from official TSV level-4 rows. On `scan_strip.png`
      The legacy component path remains the default because the German
      official-print gate regressed under the newer baseline matcher. With
      `CRISPEMBED_TESSERACT_COMPONENT_BASELINE=1`, the baseline experiment now
      has 12/12 indexed rows with mean IoU `0.813562` after vertical crop
      tightening; the projection fallback has 12/12 with mean IoU `0.865993`.
      Its first-line crop is `[48,0,434,20]` and short final-row crop
      `[27,237,72,22]`; both page ends decode coherently, although the
      baseline variant drops the final exclamation mark. Character choices and
      quote/spacing differences remain a
      decoded-text parity gate. A page-level beam A/B at widths 1, 5, 10, and
      25 keeps the same first-line choices; generic CTC beam search remains
      opt-in and is not the cause of the remaining CLI discrepancy.
      The geometry comparator now passes detector and recognizer models
      separately through the CLI (`--ocr-det`/`--ocr-rec`); its component
      prototype measurement is 12/12 lines with indexed mean IoU `0.826222` on
      `scan_strip.png`, rather than the previous false zero-row result.
      The comparator also accepts optional `--min-native-lines` and `--min-iou`
      gates; the component fixture passes at 12 lines and IoU `0.82`, while
      diagnostic runs without thresholds remain non-gating.
      Geometry reports now include detector/recognizer SHA-256 hashes and the
      indexed reading-order policy, matching the aggregate page-metrics
      provenance record without serializing local model paths.
      The geometry comparator now also reports mean/max absolute crop deltas
      and mean absolute inter-line gap deltas, with optional max-delta gates;
      this keeps spacing/crop drift visible even when row count and IoU pass.
      Its projection/component/baseline policy flags are now mutually
      exclusive and clear inherited policy variables before each run, so an
      A/B result cannot silently use a stale segmentation mode.
      It now reports monotonic reading-order checks for both TSV and native
      boxes and can gate them with `--require-reading-order`; indexed IoU is
      therefore no longer the only signal for an ordering regression.
      `tests/test_tesseract_page_geometry.py` covers the ordering, crop-delta,
      spacing-delta, and equal-count ordering-regression cases without model
      files.
      The regression workflow now runs this test in the Tier 0 smoke job and
      triggers on changes to the comparator test, keeping these gates active
      in PR and `main` CI.
      The aggregate page-metrics harness now selects `legacy-fallback`,
      `component`, `baseline`, or `projection` explicitly and clears stale
      policy environment variables between runs. On `scan_strip.png`, the
      measured CER/WER are `0.0179/0.0841`, `0.0322/0.1121`,
      `0.0179/0.0841`, and `0.0250/0.1121` respectively; legacy-fallback
      remains the best default.
      Each aggregate JSON record now includes detector and recognizer
      SHA-256 hashes plus the ordering-comparison policy, without writing
      local model paths into reports.
      The aggregate harness now supports opt-in minimum-region and maximum
      CER/WER acceptance gates and reports their individual results, allowing
      page-quality regressions to fail explicitly instead of being hidden in
      diagnostic JSON.
      The model-free geometry regression also covers aggregate gate pass/fail
      semantics and verifies that all quality gates remain opt-in.
      It also exercises all four page-segmentation policy labels, keeping the
      legacy default and each experimental mode explicit in reports.
- [x] Record the exact `.traineddata` SHA-256 in both converted Tesseract
      GGUF metadata and dumped reference GGUF metadata; the actual reference
      run and controlled-line stage/output parity are complete; page parity
      remains open.
- [x] Align the diagnostic Tesseract reference dumper and native recognizer
      with the actual Leptonica `pixScaleGrayLI` fixed-16 bilinear contract
      (top-left sampling, integer weights, replicated edges). The previous
      half-pixel resize was wrong and caused the first real divergence on a
      thin receipt crop.
- [x] Trace the receipt discrepancy against upstream Tesseract/Leptonica:
      `commons-receipt-line-3` moved from input cosine `0.990245` and
      `after_conv_fc` `0.983611` to all 9 diff stages at cosine `1.000000`,
      with matching mine/ref norms. Two additional CC0 Commons fixtures are
      vendored with URLs, licenses, and SHA-256 metadata; full CLI page and
      decoder-choice parity remain open.
- [x] Trace the remaining `Brighton`/`Drighton` discrepancy to the exact
      installed model's `training_flags=65`: Tesseract's `TF_INT_MODE` is set,
      so the CLI quantizes inputs and per-output weight rows to int8 before
      its recode beam. GGUF/reference metadata now records `training_flags`
      and `int_mode`; the current F32 graph remains measurable against the
      F32 Python reference but is not yet CLI-logit parity.
- [x] Add the first int8-equivalent Tesseract activation path and an int-mode
      Python `-ref.gguf`; input/intermediate/output boundaries pass the 0.99
      diff gate (int-mode logits cosine `0.997227`). This shifts native output
      to `Lhey ... Drighton`, but does not yet reproduce the CLI's
      `ihey ... Brighton`.
- [x] Add an opt-in CTC prefix beam (`CRISPEMBED_TESSERACT_BEAM_WIDTH`) and
      test widths 2, 3, 5, 10, 16, 25, and 50. It leaves the int-mode result
      unchanged, proving generic CTC beam search alone is not the CLI choice
      mechanism.
- [x] Close the int-mode network logit gap with Tesseract's lookup-table
      nonlinearities and exact quantized matrix arithmetic. Recode/dictionary
      scoring remains a separate pending gate below.
- [x] Added Tesseract's 1/256 LUT nonlinearities and reconstructed per-row
      int8 matrix accumulation. Native/Python int-mode parity improved to
      logits cosine `0.998405` with identical decoded output; CTC and
      Viterbi/recode-style diagnostic beams at widths 2-50 still do not select
      the CLI's `Brighton`.
- [x] Compare native against an instrumented official Tesseract PSM7 run at
      the internal activation boundary. The earliest divergence was the
      `Convolve` layer's out-of-image cells: Tesseract fills them with its
      seeded `TRand`, not zeros. GGUF now preserves `sample_iteration`, and
      both converter/reference/native paths reproduce the exact seeded padding.
      On the 601x36 official PSM7 crop, every captured stage passes at cosine
      1.000000 (logits max error 6.6e-7, cosine 1.000000), and native decodes
      the same `Brighton` path as official Tesseract. This closes arithmetic
      and preprocessing parity for this controlled line; recode/dictionary
      scoring and full-page segmentation remain separate gates.
- [x] Preserve the serialized Tesseract recoder map/offsets in the runtime and
      add an opt-in recoder-prefix legality layer to the diagnostic beam.
      Width-25 constrained decoding reproduces `Brighton` on the official
      crop with all 9 network diff stages still passing. Full RecodeBeamSearch
      certainty aggregation and DAWG dictionary scoring remain pending and
      are not enabled in production.
- [x] Harden `test-tesseract-lstm-diff` so decoded metadata mismatches fail the
      test. The Python reference now matches native `stb_image` RGB-to-gray
      conversion (`(77R+150G+29B)>>8`); all six direct line fixtures pass
      decoded parity with exact input tensors and stage cosines at or above
      `0.998821`.
- [x] Sweep six existing CC0/public-domain English line fixtures through the
      exact int-mode native/Python harness: all 6 references pass the
      captured-stage 0.99 parity gate, and the constrained width-25 beam remains
      stable on the official `Brighton` crop. Official CLI PSM7 output was
      separately measured with `language_model_ngram_on=0` and `=1`; both
      settings produced identical text on all six lines. The remaining
      differences (crop width, spacing, and case) are page-segmentation/CLI
      geometry, not DAWG scoring evidence.
- [x] Run exact hashed Homebrew English references after the Leptonica fix:
      the controlled line fixture decodes identically in native/Python as
      `_ “ ihey are going to be encamped near Drighton ;`, with all 9 stages
      passing the 0.99 gate and logits cosine `0.999863`. The full-page image
      is not a valid single-line recognizer fixture; page segmentation remains
      a separate acceptance gate.
- [ ] Record detector/ordering/recognizer provenance and checkpoint licenses;
      never relabel the cstr DBNet artifact or publish it under another account.

- [x] Fix native Tesseract confidence propagation. The converter/reference
      already expose CTC softmax probabilities, but `src/tesseract_lstm.cpp`
      discarded them and appended `0.0` for every decoded character. Greedy
      decoding now returns the selected timestep probability, so page-level
      confidence is no longer spuriously zero. This was a runtime bug, not a
      GGUF conversion bug.
- [x] Define the native beam confidence contract. A prefix/recode beam output
      is a sequence-level hypothesis assembled across timesteps, so a
      per-character confidence cannot be copied from one timestep. Native now
      exposes a length-normalized CTC sequence probability separately and
      leaves beam `char_conf` empty. Greedy decoding continues to expose
      selected timestep probabilities. This follows Tesseract's distinction
      between character certainty and word-level aggregation; it is not a
      claim of exact `WERD_RES::certainty` parity.
- [ ] Validate beam confidence against the official engine's certainty
      aggregation and implement per-character posterior/marginal scores only
      if that comparison establishes a stable mapping. Keep beam decoding
      opt-in until recoder and DAWG scoring are also matched.
      `test-confidence --tesseract-image MODEL LINE.png` now exercises the
      direct recognizer contract for greedy/beam comparisons without page
      segmentation overhead; it is diagnostic until a non-empty, transcribed
      line fixture is wired into the acceptance gate.
      `tools/compare_tesseract_line_confidence.py` now measures the same
      contract against official TSV on three English line fixtures: greedy text
      matches on two of three, while the first line remains `Lhey`/`Drighton`;
      greedy sequence-vs-official-mean-word confidence deltas were `+0.0053`,
      `-0.0847`, and `-0.0643`. Beam confidence is intentionally reported as a
      separate sequence probability (not a fabricated per-word certainty), so
      this does not close the official certainty gate. Tesseract source review
      now confirms the native greedy `word_confidence` mapping: minimum
      selected-path `log(probability)`, followed by `clamp(100 + 5*certainty,
      0, 100)`. On the direct second-line fixture this is `0.965889` versus
      official `0.959698`; page-level aggregation and beam certainty remain
      open.
      On the available German Fraktur line fixture, official Tesseract
      produces `1` with mean word confidence `0.5886`; native greedy produces
      `GI` with sequence confidence `0.5985` and two timestep confidences;
      native beam-8 produces `GIIEE` with sequence confidence `0.5808` and no
      per-character confidences. The confidence scale is close, but decoded
      text is not yet parity, so beam remains diagnostic and opt-in.
      The diagnostic now also reports native character min/mean probabilities;
      on the F32 artifact these are `0.0902`/`0.1978` for greedy, while beam
      correctly reports zero per-character values and sequence confidence
      `0.5592`. These are diagnostics only, not a production calibration.
      After correcting the aggregation to exclude blank/repeated CTC
      timesteps, native Fraktur greedy `word_confidence` is `0.8797` versus
      official TSV `0.5886`; beam remains `0`. Semantics now match the
      min-emitted-character rule, but calibration and decoded-text parity are
      still open.
      The line-confidence comparator now provides opt-in gates for greedy
      word-confidence calibration and the beam sequence-only contract; its
      model-free tests cover pass, calibration failure, fabricated beam
      character confidence, and missing-beam cases.
      It now records recognizer SHA-256 provenance and official word-confidence
      min/median/max alongside the mean, making calibration spread visible
      without treating those distributions as beam-character parity.
      It now accepts an explicit `--tessdata-dir` for the official subprocess;
      without that path, a stale `TESSDATA_PREFIX` can silently yield zero TSV
      words and invalidate the confidence comparison. With the explicit
      Homebrew tessdata directory on `german-line-tiny.png`, official output is
      `1` at confidence `0.588557`, native greedy is `G` with word confidence
      `0.883064`, and native beam-8 is `GEIEE` with sequence confidence
      `0.535476` and no character confidences. The beam contract passes, but
      text and greedy confidence calibration remain open quality TODOs.
      The comparator also provides `--require-official-words`, and its model-
      free contract test rejects empty official TSV references instead of
      allowing a misconfigured data path to appear to pass.
      It also provides `--require-greedy-text-match`; confidence/beam checks
      must not be reported as OCR-quality acceptance when native text differs.
      On the explicit German tiny-line fixture, enabling both
      `--require-official-words` and `--require-greedy-text-match` correctly
      exits 1 (`official_words_present=true`, `greedy_text_matches=false`),
      preserving the known native `G` versus official `1` quality gap.
      Both page-metrics and line-confidence comparators now emit elapsed
      milliseconds for each official subprocess and native subprocess/line
      run, so quality claims can be paired with measured cost.
      Page metrics can additionally request native detect/group/crop/recognize
      timings with `--benchmark`; official Tesseract remains an external CLI
      timing baseline unless built with matching internal instrumentation.
      The comparator uses the orchestrator-level benchmark switch, keeping
      recognizer-only timing separate from the full detector-to-recognizer
      pipeline timing.

The gated page-segmentation experiment currently gives 21 regions, 1,128
characters, and 0.836 mean confidence on the German official-print fixture.
The projection fallback gives 24 regions, 1,606 characters, and 0.702 mean
confidence. The reproducible TSV comparison reports 25 official lines, 141
words, 881 non-whitespace word characters, and 0.866 mean word confidence;
the native default blob path reports 21 regions, 1,128 characters, 0.836
confidence, CER 0.307, and WER 0.404 against the official text. Neither
experimental path is yet a quality match, so DBNet remains the production
default. The comparison is reproducible with
`tools/compare_tesseract_page_metrics.py`.

The component row-gap sweep (0, 2, 4, 6, and 8 pixels) did not improve the
quality gate: the best alternate produced 23 regions but lower confidence
(0.805). The default row fitter remains unchanged and the tuning variable is
not enabled in production.

The subsequent baseline-row matcher regression produced 49 regions and failed
the native gate. It is now preserved behind
`CRISPEMBED_TESSERACT_COMPONENT_BASELINE`; the legacy component grouping is
the default experimental component path again and restores 21 regions, 1,128
characters, 0.836 confidence, and 78/78 model-gated orchestrator checks.
The model-free orchestrator suite now also guards the two-row component
geometry and reading order, preventing this default/gated-path regression from
returning silently.
The component adapter now falls through to the baseline matcher only when
legacy grouping returns no boxes; this preserves the measured legacy
German-page behavior while avoiding an empty classical result on harder
layouts. The fallback remains behind the classical page-segmentation gate and
DBNet is unchanged.

Latest reproducible German-page rerun (2026-08-02, current `frk` Q8 artifact)
measured official Tesseract at 25 lines/881 chars, confidence `0.8658`, and
`9.34 s`; native measured 23 regions/1,235 chars, confidence `0.768`, CER
`0.5279`, WER `0.5390`, and `38.69 s` native stage. Native per-stage timing was
detect `102.0 ms`, crop `249.7 ms`, and recognize `38,338.3 ms`. This is worse
in both output quality and speed. The older 21-region/`0.307` CER result is
retained as a historical measurement because artifact/control conditions must
be normalized before calling it a regression. TODO: pin the exact recognizer
artifact and benchmark controls in the matrix, then fix the Fraktur page path.

The same fixture/control was then run against the available recognizer
variants. Standard `frk-q8_0` produced 23 regions/1,235 chars, confidence
`0.768`, CER `0.5279`, WER `0.5390`, and native stage `38.69 s`; `frk-f32`
produced 23/1,164, confidence `0.767`, CER `0.4672`, WER `0.5461`, and
`102.41 s`; `frk-int8-source-q8-candidate` matched the F32 output metrics and
produced `64.44 s`. Official output remained 25 lines/881 chars in all runs.
This isolates a precision/artifact quality tradeoff: standard Q8 is faster but
not output-equivalent, while F32/source-Q8 is more accurate on CER but much
slower and still not page-parity. TODO: choose and optimize a high-precision
Fraktur artifact only after same-artifact warm/cold benchmarks and decoded
output gates are stable.

The first reproducible mixed-precision experiment keeps Q8 as the base and
restores only `lstm.0.weight_hh` from F32 with
`models/mix-tesseract-gguf.py`. On the same page it produced 23 regions/1,146
chars, confidence `0.765`, CER `0.4603`, WER `0.5390`, and native recognition
`23.42 s` (detect `58.2 ms`, crop `55.4 ms`); official Tesseract produced
25/881, confidence `0.8658`, and `5.55 s` in that run. This improves CER over
Q8 (`0.5279`) and is much faster than full F32, but remains worse than the
reference in regions, text, confidence, WER, and speed. Keep it gated and
test additional small recurrent critical sets before promotion.

During the next sync, remote `main` had accidentally replaced the 1,025-line
int-mode/scratch implementation with an older 659-line F32-only runtime. The
regression was caught by the scan benchmark: recognition rose to `50.15 s`
for 12 Fraktur lines. The known-good runtime was restored and protected by
`tests/test_tesseract_runtime_contract.py`; the current LUT-enabled int-mode
run is `34.32 s` recognition with unchanged output (12 regions, CER `0.03375`,
WER `0.15044`). This remains slower than official Tesseract and is an active
optimization TODO, but the critical fast path is now guarded against silent
branch overwrites.

The generic quantizer now accepts repeatable `--keep-pattern` fnmatch rules.
This makes critical-weight retention reproducible for every model family while
leaving the default policy unchanged; the Tesseract mixed-precision packer
remains the safer byte-preserving path for existing artifacts.

#### Cross-check survey — quality and cost status

Every checked path must be classified by both decoded output quality and
measured cost; a cosine or successful exit code alone is not an OCR-quality
claim.

- **Controlled Tesseract line / Python reference / native GGUF:** on par for
  the validated exact line contract. The captured stage tensors and logits
  pass the existing `crispembed-diff` gates (cosines at or above `0.99`, with
  the proven controlled run at `1.000000`), and decoded output matches. The
  remaining gap is per-step timing: the diff harness records parity but does
  not yet emit reference and native elapsed time for every graph stage. TODO:
  add stage timing without weakening the cosine gate.
- **English Tesseract line-confidence path:** mixed, not full parity. Greedy
  decoded text matches two of three checked line fixtures; the remaining
  outputs contain `Lhey`/`Drighton` substitutions. Previously measured greedy
  sequence-confidence deltas versus official TSV word confidence were
  `+0.0053`, `-0.0847`, and `-0.0643`. This is worse on the affected lines,
  and the confidence calibration TODO remains open. The comparator now emits
  official/native elapsed milliseconds, but a controlled same-crop timing
  table for all three fixtures is still TODO.
- **German Fraktur line-confidence path:** worse, not on par. Official output
  is `1`; native greedy is `GI`, and native beam-8 is `GIIEE`. Native greedy
  word confidence is `0.8797` versus official TSV `0.5886`; beam correctly
  exposes sequence confidence only and zero character confidences, but that
  semantic contract is not certainty parity. TODO: obtain a transcribed,
  same-crop Fraktur line fixture and validate beam/greedy output and timing
  against the official engine.
- **`scan_strip.png` full-page legacy/fallback path:** close but worse. Both
  paths produce 12 regions; native produces 567 characters versus official
  453, CER is `0.0179`, WER is `0.0841`, and confidence is `0.895` versus
  `0.9108`. The new timed run measured official TSV at `342.6 ms` and native
  DBNet→Tesseract at `1105.2 ms` (`3.2x` slower). TODO: profile detector,
  crop/warp, and recognizer separately; the native path must close this cost
  gap before any default-path promotion.
- **`scan_strip.png` gated alternatives:** projection and baseline preserve
  12/12 rows but remain worse or non-superior in decoded output; measured
  geometry means are projection IoU `0.865993`, baseline IoU `0.813562`, and
  component IoU `0.826222`. Aggregate CER/WER are respectively
  `0.0250/0.1121`, `0.0179/0.0841`, and `0.0322/0.1121` for projection,
  baseline, and component. Per-policy native stage timings are now recorded
  below; retain all modes gated because no alternate is a quality and speed
  improvement.
- **German official-print page:** worse, not on par. The native default
  classical path reports 21 regions, 1,128 characters, confidence `0.836`,
  CER `0.307`, and WER `0.404`; official Tesseract reports 25 lines, 881
  non-whitespace word characters, and confidence `0.866`. Projection reports
  24 regions, 1,606 characters, and confidence `0.702`. TODO: add paired
  reference/native cold-load and warm per-stage timings for this fixture and
  improve segmentation/text before considering promotion.
- **Portfolio engine sweep:** this is a separate cross-engine benchmark, not
  Tesseract parity. The local M1 Metal sweep completed 11 engines, with 2
  timeout/error entries and explicit missing-model/sample statuses. TODO:
  attach the same quality/cost classification to every available engine and
  keep specialist/VLM outputs separate from plain-text OCR.

The first per-stage `scan_strip.png` benchmark now exists for all four native
policies. Official Tesseract TSV elapsed times were approximately
`315.9–349.9 ms`; native pipeline stage totals were legacy `310.7 ms`,
component `266.8 ms`, baseline `282.2 ms`, and projection `360.1 ms`.
Recognizer time dominated each native stage (`260.3–353.8 ms`), while native
detect and crop were each about `3–4 ms`. The comparator subprocess elapsed
times are higher than those stage totals because the model-gated test binary
also performs setup/fixture work; they must not be presented as pure OCR
latency. Official internal detect/recognize split timings are still unavailable
from the stock CLI, so a per-step apples-to-apples speed claim remains TODO.
The quality/cost table is currently: legacy best quality and close stage cost;
baseline same measured CER/WER but not a geometry/decoded-output improvement;
component worse CER/WER; projection slower and worse CER/WER. None is ready for
default promotion.

The current line diagnostic on the available Fraktur full-page/PSM7 input
measured official `278.6 ms`, native greedy `244.3 ms`, and native beam `110.3
ms`; these are not a same-crop benchmark and must not be used as a speed claim.
TODO: repeat timing on the cleared transcribed line fixtures and report cold
load, warm greedy, warm beam, and per-stage reference/native costs together.

The worker sweep on `scan_strip.png` preserves CER/WER while reducing native
stage total from `690.3 ms` at one worker to `300.7 ms` at four and `292.1 ms`
at eight. The next performance implementation task is therefore recognizer
batching/graph and immutable-weight reuse; detector and crop are only a few
milliseconds in the instrumented path. Do not trade away the recorded output
quality gates while optimizing this hotspot.

An activation-scratch reuse prototype was measured against the existing path:
it preserves CER/WER exactly. A paired run measured `279.1 ms` scratch versus
`282.3 ms` default, but earlier repeated runs were roughly `329–338 ms` versus
the prior `~300 ms` four-worker measurement; this is not a reliable gain. It
is retained behind `CRISPEMBED_TESSERACT_REUSE_SCRATCH` and is disabled by
default until a repeated controlled benchmark demonstrates improvement.
`tools/benchmark_tesseract_page.py` now automates repeated policy/worker runs
and summarizes median/p90 official CLI, native subprocess, native stage, and
recognizer timings alongside CER/WER/confidence deltas.

Beam width 8 is not a performance candidate on this workload: the live
full-page run reached several seconds to tens of seconds per line, versus the
normal greedy recognizer's sub-second-to-low-single-digit-second line timings.
The beam remains an opt-in diagnostic path until recoder/DAWG parity and a
usable latency budget are both demonstrated.

English Gen-2 now has a committed `test-easyocr-diff` harness, passes the agreed
0.99 per-stage cosine gate in F32 and folded-F16 forms, and decodes `5a`.
The VGG graph now derives feature channels and sequence width dynamically for
the remaining VGG recognizer variants.
Gen1 ResNet naming/folding and its residual graph path now pass a real Latin
Gen-1 checkpoint through conversion, Python reference dumping, per-stage
`crispembed-diff` (including magnitudes), and decoded-output parity (`=#4#4#`).

This workstream is informed by the external document-parser comparison, but keeps CrispEmbed's
ggml portability (CPU, CUDA, Metal, Vulkan, and WASM). Items are scoped so each
can land and be measured independently.

### O1 — Restore a trustworthy OCR baseline [COMPLETED]

- Fix duplicate region emission in the batched DBNet + TrOCR path.
- Add a regression test for one output region per detected region and no
  duplicated reading-order text.
- Record baseline latency and region/text counts in `PERFORMANCE.md`.

**Started:** DBNet postprocessing now handles degenerate one-point contours;
the local fox fixture improves from 0 to 10 detected regions. The remaining
baseline work is an automated model-backed assertion and sequential/batched
comparison.

**Done when:** batch and sequential recognition produce equivalent region counts
and no duplicate text on the OCR fixture set. The benchmark harness now accepts
`--expect-regions` and repeated `--expect-text` assertions for CI.

### O2 — Define a structured document result contract [COMPLETED]

- Add a C++ `ocr_document` result containing page dimensions, text regions,
  layout regions, tables, formulas, confidence, and engine provenance.
- Keep the existing orchestrator result and C API source-compatible; provide an
  adapter first, then migrate callers.
- Add serialization tests for empty, text-only, and mixed structured results.

**Started:** `ocr_orchestrator::result` now carries page dimensions and optional
layout regions. Layout inference is lazy and remains disabled unless
`config.layout_model` is set; existing callers and default latency are unchanged.

**Done when:** callers can consume one structured result without depending on a
specific OCR engine.

### O3 — Add CPU-only region routing after layout detection [COMPLETED]

- Introduce a pure routing module with `text`, `table`, `formula`, and
  `fallback` destinations.
- Route by layout label, confidence tier, containment/overlap, and explicit
  per-request feature policy; suppress duplicate text when a specialized
  recognizer owns a region.
- Unit-test every decision seam without model weights.

**Started:** `ocr_orchestrator::result` now carries the model-free routing plan;
table/formula/image policy is explicit in `config` and text-only by default.

**Done when:** a synthetic page produces a deterministic routing plan and the
existing specialized engines can be dispatched from it.

### O4 — Remove temporary image files from stage handoffs [COMPLETED]

- Add an in-memory RGB image/crop view shared by cleanup, detection, and
  recognizers; retain file APIs as load-and-forward wrappers.
- Make cleanup output ownership explicit and avoid unnecessary copies.

**Started:** `ocr_detect::detect_rgb` and `ocr_pipeline::run_raw` now accept
borrowed interleaved pixels; file APIs forward through them. The orchestrator
cleanup handoff still uses a temporary PNG and is the next O4 slice.

**Done when:** cleanup → detection/recognition runs without creating
`/tmp/crispembed_ocr_*.png`, with CPU/Metal output parity.

### O5 — Make capabilities and failures explicit [COMPLETED]

- Add an OCR capability query for loaded engines, languages, output types, and
  structure stages.
- Validate incompatible requests before inference; use stable errors instead of
  silent empty structure results.
- Add image dimension/pixel guards and per-item batch error isolation.

**Started:** enabling table/formula routing now fails at initialization unless
the required layout and specialized GGUF backends are configured.

**Done when:** every advertised feature is executable or rejected with a stable,
test-covered reason.

### O6 — Add reusable pipeline pooling and batch execution [COMPLETED]

- Define a bounded OCR pipeline pool for server use; retain the current path for
  single-threaded and WASM builds.
- Batch compatible crop recognition, cap batch size, and isolate bad inputs.
- Add queue/deadline metrics before changing defaults.

**Started:** DBNet+TrOCR inference contexts now serialize mutable decoder state
with an internal mutex, preventing concurrent callers from corrupting KV/cache
state. `ocr_pipeline_pool` now provides bounded isolated contexts with blocking
slot acquisition. The basic C OCR API selects the pool size from
`CRISPEMBED_OCR_POOL_SIZE` (default `1`); server-level queue/deadline metrics
remain a follow-up operational enhancement.

**Done when:** concurrent requests do not share mutable decoder state and batch
  throughput improves without changing decoded text.

### O7 — Establish unified accuracy/performance gates [COMPLETED]

- Add fixtures for receipt, form, dense page, screenshot, photo, table, and
  formula workloads.
- Measure CER/WER or exact-match, region recall, structure accuracy, p50/p95
  latency, memory, and batch throughput.
- Add regression thresholds and decoded-output checks for optimizations.

**Started:** `tests/ocr_benchmark.py` runs the real detector and pipeline test
binaries and reports region counts, decoded regions, and stage timings as text
or JSON. It uses local GGUFs and does not download models implicitly.

**Done when:** one reproducible command reports OCR quality and cost, suitable
for CI. **Complete:** `tests/ocr_benchmark.py` provides this command and JSON
output.

### O8 — Make corpus provenance and real-world coverage explicit [IN PROGRESS]

- Keep deterministic/reference fixtures for unit and per-stage parity checks,
  but do not use them as claims about real-world OCR quality.
- Add at least one public-domain/CC0 input for every production stage: text
  detection/recognition, layout, tables, cleanup, orientation, handwriting,
  multilingual routing, super-resolution, PDF routing, formulas, and OMR.
- Record source page, license, URL, SHA-256, and annotation status for every
  vendored asset. Add derived rotation/skew variants only from public-domain
  inputs.
- Acquire larger CC0 receipt and Arabic document sets separately, with a
  documented acceptance/download step instead of silently bundling them.

**Started:** `tests/regression/cc0_sources.json`, the fetched
`tests/regression/images/cc0/` seed set, and `corpus_manifest.json` now cover
receipts, forms, tables, Arabic printed/handwritten text, handwriting, cleanup,
orientation, layout, specialist lanes, and a dedicated German lane (modern
photo document, historical German print, and Kurrent handwriting). Gold
transcription review remains open for these robustness fixtures.

### O9 — Benchmark every available engine on shared inputs [IN PROGRESS]

- Use the checked-in regression manifest to enumerate every engine with a
  sample and local GGUF; report missing samples/models as explicit skips.
- Record cold load and warm inference time, return status, output excerpt, and
  CER/exact match when a gold transcription exists.
- Keep full-page VLM, ordinary OCR, math, and OMR scores separate; do not
  compare specialist outputs as plain-text OCR.
- Run the same engine sweep on the public-domain corpus after human gold
  annotations land, then add per-engine quality/latency thresholds.
- Maintain a complete matrix for model-backed engines even when a GGUF is not
  cached; the benchmark can fetch the manifest-pinned artifact with
  `--download-missing`.

**Started:** `tests/ocr_engine_benchmark.py` completed the local M1 Metal
sweep: 11 engines completed, 2 timed out/errored, and the remaining entries
were explicitly reported as missing samples, missing models, or model-needed
ports. SmolDocling is live-tested; Tesseract-LSTM is measured through DBNet
line crops; Unlimited-OCR is being fetched for its live run. Tesseract-LSTM
and PARSeq are present as recognizer-only rows; the DBNet+TrOCR document
baseline is measured separately by `tests/ocr_benchmark.py`. Results are
written as JSON with no silent omission. Unlimited-OCR subsequently completed
on M1 Metal from the system volume in 45,967 ms with correct two-region text
output; its GGUF was restored to the backup volume afterward.

The checked-in matrix now has a model-free CI coverage guard at
`tests/regression/test_engine_matrix.py`: all 23 portfolio engines must remain
present with a lane, runtime, fixture, and explicit availability status.

#### O9 measured survey — native vs reference status (2026-08-01)

This is the current evidence boundary. Timings below are cold end-to-end M1
Metal process times unless marked otherwise; they include model load and are
not directly comparable to warm service throughput. “Exact” means exact output
against the pinned reference/fixture, not a claim of SOTA quality. Missing
reference gold or a missing model is recorded as unmeasured, not as a pass.

| Path | Native timing / reference comparison | Output quality | Explicit follow-up |
|---|---|---|---|
| DBNet + TrOCR | 8.05 s cold; ~5.0 s warm on fox; DBNet postprocess optimization reduced 43.3 s → 1.54 s | 10/10 regions and 10/10 recognized; ordinary document baseline | Add shared German CC0 CER/WER and warm p50/p95 against Python/RapidOCR |
| Tesseract-LSTM via DBNet line crops | 7.55 s cold / 8.09 s warm in the first sweep; expanded run 32.0 s cold due process/model conditions | Line crop CER 0.040; controlled line arithmetic/reference parity is exact, but page path is worse: 21 native regions vs 25 official lines, CER 0.307/WER 0.404 | Normalize benchmark conditions; match page segmentation, crop geometry, spacing, and CLI decode |
| PARSeq | 0.921 s first sweep / 6.25 s expanded cold | Recognizer-only smoke (`Gooducalicanos.com`), no gold quality score | Add line-crop harness and CER/exact-match against scene-text gold |
| GOT-OCR2 | 15.662 s / 22.073 s cold | Exact fox transcript | Warm/p95 benchmark and full German CC0 quality lane |
| GLM-OCR | 32.884 s / 38.086 s cold | Exact fox transcript | Warm/p95 benchmark and German CC0 quality lane |
| InternVL2-1B | 24.908 s / 28.523 s cold | Worse than reference presentation: CER 0.540 and prompt text included | Strip prompt wrapper, compare normalized text, then optimize tile/vision residency |
| Qwen2-VL-3B | 70.757 s / 90.113 s before timeout | No transcript within the 45 s budget; quality unscored | Model-size/quantization tier benchmark and timeout budget decision |
| LightOnOCR | 31.561 s / 69.289 s cold | Plausible but unscored | Add gold transcription and separate prompt-wrapper normalization |
| SmolDocling | 16.334 s expanded | Worse: duplicated DocTags regions, payload CER 0.86 | Deduplicate/parse DocTags before any speed work |
| Unlimited-OCR | 45.967 s system-volume run; 40.391 s `UOCR_MMAP=1` backup-volume run | Correct two-region output; CER 0.010, one harmless title-box coordinate drift | Keep mmap path; split SAM/CLIP/projection/decoder timings and compare warm runs |
| MixTeX | 7.523 s / 13.286 s cold | Exact specialist LaTeX | Add warm/p95 and German/CN math fixtures; window attention remains CPU-scheduled |
| Flova | 16.153 s / 36.293 s cold | Exact specialist LilyPond | Add warm/p95 and handwritten German/music fixture coverage |
| Pix2TeX / Texteller | Pix2TeX 5.520 s / 8.980 s; Texteller 11.403 s / 18.491 s | Pix2TeX exact; Texteller worse/unusable (CER 7.293) | Keep Texteller out of quality tier; add formula-domain error analysis |
| SMT | 0.37 s incremental vs 1.98 s full recompute for ~100 tokens (5.4× faster) | 96.3% GrandStaff; native/HF token agreement 100% on validated samples; q8_0 exact, q4_k ~32% and rejected | Keep F32/Q8; do not expose Q4_K as quality tier |
| Polyphonic-TrOMR / Flova OMR / Transcoda | No common live timing in the O9 sweep | Per-stage/decoded parity documented; TrOMR and Flova byte-exact on validated references; Transcoda model lane still missing in the matrix run | Add common real-photo/historical-score warm/cold benchmark |
| PP-FormulaNet, HMER, BTTR, PosFormer, Texo | No common live timing in the O9 sweep | Stage parity exists for selected fixtures; no shared gold quality/throughput claim | Run model-backed math benchmark with exact LaTeX and CER/exact-match |
| PP-OCRv6 pipeline | German CC0 Metal: detector 6.9 s, quad crop 3.4 ms, CPU orientation 358.6 ms, recognition 455.2 ms; tiny graph diagnostic was previously 3.46 s vs 2.44 s CPU on `HI` crop; small/medium stem+backbone graph warm time is CPU 10.377/10.791 s versus Metal 0.222/0.650 s on the fox crop; full gated SVTR graph is 0.214/0.512 s on Metal | CPU accept-gate pipeline: 30 regions/139 chars, confidence 0.93, 141/141 smoke; detector graph remains worse (cos 0.99113, 31 vs 30 boxes); tiny recognizer graph reaches accepted-output parity on CPU and Metal for `HI`/fox; small/medium full SVTR graph reaches logits cos 0.999982/0.999986 on Metal with unchanged decoded output; three additional Arabic/receipt/German direct fixture runs have CPU fallback=CPU graph=Metal graph text | Fix detector box parity; add gold-backed cropped-line references and automate multi-fixture graph acceptance before changing the default gate |
| PP-LCNet orientation | CPU 0.36 s vs explicit Metal graph 1.15 s for 30 crops | 9/10 Metal parity fixtures; one uneven-illumination Arabic outlier (1.0679/3.2166 logits); CPU graph passes | Resolve Metal SE/depthwise numerical drift or keep CPU default; measure warm reuse only after reliability |
| EasyOCR/DBNet/LayoutLM handoff | No external Python page timing yet | Native model-backed run: 12 lines/98 words; geometry IoU 0.916 against TSV baseline, but native emits 98 vs Tesseract 106 words; external EasyOCR text parity pending | Install/reference Python environment and compare full manifests, CER, geometry, and warm timings |

Portfolio-wide TODOs created by this survey: (1) normalize cold-load versus
warm-inference measurement and report p50/p95; (2) attach a gold transcription
or exact structured reference to every runnable lane; (3) run the explicit
`model-needed` lanes with `--download-missing`; (4) do not optimize a lane
whose output is currently worse until its structural postprocessing/parity
failure is fixed; and (5) keep specialist metrics separate from plain-text OCR.

### O10 — Preprocessing inventory, parity, and live outcome gates [IN PROGRESS]

The OCR front-end needs its own measured regression track. Our restoration
inventory is broader than the lightweight OCR reference pipelines, but we are
missing several inexpensive geometry/orientation safeguards that often matter
more than another restoration model.

#### Existing CrispEmbed preprocessing

- Classical scan cleanup: dual-detector deskew consensus, border/content crop,
  background whitening, Otsu/Sauvola binarization, and fast binary morphology.
- Page analysis: PDF effective-DPI profiling, page split detection, content
  bounding box detection, source-type classification, and classical dewarp.
- Orientation: heuristic 0°/180° text-crop correction and rotated detection
  boxes; no learned page-orientation model yet.
- Learned restoration: NAFNet denoise, SCUNet, Restormer, InstructIR, and
  AdaIR.
- Super-resolution: PAN, TBSRN, HAT, DAT, ESRGAN, SwinIR, and SAFMN.
- Learned/classical dewarp: TPS dewarp and the classical baseline.
- VLM policy: full-page VLMs skip destructive scan cleanup and perform their
  own letterboxing/resizing; variable-resolution VLMs honor the max-pixels
  budget.

#### Reference capabilities to reproduce or explicitly reject

- Detector geometry: configurable minimum/maximum side limits, short-side
  target sizing, minimum-height padding, wide/short-image padding, and
  aspect-ratio-preserving letterbox policy.
- DB postprocessing: configurable segmentation threshold, box threshold,
  unclip ratio, optional dilation, candidate cap, and fast/accurate score mode.
- Line orientation: a dedicated 0°/180° classifier, confidence threshold, and
  an explicit all-lines mode for mixed-orientation documents.
- Page orientation: a learned 0°/90°/180°/270° classifier for PDF pages and
  photographed pages.
- Crop preparation: one shared policy for classifier geometry, recognizer
  geometry, aspect-preserving padding, and full-resolution recognition crops.
- PDF ingestion: native page rendering, page-image rotation, worker-pool
  accumulation, and the same preprocessing/OCR path as image inputs.
- Operational controls: per-stage enable/disable flags, hard errors for
  unavailable optional stages, request deadlines, and stage-level metrics.

#### Implementation slices

1. **O10.1 — Live preprocessor benchmark harness.** **Implemented in
   `feat/ppocr-next-20260731`; merged to `main`.** Add
   `tests/ocr_preprocessor_benchmark.py`. For every real CC0/German fixture,
   run raw input, classical cleanup variants, deskew, binarization, dewarp,
   denoise, and every locally available SR/restoration model. The harness
   accepts `--include-derived` to sweep every traced O10.2 robustness variant.
   Record stage
   latency, output dimensions, pixel statistics, detector regions, OCR text,
   confidence, and CER/exact match when gold text exists. Also report text
   delta versus the raw-image baseline when no verified gold transcription is
   available. Synthetic degradations remain unit stress tests, not quality
   claims.

2. **O10.2 — Problematic-input corpus.** **Implemented in
   `feat/ppocr-next-20260731`;** extend the public-domain corpus with
   verified derived variants: ±4°/±8° skew, dark border, uneven illumination,
   haze, speckle, low-DPI downsample, JPEG damage, 90°/180°/270° rotation,
   perspective/curved-page distortion, and mixed upright/upside-down lines.
   Every derived file must retain its parent SHA-256 and transformation recipe.

3. **O10.3 — Detector geometry policy.** **Implemented in `cf5f79b` and the
   follow-up routed-stage wiring.** Add a shared configuration object and
   C API fields for `min_side_len`, `max_side_len`, `min_height`,
   `width_height_ratio`, padding mode, `unclip_ratio`, dilation, score mode,
   and candidate cap. Default to safe current behavior; expose compatibility
   presets for short text strips, wide receipts, dense scans, and photos.
   The detector now provides `detect_options`/`rapid_defaults`, and the C API
   stage struct forwards these controls through DBNet, Tesseract, and PARSeq
   detector paths. Fast mode avoids pathological contour tracing; accurate
   polygon scoring remains available explicitly.

4. **O10.4 — Learned line orientation.** **Classical telemetry slice implemented
   in `feat/ppocr-next-20260731`;** port a small permissively licensed
   0°/180° line-angle classifier to GGUF/ggml. Integrate it after detection
   and before every line recognizer, including Tesseract-LSTM crops. Retain
   the current heuristic as a no-model fallback. Add per-line angle,
   confidence, and whether a rotation was applied to structured results.
   The existing classical 0°/180° safeguard is now shared by TrOCR,
   Tesseract-LSTM, and PARSeq line crops, and structured angle/confidence
   metadata is exposed per region. The learned classifier remains outstanding.

5. **O10.5 — Learned page orientation.** **Model-free fallback implemented in
   `feat/ppocr-next-20260731`;** port a small four-way page-orientation
   model. Apply it before PDF/image routing only when confidence clears a
   configurable threshold. Never rotate VLM inputs implicitly unless the
   caller enables the option, because VLM letterboxing is model-specific. The
   fallback exposes `crispembed_detect_page_orientation` and
   `/preprocess/orientation`, and never rotates
   input implicitly; a learned PP-LCNet classifier remains outstanding.

6. **O10.6 — Shared crop preprocessing.** **Implemented in
   `feat/ppocr-next-20260731`;** consolidate classifier and
   recognizer crop resizing/padding into one tested module. Support
   aspect-preserving and stretch modes, fixed height, maximum width, and
   grayscale/RGB contracts. Add parity fixtures for short, tall, wide,
   upside-down, and tightly clipped lines.

7. **O10.7 — PDF render/autorotate path.** **Image-page autorotation and
   multi-page DPI API slices
   implemented in `feat/ppocr-next-20260731`;** add native page rendering and
   page-level accumulation where the platform supports it. Reuse PDF DPI
   profiling to select render DPI, then apply page orientation and the normal
   document pipeline. `/ocr/document` now accepts explicit
   `"autorotate": true`, applies the confidence-gated fallback, and hands
   rotated temporary pages through the normal renderer. Keep the existing
   parser-only path for minimal builds. Bindings can now request all-page DPI
   metadata with ownership-safe `crispembed_pdf_all_pages_dpi` APIs.

8. **O10.8 — Stage routing, safeguards, and metrics.** **Benchmark accept-gate,
   VLM cleanup safeguard, and structured stage-metrics slices implemented in
   `feat/ppocr-next-20260731`;**
   **benchmark accept-gate slice
   implemented in `feat/ppocr-next-20260731`;** make preprocessing selection
   evidence-based: classical cleanup for scans, no destructive cleanup for
   VLMs/photos, denoise for noisy captures, SR only for low-DPI inputs, and
   orientation only above confidence thresholds. Add accept-gate comparisons
   so a preprocessor is rejected when it lowers confidence or worsens CER
   beyond the configured tolerance. The benchmark now records input/output
   checksums and dimensions plus conservative helped/neutral/harmed/
   unavailable/error outcome labels; changed text without verified gold is
   reported as unavailable rather than claimed as an improvement. Full-page
   VLM stages now skip destructive cleanup unless a future explicit VLM
   preprocessing override is added. The native/C pipeline API now exposes
   per-stage elapsed time, cleanup-applied flag, gate result, text yield, and
   confidence.

#### Required benchmark output

Each fixture/stage row must include:

- input and output dimensions, channels, and file checksum;
- cold load time, warm stage time, and peak/working-set estimate where
  available;
- detector box count, recognized region count, mean confidence, and text;

The live engine benchmark now includes both English and German
Tesseract-LSTM detector+line-crop rows, with the German model downloaded from
the pinned permissive registry entry when requested. Tesseract benchmark rows
use the F16 DB detector rather than a policy-disallowed Q4 detector.
- gold CER/exact match when verified, otherwise raw-baseline text delta;
- `helped`, `neutral`, `harmed`, `unavailable`, or `error` classification;
- stderr tail and stable failure reason for model/backend failures.

#### Acceptance gates

- Every production preprocessor has at least one real CC0/German live fixture.
- Every problematic-input variant runs through raw plus all applicable stages.
- No default preprocessor may worsen verified CER beyond its configured gate.
- A stage that cannot run is reported explicitly; it is never silently skipped.
- Orientation, geometry, cleanup, and restoration effects are reported
  separately, so a strong recognizer cannot hide a harmful preprocessor.
- Results are reproducible from one command and committed as benchmark JSON;
  large GGUFs support the external-volume no-copy path via `UOCR_MMAP=1`.

#### O10.9 — Named model candidates, licensing, and quality tiers

License policy: MIT, Apache-2.0, BSD-2/3-Clause, ISC, and similarly
permissive licenses are acceptable candidates for the core distribution. Do
not add NC/ND, research-only, or unclear checkpoint artifacts to the default
model registry. Repository-code licensing and pretrained-weight licensing must
be recorded separately before publishing a GGUF.

| Stage | Candidate model | License/reuse status | Quality position | Decision |
|---|---|---|---|---|
| Page orientation | `PP-LCNet_x1_0_doc_ori` | PaddleOCR is Apache-2.0; verify the exact exported checkpoint terms | Strong practical choice: four-way 0°/90°/180°/270° classifier, official docs report 99.06% on its test set | First port candidate |
| Line orientation | `PP-LCNet_x1_0_textline_ori` | Same Apache-2.0 project/weight-provenance audit required | Strong practical choice for per-line 0°/180° correction | First port candidate |
| Text detection | `PP-OCRv6` det | Apache-2.0 PaddleOCR code; model artifact provenance must be pinned and audited | Current practical high-quality/throughput baseline; supports multilingual deployment | Port/benchmark when PP-OCRv6 branch lands |
| Text recognition | `PP-OCRv6` rec | Same code/weight distinction as detector | Current practical high-quality/throughput baseline; one unified family is preferable to many language-specific recognizers | Port/benchmark with detector |
| Text detection fallback | `cstr/dbnet-ic15-GGUF` (`DBNet` ResNet-18) | Apache-2.0 declared for the converted artifact; source and dataset provenance documented; Challenge 4 is distinct from ICDAR2015-TextSR ODbL | Mature, reliable fallback; generally below current PP-OCR quality on difficult documents | Cleared; Q8/F16 default, Q4_K debug-only |
| Denoising | `NAFNet` | Upstream repository/checkpoint terms require explicit audit before redistribution | Strong efficient restoration baseline; upstream describes it as state-of-the-art for its restoration tasks | Keep only with artifact audit |
| Denoising/deblurring | `Restormer` | MIT repository license | Strong high-resolution denoising/deblurring/deraining model; official repo calls it SOTA for those tasks | Safe preferred learned restorer |
| Denoising | `SCUNet` | Verify upstream repository and checkpoint terms before registry inclusion | Lightweight practical denoiser, attractive for CPU/Metal | Keep as optional pending audit |
| Super-resolution | `HAT` / `HAT-S` | Apache-2.0 repository | Stronger quality-oriented SR candidate; HAT-S is a useful smaller tier | Safe preferred quality SR candidate |
| Super-resolution | `SwinIR` | Apache-2.0 repository; verify model-data/checkpoint terms | Strong broad baseline for classical/real-world SR, denoising, and JPEG artifact reduction | Safe preferred general SR candidate |
| Super-resolution | `Real-ESRGAN` | BSD-3-Clause repository; each released checkpoint still needs provenance audit | Strong practical real-world SR, but can hallucinate texture and harm OCR | Optional photo-only fallback, never unconditional |
| Super-resolution | `DAT` | Use only an explicitly permissive checkpoint/export; otherwise audit-required | High-quality transformer SR candidate, heavier than HAT-S/SwinIR | Optional quality tier after license audit |
| Super-resolution | `PAN` | Apache-2.0 model card/export candidate available | Very small and fast; useful low-resolution baseline, not current SOTA | Keep as fast CPU tier |
| Super-resolution | `SAFMN` | Apache-2.0 source/export; current GGUF card records Apache-2.0 | Excellent efficiency/size tradeoff, not absolute SOTA | Keep as default lightweight SR candidate |
| Text SR | `TBSRN` | Checkpoint license/provenance must be audited | Text-focused SR is more relevant to OCR than generic photorealistic SR | Keep only behind OCR CER gate |
| Learned dewarp | `UVDoc` / document-unwarping models | Candidate only after exact checkpoint license audit | Better fit than generic image restoration for curved pages | Prefer classical dewarp first; port if real fixtures show need |
| PDF orientation/render | PDFium + `PP-LCNet_x1_0_doc_ori` | PDFium and classifier terms must be retained in notices; classifier checkpoint audit required | Strong operational solution rather than an image-restoration model | Implement native render/autorotate path |

#### O10.9a — MMOCR model/checkpoint license audit

MMOCR's repository is Apache-2.0, but that is the license of the toolbox
code, not automatically the license of every downloaded checkpoint. The
official model-zoo page lists 48 checkpoints and links the weights separately,
without providing a per-checkpoint SPDX grant. Therefore every MMOCR weight
below is **audit-required**, unless a separately verified checkpoint license
is recorded in `tests/regression/manifest.json` and the model registry.

| MMOCR model family | Checkpoints in the official zoo | Current reuse status | CrispEmbed decision |
|---|---|---|---|
| DBNet | ResNet-18/50, DCNv2, oCLIP; ICDAR2015/SynthText/TotalText | The specific `cstr/dbnet-ic15-GGUF` ResNet-18 artifact declares Apache-2.0 and documents MMOCR + ICDAR2015 Incidental Scene Text provenance; other zoo checkpoints remain separate audits | Existing port remains; Q8/F16 default | Cleared for the specific cstr artifact; do not generalize to every zoo checkpoint |
| DBNet++ | ResNet-50, DCNv2, oCLIP; ICDAR2015 | Same checkpoint uncertainty; stronger detector but larger | Optional quality tier only after audit |
| Mask R-CNN | CTW1500/ICDAR2015, ResNet-50/oCLIP | Code/framework may be Apache-2.0; checkpoint and backbone provenance unresolved | Do not add yet |
| DRRG | CTW1500 | Checkpoint license not stated in zoo | Do not add yet |
| FCENet | CTW1500/ICDAR2015/TotalText, ResNet-50/DCNv2/oCLIP | Checkpoint license not stated in zoo | Candidate for curved text only after audit |
| PANet | CTW1500/ICDAR2015, ResNet-18 | Checkpoint license not stated in zoo | Low priority; audit before use |
| PSENet | CTW1500/ICDAR2015, ResNet-50/oCLIP | Checkpoint license not stated in zoo | Low priority; audit before use |
| TextSnake | CTW1500, ResNet-50/oCLIP | Checkpoint license not stated in zoo | Candidate for arbitrary shapes only after audit |
| ABINet | Vision-only and iterative; ST/MJ | Checkpoint license not stated in zoo; language-model/data provenance especially important | Do not bundle pending full audit |
| ASTER | ResNet-45, ST/MJ | Checkpoint license not stated in zoo | Rectification idea is reusable; checkpoint excluded pending audit |
| CRNN | Mini-VGG, MJ | Checkpoint license not stated in zoo; dataset/training terms unresolved | Do not add; Tesseract remains the tiny permissive lane |
| MASTER | ResNet-31, ST/MJ/SA | Checkpoint license not stated in zoo | Do not add pending audit |
| NRTR | Modality-transform and ResNet-31 variants, ST/MJ | Checkpoint license not stated in zoo | Do not add pending audit |
| RobustScanner | ResNet-31, ST-sub/MJ-sub/SA-real | Checkpoint license not stated in zoo | Do not add pending audit |
| SAR | ResNet-31 parallel/sequential, ST-sub/MJ-sub/SA-real | Checkpoint license not stated in zoo | Do not add pending audit |
| SATRN | Shallow and small, ST/MJ | Checkpoint license not stated in zoo | Do not add pending audit |
| SVTR | Small/base, ST/MJ | Checkpoint license not stated in zoo; promising scene-text quality/size tradeoff | First new MMOCR recognizer to investigate, but no auto-download until weights are cleared |
| SDMGR | Visual/novisual/open-set, WildReceipt | Checkpoint, dataset, and KIE-label provenance unresolved | Do not add; existing KIE pipeline is preferred |

The official zoo reports the strongest listed detection result for DBNet++
ResNet-50-oCLIP at 0.8882 ICDAR2015 hmean-IoU, and lists SVTR-small/base as
scene-text recognizers; these are quality signals only, not license grants.
The architecture/code may be studied or clean-room reimplemented, but the
specific weights remain blocked until their provenance is documented.

#### Explicitly excluded or non-default candidates

- `Texo-Distill` is AGPL-3.0 and remains outside the permissive core model
  registry; use `PP-FormulaNet-L`, `PP-FormulaNet-S`, or another audited
  Apache/MIT/BSD formula model instead.
- Any `CC-BY-NC`, `CC-BY-NC-SA`, research-only, or unclear checkpoint is not a
  default alternative even when its architecture is attractive. It may be
  supported in a user-supplied/private model path if the caller accepts the
  license, but it must not be bundled or auto-downloaded by the default
  registry.
- Generic GAN/diffusion SR models must not be called automatically on OCR
  inputs. They can create plausible but incorrect glyph detail; acceptance is
  downstream OCR CER/confidence, not visual sharpness.

#### Quality/SOTA policy

“SOTA” is task-specific and must not be treated as a blanket OCR claim. The
selection policy is:

1. `PP-LCNet_x1_0_doc_ori` and `PP-LCNet_x1_0_textline_ori` for cheap learned
   orientation;
2. `PP-OCRv6` det/rec for the primary practical OCR baseline;
3. `Restormer` or `NAFNet` for restoration when live CER proves it helps;
4. `HAT`/`SwinIR` for quality SR and `SAFMN`/`PAN` for low-resource SR;
5. `Real-ESRGAN` only for photo inputs and only behind a no-harm gate;
6. classical cleanup and no preprocessing remain valid winners when the raw
   or VLM path scores better.

Every named model must receive a matrix row with: exact source URL, revision,
license, weight license, parameter count, GGUF quantization, live latency,
CER delta on problematic fixtures, and a human-reviewed accept/reject result.

#### O10.10 — Existing-model reconciliation

The following are not new port candidates: the repository already contains
runtime implementations, converters, tests, and local GGUF artifacts for them:

| Already available | Runtime / artifact status |
|---|---|
| `NAFNet` | `src/nafnet_denoise.cpp`; `models/convert-nafnet-to-gguf.py`; local `nafnet-sidd-w32-q8_0.gguf` |
| `SCUNet` | `src/scunet_denoise.cpp`; `models/convert-scunet-to-gguf.py`; local `scunet-color-f32.gguf` |
| `Restormer` | `src/restormer.cpp`; `models/convert-restormer-to-gguf.py`; local `restormer-denoise-f16.gguf` |
| `HAT` | `src/hat_sr.cpp`; `models/convert-hat-to-gguf.py`; local `hat-sr-x4-f16.gguf` |
| `SwinIR` | `src/swinir_sr.cpp`; `models/convert-swinir-to-gguf.py`; local `swinir-light-x4-f16.gguf` |
| `Real-ESRGAN` | `src/esrgan_sr.cpp`; `models/convert-esrgan-to-gguf.py`; local `esrgan-x4-f32.gguf` |
| `DAT` | `src/dat_sr.cpp`; `models/convert-dat-to-gguf.py`; local `dat-light-x2-f16.gguf` |
| `PAN` | `src/pan_sr.cpp`; `models/convert-pan-to-gguf.py`; local `pan-x4-f16.gguf` |
| `SAFMN` | `src/safmn_sr.cpp`; `models/convert-safmn-to-gguf.py`; local `safmn-x4-f32.gguf` |
| `TBSRN` | `src/tbsrn_sr.cpp`; `models/convert-tbsrn-to-gguf.py`; local `tbsrn-telescope-f16.gguf` |

PP-OCRv6 detector/recognizer GGUF files also exist in the shared model volume
(`PP-OCRv6_{tiny,small,medium}_{det,rec}-*.gguf`), including policy Q4_K and
Q8_0 variants. They are **not yet integrated into the current main-branch
runtime**; the PP-OCRv6 task remains an integration/port task, not a model
acquisition task. The port must include detector postprocessing, recognizer
dictionary handling, line orientation, and live German/Arabic/receipt tests.

The genuinely new preprocessing ports are therefore:

- `PP-LCNet_x1_0_doc_ori` — four-way page orientation;
- `PP-LCNet_x1_0_textline_ori` — learned 0°/180° line orientation;
- `UVDoc` or an equivalently permissive document-unwarping model — only if
  classical/TPS dewarp fails the curved-page fixtures;
- native PDFium rendering/autorotation integration, which is pipeline plumbing
  rather than a new image-restoration model.

Before adding another restoration model, O10 must benchmark the existing ten
models on the same degraded fixtures and promote only models that improve
downstream OCR CER/confidence. The current default candidates are therefore
`SAFMN`/`PAN` for cheap SR, `NAFNet`/`Restormer` for denoise, and `SwinIR` or
`HAT` for quality SR, subject to the license and no-harm gates above.

### Validation follow-up — external document parser [COMPLETED]

- Unit gates passed: region router, pipeline pool, orchestrator (62/62), and
  render tests.
- Live M1 Metal gate passed: DBNet detected 10/10 fox fixture regions and
  TrOCR recognized 10/10; measured warm total was 5.0–5.3 s/image, with 8/10
  exact words and 6.1% CER.
- The comparison implementation's live execution is environment-blocked, not silently skipped: the
  CPU configure probe lacks OpenCV development files, while the production
  path requires CUDA/TensorRT and this host has no NVIDIA device/usable Docker
  daemon. The documented NVIDIA numbers are recorded in
  `PERFORMANCE.md` as reference claims only.
- Next actionable benchmark item: run both engines on a shared corpus on an
  NVIDIA host, then add detector/recognizer quality and throughput thresholds
  to `tests/ocr_benchmark.py`.
- Quantization A/B resolved the current fox errors: TrOCR-small-printed Q4_K
  produced 8/10 exact words, while the same ggml pipeline with the recommended
  Q8_0 model produced 10/10. Keep Q8_0 as the default quality model; do not
  treat Q4_K as a quality-preserving OCR quantization.
- Q8 is now the benchmark/WASM/example default. The pipeline rejects filenames
  identifying TrOCR Q4_K unless `CRISPEMBED_DEBUG_ALLOW_OCR_Q4=1` is set.
  Text crops also receive a classical 0°/180° orientation check, and results
  now expose TrOCR mean/per-character confidence values.
- Added parity-facing structured output: deterministic reading-order indices
  and lightweight Markdown export are available from the orchestrator result
  and C API after each page run.
- Added modular server/API discovery: `/capabilities`, `/health/live`, and
  `/health/ready`; structured pipeline responses now include reading order and
  Markdown. Pipeline params and native server flags can independently enable
  layout, Tesseract-backed table cells, and PP-FormulaNet formulas.
- Added a `unified` pipeline stage backed by `crispembed_ocr_model_*`: any
  metadata-dispatched GGUF engine can now be selected as an escalation or
  specialist stage without adding another orchestrator-specific enum. This
  preserves the existing modular engine matrix, including Tesseract-LSTM,
  PARSeq, VLMs, math, and music engines where full-page/crop routing makes
  sense.

### Sequencing and boundaries

Land O1 first, then O2/O3 as the structured result and router foundation. O4 is
the first performance refactor; O5/O6 apply mainly to server builds. O7 starts
with CPU fixtures and expands to Metal/CUDA where hardware is available. Do not
replace ggml with TensorRT or make the core runtime NVIDIA-only.

## Goal

Replace ONNX-runtime-based embedding pipelines (fastembed, sentence-transformers)
with a single `crispembed` binary + C library that:

1. Loads any supported model from a GGUF file (auto-detect architecture)
2. Tokenizes input text (WordPiece / SentencePiece / BPE from GGUF metadata)
3. Runs the transformer encoder or decoder via ggml graph
4. Pools + normalizes → output embedding vector
5. Supports Q4_K / Q5_K / Q6_K / Q8_0 / F16 / F32 quantisation
6. Exposes a C API, CLI, HTTP server, Python, Rust, and Dart wrappers

## Architecture (v0.11)

```
Input text / image / audio
    │
    ├─► Text ──► Tokenizer (WordPiece / SentencePiece / BPE)
    │              │
    │              ├─► Encoder path (BERT, XLM-R, MPNet, NomicBERT,
    │              │     ModernBERT, GTE v1.5, DeBERTa-v2, SPLADE)
    │              │     Token + Pos [+ Type] embeddings
    │              │     N × Transformer layer (LN → MHA → FFN → residual)
    │              │     Pooling (mean / CLS) + optional heads
    │              │     → dense / sparse / ColBERT / reranker output
    │              │
    │              ├─► Decoder path (Qwen3, Gemma3, BidirLM-Omni text)
    │              │     Token embeddings + RoPE
    │              │     N × (RMSNorm → GQA → SwiGLU/GeGLU → residual)
    │              │     Last-token / mean pooling + L2 normalize
    │              │
    │              └─► LFM2 path (LFM2.5, lfm2_embed.cpp)
    │                    RMSNorm + GQA, 350M, BOS-only tokenization
    │                    → dense / ColBERT multi-vector output
    │
    ├─► Image ──► ViT path (SigLIP/CLIP: vit_embed.cpp)
    │               Conv2D patch embed → transformer → mean pool → L2
    │
    ├─► Image ──► BidirLM-Omni vision (bidirlm_vision.cpp)
    │               Qwen2VL ViT + patch merger + DeepStack
    │               → image_embeds spliced into decoder
    │
    ├─► Image ──► CNN path (cnn_embed.cpp)
    │               SCRFD/YuNet face detection (FPN + anchor decode + NMS)
    │               ArcFace/SFace/AuraFace face recognition
    │
    ├─► Audio ──► BidirLM-Omni audio (bidirlm_audio.cpp)
    │               crisp_audio Whisper-shape encoder → mean pool → 2048-d
    │
    ├─► Math  ──► DeiT encoder + TrOCR decoder (math_ocr.cpp)
    │               Printed math → LaTeX via ggml graph compute
    │
    ├─► Math  ──► HMER: DenseNet-121 + GRU attention (hmer_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME 2016)
    │
    ├─► Math  ──► BTTR: DenseNet + Transformer decoder (bttr_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME 2014, 53% exact match)
    │
    ├─► Math  ──► PosFormer: BTTR + ARM coverage (posformer_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME, improved over BTTR)
    │
    ├─► Math  ──► MixTex: Swin-Tiny + RoBERTa (mixtex_ocr.cpp)
    │               Chinese+English LaTeX OCR (25681 BPE vocab)
    │
    ├─► Math  ──► PP-FormulaNet-S: HGNetv2 + MBart (ppformulanet_ocr.cpp)
    │               57M params, 384×384 input
    │
    ├─► Math  ──► PP-FormulaNet-L: SAM-ViT + MBart (ppformulanet_l_ocr.cpp)
    │               181M params, 768×768 input
    │
    ├─► OCR   ──► DBNet + TrOCR pipeline (ocr_pipeline.cpp)
    │               Text detection → recognition → reading-order sort
    │
    ├─► OCR   ──► Surya-OCR-2 detector (surya_det.cpp)
    │               EfficientViT + SegFormer, 38M, 91 languages
    │
    ├─► OCR   ──► Qwen2.5-VL / Qwen2-VL (qwen2vl_ocr.cpp)
    │               VLM doc OCR; german-ocr-3 (3B), FireRed-OCR, Qari-OCR, Nanonets
    │
    ├─► Layout ─► RT-DETRv2 docling-heron (layout_detect.cpp)
    │               ResNet-50 + deformable xattn, 17 document classes
    │
    ├─► OCR   ──► PARSeq scene text recognition (parseq_ocr.cpp)
    │               ViT + Transformer, 24M, 94-char ASCII, Apache-2.0
    │
    ├─► OCR   ──► InternVL2 (internvl2_ocr.cpp)
    │               InternViT + InternLM2.5 VLM, 1B/2B, MIT (+ H2OVL)
    │
    ├─► OCR   ──► GLM-OCR (glm_ocr.cpp)
    │               CogVLM2 + GLM-4, 0.9B, 8 languages, MIT
    │
    ├─► OCR   ──► GOT-OCR2 (got_ocr.cpp)
    │               SAM ViT-B + Qwen2-0.5B, document+math+table, Apache-2.0
    │
    ├─► OCR   ──► LightOnOCR-2-1B (lightonocr.cpp)
    │               Pixtral ViT + Qwen3, 1B, OCR Arena #2, Apache-2.0
    │
    ├─► OCR   ──► DeepSeek-OCR-2 (deepseek_ocr2.cpp)
    │               SAM ViT + Qwen2 + MoE decoder, 3.4B, multilingual
    │
    ├─► OCR   ──► Granite Vision 3.3-2B (granite_vision_ocr.cpp)
    │               SigLIP2 + Granite-3.1-2B, OCRBench 852, Apache-2.0
    │
    ├─► OCR   ──► Tesseract LSTM (tesseract_lstm.cpp)
    │               DBNet detection + per-line LSTM, 126 languages
    │
    ├─► NER   ──► BERT/XLM-R token classification (bert_ner.cpp)
    │               Fixed-label NER: PER/LOC/ORG/MISC, auto-detected
    │
    ├─► NER   ──► GLiNER zero-shot (gliner_ner.cpp)
    │               LFM2.5/DeBERTa-v3 + BiLSTM + span matching
    │
    ├─► KIE   ──► OCR + NER pipeline (kie_pipeline.cpp)
    │               Phase 1: OCR→NER. Phase 2: LiLT layout-aware
    │
    ├─► KIE   ──► LiLT layout transformer (lilt_kie.cpp)
    │               Dual-stream RoBERTa + BiACM, 130M, FUNSD, MIT
    │
    ├─► LID   ──► Text language identification (crisp_lid)
    │               CLD3 / GlotLID, Tesseract auto-select
    │
    ├─► Table ──► Rule-based table structure (table_parse.cpp)
    │               Line detection + grid + cell OCR → HTML
    │
    ├─► OCR   ──► PaddleOCR-VL (qwen2vl_ocr.cpp) — DONE
    │               NaViT ViT + ERNIE-4.5-0.3B, 109 langs, Apache-2.0
    │               OmniDocBench SOTA 96.3% (1.6) / 0.9B variant
    │
    ├─► Math  ──► Uni-MuMER-Qwen3-VL-2B (via qwen2vl_ocr.cpp)
    │               Handwritten math → LaTeX, 2.1B, Apache-2.0, 82% CROHME
    │
    ├─► Math  ──► Uni-MuMER-Qwen2.5-VL-3B (via qwen2vl_ocr.cpp)
    │               Handwritten math → LaTeX, 3.4B, Apache-2.0, 82.25% CROHME
    │
    │   ── PLANNED ──
    │
    └─► OCR   ──► SmolDocling (256M, Apache-2.0) — DONE: SigLIP + SmolLM2, DocTags
                    Idefics3/SmolVLM, IBM Research, DocTags output (tiny, EN-only)
```

(Evaluated and **rejected** for licensing: dots.ocr — supplemental PRC
agreement (rednote/Xiaohongshu), not pure MIT; MinerU2.5-Pro — commercial
thresholds + gated HF; Hunyuan-OCR — custom Tencent license, excludes
EU/UK/South Korea. See the next-gen table below.)

## Supported architectures (v0.11)

| Architecture | Tokenizer | Key features | Example models |
|---|---|---|---|
| BERT encoder | WordPiece | Post-LN, GELU FFN | MiniLM, BGE, SPLADE |
| XLM-R encoder | SentencePiece Unigram | Post-LN, GELU, pos_offset=2 | E5, PIXIE, arctic-l-v2, granite |
| MPNet encoder | WordPiece | Post-LN, T5-style rel attn bias | all-mpnet-base-v2 |
| NomicBERT encoder | WordPiece | Post-LN, SwiGLU, RoPE | nomic-embed-text-v1.5 |
| NomicBERT MoE encoder | SentencePiece | Post-LN, MoE 8-expert top-2, GELU, RoPE | nomic-embed-text-v2-moe |
| ModernBERT encoder | BPE | Pre-LN, GeGLU, RoPE, per-layer theta | gte-modernbert-base |
| GTE v1.5 encoder | WordPiece | Post-LN, GeGLU, NTK RoPE | gte-base/large-en-v1.5 |
| DeBERTa-v2 encoder | WordPiece | Post-LN, c2p/p2c disentangled attn | mxbai-rerank-xsmall/base-v1 |
| Qwen3 decoder | GPT-2 BPE | RMSNorm, SwiGLU, RoPE, GQA | Octen, F2LLM, Jina v5, Harrier-0.6B |
| Gemma3 decoder | SentencePiece BPE | Gemma RMSNorm(1+w), GeGLU | Harrier-270M, EmbeddingGemma-300m |
| LFM2 (bidirectional) | GPT-2 BPE | Pre-norm RMSNorm, GQA, RoPE, BOS-only | LFM2.5-Embedding-350M, LFM2.5-ColBERT |
| BidirLM-Omni | GPT-2 BPE | Bidirectional Qwen3, MRoPE, DeepStack | BidirLM-Omni-2.5B |
| ViT (SigLIP/CLIP) | — | Conv2D patch embed, CLS/mean/attn pool | siglip-base, clip-vit-base |
| CLIP text | CLIP BPE | Pre-LN, causal mask, EOS pool | clip-text-base/large |
| CNN (SCRFD/YuNet) | — | FPN, anchor decode, NMS | scrfd-det-10g, yunet |
| CNN (ArcFace) | — | ResNet-100, 512-D L2 | w600k_r50, auraface-v1, sface |
| DeiT+TrOCR | — | ggml graph encoder + decoder | pix2tex-mfr |
| HMER | — | DenseNet-121 + GRU attention | hmer (handwritten math) |
| BTTR | — | DenseNet + Transformer decoder | bttr (handwritten math) |
| PosFormer | — | DenseNet + Transformer + ARM | posformer (handwritten math) |
| MixTex | BPE (25681) | Swin-Tiny + RoBERTa 4L decoder | mixtex (CN+EN LaTeX) |
| PP-FormulaNet-S | BPE (50000) | HGNetv2 CNN + MBart 2L decoder | ppformulanet (57M) |
| PP-FormulaNet-L | BPE (50000) | SAM-ViT + MBart 8L decoder | ppformulanet-l (181M) |
| DBNet | — | ResNet-18 + FPN + DB head | text detection (12M) |
| Surya-Det | — | EfficientViT + SegFormer | surya-ocr-2 detector (38M, 91 langs) |
| RT-DETRv2 | — | ResNet-50 + deformable xattn | layout-heron (17 classes) |
| Qwen2.5-VL / Qwen2-VL / Qwen3-VL | tiktoken | ViT-32L + spatial merger + Qwen LLM; runtime ne-fix for transposed-weight GGUFs | german-ocr-3 (3B), FireRed-OCR, Qari-OCR, Nanonets, PaddleOCR-VL |
| InternVL2 | tiktoken | InternViT + InternLM2.5 LLM | internvl2-1b/2b, H2OVL |
| GLM-OCR | BPE | CogVLM2 + GLM-4 decoder | glm-edge-ocr (0.9B) |
| GOT-OCR2 | BPE | SAM ViT-B + Qwen2-0.5B | got-ocr2 (0.7B) |
| LightOnOCR | tiktoken | Pixtral ViT + Qwen3 decoder | lightonocr-2-1b (1B) |
| DeepSeek-OCR-2 | tiktoken | SAM ViT + Qwen2 + MoE decoder | deepseek-ocr2 (3.4B) |
| Granite Vision | tiktoken/BPE | SigLIP2 ViT + Granite-3.1 LLM | granite-vision-3.3-2b |
| PARSeq | — | ViT + AR/NAR Transformer | parseq (24M, 94-char) |
| Tesseract LSTM | — | DBNet det + LSTM line rec | 126 languages |
| LiLT | RoBERTa BPE | RoBERTa + layout transformer + BiACM | lilt-funsd (130M) |
| BERT NER | WordPiece/SP | BERT/XLM-R + Linear classifier | bert-ner, xlmr-ner-hrl |
| Table parser | — | Rule-based morphology + grid detection | table_parse (no model) |

## Shared code with CrispASR

| Component | Source | Reuse method |
|-----------|--------|-------------|
| ggml | submodule | identical |
| GGUF loader | src/core/gguf_loader.{h,cpp} | copy |
| Attention helper | src/core/attention.h | copy (header-only) |
| FFN helper | src/core/ffn.h | copy (header-only) |
| httplib.h | examples/server/ | copy |
| crisp_audio | CrispASR build | shared library |
| crisp_punc | CrispASR/crisp_punc/ | shared library (FireRedPunc + PCS) |
| crisp_lid | CrispASR/crisp_lid/ | shared library (CLD3 + GlotLID) |
| crisp_truecase | CrispASR/crisp_truecase/ | shared library (stat + CRF + BiLSTM) |

## File layout (current)

```
CrispEmbed/
├── CMakeLists.txt
├── README.md
├── PLAN.md                     architecture + roadmap (this file)
├── HISTORY.md                  completed milestones
├── LEARNINGS.md                technical notes
├── PERFORMANCE.md              benchmarks
├── ggml/                       (submodule)
├── src/
│   ├── crispembed.{h,cpp}      C API + encoder graph + OCR-model dispatch
│   ├── decoder_embed.{h,cpp}   decoder graph (Qwen3/Gemma3/BidirLM)
│   ├── lfm2_embed.cpp          LFM2.5 dense + ColBERT multi-vector
│   ├── bidirlm_vision.cpp      BidirLM-Omni vision tower
│   ├── bidirlm_audio.cpp       BidirLM-Omni audio tower
│   ├── vit_embed.{h,cpp}       SigLIP/CLIP ViT vision encoder
│   ├── clip_text_embed.{h,cpp} CLIP/SigLIP text encoder
│   ├── cnn_embed.{h,cpp}       SCRFD/YuNet/ArcFace/SFace
│   ├── image_preprocess.{h,cpp} C++ image preprocessor
│   ├── math_ocr.{h,cpp}        DeiT+TrOCR printed math OCR
│   ├── hmer_ocr / bttr_ocr / posformer_ocr / mixtex_ocr / ppformulanet*  math OCR
│   ├── qwen2vl_ocr / internvl2_ocr / glm_ocr / got_ocr / lightonocr      VLM OCR
│   ├── deepseek_ocr2 / granite_vision_ocr / parseq_ocr / tesseract_lstm  OCR engines
│   ├── tokenizer*.{h,cpp}      WordPiece + SentencePiece + BPE
│   └── core/                   shared helpers (gguf_loader, bpe, mel, cpu_ops)
├── examples/
│   ├── cli/main.cpp            CLI binary
│   └── server/server.cpp       HTTP server (4 API dialects)
├── models/                     GGUF conversion scripts
├── python/crispembed/          ctypes wrapper
├── crispembed-sys/             Rust FFI bindings
├── crispembed/                 Rust safe wrapper
├── flutter/crispembed/         Dart/Flutter FFI plugin
├── tools/quantize.cpp          C++ quantizer
└── tests/                      parity + benchmark scripts
```

## Pending work

Only genuinely-open, in-progress, or reference material lives below. **Completed
milestones — the imatrix quant rollout (C1), batched-encoder throughput (C3),
prefix KV cache (C4), mtmd-preprocessing port (C5), flash-attn epilogue audit
(C6), mmproj interop, the June-2026 optimization-TODO sweep, per-backend perf
passes, the SR conv→ggml sweep, the regression-guardrail closure, the CUDA
device-pointer fixes, and the scan_cleanup / unpaper feature ports — have moved
to `HISTORY.md`** (deep technical notes in `LEARNINGS.md`). Before starting any
item: read LEARNINGS "measure the DOMINANT cost before fixing a flagged
micro-gap" and "the build dir was silently CPU-only"; verify
`GGML_METAL:BOOL=ON` in `build/CMakeCache.txt`; check `git worktree list` +
`git log main..<branch>` for a concurrent session's finished work; all edits in
a worktree (ggml symlink dance, see CLAUDE.md).

### Shipped ecosystem-compat work (A1–A4, modern-bert, A3 parity harness) — see HISTORY.md

The A1–A4 JSON/hparam/CI hardening, the community `modern-bert` BPE-tokenizer
fix, the encoder ground-truth parity harness, and the e5-small/granite matrix
closure all shipped 2026-07-16 (verified in code 2026-07-20). Details + preserved
specifics: HISTORY.md "July 20, 2026" + "July 16, 2026"; deep-dives in LEARNINGS.md.

### Community-matrix coverage roadmap — candidate archs to add

Matrix entries (10): bge-small + all-MiniLM (`bert`, split-QKV/abs-pos/WordPiece),
nomic-v1.5 (`nomic-bert`), nomic-v2-moe (`nomic-bert-moe`), gte-modernbert
(`modern-bert`, gpt2-BPE), granite-107m (`bert` + `t5`/SPM, CLS), gte-base-en-v1.5
(`NewModel` tanh-GeGLU), Qwen3-Embedding-0.6B (`qwen3` decoder), LFM2.5-Embedding
(`lfm2`), embeddinggemma-300m-qat (`gemma-embedding`). Each remaining family below
exercises a DISTINCT loader/graph path not yet guarded against a
third-party GGUF. Ordered by coverage value; every one is a load + shape +
garbage-guard + HF per-stage entry (the granite recipe), each MUST be gated on the
per-stage structural cosine (a garbage-guard-only pass hides an e5-style shift).
Availability probed 2026-07-16 (repos listed are candidates, not yet validated):

| Candidate | arch / path it covers | Fits dense driver? | Candidate community GGUF | Watch-out |
|---|---|---|---|---|
| **Qwen3-Embedding-0.6B** | `qwen3` DECODER embed — last-token pool, **causal**, gpt2-BPE decoder path (distinct from modern-bert's ENCODER BPE) | ✅ (last-token) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (official) + many | **ADDED + validated (2026-07-16), CLEAN — no loader change.** decoder_embed.cpp already takes blk.N.* + the gpt2-BPE KV-merges path is handled. Final cosine vs HF: q8 mean 0.999727, **f16 mean 1.000000** (graph exact); garbage margin 0.58 |
| **EmbeddingGemma-300m** | `gemma-embedding` — mean pool, **Dense bottleneck + Matryoshka** projection | ✅ (mean) | `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF`, `unsloth/…`, `lmstudio-community/…` | **ADDED + SHIPPED (2026-07-17, `138ee0c`).** Real bug was the tokenizer (SPM loaded as char-level BPE), not Dense/norm; arch-gated routing to `decoder_embed.cpp` + SPM-BPE bigram-merge mode + Dense baked via `models/add-st-dense-to-gguf.py`. HF-full parity **0.985**; registry `embeddinggemma-300m-qat`; matrix entry (HF gate), 10/10 PASS; HF `cstr/embeddinggemma-300m-GGUF`. Full write-up: HISTORY.md 2026-07-17; deep-dive: LEARNINGS.md "Community `gemma-embedding`". |
| **LFM2.5-Embedding-350M** | `lfm2` bidirectional hybrid — ShortConv + attention, **BOS-only wrap** | ✅ (CLS, pooling_type=2) | `LiquidAI/LFM2.5-Embedding-350M-GGUF` (official) | **FIXED + SHIPPED (2026-07-16), added to matrix.** Was a loader gap — `lfm2_embed` requires our `lfm.*` tensor names + `lfm2.<our>` hparam keys + a `lfm2.layer_types` c/a string; the official llama.cpp export uses `blk.N.*` + canonical `lfm2.*` keys + no layer-types string. Same class as modern-bert (alias gap), bigger. **Complete fix recipe (exact tensor + hparam maps, layer-type-from-tensor-presence, per-stage gate) in the "FOUND (2026-07-16): official `lfm2`…" subsection just below.** GGUFs already downloaded. Needs a quiet box for the build + `test-lfm2-diff` per-stage validation |
| **GTE-v1.5 (gte-base-en-v1.5)** | `NewModel` NTK-RoPE + GeGLU **tanh** (the path the modern-bert `geglu_erf` gate was explicitly kept OFF for) | ✅ | `cstr/gte-base-en-v1.5-GGUF` (our own; llama.cpp ❌ so third-party rare) | **ADDED + validated per-stage (2026-07-16).** q8 vs HF fp32: emb_ln_out gate 0.999927, all layers PASS (encoder_out 0.9926). Guards the tanh-GeGLU branch stays correct next to modern-bert's erf branch. Arch coverage (own GGUF), not ecosystem-compat |
| **MPNet (all-mpnet-base-v2)** | MPNet two-stream / T5-style rel-attn bias — **we are unique** | ✅ | `cstr/all-mpnet-base-v2-GGUF` (our own; no third-party — llama.cpp ❌) | **ADDED (2026-07-20)** — matrix guard `all-mpnet-base-v2`; HF final-cos 0.997 realistic / 0.987 short (f32==q8_0 → structural residual, not quant); guards the unique rel-attn-bias graph. Arch coverage, not ecosystem-compat |
| **XLM-R-large / multilingual-e5-large** | `bert`+SPM XLM-R at 1024-dim | ✅ | `soichisumi/…-Q8_0-GGUF`, `phate334/…`, `walsons/…` | **EXPECT the e5-small position-offset FAILURE** (XLM-R needs offset 2; community `bert`-arch GGUFs omit `position_offset`). Add ONLY if a community GGUF declares the offset — else it documents the same known gap |
| **SPLADE-v3 (sparse)** | MLM/sparse head — `has_sparse` path, NOT dense | sparse metric (sparse-cos), not the garbage guard | `mradermacher/Splade-V3-GGUF` — **HEADLESS, unusable (2026-07-20)** | **Driver already does SPLADE** (CLI `--sparse`, `crispembed_encode_sparse`, `splade-pp-en-v1` ships at sparse-cos 0.996, `audit_gguf_heads` guards the head through quant). The COMMUNITY GGUF can't be supported: `mradermacher/Splade-V3-GGUF` (arch `bert`, 197 tensors, inspected) has **NO `cls.predictions.*`/MLM head** — llama.cpp drops it at convert, so it loads as a plain dense encoder (same class as e5-small/EmbeddingGemma "community export drops the head"; no loader alias recovers an absent tensor). Only OUR converter (`convert-bert --crisp`, reads checkpoint files) keeps the head. **`naver/splade-v3` ADDED + SHIPPED (2026-07-20):** `convert-bert --crisp` (MLM head detected+kept) → f16/q8_0/iq4_xs+imatrix, sparse-cos vs HF **1.0000 / 1.0000 / 0.9971**; HF `cstr/splade-v3-GGUF` (CC-BY-NC-SA-4.0 card + attribution); registry `splade-v3`/`splade-v3-q8` (NC → `--accept-license`), `upload_to_hf.py` + `audit_gguf_heads` entries. |
| **DeBERTa-v2** | disentangled c2p/p2c rel-attn (`rel_embd`, `position_buckets`) — **we are unique**, highest-complexity encoder path | ✅ | **none found** (llama.cpp ❌, no community GGUF exists) | Blocked on the absence of any third-party GGUF; only our own conversion exists |

Status: **Qwen3-Embedding, LFM2.5-Embedding, granite-107m, GTE-v1.5, EmbeddingGemma,
and MPNet all ADDED**; **e5-small CLOSED** (under-specified export). **Genuinely
remaining candidates:** **XLM-R-large** expected to reproduce the e5 offset gap
(add only as a documented negative, or if a community GGUF declares
`position_offset`); **DeBERTa-v2** blocked on GGUF availability (no third-party
export exists). Do each
on a quiet box (250K-vocab SPM reads + HF forwards are slow under contention) and
gate on the per-stage structural cosine. **SPLADE-v3 is NOT a remaining driver
gap** — sparse retrieval ships (`splade-pp-en-v1`, sparse-cos 0.996); the community
`Splade-V3-GGUF` is headless (documented above), so only an optional converter-add
of our own `naver/splade-v3` GGUF remains.

### Transcoda OMR decode enhancements (deferred, 2026-07-13)

The shipped `transcoda_ocr` engine uses greedy decode (byte-identical to the HF
reference; persistent device-KV, 2.4–4×). The paper's two higher-accuracy decode
modes are **deferred** — both are large, and neither is byte-exactly validatable,
so they were intentionally NOT shipped (byte-exact-or-bust discipline). Concrete
plans for a follow-up session:

- **Beam search (width 3)** — the paper's headline (OMR-NED 18.46% vs greedy
  ~higher on Verovio-synth). HF config: `num_beams=3, length_penalty=1.0,
  repetition_penalty=1.1, early_stopping=True`.
  - *Where*: a `decode_beam(ctx, n_beams)` in `src/transcoda_ocr.cpp`, gated
    `TRANSCODA_OCR_NUM_BEAMS=N` (opt-in; greedy stays the default). Per-beam
    next-token logits via either B independent persistent KV caches (extend
    `pk_*` to a `[..., B]` beam dim) or the full-recompute `run_decoder` per beam
    (simplest, O(B·L²) — fine for opt-in).
  - *Algorithm* (mirror HF `BeamSearchScorer`): keep B live beams (init scores
    `[0,-inf,-inf]`), each step apply per-unique-token rep-penalty + `log_softmax`,
    add to beam score, take top-`2B` over the flattened `B×vocab`, route eos
    candidates to a finished pool with score `/(len**length_penalty)`, keep the
    top-B non-eos as the next beams; early-stop when B finished hypotheses exist;
    return the best finished (or best live) hypothesis.
  - *Validation*: (1) on the confident synth page `sample_page.png`, HF beam-3 ==
    greedy, so mine must be **byte-exact == greedy** there (a real regression
    gate); (2) on a real Polish scan (`btrkeks/polish-scores`, license "other" —
    LOCAL validation only, do NOT commit the image), HF beam-3 diverges from
    greedy at accent/ornament tokens (`16b#JJ`→`16bJJ`) and spine markers
    (`*^`/`*v`) — target **CER-close** to the HF beam-3 dump (byte-exact over a
    512-token uncapped scan is not realistically achievable; cascading). HF
    references already captured: `scratch-transcoda/oracle_beam3.kern.txt`,
    `polish_beam3.kern.txt`.

- **Grammar-constrained decode** — guarantees structurally-valid `**kern`
  (paper's `grammars/kern.gbnf` via xgrammar logits processors). Large: needs a
  GBNF parser + a per-step token-mask constraint engine (llama.cpp's
  `llama-grammar` is the reference, ~1k LOC). *Where*: a `kern_grammar.{h,cpp}`
  constraint module + a mask hook in the decode loop, gated
  `TRANSCODA_OCR_GRAMMAR=1`. *Validation*: structural only (every output parses as
  valid kern); no byte-exact HF target (xgrammar's tie-breaking differs). Lowest
  priority — greedy already emits valid kern on clean inputs.

### Optical Music Recognition (OMR) — models to port (2026-07-12)

OMR is "OCR for staff notation": the winning modern approach is exactly the
TexTeller shape — vision encoder + autoregressive transformer decoder emitting
a linearized notation token sequence. This reuses the existing
VisionEncoderDecoder machinery (`math_ocr.cpp` path). Output format is
irrelevant to us (bekern / **kern / MusicXML / LilyPond are all parseable
downstream), so we optimize for arch fit + license, not output dialect.

**Two distinct problems:** printed staff notation (tractable, MIT weights
exist) and handwritten (hard; the real license risk is on the *training
data*, not the code — see landmine below).

**Licensing methodology — AGPL *code* is NOT a blocker (verified 2026-07-13).**
The gate is the **weights** license (we redistribute GGUFs) and the **engine
authorship**, which are independent of an upstream repo's *code* license:
- If the **weights** are permissive (MIT / Apache / **CC BY**), the GGUF is
  redistributable regardless of the training-code license. AGPL/GPL on the
  *code* does not attach to CC-BY *weights*. (Training-data license only matters
  if we redistribute the data or retrain — not for shipping pretrained weights.)
- The **engine** is written **clean-room**: run the upstream Python as an
  *oracle* (reference-activation dumps — no derivative) and implement from a
  **facts-only spec** (architecture, tensor shapes, op order, hparams, eps/scale
  — all uncopyrightable) + the paper + configs. Never transcribe AGPL source
  line-by-line. Two-team wall: the brief-writer may read the AGPL `.py`; the
  implementer sees only the facts brief. (Permissive blueprints don't need this.)
- Hard rejects shrink to: **gated / unlicensed / non-permissive weights**, or an
  **11B+ base under a restrictive model license**, or a **non-single-model
  pipeline** (poor ggml fit).

| # | Model | Params | License (code / weights) | Architecture | Output | Handles | Effort | Status |
|---|-------|--------|--------------------------|-------------|--------|---------|--------|--------|
| 1 | **Sheet Music Transformer (SMT)** | 21.4M | **MIT / MIT** | ConvNext encoder + Transformer decoder | bekern | Printed polyphonic | Low | **DONE** — `src/smt_ocr.cpp` shipped (per-stage cos 1.0, 96.3% GrandStaff) |
| 1b | **SMT++ full-page** | ~10.9M | **MIT / MIT** — `PRAIG/smt-fp-grandstaff` (public, **not gated**, verified HF card) | full-page extension of SMT (curriculum-trained) | bekern | **Full-page pianoform** (no separate layout stage) | **Low–Med** | **DOABLE — top permissive target.** Verify arch delta vs base SMT first: deep-research *refuted* (2-1) the "same-arch, curriculum-only" claim, so confirm the graph before assuming free reuse. If same graph → near-free extension of shipped SMT |
| 2 | **Transcoda-59M-zeroshot** | 58.8M | **AGPL code / CC BY 4.0 weights** (`btrkeks/transcoda-59M-zeroshot-v1`, verified HF card) | ConvNeXt-V2-Tiny enc + 8L Transformer dec (d512/8h, **RoPE**) | **kern | **Full-page + historical scans** (zero-shot); **current OMR-NED SOTA** (Polish 63.97%, Verovio 18.46% — beats SMT++ & Legato) | **Med** | **DOABLE — accuracy leader.** Weights CC BY 4.0 → GGUF redistribution clean (attribute). Engine **clean-room** (code is AGPL). Arch fully in-tree: ConvNeXt-V2 ≈ SMT's ConvNext, RoPE decoder ≈ Qwen3; add 3000-token kern BPE tok; optional GBNF grammar-constrained decode. Training data `polish-scores` = `license: other` (irrelevant to CC-BY weight redistribution) |
| 3 | Polyphonic-TrOMR (NetEase) | ~22M | **Apache-2.0 / Apache-2.0** | ViT + multi-head Transformer decoder (rhythm/pitch/lift/note) | symbolic text | Printed polyphonic photos | Medium | **DONE** — `src/tromr_ocr.cpp` (cos 1.0 / 100% argmax / byte-exact); `cstr/tromr-GGUF` |
| 4 | **Flova/omr_transformer** | 143M | Apache-2.0 / Apache-2.0 | Donut VED (DonutSwin + 4L mBART) | LilyPond | artificial + **handwritten** + whiteboard (monophonic) | Medium | **DONE** — `src/flova_ocr.cpp` (cos 1.0 / 40-40 argmax / byte-exact); `cstr/flova-omr-GGUF` (f32 + q8_0); CLI + registry wired |
| 5 | oemer | 2× U-Net | MIT / MIT | 2 segmentation U-Nets + numpy reconstruction | MusicXML | Printed, photos, skewed | High | Reference-only — multi-model + rule-based reconstruction, poor ggml fit |
| ~~6~~ | ~~Legato~~ | ~11B | MIT (trained delta) / **Llama-3.2 license + GATED** | frozen Llama-3.2-11B-Vision + trained decoder | ABC | full-page | — | **REJECTED** — 11B base under Meta's Llama license + contact-gated weights; MIT covers only the delta. Too big + non-permissive base |
| ~~7~~ | ~~starry / FindLab~~ | — | **no code license / gated, unlicensed weights** | 7-microservice pipeline (PyTorch+TF+ONNX) | LilyPond/kern | complex polyphonic | — | **REJECTED** — not a single model (poor ggml fit) *and* weights token-gated with no stated license |
| ~~8~~ | ~~Clarity-OMR~~ | — | (unverified) | PDF→MusicXML **pipeline** | MusicXML | printed | High | Reference-only — multi-stage pipeline, not a single VED model |
| ~~9~~ | ~~homr (liebharc)~~ | — | **AGPL-3.0** (code) | pipeline + TrOMR | MusicXML | printed/camera | — | **REJECTED** — pipeline (poor ggml fit); the underlying TrOMR is already shipped separately (Apache-2.0) |

**Recommended priority (updated 2026-07-20 — SMT/SMT++/TrOMR/Flova/Transcoda ALL shipped):**

1. ~~**SMT++ full-page**~~ **SHIPPED** — `smt_ocr.cpp` (arch reuse confirmed) +
   registry `smt-fp`; HF `cstr/smt-fp-grandstaff-GGUF`. (Was the "best next step";
   done.)
2. ~~**Transcoda-59M**~~ **SHIPPED** — clean-room `src/transcoda_ocr.cpp` +
   registry `transcoda`; HF `cstr/transcoda-omr-GGUF` (CC-BY-4.0). The only genuine
   remaining Transcoda work is the *optional* beam-3 + GBNF `**kern`
   grammar-constrained decode (still greedy-only; see "Transcoda OMR decode
   enhancements" — the sole open OMR-engine lever).

3. **Handwritten *polyphonic* — the real remaining gap.** No permissive model
   fills it: Flova (shipped) is monophonic-toy; the strong performers are all
   rejected (Legato = Llama-11B/gated, starry = gated/unlicensed pipeline, homr =
   AGPL pipeline). Reach it by *fine-tuning* a shipped graph (SMT or Transcoda)
   on synthetic + license-clean handwritten-style data — same engine, new weights.

3. **Polyphonic-TrOMR — DONE (2026-07-13).** Genuinely accurate model (reads
   clefs/keys/rhythms/pitches correctly on real photos). The ggml engine
   `src/tromr_ocr.cpp` (ResNetV2 SAME-pad backbone + hybrid ViT encoder →
   x-transformers 12-sublayer decoder with SIGLU attn-on-attn + GEGLU FF → 4
   parallel heads, autoregressive over rhythm/pitch/lift streams) is written,
   wired (dispatcher + CMake + `test-tromr-diff` + CLI `--ocr` auto-detect), and
   **validated CPU-only vs the reference model**: every diff-harness stage cos
   **1.0** (backbone, ViT context, all 12 decoder blocks, all 4 logit heads),
   **100% per-position argmax agreement** teacher-forced (66/66, 85/85), greedy
   decode **byte-exact** vs the authors' `examples/{1,2,3}.txt`, Metal == CPU.
   q8_0 also decodes byte-exact. ~~**Remaining:** HF upload `cstr/tromr-GGUF`
   (f32 + q8_0) + `model_mgr.cpp` registry entry.~~ **DONE** — `tromr` registry
   entry (model_mgr.cpp) points at `cstr/tromr-GGUF`.
   Corrections vs the (now-removed) handover brief found in validation: ViT scale
   is **32^-0.5** not 64^-0.5; the converter emitted tensor names >64 chars that
   the ggml loader rejects (`GGML_MAX_NAME`) → shortened the backbone prefix to
   `enc.bb`; the quantizer must keep `enc.bb`/`enc.proj` convs unquantized
   (flatten+quantize → reshape-to-4D abort). See LEARNINGS.md.
   Weights: `tromr/workspace/checkpoints/img2score_epoch47.pth` (86.3 MB)
   committed directly into the Apache-2.0 repo (not LFS → covered by the repo
   license), with a 4-file tokenizer set (`tokenizer_{lift,pitch,rhythm,note}.json`).
   Architectural wrinkle vs SMT: TrOMR is **not** a single autoregressive stream
   — it has *parallel classification heads* (rhythm / pitch / lift / note) per
   decoder timestep, so the port needs 4 output projections + a merge step, not
   one LM head. `homr` wraps this same model but is AGPL — weights taken from
   the NetEase repo, not homr.

**Reuse map (assessed 2026-07-12, feat/smt-omr worktree):** ~70% of the SMT
port reuses existing infra —
- **Decoder + decode loop + C ABI:** `src/math_ocr.cpp` is already SMT's exact
  shape ("Hybrid CNN + ViT encoder → cross-attention Transformer decoder → token
  sequence"): KV-cached decoder, greedy + beam decode, batched encode, per-token
  confidences. SMT's "classic Transformer decoder" == TrOCR == this; port by
  config, not new graph code.
- **Converter:** `models/convert-trocr-safetensors-to-gguf.py` already handles
  the decoder + top-level `decoder_start_token_id`. New `convert-smt-to-gguf.py`
  = that file + a ConvNext encoder tensor mapping.
- **ConvNext encoder (the one new piece):** CrispASR has ConvNeXt blocks in
  `f5_tts / vibevoice / qwen3_tts / kugelaudio / outetts_wavtok` (1-D/audio, but
  identical block: dwconv → LN → pwconv → GELU → pwconv → layer-scale → residual)
  + `core/activation.h`; CrispEmbed has mature `ggml_conv_2d` engines (`swinir`,
  `nafnet`, `cnn_embed`, `adair`, `tbsrn`) for the 2-D image side. Adapt, not
  invent.
- **Shared load/preproc/vocab:** math_ocr grayscale-resize-normalize;
  `core/{gguf_loader,cpu_ops,bpe}.h`; bekern = fixed lookup vocab (simpler than
  any in-tree BPE).
- New work = 2-D ConvNext encoder + bekern vocab + encoder-side converter.

**Confirmed SMT architecture (2026-07-12, from SMT++ source + safetensors header):**
Total **21.4M params, F32, 360 tensors, 85.5 MB** `model.safetensors`. Greedy
manual decode (no HF `.generate()`), seed `<bos>=4426`, stop `<eos>=8822`,
`pad=0`, up to `maxlen=1281` steps.
- **⚠ Convert against SMT++ tensor names, NOT SMT-main.** The shipped
  grandstaff/camera-grandstaff weights only match `SMT-plusplus/smt_model/
  modeling_smt.py` (`input_attention`/`cross_attention`/`ffNet`/`out_layer`); the
  SMT-main repo has a rewritten module whose names match no checkpoint.
  `smt-string-quartets` ships **no weights** (README only).
- **Encoder** = stock HF `ConvNextModel(num_channels=1, num_stages=3,
  hidden_sizes=[64,128,256], depths=[3,3,9])`. Plain ConvNeXt, no attention. Stem
  Conv2d(1→64,k4,s4)+LN; stage-1/2 downsample Conv2d(k2,s2); **16× H/W reduction**.
  Last stage already outputs 256 = `d_model`, so **no encoder→decoder projection**.
  `encoder.layernorm` (pooler LN) is in the ckpt but **dead** on the inference path
  (`last_hidden_state` is pre-pooler). Tensors:
  `encoder.embeddings.patch_embeddings.{weight[64,1,4,4],bias}`,
  `encoder.encoder.stages.{0,1,2}.layers.{i}.{dwconv,layernorm,pwconv1,pwconv2,layer_scale_parameter}`,
  `encoder.encoder.stages.{1,2}.downsampling_layer.{0=LN,1=Conv2d}`.
- **Decoder** = 8 layers, d_model=256, **4 heads** (hd=64), **FFN dim=256 (1×, not
  4×)**, activation **ReLU** (+ `end_relu` before the head). Post-norm:
  self-attn→norm1→cross-attn→norm2→FFN→norm3. Token emb `nn.Embedding[20578,256]`;
  **embeddings NOT tied** to head. LM head = `Conv1d(256→20578,k1)` →
  `decoder.out_layer.weight[20578,256,1]` (squeeze trailing 1 → Linear) + bias.
  Tensors: `decoder.embedding.weight`, `decoder.decoder.layers.{0..7}.
  {input_attention,cross_attention}.{lq,lk,lv,out_proj}.{weight,bias}`,
  `.ffNet.{0,3}.{weight,bias}`, `.{norm1,norm2,norm3}.{weight,bias}`.
- **Positional encodings are NOT in the checkpoint — bake as constants.** (a) 1-D
  sinusoidal added to decoder token embeddings; (b) 2-D sinusoidal
  (`dim=256`, first 128ch=row H, last 128ch=col W, `div=exp(-arange(0,dim//2,2)/dim·ln1e4)`).
- **⚠ Cross-attention key≠value:** encoder output flattened over H×W;
  the 2-D PE is added to the **KEYS only**; **VALUES are the raw** flattened
  features. Query = decoder states. Cross-attn has no mask; self-attn is causal.
- **Preprocessing:** grayscale, **always color-invert** (`RandomInvert(p=1.0)` —
  mandatory, not augmentation), `ToTensor` → **[0,1], NO mean/std normalize**.
  `cv2.resize` bilinear at `reduce_ratio=0.5`, height floored/capped ~256px
  (`maxh=256`, `maxw=3056`).
- **bekern vocab** = fixed word-level lookup (NOT BPE), `out_categories=20578`,
  identical across grandstaff/camera. `w2i`/`i2w` embedded in `config.json`
  (875 kB) and as `vocab/*.npy`. Split GT on whitespace/`·` delimiter; layout
  tokens `<b>` break / `<s>` space / `<t>` tab.
- **SMT vs SMT++:** identical neural graph; SMT++ gains are training-side
  (curriculum + synthetic full pages). Full-page = same graph, bigger images +
  longer decode + layout tokens, no extra module. **Target single-system
  grandstaff first** (the only checkpoints with published weights).

**Port progress (2026-07-12, feat/smt-omr worktree):**
- ✅ `models/convert-smt-to-gguf.py` — torch-free, verbatim SMT++ names, squeezes
  `out_layer` 1×1 conv→Linear, bakes 1-D decoder PE, records `smt.scale_attention=
  False`. Verified GGUF: arch `smt_ocr`, 361 tensors, 20578-tok vocab, 83 MB.
- ✅ `tools/dump_smt_reference.py` — loads REAL SMT++ model (hooks, not a
  re-forward), dumps 18 per-stage F32 tensors → `smt_ref.gguf`. Validated on a
  real GrandStaff test image: enc 336×128→`(256,8,21)` (16× reduction, 168 mem
  tokens), decode emits correct bekern (`**ekern_1.0 <t> … *clefG2 <b> …`).
  Test assets in scratchpad: `smt-grandstaff/`, `SMT-plusplus/` clone, `gs_test0.png`
  (+ `.gt.txt`), `smt_ref.gguf`. Note: cloned `SMTConfig` needs a
  `super().__init__(**kwargs)` patch to load under transformers 4.57.
- ✅ `src/smt_ocr.{h,cpp}` ggml engine (ConvNext encoder + cross-attn decoder +
  greedy decode) + `tests/test_smt_diff.cpp` + CMake wiring. **Full per-stage
  parity vs `smt_ref.gguf` (CPU):** enc_stage0/1/2 + enc_output + mem_key
  cos_min ≥ 0.999996; dec_tok_emb + dec_layer0–7 + logits cos_min = **1.000000**.
  Native greedy decode emits correct bekern (header/clefs/meter/barlines match
  GT exactly; `*k[]` vs GT `*k[b-]` is the model's own prediction — the Python
  ref emits `*k[]` too). Bugs found & fixed during bring-up: (a) off-trunk
  `enc_stageN` snapshots weren't in the graph (`to_tokens` forks off the trunk)
  → `ggml_build_forward_expand` each; (b) `crispembed_diff.h` GGUF reader only
  decodes F32 (its I32 branch checks a stale type id 5, but this ggml tags I32
  as 26) → dumper now stores `token_ids` as F32.
- ✅ Preprocessing parity: `recognize_raw` now does cv2-bilinear resize +
  RandomInvert + BGR-as-RGB grayscale → native decode is **token-identical to
  HF** on real GrandStaff scores (100% on 3/4; 4th matched to the ref cap), CPU
  and Metal.
- ✅ Wiring: `src/crispembed.cpp` dispatcher (`arch=="smt_ocr"` → all 4 switches),
  so `crispembed -m smt.gguf --ocr score.png` works end-to-end (verified 69/69 vs
  HF); `smt_ocr_recognize_raw` added; `examples/cli/model_mgr.cpp` registry entry
  (`smt-grandstaff`). Server/bindings inherit via the generic `crispembed_ocr_model_*`.
- ✅ Quantize: `tools/quantize.cpp` keeps SMT conv kernels (`dwconv`/`downsampling`)
  and the baked PE (`positional`) F32; engine reshapes the quantizer's flattened
  2-D conv headers back to 4-D. **q8_0 (24 MB) decodes identically to HF (100%);
  q4_k (17 MB) is too lossy for the AR decode (~32%) — ship f32 + q8_0 only.**
- ✅ KV-cache: incremental decode (cross K/V precomputed once, self K/V grown per
  step via concat). Token-identical to the full-recompute path (kept behind
  `SMT_OCR_FULL_DECODE=1` for A/B) and to HF, CPU + Metal. **5.4× faster** (0.37 s
  vs 1.98 s for ~100 tokens); the gain grows with sequence length.
- ✅ GGUF upload: `cstr/smt-grandstaff-GGUF` (f32 83 MB + q8_0 24 MB + MIT model
  card; card license verified `mit`). Registry auto-download works end-to-end.
- ✅ **Preprocessing fixed → SMT WORKS at 96.3%.** The engine had been *inverting*
  the image (SMT-plusplus's `convert_img_to_tensor` has `RandomInvert(p=1.0)`), but
  `smt-grandstaff` is an **SMT-main** model whose preprocessing is `Grayscale→
  ToTensor` with **NO invert**. Inverting → ~30%; correct (non-inverted) → **96.3%**
  on the clean `antoniorv6/grandstaff` test split (per-image 91.8/96.2/96.7/99.6%).
  Full pipeline: RGB (no cv2-BGR swap), `reduce_ratio=1.0`, `width=min(w,3056)`,
  `height=max(h,256)`, grayscale, no invert. Fixed in `recognize_raw` + the dumper.
- ✅ **Fully validated:** per-stage diff cos=1.0; C++ decode == Python blueprint
  (100% token agreement, 10 fresh images); **C++ engine vs ground truth = 96.3%.**
  SMT-plusplus's unscaled forward confirmed correct (SMT-main's forward → 0% garbage
  on this checkpoint). The port was exact all along — the invert was the only bug.
  Lesson: [[validate-intermediates-and-outputs]] — a "reads-structure-not-detail"
  pattern across models was a preprocessing/input bug, not model quality; derive
  preprocessing from the model's OWN repo (SMT-main, not the SMT-plusplus fork).

**Landmines:**
- **⚠ SMT attention is UNSCALED.** `MHA.forward` computes `bmm(q,k)` then softmax
  with **no** `1/sqrt(head_dim)` — `self.scale_factor` is defined but never
  applied (verified in source, not the abstract). The C++ must NOT scale QK^T
  (converter records `smt.scale_attention=False`). Also: token embeddings are
  **not** scaled by `sqrt(d_model)` (no `scale_embedding`).
- **Cross-attn key≠value:** memory_key = flattened encoder features **+ 2-D PE**;
  memory_value = **raw** flattened features. Easy to wire both to the same tensor.
- **Encoder `last_hidden_state` is pre-pooler-LN** → `encoder.layernorm` in the
  ckpt is dead weight; don't apply it. Feature map is `(256, H/16, W/16)`.
- **Handwritten training-data license trap:** the canonical handwritten OMR
  datasets — **MUSCIMA++ / CVC-MUSCIMA — are CC BY-NC-SA (non-commercial)**.
  Training weights on them contaminates the *weights* for commercial use (same
  pattern as the old PosFormer/BTTR/HMER math models). PrIMuS / Camera-PrIMuS /
  GrandStaff are printed/synthetic and license-clean. Keep handwritten training
  data NC-free from day one if shipped weights must be commercially usable.
- **VisionEncoderDecoder `decoder_start_token_id`** comes from the *top-level*
  config, not the nested decoder config (the TexTeller start-token bug that
  poisoned position-0 KV — see the TexTeller 3.0 entry above). SMT's converter
  must resolve the start token the same way.
- Watch F16 Metal matmul overflow on large activations (see
  [[metal-mul-mm-f16-overflow]]) as with all VED ports.

**Sources:** SMT [github.com/antoniorv6/SMT](https://github.com/antoniorv6/SMT) ·
[SMT++](https://github.com/antoniorv6/SMT-plusplus) ·
[HF smt-grandstaff (MIT)](https://huggingface.co/antoniorv6/smt-grandstaff) ·
[PRAIG collection](https://huggingface.co/collections/PRAIG/sheet-music-transformer-6853c4ca1bd7980a91677dfd).
oemer [github.com/BreezeWhite/oemer (MIT)](https://github.com/BreezeWhite/oemer).
TrOMR [github.com/NetEase/Polyphonic-TrOMR (Apache-2.0, weights `img2score_epoch47.pth` 86 MB in-repo)](https://github.com/NetEase/Polyphonic-TrOMR).
[Flova/omr_transformer (Apache-2.0)](https://huggingface.co/Flova/omr_transformer).
homr [github.com/liebharc/homr (AGPL-3.0)](https://github.com/liebharc/homr).

### OCR — next-gen models to port

| # | Model | Params | OmniDocBench | License | Architecture | Status |
|---|-------|--------|-------------|---------|-------------|--------|
| ~~1~~ | ~~dots.ocr~~ | ~~3B~~ | ~~88.4%~~ | ~~NOT pure MIT~~ | — | REJECTED: supplemental PRC license (rednote/Xiaohongshu) |
| 2 | **PaddleOCR-VL-0.9B** | 0.9B | — | Apache-2.0 | NaViT + ERNIE-4.5-0.3B | **DONE + verified E2E** (2026-07-02): reuses qwen2vl_ocr engine; fox.png → "The quick brown fox…" on CPU+Metal. Was SIGSEGV-ing (ERNIE head_dim=128≠D/heads) + empty output (SPM vocab loaded as GPT-2 BPE); both fixed. Q8_0/Q4_K on HF |
| 3 | **PaddleOCR-VL-1.6** | 0.9B | 96.3% SOTA | Apache-2.0 | NaViT + ERNIE-4.5-0.3B (same arch, improved training) | **DONE**: same engine/fixes as 0.9B; Q8_0/Q4_K on HF |
| ~~4~~ | ~~MinerU2.5-Pro~~ | ~~1.2B~~ | ~~90.7%~~ | ~~NOT pure Apache~~ | — | REJECTED: commercial thresholds, mandatory attribution, gated HF |
| 5 | **SmolDocling** | 256M | — | Apache-2.0 | Idefics3/SmolVLM, IBM Research | DONE: engine + parity cos=0.9999, HF `cstr/smoldocling-GGUF` |
| ~~6~~ | ~~Hunyuan-OCR~~ | ~~1B~~ | — | ~~Custom Tencent~~ | — | REJECTED: excludes EU/UK/South Korea |
| 7 | **Qari-OCR** | 4B | Apache-2.0 | Qwen2-VL fine-tune (Arabic only) | **DONE (shipped)** — registry `qari-ocr` → `qari-ocr-2b-q4_k.gguf`. Vision parity fixed; direct "output only text" prompt; filename-independent `general.name` detection. |

~~**Remaining**~~ **DONE (both shipped)**: FireRed-OCR (registry `firered-ocr` / `firered-ocr-q4k`, Qwen3-VL 2B) and german-ocr-3 (registry `german-ocr-3.1`, Qwen2.5-VL) both reuse the qwen2vl_ocr engine; runtime ne-fix handles GGUF converters that store weights in PyTorch (out, in) order. (NB: `src/fireredpunc.cpp` is a different model — BERT punctuation, not FireRed-OCR.)

#### OCRBench leaderboard reference (small VLMs, ≤3B)

| Rank | Model | LLM | Params | OCRBench | License | Status |
|------|-------|-----|--------|----------|---------|--------|
| 1 | Granite Vision 3.3-2B | Granite-3.1-2B | 3B | 852 | Apache-2.0 | **Ported** |
| 2 | InternVL2.5-2B* | InternLM2.5-1.8B | 2.1B | ~830 | MIT | **Ported** |
| 3 | MiniMonkey | InternLM2-1.8B | ~2B | 806 | — | Low priority |
| 4 | H2OVL-Mississippi-2B | H2O-Danube-1.8B | 2.1B | 782 | Apache-2.0 | **Ported** |
| 5 | InternVL2-1B | Qwen2-0.5B | 0.9B | 779 | MIT | **Ported** (edge) |
| 6 | InternVL2-4B | Phi-3-mini | ~4B | 776 | MIT | Low (too big) |
| 7 | H2OVL-Mississippi-0.8B | H2O-Danube3-0.5B | 0.8B | 751 | Apache-2.0 | Low (tiny) |

*InternVL2.5-2B not on the original leaderboard slice but scores higher than
InternVL2-2B (768).

### llama.cpp parity — support matrix (reference)

A living audit of which CrispEmbed architectures llama.cpp supports (upstream
`ggml-org/llama.cpp` @ ~`4fc4ec5`, July 2026), how it implements them, and where
we remain unique. Deep technical notes live in `LEARNINGS.md → "llama.cpp
implementation reference"`. **The convergence backlog derived from this audit
(C1 imatrix quant, C3 batched throughput, C4 prefix KV, C5 mtmd preprocessing,
C6 flash-attn epilogue, mmproj interop) all shipped — see HISTORY.md.** This
section is kept only as the capability reference. Any future borrow must still
land behind an A/B on BOTH speed and quality, on CPU and Metal.

#### Support matrix (CrispEmbed arch → llama.cpp)

Text-embedding encoders:

| CrispEmbed | in llama.cpp | llama.cpp arch id | note |
|---|---|---|---|
| BERT | ✅ | `bert` | one shared `bert.cpp` graph, config from GGUF |
| XLM-RoBERTa | ✅ | `bert` | RoBERTa/XLM-R fold into `bert`; pos-offset + SPM handled |
| NomicBERT | ✅ | `nomic-bert` | SwiGLU + RoPE |
| NomicBERT-MoE | ✅ | `nomic-bert-moe` | PR #12466; 8-expert top-2 |
| ModernBERT | ✅ | `modern-bert` | SWA global/local + per-layer RoPE θ |
| MPNet | ❌ | — | T5-style rel-attn bias unimplemented — **we are unique** |
| GTE-v1.5 (`NewModel`) | ❌ | — | NTK-RoPE `NewModel` unsupported (#6821) — **we are unique** |
| DeBERTa-v2 | ❌ | — | disentangled c2p/p2c has no ggml graph — **we are unique** |
| SPLADE (sparse) | ❌ | — | MLM head dropped at convert — **we are unique** |
| bge-m3 sparse+ColBERT | ❌ (dense only) | — | tri-head only in fork `iz0eyj/llama.cpp-mv` — **we are unique** |

Decoder / hybrid embedders:

| CrispEmbed | in llama.cpp | arch id | note |
|---|---|---|---|
| Qwen3-Embedding | ✅ | `qwen3` (embed mode) | last-token, **causal** (Qwen3-Emb is trained causal — correct); Instruct/Query prefix is caller-side |
| EmbeddingGemma | ✅ | `gemma-embedding` | Dense/Matryoshka projection supported via `--sentence-transformers-dense-modules`; mean, non-causal |
| LFM2 / LFM2.5 | ✅ | `lfm2` (+`lfm2moe`) | PR #14620; ShortConv via `ggml_ssm_conv`, conv tensors F32 |
| LFM2.5-Embedding | ✅ | `lfm2` embed | official LiquidAI GGUFs, bidirectional |
| LFM2.5-ColBERT | ⚠️ partial | `lfm2` + `--pooling none` | per-token out; MaxSim client-side |
| BidirLM-Omni | ❌ | — | not present — **we are unique** |

Reranking: `--pooling rank` (RANK=4), `/v1/rerank` (PR #9510). bge-reranker-v2-m3
/ base, jina-v2, ms-marco-MiniLM ✅. Qwen3-Reranker ✅ (needs `cls.output.weight`
+ template). mxbai-rerank (DeBERTa-v2) ❌.

Vision / VLM-OCR (via `libmtmd`, projector-id keyed):

| CrispEmbed | in llama.cpp | projector id | note |
|---|---|---|---|
| Qwen2/2.5-VL | ✅ | `qwen2vl_merger` / `qwen2.5vl_merger` | 2D RoPE `build_rope_2d()`, window-attn |
| Qwen3-VL (+MoE) | ✅ | `qwen3vl_merger` | **DeepStack + IMROPE** — same family as our BidirLM-Omni |
| InternVL2/2.5/3 | ✅* | `internvl` | OpenGVLab (non-HF) checkpoints only |
| GLM-4V / GLM-OCR | ✅ | `glm4v` | AIMv2 tower, **dynamic** resize — ours matches now (Glm46VImageProcessor Qwen2VL smart-resize, shipped `dfd5653`; verified OCR 2026-07-13) |
| Granite Vision 3.x | ✅ | `mlp` (LLaVA-Next) | multi-level feature concat + anyres |
| SmolVLM/SmolDocling/Idefics3 | ✅ | `idefics3` | SigLIP + pixel-shuffle |
| Pixtral / LightOnOCR-1B | ✅ | `pixtral` / `lightonocr` | LightOnOCR-2 declined (#18943) |
| DeepSeek-OCR / Unlimited-OCR | ✅ | `deepseekocr` / `deepseekocr2` | hybrid SAM+CLIP DeepEncoder |
| PaddleOCR-VL | ✅ | `paddleocr` | NaViT + M-RoPE (`ggml_rope_multi`) |
| GOT-OCR2 | ❌ | — | SAM path exists only inside DeepSeek-OCR — **we are unique** |
| CLIP/SigLIP standalone image **or** text embed | ❌ | — | mtmd is tower-only (per-patch, LLM-sized); no text tower — **we are unique** |
| Math OCR (pix2tex/TrOCR/HMER/BTTR/PosFormer/MixTex/PP-FormulaNet/PARSeq/Tesseract/Pix2Struct) | ❌ | — | enc-dec/CTC out of llama.cpp's class — **we are unique** |

**Reverse interop (import a stock llama.cpp mmproj INTO CrispEmbed):** shipped +
validated for the three rows where both a working CrispEmbed loader and a
downloadable mmproj exist — `qwen2vl_merger`, `idefics3`, `internvl` — via the
auto-detecting `models/merge-llamacpp-gguf.py` (see the status block below and
README "Importing a stock llama.cpp VL model"). Qwen2-VL is bidirectional
(export too). The rest need either a non-crashing dynamic-preproc loader
(`glm4v`) or an mmproj llama.cpp doesn't ship (`GOT-OCR2`).

Entirely outside the ggml ecosystem (CrispEmbed-only): **face** (YuNet/SCRFD/
AuraFace/SFace), **detection/layout** (DBNet/RT-DETRv2/Surya-Det), **NER/KIE**
(GLiNER/LiLT; BERT-NER only an *unmerged* PR #19725), **LID** (CLD3/GlotLID),
**punctuation** (FireRedPunc/Fullstop/PCS), and **image restoration/SR** (NAFNet/
SwinIR/HAT/Restormer/SCUNet/SAFMN/DAT/InstructIR/AdaIR — only ESRGAN/RRDBNet
exists, and in `stable-diffusion.cpp`, not llama.cpp).

### Feature gaps vs fastembed-rs

| Gap | Impact | Effort | Notes |
|---|---|---|---|
| ~~Qwen3-VL multimodal~~ **DONE** | — | — | Qwen3-VL OCR/VLM shipped: engine (`qwen2vl_ocr.cpp` DeepStack + interleaved-mRoPE + qk-norm) + registry `qwen3vl-2b`. (Only a Qwen3-VL *embedding* model — not the OCR path — would still be open, if ever wanted.) |

### DeepSeek-OCR-2 performance (remaining levers)

The pipeline is now mostly on Metal (encoder, MoE decode, SAM convs + patch
embed, LM head) — full OCR ~9 min (never completed) → ~12 s warm. Profiled
warm breakdown: load ~9 s cold / 0.8 s warm · SAM ~4.7 s · decode ~3.8 s ·
enc+proj ~1.1 s. Remaining levers, ranked by leverage:

- [x] **#1 Load-path prefetch — DONE, but not the bottleneck.** Added
  `madvise(MADV_SEQUENTIAL/WILLNEED)` to `core_gguf::load_weights` (correct
  practice, helps genuinely disk-bound cold loads on other systems). On *this*
  machine it didn't move the needle, and the diagnostic explains why: the disk
  reads 2.1 GB in **1.17 s** and a warm load is **0.8 s** — so the ~9–18 s cold
  loads are **memory-pressure / swap**, not readahead. During a run the process
  holds ~5 GB (2.1 model + 1.3 stacked experts + 0.65 embed-f32 + Metal) on a
  16 GB box, so file pages and new allocations contend and swap. → the real load
  lever is **reducing the footprint** (#3, #4), not prefetch.
- [~] **#2 Decode graph reuse — PARTIAL (KV persistent; graph NOT).** Corrected
  2026-07-20 (code-verified): the **KV cache is persistent** device tensors
  (`alloc_ds_kv_cache`, written in-graph via `ggml_cpy`/views) — but the decode
  **graph is still rebuilt + freed every layer, every token** (`build_llm_layer_attn`
  → `ggml_free(lag.gctx)` in the per-layer loop). No persistent/single multi-layer
  graph yet; the `g_ds_build_us`/`g_ds_compute_us` profiler exists precisely to
  decide if the persistent-single-graph port is worth it. So **the persistent
  single-step decode graph + F16 KV (cache is F32 today) are genuinely OPEN for
  deepseek specifically** — the one engine the GPU-decode "done" note above does
  NOT fully cover.
- [x] **#3 Per-row embedding dequant — DONE (core win).** The decode hot path
  `get_embedding` lambda is per-row (`ggml_backend_tensor_get` + `to_float`),
  replacing the 655 MB full-table copy held across decode. (Sub-detail corrected:
  the prompt-assembly `put_tok` still full-table-dequants once via `to_f32`, freed
  after — not per-row; line refs drifted to ~2424/~1897.)
- [x] **#4 Converter-emitted stacked experts (memory) — DONE
  (`feat/ds-ocr2-stacked-experts`).** Converter emits `ffn_{gate,up,down}_exps
  [in,out,n_exp]` (byte-identical to `stack_moe_experts`); loader loads them
  directly + per-expert views for the `DS_MOE_CPU` fallback + backward-compat.
  Kaggle-reconverted + byte-validated vs source; f16/q4_k on HF as `-stacked`
  (non-clobber). **M1 Metal q4_k A/B: peak footprint 5.27→3.97 GB (−1.30 GB),
  decoded output identical on all 3 loader paths.** Registry auto-download default
  **promoted to `deepseek-ocr2-q4_k-stacked.gguf`** (loader backward-compatible).
  Deep-dive in LEARNINGS.
- [ ] **#5 SAM flash-attention (marginal, skip unless needed).** The SAM
  attention uses a decomposed rel-pos bias (rel_h/rel_w added to scores), which
  blocks `ggml_flash_attn_ext` unless the bias is materialized as a [T,T] mask —
  fiddly, and the win is small (~3–4 s SAM is mostly the genuine 4096-token
  global attention compute).

All deepseek perf paths are env-gated with validated CPU fallbacks
(`DS_QWEN2_SCALAR`, `DS_MOE_CPU`, `DS_SAM_CONV_CPU`, `DS_LMHEAD_CPU`, `DS_MMAP`,
`DS_REF` parity harness, `DS_DBG` timers).

### Open performance levers

Each needs a target GGUF (q8_0 preferred, to isolate from q4_k noise) and a
before/after parity + latency measurement — never land a "perf" change on a
compile-only check. A/B every change against ground truth and gate behind an env
var (see `../crispasr-crispembed-dev.md` "A/B every perf optimization").

- **ENCODER (embedding) path — the domain the 2026-07-16 community-GGUF work
  landed in, and NOT otherwise in this backlog (encoders are fast: 6–22 layers,
  batched).** One concrete micro-lever spotted:
  - **MoE FFN redundant `ggml_repeat` (nomic-bert-moe / nomic-embed-text-v2-moe).**
    The MoE FFN in `src/crispembed.cpp` explicitly expands the input
    `cur [H,TB] → [H,K,TB]` with `ggml_repeat` before `ggml_mul_mat_id`. llama.cpp's
    canonical MoE reshapes to `[H,1,TB]` and lets `mul_mat_id` BROADCAST the
    singleton expert-slot dim, so the repeat materializes K copies of the
    activations per MoE layer for nothing (6 MoE layers × K=2 on nomic-v2-moe).
    Gate landed on `main` (`5abc4de`), broadcast path behind
    `CRISPEMBED_MOE_NO_REPEAT=1` (default keeps the repeat).
    **Correctness VALIDATED (2026-07-16):** default vs `CRISPEMBED_MOE_NO_REPEAT=1`
    on `nomic-embed-text-v2-moe` is **BYTE-IDENTICAL (max_abs_diff=0.0, cos=1.0)**
    at BOTH f16 and q4_k (50-token input) — the broadcast is exactly the repeat, so
    HF cosine is unchanged by construction. **Latency INCONCLUSIVE / neutral:** a
    7-run bench A/B (graph-compute, T=50, Metal) gave repeat median ~188 ms vs
    norepeat ~195 ms but with ±100 ms run-to-run swings at load ~9 — the
    distributions fully overlap, so no reliable delta (matches the "may be
    perf-neutral" expectation; the repeat materializes only ~1.8 MB total). **Flip
    decision deferred:** per the dev-guide rule (flip only when it wins on speed AND
    quality), a clean flip needs a genuine quiet box (load <3) for a back-to-back
    median; until then keep opt-in — correctness is no longer the blocker, only a
    trustworthy latency number is.
#### HEADLINE remaining lever — GPU recognizer AR decode (scoped 2026-07-20)

PERFORMANCE.md calls the per-region CPU-bound token loop "the real speed path".
**This is NOT greenfield:** `internvl2_ocr` (maturity rank 1) already ships the
target pattern — `ggml_flash_attn_ext` LLM decode + **F16 KV in ggml tensors
(zero-copy view + `ggml_cpy` writes)** + prefill/decode separation + sched GPU
dispatch; `glm_ocr` (rank 2) and `got_ocr` (rank 3) confirm it. The project =
**propagate that proven pattern to the laggard engines**, ranked by leverage
(PERFORMANCE.md "Optimization maturity ranking" + "Opportunities"). Beyond the
KV swap, the top layer is a **persistent single-step decode graph** (build once,
`gallocr` once at max KV, dispatch sched-free per step, **re-set ALL inputs each
compute** — the moonshine/OMR pattern; already proved here: smt-fp 18×, transcoda
2.4–4×, byte-identical).

**✅ CODE-VERIFIED 2026-07-20 — ALREADY DONE; PERFORMANCE.md's maturity table is
STALE.** Auditing the actual LLM-decode path of every engine (not the table): they
**all default to a ggml F16-KV GPU decode**, with the `core_vlm` CPU-scalar path
kept only as a gated fallback. So this "headline project" is closed. Evidence:
- **`qwen2vl_ocr`** — F16 GPU KV (`GGML_TYPE_F16` on `ctx.backend`) + `ggml_flash_attn_ext`
  + `build_decode_step_graph` (0 `core_vlm`).
- **`smoldocling_ocr`** — `sd_run_llm_body` ggml graph handles decode T=1 with F16
  backend KV; `use_ggml = (llm_sched && sd_alloc_kv_cache())` is the DEFAULT,
  `sd_llm_decode_step` (`core_vlm`) is the fallback.
- **`granite_vision_ocr`** — `gv_run_llm_body` ggml + F16 backend KV is DEFAULT
  (`if (!getenv("CRISPEMBED_GRANITE_LLM_SCALAR")) use_graph = gv_alloc_kv_cache()`),
  diff-validated cos 0.9999 vs granite-llm-ref; `core_vlm` is the opt-out. The old
  "10–50× / entire LLM CPU-scalar" is stale.
- **`pix2struct`** — KV cache + DequantCache (Phase 2/3), not "no KV, O(T²)".
- **`deepseek_ocr2`** — ggml per-layer graphs + flash + `alloc_kv` (not `core_vlm`).
- **`internvl2`/`glm`/`got`/`lightonocr`** — the reference implementations.

**Only genuine sliver left (micro, not the headline):** `deepseek_ocr2` builds
per-layer graphs (≈12 builds/token) rather than one multi-layer graph — a graph-shape
tidy, F16 KV already present. And the *persistent single-step graph* (build once,
reuse) is only in qwen2vl/lightonocr/deepseek; the others rebuild the step graph each
token but already on-GPU. Both are marginal vs the closed headline. **PERFORMANCE.md's
"Optimization maturity ranking" + "Opportunities" tables need a refresh to match.**

**Tier 2 — polish:** `lightonocr` GPU dispatch (has persistent F16 KV, GPU=No);
`internvl2` native GQA in flash (skip `ggml_repeat`).

**Landmines (non-negotiable):**
- **CUDA contiguity (LEARNING 35):** `ggml_get_rows` needs a contiguous index
  (`ggml_cont` before it). "Correct on CPU AND Metal" is NOT sufficient — CUDA has
  stricter per-op asserts; the decoded-roundtrip MUST run on a real CUDA box
  (Kaggle P100) before flipping any GPU default. [[flashattn-ext-already-permutes]]
- **Metal `set_output` snapshots LIE** on the sched — bisect on the genuine
  truncated output (`..._MAX_LAYERS=N`), not per-intermediate snapshots.
  [[set-output-on-view-stale]]
- **Metal `mul_mm` F16 overflow** (large ×N activations) → scale 1/256 pre-matmul,
  ×256 post. [[metal-mul-mm-f16-overflow]]
- **CPU-pinned decode re-copies GPU weights every token** — `load_weights_split`
  (encoder→GPU, decoder→CPU) to kill cross-backend traffic; **per-step GPU dispatch
  is launch-bound for tiny models** — the persistent graph is the win, not
  sched-free per-step. Measure the CPU baseline on the right BLAS first (parakeet
  lesson: a "GPU idle" gap was half a CPU-BLAS artifact).
- ggml scheduler: run side graphs before alloc; never reset between alloc and
  compute on the same graph.

**Validation gates (per change; env-gate every path, NEVER delete the scalar one):**
1. Per-stage `crispembed_diff` structural parity (cos ≥ 0.999).
2. **Decoded-output roundtrip is the ONLY acceptance test** — OCR a real doc, read
   the text. Test BOTH f16 AND q4_k.
3. A/B back-to-back under IDENTICAL load on a quiet box (loaded timing lies ±20%);
   final GPU-default flip gated on a Kaggle CUDA decoded-roundtrip.
4. Add a regression entry with `expected_text`; keep `<ENGINE>_CPU_DECODE=1` fallback.

**Sequencing:** ~3–5 focused sessions, each needs a quiet box + one Kaggle run.
S1 `smoldocling_ocr` (core_vlm→ggml LLM decode) → S2 `granite_vision_ocr` (10–50×, instrument-first)
→ S3 `deepseek_ocr2` (single graph + F16 KV). qwen2vl/pix2struct already done; the
the pattern first.

- **SR/restoration — fused ggml graphs: COMPLETE (2026-07-13).** Every engine
  now runs a fused ggml graph, not per-conv mini-graphs. Ported this session:
  - **SAFMN** (`8594cee`): whole forward = ONE fused graph (erf-GELU) — **2.2×
    faster AND more accurate (cos 1.000000 vs 0.994)**. Tiny/overhead-bound, so
    fusion is a big win; Metal is a net loss here (default CPU, `SAFMN_SR_METAL`).
  - **NAFNet** (`14a8393`) + **InstructIR** (`e1eb1dc`): fused per-block graph,
    cos ≥ 0.999998, output identical to legacy. NAFNet-family = **compute-bound**,
    so fusion is perf-NEUTRAL (cleaner, not faster). NAFNet defaults to Metal
    (modest ~15%; `NAFNET_CPU`); InstructIR is CPU-only (GPU conv_2d hits a Metal
    f32×f16 mul_mv pipeline issue). Gates: `NAFNET_LEGACY` / `INSTRUCTIR_LEGACY`.
  - **Restormer**: was ALREADY fused — `rst_transformer_block_ggml` (MDTA + GDFN
    in one graph) is the default; `RESTORMER_SCALAR` is the fallback (cos 0.999997
    both). Only the stale "CPU-scalar" header was corrected.
  - **scunet, swinir, tbsrn, hat, adair, dat**: already build a single graph
    (`forward_expand=1`, no per-conv helpers) — verified sensible (swinir 0.9984,
    dat 0.99999, hat 0.89 q8_0). No work needed; the "CPU-scalar" labels were loose.
  **Key finding:** the fusion win depends on overhead-bound (tiny SAFMN → 2.2×)
  vs compute-bound (NAFNet/InstructIR → perf-neutral). Metal helps only where
  per-dispatch overhead is small relative to compute. Env gates per engine.
- **SR-on-GPU — conv weight residency (research, deferred).** The entire SR
  family computes convs on a CPU-only `enc_sched` with CPU-resident F32 kernels;
  there is no GPU sibling to match. Real SR-on-GPU needs Metal `ggml_conv_2d` for
  these shapes + a GPU-resident weight/graph path the family currently avoids —
  research, not a residency toggle. Reprioritized down.
- **Decode-step graph cache — remaining decoders.** Shipped (sched-free gallocr,
  reserved once at max KV, byte-identical, per-engine env gate) for got_ocr,
  internvl2, glm_ocr, lightonocr, math_ocr. **Still open, each needs the
  single-backend decode check first:** `smoldocling` (CPU LM head outside the
  graph), `granite` (shares the vision sched), `deepseek_ocr2` (per-layer-per-step
  → needs the persistent-graph variant). Modest win (~3% light decoders, ~0% heavy;
  real value is load-insensitivity). `qwen2vl` does NOT fit (multi-backend decode).
- **ggml-metal ICB replay / op-count reduction (the real Metal decode lever).**
  Warm Metal decode is ~82% GPU-execute (per-kernel launch across ~355 sequential
  ops), so ICB (which only collapses the ~18% host-encode) caps at ~18% and is
  NOT justified for CrispEmbed's light decoders. The tractable in-tree lever is
  **fewer, bigger ops per step** — fuse per-layer norm/scale/bias chains, QKV,
  the GLU elementwise chain, prefer `ggml_soft_max_ext`. Per-decoder graph surgery
  in each `build_decoder_step_graph`; verify output cos ≈ 1.0 + node-count +
  latency per model. Re-measure heavy decoders with `CRISPASR_METAL_PROFILE=1`
  before any ICB work. **Caveat (measured 2026-07-13):** the math_ocr ~30%
  cont-removal does NOT generalize to decoder-only VLM engines — got_ocr's cached
  decode already feeds K/V as cache views, so only Q's cont was removable
  (byte-identical, but latency within noise; `5011848`, `GOT_OCR_ATTN_CONT=1`).
  **Op-fusion measured marginal too (2026-07-13):** (a) Metal already auto-fuses
  (`use fusion=true`; `kernel_norm_mul_add`, `kernel_bin_fuse` kernels handle the
  norm/scale/bias + GLU elementwise chains at dispatch), so graph-level elementwise
  fusion is redundant there; (b) attention is already flash-fused; (c) these
  decode steps are compute-bound (got_ocr ~89% GPU-execute), capping any dispatch
  reduction at the ~11% host slice; (d) the trocr decoder is already lean (319
  nodes, 55 ms/16 tok — the ViT *encoder* at 212 ms is trocr's real cost, not the
  decoder). The only non-auto-fusable win is **QKV concat-matmul** (3→1), but a
  probe (`GOT_OCR_QKV_FUSE`, 2026-07-13) confirmed it's not worth it: `ggml_concat`
  **mishandles q4_k** (garbage output) and re-concatenating per step is 3× slower,
  so a correct fusion needs manual load-time q4_k row-block byte-stacking — and on
  a memory-bound T=1 decode that only saves ~2 matmul launches/layer (~4%).
  Deferred; see HISTORY.
  (DeepSeek-OCR-2's MoE-compute lever is detailed in its own subsection above.)
- **unlimited_ocr — remaining deferred items.** `UOCR_PD=1` persistent T=1 decode
  graph (blocked on a small flash-attn padded-vs-exact-KV numerical drift that
  changes argmax by ~step 3; ~14% decode win if solved); `UOCR_OPT_GGML_WINDOW=1`
  (SAM window partition in-graph, ~2–5%, deferred); SAM flash-attn (won't — the
  decomposed RPE bias defeats the O(T) benefit).
- **text_sr — blocked on a public checkpoint** (NAFNet text-SR; registry URL
  empty, no shipped GGUF). Conv paths are guarded transitively by the `nafnet`
  entry; PixelShuffle/bicubic tail unguarded. To train one on clean (Apache/MIT)
  data see `docs/text_sr_training_data.md`.
- **esrgan tile-loop parallelism (concurrency project, deferred).** Intra-op
  threading measured SLOWER (tiled convs don't thread-scale). The real lever is
  running whole 128px tiles concurrently → needs per-thread backend+sched
  replication (the tile loop shares one `ctx->enc_sched`). Verify on a quiet box.
- **TrOCR recognizer accuracy/speed.** eos/length-penalty parity is still TODO
  (the trigram-repeat bug is fixed). The bigger levers: swap DBNet-ic15
  (scene-text) for a document-text detector on dense pages; steer document OCR to
  the doc-VLMs (PaddleOCR-VL / SmolDocling); GPU (WebGPU/Metal) recognizer decode
  is the real speed path (the per-region AR token loop is CPU-bound).

### Open correctness / infrastructure

- **CUDA regression — the 4 FAILs are RESOLVED / explained (P100-verified 2026-07-13).**
  A diagnostic kernel (`tools/kaggle/crispembed-cuda-diag`, Tesla P100 / Pascal
  sm_60) diagnosed each under its env gates, then a 2nd run verified the fix:
  - **`layout-heron` — FIXED (`49cb38a`).** The flash→manual attention fallback
    removed the `fattn.cu:602` abort; P100 CUDA now runs `test-layout-diff` to
    **8/8 stages PASS, DIFF PASSED** (dec_0_cross_out 0.977). ✅
  - **`glm-ocr` + `internvl2` — FIXED (`7998f3c`): it was a stdout banner, NOT
    vision garbage.** Both engines printed their load banner (`glm_ocr: loading…
    Vision:… LLM:… KV cache… Ready`) via `printf` → **stdout**, and `run_one`'s
    `--ocr` text-match captures stdout — so `actual` = the banner (cer 4.3/5.4,
    mis-read as "Class-B CUDA vision garbage"). The P100 diagnostic proved both
    OCR the fox **correctly** on CUDA *and* CPU; only the harness saw the banner.
    Routed all banners to stderr to match the passing engines (qwen2vl_ocr, …). ✅
  - **`granite-vision` — text OCR PASSES**; the projector diff drift is
    cross-toolchain FP strictness (identical CUDA=CPU=scalar on P100), threshold
    already 0.95. ✅
  - **Bottom line: NONE of the 4 were real CUDA vision divergences.** It was one
    genuine CUDA bug (layout flash-abort on Pascal) + a stdout-banner harness bug
    (glm/internvl2) + cross-toolchain FP threshold strictness (granite). The
    diagnostic-first approach (test on the box via env gates) was essential — a
    blind "fix the Class-B vision divergence" would have chased a non-existent bug.
  - **RESULT: portfolio 14 → 0 FAIL** across the fix waves (harness `be6ec54`;
    parser `2af57b1`; layout flash→manual `49cb38a`; banner→stderr `7998f3c`;
    parser value-dump/nameless `c26abc4`; layout perm-tolerant `debug/layout-cross`).
    glm-ocr, internvl2, granite all PASS on P100 now. **All original FAILs fixed** —
    every "Class-B" one was a harness/output bug, not CUDA vision divergence.
  - **The last FAIL (`layout-heron` `dec_0_cross_out`) — ROOT-CAUSED + FIXED
    (`debug/layout-cross`).** NOT flaky and NOT an inference bug. The apparent
    "non-determinism" (0.977 v2 vs −0.034 v14 on P100; −0.08/−0.19 on Metal
    manual/flash) is a **query-permutation comparison artifact**. The 300 decoder
    queries are chosen by `partial_sort` over ~8400 near-tie encoder proposals
    (`layout_detect.cpp:1318`); a tiny backend FP delta in enc_output (Metal/CUDA
    vs the CPU/Python reference — max_abs 0.02, cos 0.99999) reshuffles near-tie
    ranks, so "query i" in our output is a *different physical proposal* than the
    reference's "query i". Instrumented proof: the initial queries themselves show
    per-query cos mean 0.78 / 111 below 0.9 (matching cross_out's mean 0.79), the
    top-5 ranks agree, and the cross_out **values are correct** (best-cosine
    matching each ref query → cos_mean 0.999, 299/300 unique = clean bijection).
    Final boxes are unaffected (score-sort + NMS). **Fix:** `test_layout_diff.cpp`
    compares this stage permutation-tolerantly (`perm_tolerant_cos`); now PASS on
    Metal (0.947/0.999), Metal+flash (0.947), CPU (0.967/0.999). Guardrail keeps
    full power — simulated scrambles (feature-shuffle/sign-flip/roll) collapse to
    ≤0.08 vs the 0.85 gate, and s3..enc_output still guard the encoder-scramble
    class strictly at 0.99. Manifest threshold 0.97→0.85 + comment corrected (the
    old "backend-independent" note was wrong).

  Original diagnostic detail (the run that overturned 3 of the 4 assumptions):
  - **`layout-heron` — REAL CUDA bug (fixable).** `test-layout-diff` aborts:
    `ggml/src/ggml-cuda/fattn.cu:602 fatal error` in `ggml_cuda_flash_attn_ext`
    → `GGML_ABORT` because Pascal (sm_60) has **no flash-attention kernel**
    (`get_best_fattn_kernel == BEST_FATTN_KERNEL_NONE`). With
    `LAYOUT_DETECT_FORCE_CPU=1` **all 8 stages PASS (cos 1.0)** — so the graph is
    correct; the engine just runs `flash_attn_ext` on a single CUDA backend that
    bypasses the scheduler's `supports_op` CPU-fallback. **Fix:** don't use the
    CUDA flash kernel where it's unsupported — either (a) route layout attention
    through a scheduler that honours `ggml_cuda_flash_attn_ext_supported` (returns
    false on sm_60 → runs on CPU), or (b) give `layout_detect` a manual masked
    attention fallback (`mul_mat`+`soft_max_ext`+`mul_mat`, mask=nullptr = full
    attn) selected when flash is unsupported. Verify: `test-layout-diff` PASSES on
    P100. NOTE T4 (Turing sm_75) HAS flash — this only bites Pascal.
  - **`granite-vision` — NOT a CUDA bug.** The projector stages fail **identically
    on CUDA, `GRANITE_VIS_SCALAR`, AND full-CPU (`GRANITE_CPU`)** on the P100 box
    (cos 0.952 / 0.958 / 0.955 — same to 2 dp across all three), while they PASS on
    the Mac. So it is a **cross-toolchain FP-strictness gap** (Kaggle gcc vs Mac
    clang on high-magnitude projector activations, max_abs ~2.7–4.3), NOT a CUDA
    divergence, and the **OCR text passes** (cer 0.163). **Fix:** relax the
    projector-stage diff thresholds (≈0.95, they gate a real crater by going
    negative) — a parity-harness strictness fix, not a model change.
  - **`glm-ocr` — NOT a CUDA bug.** `test-glm-ocr-diff` vis_layers 14–23 fail at
    cos 0.96–0.98 **identically on CUDA and CPU** on P100 (vis_layer_23: CUDA
    0.9630 vs CPU 0.9632; max_abs up to 217) — same cross-toolchain strictness as
    granite. And on a clean generated fox image glm reads it **correctly on CUDA**
    (`"The quick brown fox jumps over the lazy dog 12345"`). So glm's vision is not
    CUDA-garbage. Its portfolio FAIL is the **text-match on the repo `fox.png`
    (800×200)** specifically — untested CPU-vs-CUDA yet (see below).
  - **`internvl2` — reads a generated fox CORRECTLY on P100 CUDA** (identical to
    CPU). No ref uploaded, so no per-stage diff. Its portfolio FAIL is likewise the
    text-match on the repo `fox.png` (800×200), not universal vision garbage.
  - **Open sub-question (glm + internvl2 portfolio garbage):** the repo `fox.png`
    is 800×200 (the diagnostic used a 640×96 render). Next diagnostic run must OCR
    the **repo** `tests/regression/images/fox.png` under default vs `*_FORCE_CPU`
    for both engines — if CPU is also garbage there, it's a Kaggle-BUILD issue
    (like granite/glm diff), not CUDA; if only CUDA is garbage, it's a genuine
    larger-image CUDA vision divergence to localize. The vis-diff being CPU=CUDA
    identical strongly suggests the former.
  - **Full data:** the diagnostic log is on Kaggle
    (`chr1s4/crispembed-cuda-diagnostic-4-remaining-fails`, transcript in
    `/kaggle/working/diag.log`); see HISTORY.md.
- **DBNet detector — mostly resolved (2026-07-13).** The CPY abort was already
  fixed (`dequant_rows_f32` via get_rows); the real cost was the CPU postprocess
  (43 s → 1.5 s, scanline box scoring `74b8ac5`, see HISTORY). Detection graph
  compute is only ~3 s on CPU and Metal `conv_transpose_2d` is still ~13× slower,
  so **CPU stays the correct default** — a faster Metal `conv_transpose_2d` (or a
  1/4-res prob-map + cheap upscale) is the only remaining, low-value, upstream
  lever for GPU-default detection.
- **bidirlm-omni GGUF re-quant follow-up.** The text-tower converter bug is fixed
  and `bidirlm-omni-2.5b-q8_0.gguf` re-uploaded (text cos 1.0 f16 / 0.9992 q8_0),
  but the repo's f16 + imatrix q4_k/q5_k/q6_k and the whole `-textonly` repo are
  still the OLD (text-broken) conversion — regenerate them from the fresh f16
  (imatrix variants via the imatrix pipeline). Kaggle-only (16 GB Mac OOMs).
- **Regression-guardrail residuals.** `bert_ner` dumper written but its ref is
  download-blocked; face *recognition* (arcface/sface) unguarded (no local rec
  GGUF; detection is guarded). All SR/restoration (11) + esrgan/safmn + lilt +
  lfm2 + the closed engines are auto-guarded in `tests/regression/manifest.json`.
- **`core/vlm_decoder.h` — deferred.** A unified scalar decode loop; only 2 scalar
  engines remain, so abstracting is premature. Revisit if a 3rd appears.

- **Tesseract seeded-artifact regeneration (2026-08-01).** The 12 installed
  canonical `.traineddata` sources were SHA-256 matched to the existing GGUFs.
  Fresh Miniconda F32/F16 conversions and metadata-repaired Q8_0/Q4_K
  companions are now in `/Volumes/backups/ai/crispembed-gguf/` as `*-seeded.gguf`.
  Forty-two companions are readable and carry nonzero `sample_iteration`; old
  files remain available for rollback. The old Fraktur `mixed-lstm0ih` candidate
  is truncated and was excluded. Per-language diff and decoded-output gates are
  still required before promotion to canonical names.

- **Tesseract int-mode kernel optimization — COMPLETED.** Cache each LSTM
  gate's int8 weights, bias quantization, and scale at model load, then
  quantize each input/hidden vector once per timestep. The prior implementation
  recomputed scales and rounded every weight inside every gate dot product.
  Benchmark and decoded-output parity must pass before promotion.

- **Tesseract int-mode kernel optimization (2026-08-02).** Packed each cached
  gate row as `[W_ih | W_hh]` and each timestep activation as `[input | hidden]`,
  reducing the hot path to one contiguous int8 accumulation per gate while
  preserving the existing quantization, LUT, and fallback semantics. On the
  seeded English Q8 model and `scan_strip.png`, three CLI runs decoded `S` in
  both modes: cached LSTM `24.2/16.3/18.9 ms`, uncached `896.3/443.5/233.7
  ms`. The output contract is unchanged; the packed cache remains the default
  and `CRISPEMBED_TESSERACT_DISABLE_INT_CACHE=1` remains the diagnostic
  fallback.

- **Tesseract LSTM scratch reuse (2026-08-02).** Added a per-context,
  environment-gated `lstm_scratch` arena for the hidden/cell/gate and int8
  activation vectors used by SummLSTM and the recurrent layers. The default
  allocation path is unchanged; `CRISPEMBED_TESSERACT_REUSE_SCRATCH=1` reuses
  buffers only within one recognizer context. On a dimension-matched seeded
  English fixture, reuse and fresh allocation both decoded `Etaansen `, so the
  output contract remains exact. The path is covered by the runtime contract
  test and remains opt-in until a repeated page benchmark demonstrates a
  material allocation reduction.

- **Tesseract composed-recoder output — IN PROGRESS.** The Chinese seeded F32
  reference passes all tensor stages but exposed native dropping of unmapped
  multi-code recoder classes (`native=''`, Python=`<141>`). The native
  diagnostic fallback now preserves `<class>`; full recode-beam composition
  and dictionary scoring remain a production-quality TODO.

- **Tesseract HF artifact publication (2026-08-01).** All 51 corrected
  canonical F32/F16/Q8_0/Q4_K files are uploaded to the intended
  `cstr/tesseract-lstm-GGUF` and `cstr/tesseract-frk-GGUF` repositories. Remote
  metadata spot-checks confirm nonzero `sample_iteration`; no `mlx-community`
  repository was used.

- **Tesseract seeded F32 sweep (2026-08-01).** Corrected references and native
  diffs now pass decoded parity for all 12 canonical languages on the
  controlled line. German's former 3/150 mismatch and Spanish's one-blank
  mismatch were stale-reference effects from NumPy float32 LUT generation;
  Korean's former 6/200 mismatch was fixed by aligning the production native
  LUT with upstream generated double-precision entries cast to `TFloat`. All
  12 runs report 9/9 stages and exit 0.

  The cache now has an explicit `CRISPEMBED_TESSERACT_DISABLE_INT_CACHE`
  fallback for controlled parity comparisons; cached mode remains the default.
  Controlled scan-strip validation passed: cached and uncached decoded text
  were both `SEEEES`; LSTM time was `35.4 ms` cached versus `1,035.6 ms`
  uncached (`29.3x` faster). Promote the cache as the default and retain the
  disable gate for future architecture-specific diagnostics.
  Full-page scan-strip validation also passed: both modes produced 12 regions,
  566 chars, CER `0.03375`, and WER `0.15044`; cached stage time was `22.11 s`
  versus `157.59 s` uncached (`7.1x` faster). Detection and crop together
  remained below `50 ms`, so recurrent recognition is the active bottleneck.

  The page comparator now preserves normalized official and native decoded text
  in its JSON output. On `scan_strip.png`, the remaining errors are concrete
  crop/decode differences (`50`→`80`, `ay`→`8ay`, `Such`/`such`, `Scheme`/
  `scheme`, and punctuation/hyphen spacing), not a region-count mismatch.
  Use these actual strings to guide crop and decoder fixes rather than treating
  CER/WER alone as a sufficient quality signal.

- **Tesseract crop-border A/B (2026-08-01).** `CRISPEMBED_TESSERACT_CROP_PAD`
  is now an opt-in gate around the Fraktur line-crop border; the production
  default remains 2 pixels. On `scan_strip.png`, 0/1/2/4 pixels produced
  12/12/12/12 regions and CER/WER `0.07460/0.30088`, `0.04796/0.20354`,
  `0.03375/0.15044`, and `0.03552/0.15044`, respectively. Keep 2 pixels as
  the default. Recognition output remains worse than official Tesseract in
  substitutions and punctuation despite matching region count; decoder,
  recoder, and line-image parity remain the active quality TODOs.

- **Tesseract page comparator hardening (2026-08-01).** The page CER/WER
  harness now supports explicit `--tessdata-dir`, clears stale inherited
  `TESSDATA_PREFIX`, and decodes subprocess diagnostics with replacement for
  invalid bytes. An explicit-tessdata scan-strip rerun remains stable at
  official 12 lines/113 words/451 chars versus native 12 regions/566 chars,
  CER `0.03375`, WER `0.15044`; no tessdata warning remains. It now also
  supports `--require-text-match`, and stores normalized official/native text
  in the comparison object so approximate CER/WER gates cannot be mistaken
  for exact page-output parity.

- **Tesseract confidence harness hardening (2026-08-01).** The line-confidence
  comparator now tolerates non-UTF-8 Tesseract diagnostics and removes stale
  inherited `TESSDATA_PREFIX` when an explicit tessdata directory is supplied.
  A valid cropped Fraktur line run produced official/native text differing only
  by spacing (`1 hey` vs `1hey`), but official mean word confidence was
  `0.7060` versus native greedy word confidence `0.9726`; beam sequence
  confidence was `0.9924` with zero fabricated character confidences. Keep the
  confidence calibration gate open until line-vs-word aggregation is compared
  against the official certainty contract.

- **Tesseract page-segmentation and beam A/B (2026-08-01).** Projection
  improved the scan-strip comparison to CER/WER `0.03197/0.12389` from the
  legacy `0.03375/0.15044`, with the same 12 regions; baseline matching was
  unchanged and slower. Keep projection behind
  `CRISPEMBED_TESSERACT_PAGESEG_PROJECTION` because the improvement is not yet
  official-output parity. Beam width 8 on projection was text-identical to
  greedy (`0.03197/0.12389`) but increased recognition from `9.66 s` to
  `29.75 s`; keep it diagnostic-only. Next TODO: line-image/crop geometry and
  Tesseract-compatible decoder/recoder semantics.

- **Tesseract page-box geometry A/B (2026-08-01).** Added gated
  `CRISPEMBED_TESSERACT_PAGESEG_BOX_PAD` with the existing 3 px expansion as
  the default. Legacy-page tests at 1/2/3 px all emitted 12 regions, 566 chars,
  and identical CER/WER `0.03375/0.15044`; 1 px and 2 px were also slower than
  the default in the measured runs. Keep the control for alternate scan
  resolutions, but the current fixture points away from box expansion and
  toward line-image preprocessing/decoder semantics.

- **Seeded page-gate rerun correction (2026-08-01).** The earlier report of
  only 2 boxes/lines was invalid evidence: `test-ocr-orchestrator` was stale
  after the remote pageseg changes. After rebuilding the actual target, with
  proof `[76/76] Linking CXX executable test-ocr-orchestrator`, the canonical
  Q8 DBNet IC15 detector plus both corrected Fraktur seeded recognizers emitted
  the established 12 boxes/lines. The pipeline gate passed in both runs, but
  the exact-text gate remains non-green. F32 measured CER/WER
  `0.03922/0.13274`, 12,373 ms total, and confidence delta `0.01647`; Q8
  measured the same CER/WER, 14,560 ms total, and confidence delta `0.01447`.
  Native text still differs in punctuation, spacing, and several glyphs, so
  the remaining TODO is recognizer line-image/decoder parity, not detector
  geometry or a precision-only failure. The rejected stale-binary result must
  not be used to diagnose detector compatibility.

- **Tesseract line-input diagnostic (2026-08-01).** Added the opt-in
  `CRISPEMBED_TESSERACT_CROP_DUMP_DIR` hook to dump the exact grayscale crops
  passed to the native LSTM recognizer. On `scan_strip.png`, the rebuilt Q8
  run produced 12 crops (heights 22–32 px; final crop 76×25 px), matching the
  valid 12-line geometry. This rules out the stale 2-box observation and makes
  Tesseract's internal page-segmentation/line normalization the next parity
  comparison target; do not attribute the text gap to GGUF precision yet.
  The hook now also writes `crops.tsv` with source boxes and pixel ranges;
  the verified Q8 run wrote 12 crop records plus a header and exposed the
  first line as an edge-clipped box at `y=0`. Compare this manifest against
  official TSV line geometry before changing recognizer preprocessing.

- **Tesseract crop ink-trim A/B (2026-08-01).** The opt-in
  `CRISPEMBED_TESSERACT_CROP_TRIM_INK` experiment trims vertical paper around
  dark pixels while preserving native box geometry. On the canonical Q8
  scan-strip run it reduced native recognition from `11,351.6 ms` to
  `10,407.1 ms`, but worsened CER/WER from `0.03922/0.13274` to
  `0.04278/0.14159` and increased the character delta from 116 to 121.
  Reject this preprocessing for production; the remaining mismatch is not
  explained by excess vertical paper alone.

- **Tesseract page box-pad A/B (2026-08-01).** Setting the legacy component
  `CRISPEMBED_TESSERACT_PAGESEG_BOX_PAD=0` retained 12/12 regions and produced
  byte-identical native text and unchanged CER/WER `0.03922/0.13274` versus
  the default pad. Reject box expansion as the explanation for the token
  mismatch; the next target is row-boundary construction and baseline
  assignment, not another crop-padding variant.

- **Tesseract component-row A/B (2026-08-01).** The existing opt-in
  `--component`/`CRISPEMBED_TESSERACT_COMPONENT_PAGESEG` policy retained 12
  regions but degraded scan-strip quality to CER/WER `0.10873/0.20354` and
  corrupted the first line (`40eNArEBOg 10DE EES EEN`). Reject it; the legacy
  row-clustering path remains the active baseline. A separate alternate run
  with a malformed model path emitted no metrics and was discarded as invalid
  evidence.

- **Tesseract crop-geometry comparator (2026-08-01).** Added
  `tools/compare_tesseract_crop_geometry.py`, which compares `crops.tsv`
  against official Tesseract TSV level-4 rows. On the current 12-line
  scan-strip fixture, counts match, but native boxes average `dx=-2.08`,
  `dy=+1.83`, `dw=+4.33`, `dh=+1.50`; the largest deltas are row 5 width
  `+80`, row 10 vertical offset `+14`, and row 5 height `+12`. This identifies
  local row construction/splitting errors, not a global crop-padding issue.

- **Tesseract row-blob-bounds A/B (2026-08-01).** Direct debug showed row 5
  assigned blobs at `x=29..343` but the vertical ink scan widened its crop to
  `x=27..428` using neighboring-row pixels. The gated
  `CRISPEMBED_TESSERACT_PAGESEG_ROW_BLOB_BOUNDS` mode prevents that expansion.
  On scan-strip it improved CER/WER from `0.03922/0.13274` to
  `0.03209/0.11504`; geometry mean `dw` improved from `+4.33` to `+2.42`,
  worst width delta from `+80` to `+13`, while counts stayed 12/12. Keep it
  gated pending additional page fixtures; exact text parity is still open.

- **Tesseract composed-recorder (2026-08-01).** Added opt-in
  `CRISPEMBED_TESSERACT_RECODE_COMPOSE`, which segments collapsed CTC output
  classes against the serialized multi-code recoder and emits complete
  unichar tokens. The existing single-code/fallback decoder remains the
  default. Fraktur default and opt-in outputs are byte-identical on the
  controlled line; a Chinese smoke run passes both modes without crashes, but
  did not emit a multi-code class, so full composed-recoded quality parity and
  dictionary/DAWG scoring remain open.

- **Per-line page comparator correction (2026-08-01).** Official TSV words
  are now grouped by page/block/paragraph/line; the previous diagnostic
  accidentally included `word_num` and compared 113 words as 113 lines. The
  corrected row-bounds run still emits 12 native and 12 official lines, but
  only 3/12 are exact. The first divergence is line 0 (`<< 4 ...` official
  versus `“< A ...` native); lines 4, 7, and 9 match exactly. Page CER/WER
  remains `0.03209/0.11504`. The active TODO is line crop/normalization or
  decoder semantics; no recognizer math change is justified before a
  tensor-level diff of the corresponding crop.

- **Line-0 crop tensor diff (2026-08-01).** Dumped native crop 0 and created
  a fresh Python reference from `/opt/homebrew/share/tessdata/frk.traineddata`.
  `test-tesseract-lstm-diff` passed input, convolution, conv-FC, maxpool, all
  four LSTM stages, and logits; the minimum cosine was `0.997755`, mine/ref
  norms were `35.8611/35.8704` at the lowest recurrent stage, and both native
  and Python decoded `“< A hey are gomg to be encamped near Brighton ;`.
  The official Homebrew Tesseract CLI cannot reopen local image files in this
  environment (PNG, PGM, and TIFF all fail in Leptonica), but its page TSV
  line differs. This proves the line-0 quality discrepancy is in official
  page segmentation/line normalization, not the native GGUF recognizer.
  The page comparator now has `--crop-dump-dir` for fresh reproducible dumps,
  and `tools/compare_tesseract_crop_diff.py` automates regeneration of a
  per-crop Python reference plus the native tensor diff without invoking the
  broken local-image CLI path.

- **CC0 German page cross-check (2026-08-01).** The same canonical Q8 DBNet
  plus seeded German recognizer was run on
  `tests/regression/images/cc0/german_official_print.jpg`. Official Tesseract
  produced 28 lines/153 words/897 characters; native produced 23 lines/862
  characters, with CER/WER `0.32984/0.67974`. The native benchmark was
  `detect=982.4 ms`, `crop=670.0 ms`, `recognize=19594.7 ms`,
  `total=21247.2 ms`. Because the line counts differ, index-paired line CER
  is not a recognizer-quality measure here: this fixture opens a separate
  detector/line-geometry coverage gate. The comparator now marks such
  per-line alignment as invalid instead of implying a valid line-by-line
  pairing.

- **Explicit native Tesseract-like route (2026-08-01).** Added
  `--native-pageseg` to the page comparator; it sets the classical route and
  reports `detector_route=native-tesseract-pageseg`, rather than silently
  treating every run as DBNet. On `scan_strip.png`, the native route emitted
  12/12 lines with CER/WER `0.03209/0.11504`, 3/12 exact lines, and stage
  timing `detect=12.6 ms`, `crop=644.8 ms`, `recognize=11856.4 ms`,
  `total=12513.8 ms`. This matches the prior classical result and confirms
  that DBNet is not involved in this route; official-output parity remains
  open at page segmentation/line normalization and decoder semantics.

- **Native-route benchmark propagation (2026-08-01).** Extended
  `tools/benchmark_tesseract_page.py` with `--native-pageseg` and explicit
  `detector_route` output, so repeated A/B runs cannot silently fall back to
  DBNet. The wrapper and route-selection behavior are covered by the focused
  Miniconda test suite.

- **Native route on CC0 German page (2026-08-01).** The explicit
  `--native-pageseg` route on `german_official_print.jpg` also emitted 23
  lines/862 characters, versus official Tesseract's 28 lines/897 characters;
  CER/WER remained `0.32984/0.67974`. Native stage timing was
  `detect=1014.9 ms`, `crop=605.7 ms`, `recognize=14263.4 ms`,
  `total=15885.6 ms`. Thus both our DBNet and native row routes currently
  share the same five-line coverage gap on this fixture; neither result is a
  valid recognizer-quality comparison until line geometry is aligned.

- **German crop geometry guard (2026-08-01).** The native-route crop manifest
  has 23 rows while official TSV has 28. The old geometry tool's index-paired
  summary therefore produced meaningless deltas (for example mean `dy=257.7`)
  and exited nonzero only because of the count mismatch. It now reports
  `alignment_valid=false` and `paired_rows`, so those deltas cannot be treated
  as geometry measurements until a line-matching strategy handles merges and
  missing rows.

- **Merge-aware German geometry diagnostic (2026-08-01).** Added
  `--match-by-geometry` to the crop comparator. On the native German manifest
  it matched 23 native rows monotonically and identified five unmatched
  official rows (`0,2,3,4,26`) instead of treating merged/missing lines as
  same-index pairs. The tool remains strict (exit 1 while counts differ), and
  its matched deltas are diagnostic only until the row matcher accounts for
  true one-to-many merges.

- **Nested-row classification (2026-08-01).** Source-pixel review shows the
  German groups include official TSV decoration rows nested inside larger
  boxes, not necessarily missing text lines. `merged_official_groups` now
  identifies the largest primary official box and nested official indices;
  do not split a native crop merely because TSV emitted nested marks.

  On the German page, native row 0 has primary official index 1 with nested
  indices 2 and 4; native row 9 has primary index 13 with nested index 12;
  native row 22 has primary index 26 and no fully-contained nested row. The
  remaining unmatched official row 0 is not explained by a nested decoration.

- **One-to-many merge reporting (2026-08-01).** The geometry diagnostic now
  reports native rows whose vertical span covers multiple official rows as
  `merged_official_groups`, separating true row merges from simply missing
  official rows. This is diagnostic only; no production row-splitting default
  is changed until the merged groups are reviewed against the source pixels.

- **German merge candidates (2026-08-01).** On the native CC0 German
  manifest, the diagnostic identifies native row 0 covering official rows
  `1..4`, native row 9 covering `12..13`, and native row 22 covering `26..27`.
  Official row 0 remains unmatched. These are the first concrete source-pixel
  targets for a future row-splitting adapter; no production default changes
  are justified from geometry alone.
