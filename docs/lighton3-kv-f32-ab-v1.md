# LightOn3 CPU cache-precision A/B — first case

The complete [epsilon experiment](lighton3-pixtral-epsilon-ab-v1.md) is a retained
**negative** acceptance result: all12 corrected numerical arms pass and all6
F32 decodes match, but F16 form-transcription changes at token6 (original2918,
native3377), switching its generated description from English to French.
[Complete reviewed evidence](../tests/results/lightonocr3-1b/epsilon-ab-v1/full-review-41889499814.json)
retains24 rows/72 native requests, all old-path bit controls and strict failures.
Neither epsilon default nor model/publication qualification changes.

## Causal question

The pinned original runs float32 with `use_cache=True`. Native stores all K/V
in persistent F16 tensors even for an F32 GGUF. Prefill attention uses freshly
computed F32 values, while later decode reads that half-precision cache. The
form-transcription first-logit maximum error is about7e-5 after the epsilon fix;
step6 reaches0.012965 F32/0.013396 F16. This identifies a precision difference
worth testing; it does not yet prove the cause of the language change.

The checkpoint header records532 BF16 tensors. Merely changing stored weight
precision is not a demonstrated fix: cache and matmul activation precision are
separate. Test actual original/native intermediates before selecting tensors.

## Fixed first-only protocol

[Protocol JSON](lighton3-kv-f32-ab-v1.json) pins native/test/driver hashes,
protected source objects, the old failed producer, source image and both GGUFs.
Only the two native source files differ from0473dd5e8; legacy epsilon and all
other model arithmetic stay fixed. `CRISPEMBED_LIGHTONOCR3_KV_F32=1` selects
F32 persistent K/V only for the admitted new profile on a CPU backend. Default,
empty/0, legacy profiles and GPU backends keep F16. Cache storage doubles at the
same token capacity; this is not a weight-format or release change.

The input is the failed full-epsilon raw artifact13355422898 from run41889499814,
source0473dd5e813fdf48d2bafbd16811271cf6c42599,8831289731 bytes, SHA256
`b57f2b906da12f20fc1ef5462c4800b801a5349ed38eee790b3ecdbf1b057078`.
Its expected run conclusion is failure, with completed arrays. Authenticate all
producer/API/archive fields before selecting members. Only
`references/form-transcription/` and the old corrected-epsilon native paths
`form-transcription/{f32,f16}/gate1/{forced,autonomous,plain}/` are admitted.
No new Python model forward or gold is created.

Run only `form-transcription`, F32/F16 × cache0/1 × forced/observed-free/plain-free:
**12 native request processes**. Hold the corrected epsilon gate at1, CPU threads4,
portable backend and the original128-token cap. Reconstruct exactly the old
F32/F16 GGUF hashes. Source/resource and actual GGML norm/cache stride controls
run before archive/model acquisition. Require12GiB RAM and40GiB free scratch.
Heavy work stays hosted; children do not inherit Actions/Hub/Kaggle tokens.

- Cache0 must replay every retained corrected-epsilon tensor byte, non-timing
  trace field and decoded byte, **including the French F16 negative**.
- Cache1 must preserve all prefill tensor bytes, including step0 logits, versus
  cache0; only later decode logits may change. Both observed and independent
  ordinary generation must match original IDs/text at the unchanged cap/status.
- Compare every original position/full vocabulary, strict .002/.99999 budgets,
  owner final bars .99/.98, absolute/RMS errors and magnitudes. Step6 maximum
  error must decrease for each format. Retain every negative flag.
- Report full-vector hashes, top5 values, top1/top2 margin and the exact
  2918-versus3377 margin at step6. A selected-token probe alone cannot qualify
  the model; it accompanies the complete forced and free comparisons.

Use `heavy-cpu.yml`, selector `tools/ci-heavy/lighton3_kv_f32_ab.py`, empty args/pip,
`ubuntu-latest`, `portable`. The dedicated job has repository/Actions-read only;
review and raw artifacts `lighton3-kv-review-RUN` / `lighton3-kv-raw-RUN` retain90d.
First run [41897136488](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41897136488)
is executing at3e82a1205. Re-query it before any retry.

## Stopping rule and remaining work

This first-only driver accepts no full-phase arguments. If negative, preserve it
and attribute the earliest actual added error before another precision choice.
No budget relaxation, changed gold, prompt/cap fitting or blind original replay.
After independent positive review, declare a separate remaining-panel closure
and authenticated first-proof barrier; do not repeat the first12 native calls.
The other5 image/modes need60 new native processes. Their corrected-epsilon
baseline is in the same full artifact, except form-grounding native controls,
which are in first-epsilon artifact13353235590,1459280989 bytes, SHA256
`3f6087476bdc9d18ad9d58bbfad926a1f78d3f2d2fa4f98e41e0af0804adaba0`.
All original references are in the full raw artifact. Authenticate both sources.

The old epsilon v1 protocol/driver/owned-source test remain immutable historical
controls for their source. Current CI runs its pure replay controls plus the new
KV closure, rather than asserting the old native hash on intentionally changed
source. Actual old-path replay remains a required runtime gate in this experiment.
The M1 preflight at0473dd5e8 stopped on memory pressure before compilation;
its reported hash checks supply source-preflight evidence only. Actual Apple
build/component/style evidence for the current cache source remains outstanding.
Do not run more work on that machine under its current pressure. Defaults,
Q8/Q4, EOS-completed document quality, public consumers, aliases and release
remain separate gates. Development0.17.15 stays unreleased.
