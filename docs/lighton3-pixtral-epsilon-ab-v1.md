# LightOn3 Pixtral epsilon: retained-reference A/B

## Final bounded verdict — completed negative

[Full run41889499814](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41889499814)
completes all24 rows/72 native processes across both phases, zero original
forwards. All old tensor/decoded replay controls and all12 corrected final-logit
bars pass; corrected F32 outputs match6/6, F16 outputs5/6. Form-transcription F16
first differs at token6 (2918→3377), switching its description English→French.
The [full reviewed receipt](../tests/results/lightonocr3-1b/epsilon-ab-v1/full-review-41889499814.json)
retains the negative verdict, every comparison, errors, caps and source identities.
Default remains off. No first-case success overrides this complete-panel result.

Raw artifact13355422898 is8831289731 bytes, SHA256
`b57f2b906da12f20fc1ef5462c4800b801a5349ed38eee790b3ecdbf1b057078`,
retained90d with all six original reference arrays. Small artifact13355715186 is
1117948 bytes, SHA256
`350b309d2390ceea4c6295c7b6318bca21a1040a88d1b9c3db6da2615943d6cb`.
The small ZIP,24 unique rows, first-proof import and all gate0 inventories were
independently checked. Preserve this v1 source0473dd5e8 and original protocol.
A [separate first-only cache experiment](lighton3-kv-f32-ab-v1.md) now tests the
native F16 persistent K/V versus the float32 original at fixed corrected epsilon.
It changes neither weights nor this v1 experiment's gold, budgets or verdict.

## Reviewed first-stage result and current run

[First retry41885702781](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41885702781)
**passes**, at exact sourcef27223fc276674c4621f32b709aa03ef0d035f09.
The [reviewed receipt](../tests/results/lightonocr3-1b/epsilon-ab-v1/first-review-41885702781.json)
retains all four comparisons and exact gate0 tensor inventories:188 forced and188
observed-free tensors per precision, with independent plain decoded controls.
All old tensor bytes/non-timing traces reproduce. Gate1 restores original free
IDs/text at both precisions, including the old changed form box/text.

First-norm maximum error falls7.3313713e-5→2.9355288e-6. Vision-layer23 max falls
4898.416→6.53125 F32 and4901.8125→5.42578125 F16; final minimum cosine improves
0.9999600606→0.9999982037 and0.9999598968→0.9999984228. Strict failures remain155
and165 and the original/native output is still capped at128. These are bounded
arithmetic/decoded improvements, not document-quality or strict all-stage parity.

Small artifact13352053898 is387474 bytes, SHA256
`81e01a2df42d800cc3e7c4a0bb4389a660a516acbefe41dc41f1aac35ada44a9`;
its ZIP digest and receipt source/coverage were independently checked. Raw
artifact13353235590 is1459280989 bytes, SHA256
`3f6087476bdc9d18ad9d58bbfad926a1f78d3f2d2fa4f98e41e0af0804adaba0`.
The remaining phase [41889499814](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41889499814)
completed at0473dd5e8 with the negative final verdict above, bound to that exact
positive first proof. Do not repeat it;
do not repeat the first stage or modify the owned driver/native/protocol closure.
Default remains off.

The first complete [F32/F16 panel](../tests/results/lightonocr3-1b/native-initial-41873656972/review.json)
passes the owner final-logit bars on all 12 arms. Ten decodes match; form grounding
changes coordinates and text at both precisions. Eight arms hit the unchanged
128-token cap. This is numerical port evidence, not document-quality qualification.

The pinned Pixtral implementation
[`modeling_pixtral.py`](https://github.com/huggingface/transformers/blob/3568df44fff52979f1f9cecc7f729713e940d277/src/transformers/models/pixtral/modeling_pixtral.py)
uses epsilon 1e-5 at `ln_pre`, `attention_norm` and `ffn_norm`. Native vision used
1e-6 at all three call sites. In form grounding, pixels match exactly; patch-conv
max error is 9.06e-6, first norm max 7.33e-5, vision layer0 max 0.00615025, vision
layer23 max 4898.416 at reference RMS 2834.331. The first decoded difference is
position9: original ID24 versus native23 (`999` becomes `899` in the first box).
High final cosine alone hid this output defect.

The runtime experiment changes only those vision norms, only for the admitted
LightOn3-1B profile, behind default-off `CRISPEMBED_LIGHTONOCR3_PIXTRAL_EPS=1`.
Literal `0` preserves the old path; legacy LightOn profiles always retain 1e-6.
Projector/text/QK norms, weights, prompts, positions, caps and comparison budgets
are unchanged. The candidate starts 0.17.15 development; no release is cut and
published 0.17.14/application pins remain unchanged.

## Frozen inputs and first stage

[Protocol JSON](lighton3-pixtral-epsilon-ab-v1.json) fixes the original source
`4476be6eb21ecacc3c001ed4e02928d0a98be0ea`, run41873656972, full archive
13347781304 (5,627,376,432 bytes, SHA256
`020a1c21f434702366ca895f1e3b358644f2c1e5dde5948e49e1a9edb33d48e7`),
all source objects, unchanged F32/F16 GGUF hashes and the six case/mode inputs.
The runner authenticates the source archive, validates every ZIP entry, retains
only selected reference/native members and rehashes every extracted file.
Original model forward calls are **zero**.

The first dispatch is [41885090925](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41885090925)
on `fix/ocr-qualification-nov4`, script
`tools/ci-heavy/lighton3_pixtral_eps_ab.py`, arguments `--phase first`, portable
mode, `ubuntu-latest`, no extra pip. It stopped before acquisition/build/inference
because shallow checkout lacked the baseline4476be6eb. Repairf27223fc2 fetches
that exact baseline without weakening the source guard. Retry
[41885702781](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41885702781)
completed successfully as reviewed above. The initial preflight failure is
[retained separately](../tests/results/ocr-benchmark-expansion/lighton3-eps-preflight-41885090925.json).

Source/operator/build controls run before archive/model acquisition. Actual
GGML RMSNorm on a low-variance authored input must match the independent scalar
1e-5 formula. Provider/source/resource controls require 12 GiB available RAM and
40 GiB combined scratch/output on the dedicated hosted runner. No model or build
runs on the shared development host; the job receives only repository Actions-read
authority, with that token stripped from model/build child environments.

First stage admits **form-grounding only**. Two precisions × gate0/gate1 ×
forced/observed-free/plain-free gives 12 native request processes. Original
prefixes are used only for teacher-forced comparisons; both free runs remain
independent. Gate0 must reproduce every retained native tensor byte, non-timing
trace field, ID and decoded byte before the correction is accepted. Gate1 must
improve the earliest norm, pass unchanged .99/.98 final-output bars and match
original free IDs/text, including the unchanged cap status. All strict
.002/.99999 failures and magnitudes remain in the reports.

## Remaining panel opens only after first-stage review

The first small artifact must be a successful owned manual run, with exact ZIP
size/SHA and matching protocol SHA. The runner rejects a missing, stale, partial,
negative or different-protocol first proof. A supplemental workflow barrier
also binds first source to the actual producer, checks exact unique form-grounding
coverage, the12/0 native/original counts, every gate0 inventory and gate1
numerical/decoded flags. Eight synthetic controls cover forged or incomplete
receipts without changing the frozen driver/protocol or repeating native work.
Example after independent review:

```sh
gh workflow run heavy-cpu.yml --ref fix/ocr-qualification-nov4 \
  -f script=tools/ci-heavy/lighton3_pixtral_eps_ab.py \
  -f 'args=--phase full --first-run 41885702781 --first-artifact 13352053898 --first-sha 81e01a2df42d800cc3e7c4a0bb4389a660a516acbefe41dc41f1aac35ada44a9' \
  -f runner=ubuntu-latest -f build_mode=portable -f pip=
```

The remaining phase imports the four first-stage outcomes, preserves their run
identity, and performs 60 new native request processes for the other five
case/mode inputs. Aggregate coverage must be 24 gate/precision comparisons and
72 native requests across both stages. No original inference or first-case
native repetition is authorized by this continuation. Both source gates stay
available and the correction stays default-off until a separate default decision.
If gate0 bytes or gate1 decoded output fail, preserve the failure and diagnose
the earliest actual differing operator; do not loosen a threshold or replace gold.

Small artifacts are `lighton3-eps-review-<run>`; raw artifacts are
`lighton3-eps-raw-<run>`, retained for90 days. New raw artifacts preserve copied
original arrays with their source identities and all new native arrays, excluding
weights, environments, source checkouts and build trees. The initial full archive
expires18 November2026; preserve still-needed uncopied inputs before that deadline.
Final model qualification/public consumers, EOS-completed document protocols,
Q8/Q4 and publication remain separate gates; no alias or application upgrade
follows merely from a successful epsilon experiment.

## Subsequent default decision is separate

After positive **complete cache+epsilon** panel and Apple build evidence, review
paired adoption for the admitted LightOn3-1B **CPU** profile only. The epsilon-only
full panel remains negative: do not flip epsilon while leaving the old F16 cache
as the CPU default. Cover both controls and their interaction, preserving
explicit0 as the old alternative for each. Legacy profiles and default GPU math/
cache stay unchanged until separate GPU evidence; Apple compilation does not
qualify Metal inference. Use the shared value-parsed
gate helper, update env documentation and test unset/empty/0/1 behavior. Declare
a separate source closure/current-source control protocol; preserve this v1
closure and its exact producer branch instead of repinning its owned hashes.
Verify default-unset native forced/free output against retained corrected
EPS=1/KV=1 results, covering all six case/modes and both precisions without new
original forwards. Retain exact bytes, full logits, decoded IDs/text and cap/EOS
status. Do not promote an alias or claim EOS-completed documents from the existing
128-token capped cases. Q8/Q4 and public consumers keep their separate gates.
