# PCS final checkpoint and handoff

Status: **unresolved PCS casing failure; no production fix or main-green closure**.
This checkpoint stops new experiments, model runs, builds and broad sweeps.
The current bounded operation is documentation and branch preservation only.

## Canonical source and evidence

Canonical completed attribution remains commit
`135665fc50137507c2a51ba853da84ba1eb4c605` on
`diag/pcs-gate-capture-20261118`, continuing preserved source-only commit
`7d9a952fef505f709dbf34ad7cfda175c06820a9`. The final documentation commit
adds this handoff without changing production or diagnostic computation.
The existing 301-file preservation inventory is retained without another audit.

The [evidence index](../tests/results/pcs/README.md) links the original reports,
source identities, compiler/CPU/ISA information, raw artifact identities and
all 16 row-37 F32 casing logits/bits. Preserve every positive and negative arm.

| Execution | Row-37 `o` logit | Casing | Decoded |
| --- | ---: | ---: | ---: |
| Original uncompressed ONNX | +1.083245516 | 67/67 | 6/6 reference |
| Intel native AVX512 positive control | +0.039054506 | 67/67 | 6/6 |
| AVX2 native reproduction | −0.098158233 | 66/67 | 5/6 |
| Rejected same-weight F32 matrix control | −0.238311455 | 66/67 | 5/6 |
| Original graph/collector with all 214 stored tensors | −0.238312081 | 66/67 | 5/6 |

Post/pre/segmentation agree 67/67 in the reproduced failure. Row 37 is
request 3, local token 3, ID 3664, piece `▁ok`, in one 11-token chunk.
Native mapping aligns and sentence-start conditioning is false. Actual original
versus diagnostic OFF and diagnostic OFF/ON match all four head dumps and
decoded bytes exactly. Original collector reconstruction of the frozen reference
predictions passes 6/6. The unchanged gate produces `I'm oK`, rather than `I'm OK`.

Immutable GGUF input:

- Repository: `cstr/pcs-xlmr-base-GGUF`.
- Revision: `27f17edbd7e33949f42a9a4fe4322b5af8172e6b`.
- Filename: `pcs-xlmr-base-q4_k-imatrix.gguf`.
- SHA-256: `faee4bf8469a502362b5967a2cfeb7a1732539fbd999c63b489d00efff3531da`.

Preserve these completed runs without unchanged re-execution:

| Run | Preserved outcome |
| --- | --- |
| [44272906599](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44272906599) | Compiled/linked Intel native positive and byte-neutral capture controls |
| [44273793050](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44273793050) | Exact AVX2 negative; original ONNX predictions observer-neutral |
| [44274761438](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44274761438) | F32 matrix alternative rejected; 13 encoder stage pairs retained |
| [44276100741](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44276100741) | Original singleton-axis replay preflight failure, before inference |
| [44276378964](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44276378964) | All 214 stored tensors replayed in original graph/collector; still quality-negative |

The final replay's original execution exit was successful despite its negative
quality report. The reporting correction in canonical `135665fc5` makes those
unchanged negative comparisons return exit 1; retained-report reanalysis proves
that behavior without another model run. It does not make the model pass.

## Separately authorized operator-attribution continuation

The user designated a separate PCS operator-attribution owner. The canonical
capture owner subsequently remained report-only and did not duplicate that
owner's captures, model/native builds, diff execution or audits. The following
newer outcomes were supplied by the coordinator/user, rather than independently
re-audited by the canonical owner:

- Corrected observer `986037`,
  [run 44347337837](https://github.com/CrispStrobe/CrispEmbed/actions/runs/44347337837),
  is qualified for its frozen CPU/AVX2 request: context 12/12; original-driver
  versus legacy OFF and both OFF/ON controls preserve all four head dumps and
  decoded bytes exactly. Preserve the preceding capture-alias diagnosis too.
- Subsequent commit `0b54ef8ba` establishes six scoped controls and an
  operand-QK causal contribution. This is not a PCS casing repair. Decoded `oK`,
  all ten QK budget failures, and unqualified original full gates remain.
- The exact current held-fixed attribution blocker is **missing casing-projection
  fc1/fc2 weight payloads in that captured operator panel**. The earlier all-214-
  tensor original-graph replay does not supply the missing payloads at the newer
  panel's held-fixed boundaries. Do not treat those panels as interchangeable.

These scoped observer/operator results do not supersede the canonical positive
AVX512 versus negative AVX2 and stored-weight replay boundaries. The exact
casing-causing tensor, conversion operation and conversion commit remain unknown.
No model artifact repair, production fix, release, passing fullstop gate or
merged-main PCS success is established.

## Remaining work and execution boundary

1. A resumed, explicitly authorized operator owner must resolve the missing
   fc1/fc2 payloads for the declared held-fixed panel. Bind source, model hash,
   actual inputs, conditioning and weight bytes; prove observer neutrality before
   using any new capture. Do not repeat already qualified arms for checkpoints.
2. Complete held-fixed casing attribution and identify the actual cause. Keep
   the scoped QK contribution separate from proof of the full casing failure.
   Artifact/model/quantization remediation requires its own authorized scope;
   no artifact regeneration or new experiment starts from this handoff alone.
3. Implement only a causally justified repair. Keep production PCS and the
   original corpus/reference/scorer/thresholds/tolerances unchanged until an
   actual fix is established. No baseline update, case hack or waiver.
4. Pass the unchanged PCS gate: post/pre/segmentation/casing 67/67 and decoded
   6/6, then reach and pass fullstop. Complete a normal repair PR, exact-head
   checks, owner merge and fetched-main push-only PCS verification. None of
   these repair/merge/closure steps is complete at this checkpoint.

This branch is retained as diagnostic evidence and handoff, without a repair PR
or merge. Opening its code-bearing PR would schedule fresh builds, outside this
final checkpoint's stop-new-builds instruction. A future resumed repair owner
must review the assembled final source before opening the repair PR.

Retain the 5 GiB user.slice limit, at least 1 GiB projected headroom, admission
only when current use is at most 3.8 GiB, one heavy task/shared lock, and all
original host guards. Build/model/corpus work belongs on authorized hosted
resources; no new watcher is requested. Host guard pressure notices are resource
events, not model-quality evidence. All machine paths, pane/owner identities
and local raw-receipt locations remain in private continuity outside Git.
