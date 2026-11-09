# Read-only actual shared-prefix step33 comparison

Wait for both admitted producers: original source trace
`8eeb67bdeb1bf0e55b0d9ffd833a600967615765` / CPU service v1
`lighton4b-bf16-mixed-step33-v1`, and native head A/B
[42168142185](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42168142185)
at `b98883abeccd0cae39e0026571c150e444b71786`. No consumer dispatch before
complete source/native receipts, successful native attempt1 and immutable archive
identity review. Source receipt is
`5aa5a8026226f218a6e441d6e0c6d1a8a5444b7113574943f92e94d28d3c86b4`.

The [consumer](../tools/ci-heavy/lighton4b_mixed_step33_compare.py) performs no
model/reference inference or model acquisition. Authenticate exact CPU service
bootstrap/version, source model/parameter/call controls and all34 observer-neutral
score hashes against retained178 capture. Authenticate full native archive,
producer report, frozen baseline identity, head controls and33 matching causal
IDs. Verify actual input68100 and three rotary position values60. Native ON
arrays are eligible only when its preceding33 IDs equal the source; otherwise
record that exclusion rather than compare different contexts.

Compare48 corresponding raw arrays with the actual unchanged
`crispembed-diff-lighton4b-document` target: embedding/input, positions, layer0
operators including QKV/z/a/b, all32 hidden outputs, final norm and full head
output. Keep `0.002` / `0.99999` budgets. Record absolute magnitudes, every raw
STAGE metric, earliest raw difference and budget failure. Require native OFF/ON
same-prefix nonhead arrays to be byte-identical for head-only attribution.
Source head input/norm and head output/generated-score boundaries are already
byte-exact. Exact Ref equal/rejection and actual tie-to-even controls run hosted.

Separate BF16 nearest-even controls round copies of native outputs only when
the source's actual dtype is BF16. Preserve raw metrics and arrays; these
controls do not replace gold, waive budgets or establish an operator fix.
Native dequantized parameters are F32 while the original uses BF16-rounded
parameters; prior hidden/cache states also differ. A stage residual cannot be
called a kernel bug until same-input/effective-weight controls justify it.
Record actual source/native/RNE argmax and401/1778 tie values at the head,
without injecting any source values into product inference.

The same consumer reuses the first26 retained score rows of official BF16,
genuine mixed BF16 and native OFF. Verify their25 preceding generated IDs and all
five prepared-input hashes agree; decision25 is included because its input
prefix is still identical. Later contexts are excluded. Authenticate the official
original v1 bootstrap, receipt and unchanged177-row protocol before retrieving
only26 rows. Compare official-versus-mixed (quantization plus BF16 parameter
rounding), mixed-versus-native (runtime plus effective-precision residual) and
official-versus-native separately with the same Ref budgets. In particular this
checks the old worst score row5 without a new teacher-forced/model run. Each
payload is hashed and preserved; scalar maxima/RMS cannot be subtracted to assign
causal contributions or interpreted as an isolated implementation defect.

The one dedicated existing-workflow read-only job uses Python3.12, pinned
NumPy2.3.5/Kaggle2.2.5/six1.17.0 and portable two-thread Ref build. It has no model
upload credential. Require2GiB available RAM,4GiB scratch,512MiB output;
native archive at most2GiB;20-minute script /25-minute job bound. All source
and native stage payloads are individually hashed and retained with reports for
90days. Optional `--preserve-full-source` authenticates the original full worker
version/bootstrap and preserves all178 BF16 mixed rows (176,803,840 bytes), text
and IDs in the same artifact. No model payload is uploaded. Current CPU service
retention alone has no declared90-day guarantee.

Arguments, independently reviewed after native completion:

```text
--artifact-id <native-ab-artifact>
--artifact-sha256 <native-ab-complete-zip-sha256>
--receipt-sha256 <native-ab-report-sha256>
--source-receipt-sha256 5aa5a8026226f218a6e441d6e0c6d1a8a5444b7113574943f92e94d28d3c86b4
--preserve-full-source
```

Raw numerical rejection exits1 after complete measurement; authentication or
transfer failure exits2 with partial reports retained. The observed form remains
excluded from held-out calibration and full architecture-quality selection.
No native/source model replay, default/release/app-pin change, model publication
or new PR is part of this consumer.
