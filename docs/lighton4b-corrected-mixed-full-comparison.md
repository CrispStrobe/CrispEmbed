# Corrected mixed form: pending full177-row comparison

Prepare exactly one read-only hosted comparison after native run
[42109519607](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42109519607)
finishes successfully. Its producer source is fixed to
`189fc91bb1cc3a49ec918c804c30d072135546af`; the mixed model remains
`d2eb57d1f98d5c455f46f95b6c6095f38521df19927d486848b668222e9fc474`.
No dispatch is authorized by this document while native output is pending.
The observed form remains excluded from held-out calibration/quality selection.

The [adapter](../tools/ci-heavy/lighton4b_corrected_mixed_full_compare.py) uses the
unchanged frozen BF16 full-generation protocol SHA
`4d2a738a8df75947f5a8e269799c8b3a27af19bfafb82653802a7531b7c928bc`,
original receipt `170bc06c8b5e929e45d334a5eadc284024f4265b2549f2feddb46607cb49251a`,
all 177 original score hashes and generated IDs, and the actual
`crispembed-diff-lighton4b-document` target. Maximum error `0.002` and minimum
row cosine `0.99999` remain unchanged. No original/native inference or F32 model
runs in the comparer.

This producer emits `corrected-mixed-full.json`, not the legacy producer's
`native.json`. The adapter therefore constructs a deterministic, explicitly
**comparer-derived** normalized receipt after authenticating the actual producer.
It does not relabel mixed as Q4 or claim the producer authored that receipt.
Require independent review of the successful run/source/attempt, artifact ID,
complete archive SHA-256, producer receipt SHA-256, and expected normalized
receipt SHA-256 before dispatch. Original protocol files and the old native
admission registry remain unchanged. A closed additive admission permits this
one mixed precision/receipt tuple; the shared comparer rejects changes to any
frozen reference field, budget, or legacy admission.

The adapter authenticates full model provenance, corrected rotary profile/table,
all gates, all five prepared-input digests, all 29 source vision controls and
their actual ZIP payload bytes, all 177 complete native score rows, and the raw
independent generated IDs/text. The original BF16 provider version/bootstrap,
receipt, prepared inputs, complete score inventory and EOS completion are
authenticated separately. No ambiguous ZIP member or duplicate name is accepted.
Each score row has 248,320 F32 scores / 993,280 bytes and its own SHA-256.

After an isolated portable two-thread build of the actual Ref target, run its
existing equal/small-change/rejection/tampered-payload controls. Then execute the
existing comparison loop over every original-prefix teacher-forced position,
preserving all per-row STAGE metrics, earliest difference and earliest budget
failure. Independently compare full greedy IDs/text and EOS/cap256 closure.
Numerical rejection is preserved even if free generation ends at EOS; matching
the first token or component arrays cannot imply full decoded parity.

The dedicated existing read-only comparison lane retains outputs for 90 days.
The adapter has a 1,200-second deadline inside its 25-minute job, a 2 GiB native
archive bound, original provider listing/row bounds, at least 4 GiB scratch,
256 MiB output and 2 GiB available RAM. Preserve partial blocked reports on guard
or transfer failure. No model payload is downloaded, published or uploaded.

Once all required identities have been reviewed, dispatch this exact script via
the existing `heavy-cpu.yml` workflow with `runner=ubuntu-latest`,
`build_mode=portable`, empty `pip`, and arguments:

```text
--artifact-id <completed-run-artifact>
--artifact-sha256 <reviewed-full-archive-sha256>
--receipt-sha256 <reviewed-corrected-mixed-full-json-sha256>
--normalized-receipt-sha256 <reviewed-canonical-normalized-json-sha256>
```

Canonical normalization is `normalize(producer_report, compare.protocol())`,
serialized with `json.dumps(..., indent=2, sort_keys=True) + '\n'` and hashed as
UTF-8. This can be reviewed from the small producer receipt before downloading
the raw archive to the hosted worker. Source/run are fixed in the adapter,
rather than free-form dispatch parameters. No runtime/default/release/app pin or
new PR is part of this comparison.
