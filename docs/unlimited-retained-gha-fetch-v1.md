# Exact stopped-v3 delivery and read-only Unlimited structural audit

The original mixed-Q4 v3 source410541f9 completes all16 F16 control executions
and retains a6,481,651,840-byte candidate, but rejects `v.pos_embed` storage
shape before any Q4 inference. Preserve that negative. The separate retained
audit v1 at a7de18f9 fails during environment bootstrap; corrected v2 atc406fc28
passes setup and17 controls but fails because its exact stopped-v3 panel is
absent from mounted inputs. Neither audit reads arrays/weights or runs inference.

The producer output remains accessible through exact-version SDK GET. A bounded
one-byte ranged GET returned206 with the expected6,481,651,840-byte whole extent
and first GGUF header byte. HEAD returns404 on this endpoint and cannot establish
that retained data was lost. Do not repeat the missing-mount audit or original
model capture.

`unlimited-retained-gha-fetch-v1.yml` runs only on hosted Linux CPU, with the
existing repository service secrets. It authenticates source closure, then exact
v3 Get, Status and every paginated List request. Each typed request must serialize
`versionLabel=v3`; the7899-byte submitted script has SHA256
`dd11f6e7b530489fa3bcf09b858051b47230cd3f77871b7c7beafe74ec396b6e`.
The terminal state must remain ERROR at producer410541f9.

Before array imports, fetch and authenticate the514832-byte panel at SHA256
`9c0aae284ef9af732c51bf3ccfe5996b71cb5d8b448dc1bf5f6802468cf53ba9`,
then all540 listed panel files /7,901,314,457 bytes, including the existing first
candidate at SHA256
`8b9d8f28e225a229107da1c2dd043452d5c30a3f568337eff1741181a5da4f37`.
No unrelated service outputs are transferred. Use two IO workers,64MiB exact
ranges for large files and4MiB read buffers; require status, Content-Length,
Content-Range, safe paths and every whole-file SHA. Signed URLs and credentials
stay in memory and never enter proof files or exception messages. Require10GiB
available RAM and16GiB free hosted scratch before SDK or payload access.

Only after every byte is authenticated, the new GHA-only audit applies the
unchanged retained-v3 structural analysis: all four original IDs/text, sixteen
F16 modes,58 per-vision observed/repeat stages, candidate inventory, selected
actual Q4_K tensors and every protected type/semantic axis/byte SHA. The copied
audit differs from oldc406 only in genuine source-bound GHA admission and its
module description. The old Kaggle audit, producer, original dumps, source83
production closure and all original budgets remain untouched.

Success would establish retained structure and whether physical rank
canonicalization alone explains the earlier protected-tensor rejection. It
would not qualify any Q4 numerical stage or decoded output. Preserve small
transfer/audit proofs as90-day hosted artifacts; retain original large inputs
remotely without republishing them. Subsequent Q4 forwards require a separately
declared source, a matched-host F16 control against these retained traces, actual
`crispembed-diff`/absolute/RMS/magnitude comparisons and exact original decoded
output. Reuse the first candidate; the2/8/22 selected stacked-expert recipes and
all their higher-precision exclusions remain unchanged. No quantizer rerun,
original/native model forward, build, model alias or default change belongs in
this read-only delivery workflow.

Actual40834740153 atfbd8f68f passes exact-v3 service/script/all-seven-page
admission and authenticates the panel plus537 journaled files/1,419,601,456 bytes.
It then rejects the producer's zero-byte `quantize-last1/stderr`, because the
first downloader required positive file lengths. The [initial review](../tests/results/unlimited-ocr2/retained-get-empty-member-initial-review.json)
retains that failure. The correction accepts a zero-length member only after
a real200 GET, Content-Length0, no extra byte and exact empty-file SHA; it never
skips an empty member. All540-file, source, model, structure and numerical gates
remain unchanged. The initial run never reaches array analysis or Q4 inference.
