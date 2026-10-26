# Initial successor Mac-only diagnostic retry

The reusable [successor-apple-only workflow](../.github/workflows/successor-apple-only.yml)
retries the initial Apple consumer after successful
[40386968045](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40386968045)
reproduced the frozen fifty PNG identities. It uses only the original native
producer [40339796226](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40339796226)
at40a82158146faf0cd44de2a4ee7056e7b0c6a58c. Wrong input run IDs reject before
archive, corpus or model access and retain a diagnostic rejection receipt.

Dispatch metadata phase`successor-apple-only` with
`native_run_id=40339796226` and`candidate_run_id=40386968045`. These forward to
reusable inputs`native_run` and`corpus_run`; no extra metadata input is added.
The audit starts from20e58588 with production/build/binding objects unchanged
from the immutable initial producer. Existing source/manifest/archive/member
checks remain enforced by the unchanged consumer helpers.

The single macOS15 job preserves the existing Apple consumer body: download the
actual initial Mac archive, consume the exact successful Linux corpus proof,
verify every original PNG/RGB/manifest/notice identity, fetch the pinned published
PosFormer model, and compare actual released0.17.12 against candidate unset,
literal1 and legacy0 on all fifty cases. Legacy0 remains a separate diagnostic.
All per-arm reports/logs, corpus or input-guard rejection receipts and the Mac
archive receipt upload with`if: always()` to the distinct
`successor-apple-only-<run>` artifact. No other consumer job is started.

This retry is diagnostic Mac-only evidence. It performs no aggregate and cannot
qualify a release, an integrated source, an application upgrade or recognition
quality. Final integration still requires fresh native/WASM producers at one
reviewed final source and every required fresh consumer, including Apple.
The initial pre-inference failure remains preserved.

Six small stdlib source controls verify the copied probe body, immutable run
selection and retained negatives, metadata forwarding/input count, failure
artifact coverage, and unchanged production/helpers/full-consumer workflow.
Workflow lint passes. No local build, model, corpus or inference ran, and this
prepared retry has not been dispatched.
