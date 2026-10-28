# Successor assets v2: final-source producers and consumers

Protocol: [successor-assets-v2.json](successor-assets-v2.json). Contract and
controls: `tools/successor_assets_v2.py`, `tests/test_successor_assets_v2.py`.
Workflow: `.github/workflows/successor-assets-v2.yml`.

This is a separately versioned protocol. Every v1 successor file, guard and
receipt (`docs/successor-*.md`, `tools/ci-heavy/successor_*.py`,
`.github/workflows/successor-*.yml`, `tests/results/packaging/*.json`, ...)
stays unchanged; v2 pins their Git blob identities and rejects any change.

## Pins and status

- Final base `7bfa060df57f29692c972b84acf7b73abeaa80e7`, ggml
  `a06fc0bb46b73cc10011fc90ae4054aee8438aeb`, both required to equal
  `docs/ovisocr2-final-integrated-v2.json`.
- The final-integrated-v2 validation run is **pending**. v2 producers may run,
  but nothing qualifies, no release alias is created and the app stays on
  0.17.12 until that validation and every consumer are positive.
- Producers: the seven native CPU/mobile archives (`release.yml`,
  `cpu_only=true`) and the two WASM bundles (`release-wasm.yml`, no tag), both
  dispatched at one commit whose v1 runtime closure equals the final base.
- Consumers: the unchanged v1 matrix in `successor-assets.yml` (desktop
  Linux x86_64/ARM64, macOS ARM64, Windows x86_64; Linux Rust/Dart; browser;
  Apple PNG/Mac50; Android arm64/armv7 and iOS cross-link; review).

## Dispatch

`ovisocr2-metadata.yml` already totals 50 reusable-workflow references when
nested and duplicate references are counted, so v2 is **not** a dispatcher
phase, and the existing `successor-consumer` phase is not reused because that
would require editing v1 files. The v2 workflow is standalone and can be
dispatched only after it is merged to `main`:

```
gh workflow run successor-assets-v2.yml --ref <branch whose runtime closure equals 7bfa060df57f29692c972b84acf7b73abeaa80e7>
```

One dispatch runs the stdlib controls, dispatches both producers (or adopts
existing `native_run`/`wasm_run` IDs), waits, verifies producer identity and
then calls the v1 consumer matrix.

## Expectations (declared before any run)

| Surface | Expected | Note |
|---|---|---|
| C, Python, CLI, cache | pass | Windows compiled ABI/Python never reached in the initial run |
| Rust, Dart | unproven | Linux prebuilt only |
| Windows UTF-8 paths | unproven | consumer UTF-8 reader and `cl /utf-8` repair (5702b1ae) is in the final base; native sources have no wide-path conversion and no non-ASCII path is exercised |
| Apple PNG/Mac50 | pass | compatibility only; 7/50 quality unchanged |
| Mobile cross-link | unproven | needs a fresh final-source receipt |
| Browser/WASM | **fail** | full-panel parity negative; rounding candidate default-off; must not be claimed |

All 16 earlier packaging receipts are carried forward as read-only evidence,
with their negative/positive/context verdicts listed in the protocol.
