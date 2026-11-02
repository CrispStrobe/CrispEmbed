# One-sentence live norm operands v2: component inventory admission

[Run 40867038915](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40867038915)
at `ff3b8a0171d2cdb5cfcdd65701fadaa49a99d265` (v1) authenticated all39 small
members, built native and pthread WASM components with exit0 and passed the
connected observer controls. It then rejected in `component-build` with
`Actual component rounding flags differ`
([review](../tests/results/packaging/wasm-live-norm-initial-build-review.json)).
No model, browser recovery or original forward ran. That rejection is retained
unchanged.

Cause: v1 required exactly one `wasm_native_order_norm.cpp` compile entry. The
live CMake project adds the frozen controls (`frozen-controls/`) and its own
`live-norm-replay` target, and both compile the helper. The count assumption,
not a flag, failed; v1 never saved the actual inventory.

v2 is a closed caller ([protocol](wasm-live-norm-operands-v2.json),
[workflow](../.github/workflows/wasm-live-norm-operands-v2.yml)). It changes
only component admission, and admission now runs before any build:

- After configure, the **full** native and WASM `compile_commands.json` are saved
  to `review/` before any check can reject.
- Exactly five declared (target, source, directory) entries are allowed:
  controls.cpp, the helper and diff.cpp in `frozen-controls`, plus replay.cpp and
  the helper in `live-norm-replay`. Missing, duplicate, extra, relocated or
  out-of-tree entries reject.
- Every declared entry needs `-ffp-contract=off` as its only contraction flag.
  The entry also must not contain fast-math or related flags, `-Ofast`,
  `-ffp-model=fast` or `-msimd128`. It needs the exact source path, the right
  compiler (`em++` for WASM, native C++ otherwise) and `-pthread` only on WASM.
- Each entry's arithmetic/define/std profile must equal the SHA-pinned
  run-40852835039 frozen component inventory, which is extracted from the
  authenticated previous review archive. The live target is compared with the
  frozen controls profile. Both helper compilations must match.
- The GGML backend file set and per-file profile must equal the frozen
  inventory, including the WASM `-msimd128` backend.

All other phases are the unchanged v1 functions under v2's source/service guard.
The one-sentence scope is the same: 13 norm pairs, 23 layer3 boundaries,
208 same-input and 196 natural comparisons, 32 old WASM stage hashes and native
byte anchors. A complete numerical negative still exits1. Gates, defaults,
0.002/0.99999 budgets and the full-norm64 rejection are unchanged. No ten-case
retry.

## Hosted attempts

- [41416683981](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41416683981)
  at `1b93141b`: the v1 tests, v2 tests, preflight, prepare (frozen inventory
  extraction) and browser setup pass. Component admission saved the full
  30-entry native inventory, then rejected before any build with
  `Unrecognised compile output: frozen-controls/CMakeFiles/...`. The saved
  inventory contains exactly the five declared entries with the frozen flags.
  The v2 parser had assumed `output` was relative to `directory`. CMake writes
  it relative to the top-level build. The repair reads the target from `-o` and
  requires `output == <declared dir>/<-o>`. The saved inventory is now a
  regression fixture. This rejection is kept unchanged.
