# One-sentence live norm operands v4: Ref adapter for the live 1536-wide rows

v3 ([protocol](wasm-live-norm-operands-v3.json)) completed every control,
capture and the 208 same-input comparisons (all bit-exact) in
[41427588368](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41427588368).
Replay then rejected `Bounded exact NumPy shape required` before the 196
natural comparisons. The natural pairs include the actual layer3 `fc1_raw`,
`fc1` and `gelu` tensors `[12,1536]`. The frozen, protected Ref adapter
(`tools/moedern-native-v2/diff_adapter.py`) caps every dimension at 512. That
rejection is retained.

v4 ([protocol](wasm-live-norm-operands-v4.json),
[workflow](../.github/workflows/wasm-live-norm-operands-v4.yml)) changes only the
adapter used by the Ref comparison.

- [`diff_adapter_v4.py`](../tools/wasm-live-norm-v4/diff_adapter_v4.py) executes
  the SHA-pinned frozen adapter source into a private module and replaces only
  `shape()` with an exact allowlist: `[12,384]` and `[12,1536]` (live rows) plus
  `[2,8]` (existing Ref controls). It is not an open limit. Anything else rejects,
  on write and on readback. The frozen file is unchanged.
- Metric semantics are unchanged. The independent CPython metric uses
  `width = shape[-1]`, and the C++ Ref (`tests/moedern-native-v2/diff.cpp`) uses
  `ne[0]`, the C-order last axis, as the row axis. For `[12,1536]` both compute
  max-abs and cosine over full 1536-wide rows. Nothing is reshaped or chunked.
  Budgets stay 0.002 / 0.99999, and numeric failures still exit 1.
- The Ref call body is the frozen `wasm_native_order_norm_v1.actual_ref` with
  only the adapter module swapped. A test compares the two ASTs.
- Tests prove GGUF bytes, readback and `collect` results are bit-for-bit
  identical to the frozen adapter for every allowlisted shape up to 512 wide,
  for both passing and numeric-rejection records. The old adapter rejects
  `[12,1536]` and v4 admits it. Unexpected shapes (13/16 rows, 385/512/1537
  widths, transposes, rank 1/3, non-int, `[4,4]`) are rejected. A 512-chunked
  cosine would report a different statistic than the full row.

v2 admission, the v3 observer, the v3 observer controls before model access, and
every v1 phase run unchanged in the same component-first order.

## Attempts

- [41432135298](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41432135298)
  at `77ed7122`: the attribution **completed** and is a numerical rejection
  (exit1), as designed ([receipt](../tests/results/packaging/wasm-live-norm-v4-attribution.json)).
  All weight-free controls passed before model access.
  - Local attribution passes: 104/104 replay controls, and all 208 same-input
    norm replays are bit-exact.
  - Natural comparisons: 132/196 pass and 64 fail (40 norm rows, 24 layer3
    rows). C++ Ref and independent metrics agree.
  - Error stays at or below 1.9e-6 through layers 0-2 and layer3 attention. It
    rises about 300x at FC1 (`fc1_raw` 5.73e-4) and first fails at `fc2_raw`
    (0.00331, cosine 0.99999569), then reaches 0.185 at norm12.
  - Results are identical for gate 0/1 and threads 1/2. The norm operator is
    not the source.
