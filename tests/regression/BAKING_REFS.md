# Baking diff-harness reference archives

How to produce a `<engine>-ref.gguf`, upload it, and promote a `smoke` entry to
the `diff` (real-parity) tier. This is the path for the ~25 engines that have a
`test-<engine>-diff` harness but no frozen reference yet (see
`docs/diff-harness-coverage.md`).

## What a ref is

A `*-ref.gguf` holds the reference intermediate tensors (input + per-stage
activations + output) that the original PyTorch model produces for a fixed
input. `test-<engine>-diff <model.gguf> <ref.gguf>` runs the C++ engine on the
same input and compares each stage via cosine similarity (`crispembed_diff::Ref`
in `src/crispembed_diff.h`).

## Two kinds of dumper

- **`tools/dump_<engine>_reference.py`** — loads the upstream model from
  HF/source weights, runs it, writes the ref. Needs the source model + its
  Python deps (transformers/torch/the model package).
- **`tools/dump_<engine>_reference_from_gguf.py`** — *self-consistent*: loads the
  **GGUF** weights into the reference PyTorch arch and recomputes the stages, so
  the ref matches what the C++ engine should reproduce bit-for-bit (modulo
  quant). No source download — but it still needs the PyTorch **arch class**.

> ⚠️ The `*_from_gguf` dumpers `exec()`/import the upstream **arch source**
> (e.g. `PAN_arch`, HAT/SCUNet/InstructIR modeling) which is **not vendored in
> this repo** — it lives wherever you cloned the upstream model (the dumps were
> originally driven from `/private/tmp/...`). You need that arch `.py` on hand.
> `dump_hat_reference_from_gguf.py` already takes `--arch/--gguf/--output`; the
> pan/scunet/instructir variants currently hardcode `/private/tmp` paths — pass
> the equivalent locally or adjust the top of the script.

## Bake → upload → promote

1. **Bake** (local box with torch + the arch source). Example (HAT, already
   parameterized):
   ```bash
   python tools/dump_hat_reference_from_gguf.py \
       --arch /path/to/hat_arch.py \
       --gguf /path/to/hat-sr-x4-f16.gguf \
       --output /tmp/hat-ref.gguf
   ```
   The other SR archs (`scunet`, `pan`, `instructir`, and — once dumpers exist —
   `esrgan`, `safmn`, `swinir`, `restormer`, `adair`, `dat`, `tbsrn`) follow the
   same shape; edit the hardcoded paths or add an argparse like HAT's.

2. **Upload** to that model's HF GGUF repo under `diff-harness-ref/`:
   ```bash
   huggingface-cli upload cstr/scunet-GGUF /tmp/scunet-ref.gguf \
       diff-harness-ref/scunet-ref.gguf
   ```
   (Convention: `diff-harness-ref/<engine>-ref.gguf`. Legacy refs at repo root —
   `instructir-ref.gguf`, `hat-ref.gguf`, `pan-ref.gguf`, `granite-*-ref.gguf` —
   stay where they are; the manifest pins the exact path.)

3. **Promote** the manifest entry from `smoke` to `diff` — replace its `smoke`
   block with:
   ```json
   "tier": "diff",
   "diff_bin": "test-scunet-diff",
   "argv": "{model} {ref}",
   "ref": { "repo": "cstr/scunet-GGUF", "revision": "<sha>",
            "path": "diff-harness-ref/scunet-ref.gguf" },
   "diff_thresholds": { "*": 0.999 }
   ```
   Drop the `"modality"` field so the generator treats it as curated (won't
   overwrite it). Pick the threshold from the engine's realistic floor (f16
   self-consistent refs hit ~0.999; q8_0 vision towers can be lower — see the
   granite-vision note in `PLAN.md`).

4. **Verify + refresh docs**:
   ```bash
   cmake --build build --target test-scunet-diff -j
   python tests/regression/run_one.py --backend scunet --build-dir build
   python tools/audit_diff_coverage.py --probe-hf      # refresh coverage table
   ```

## Rebake on Kaggle (alternative)

`tools/kaggle/crispembed-regression/` has a `MODE=rebake` path that runs a
per-entry `rebake` recipe and (with `UPLOAD=1`) pushes the ref to the model
repo's `diff-harness-ref/`. To use it, (a) **vendor the arch source** into the
repo so the kernel can import it, and (b) add a `rebake` block to the entry:
```json
"rebake": { "cmd": "python tools/dump_scunet_reference_from_gguf.py {model} {out}",
            "needs_gguf": true }
```
`{model}` = the downloaded GGUF, `{out}` = the staged ref path. Until the arch
is vendored, prefer the local bake above.
