"""Per-stage encoder diff: crispembed (ggml) vs the ORIGINAL (HF/PyTorch) model.

Compares two GGUF dumps of the same text, stage by stage:
  ours:      CRISPEMBED_DUMP_LAYERS_GGUF=<path> crispembed -m model.gguf "<text>"
  reference: tools/dump_encoder_reference.py --model <hf repo> --text "<text>"

Why this exists: a final-embedding cosine says THAT something diverged, never
WHERE — and it hides small structural errors under "quantization noise". The
first failing stage is the bug.

ORDER MATTERS (per the repo's diff methodology):
  1. token alignment  — both sides must tokenize to the SAME ids. A shifted
                        sequence or an extra special token produces a smooth
                        position-dependent error that looks exactly like FP drift.
  2. emb_ln_out       — the STRUCTURAL GATE (pre-block-0; only embeddings +
                        tokenization feed it). Must be ~0.99999. If it fails, no
                        per-layer number below it means anything.
  3. layer_0..N       — only interpret these once 1 and 2 pass.

Thresholds are per-stage cos_min. Defaults suit a q4_k GGUF vs an fp32 reference;
tighten with --min-cos for f16/f32 GGUFs.

Usage:
  python tests/test_encoder_diff.py --ours /tmp/ce.gguf --ref /tmp/ref.gguf
  python tests/test_encoder_diff.py --ours ... --ref ... --min-cos 0.99 --gate-cos 0.9999
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def load_gguf(path: str):
    from gguf import GGUFReader

    r = GGUFReader(path)
    tensors = {}
    for t in r.tensors:
        a = np.array(t.data, dtype=np.float32)
        tensors[t.name] = a.reshape(-1)
    kv = {}
    for k, f in r.fields.items():
        try:
            v = f.parts[f.data[-1]]
            kv[k] = v.tolist() if hasattr(v, "tolist") else v
        except Exception:
            pass
    return tensors, kv, r


def cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    if da == 0 or db == 0:
        return float("nan")
    return float(np.dot(a, b) / (da * db))


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-stage encoder diff vs HF reference")
    ap.add_argument("--ours", required=True, help="GGUF from CRISPEMBED_DUMP_LAYERS_GGUF")
    ap.add_argument("--ref", required=True, help="GGUF from tools/dump_encoder_reference.py")
    ap.add_argument("--min-cos", type=float, default=0.99, help="per-layer threshold (q4_k vs fp32)")
    ap.add_argument("--gate-cos", type=float, default=0.9999, help="emb_ln_out structural gate")
    a = ap.parse_args()

    ours, ours_kv, _ = load_gguf(a.ours)
    ref, ref_kv, _ = load_gguf(a.ref)

    print("per-stage encoder diff — crispembed (ggml) vs HF (PyTorch)")
    failures = 0

    # ---- 1. structural: same shapes / token count -------------------------
    n_ref_tok = ref_kv.get("dump.n_tokens")
    if isinstance(n_ref_tok, list):
        n_ref_tok = n_ref_tok[0]
    emb_ours, emb_ref = ours.get("emb_ln_out"), ref.get("emb_ln_out")
    if emb_ours is None or emb_ref is None:
        print("  [FAIL] emb_ln_out missing from one side — cannot gate")
        return 1
    if emb_ours.size != emb_ref.size:
        n_embd = ref_kv.get("dump.n_embd")
        if isinstance(n_embd, list):
            n_embd = n_embd[0]
        print(f"  [FAIL] token/shape mismatch: ours={emb_ours.size} ref={emb_ref.size} elems "
              f"(n_embd={n_embd}, ref tokens={n_ref_tok}) -> the two sides did NOT tokenize the "
              f"same text. Fix tokenization/prefix/special tokens FIRST; per-layer cosines below "
              f"this are meaningless.")
        return 1
    print(f"  [PASS] shape gate: both sides {emb_ours.size} elems ({n_ref_tok} tokens)")

    # ---- 2. structural gate: pre-block-0 ---------------------------------
    c = cos(emb_ours, emb_ref)
    ok = c >= a.gate_cos
    # Print BOTH magnitudes: a 10-30x norm outlier on either side says
    # "same name, wrong quantity" (a harness bug) rather than a model bug. This
    # is how the nomic emb-hook artifact was caught — cos 0.69 at the gate while
    # every layer read 1.000000, which is impossible for a real input mismatch.
    print(f"  [{'PASS' if ok else 'FAIL'}] emb_ln_out (STRUCTURAL GATE): cos={c:.6f} "
          f"|ours|={np.linalg.norm(emb_ours):.3f} |ref|={np.linalg.norm(emb_ref):.3f} (need >={a.gate_cos})")
    if not ok:
        print("         ^ pre-block-0 mismatch = the inputs differ (tokenization / embeddings / "
              "position offset), NOT the graph. Do not interpret the layers below.")
        print("         ^ BUT if the layers below all read ~1.0, the inputs cannot really differ — "
              "then THIS comparison is the bug (are both sides the same quantity, e.g. pre- vs "
              "post-LayerNorm?).")
        failures += 1

    # ---- 3. per-layer ----------------------------------------------------
    n_layer = ref_kv.get("dump.n_layer")
    if isinstance(n_layer, list):
        n_layer = n_layer[0]
    first_bad = None
    n_layer = int(n_layer or 0)
    for i in range(n_layer):
        name = f"layer_{i}"
        ours_t = ours.get(name)
        # The graph renames the LAST block's output to "encoder_out", so
        # "layer_{n-1}" is absent by construction — alias it rather than skip.
        if ours_t is None and i == n_layer - 1:
            ours_t = ours.get("encoder_out")
            if ours_t is not None:
                name += " (=encoder_out)"
        ref_t = ref.get(f"layer_{i}")

        # A missing stage must FAIL, never be skipped: silence is indistinguishable
        # from success, and the stage most likely to go missing is the last one —
        # the one that feeds pooling.
        if ours_t is None or ref_t is None:
            side = "ours" if ours_t is None else "ref"
            print(f"  [FAIL] {name:12s} MISSING from {side} — cannot compare (not a pass)")
            failures += 1
            if first_bad is None:
                first_bad = name
            continue

        c = cos(ours_t, ref_t)
        good = c >= a.min_cos
        if not good and first_bad is None:
            first_bad = name
        if not good:
            failures += 1
        print(f"  [{'PASS' if good else 'FAIL'}] {name:22s} cos={c:.6f} "
              f"|ours|={np.linalg.norm(ours_t):.3f} |ref|={np.linalg.norm(ref_t):.3f}")

    if first_bad:
        print(f"\n  FIRST DIVERGENCE: {first_bad} — that stage is the bug; everything after it is "
              f"downstream noise.")
    print(f"\n{'FAILED' if failures else 'OK'} ({failures} stage failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
