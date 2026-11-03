#!/usr/bin/env python3
"""The named Unlimited-OCR-2 quantizer policy equals its measured recipe (no weights).

tools/quantize.cpp (CRISPEMBED_QUANTIZE_UNLIMITED_OCR2_POLICY) and
tools/unlimited_ocr2_quant_recipes.json must agree: the C++ rule table and base type
equal the JSON entry, and the rules resolve every tensor class to the same type as the
measured recipe they stand for (`equivalent_to`), last match wins. Also checks the
sweep helper's cosine scoring on synthetic logits.
"""
from fnmatch import fnmatchcase
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["v.pos_embed", "c.pos_embed", "v.blk.0.attn_qkv.weight", "c.blk.23.ffn_up.weight", "l.embed_tokens.weight",
         "l.lm_head.weight", "l.blk.0.ffn_down.weight", "l.blk.0.ffn_gate.weight", "l.blk.1.attn_q.weight",
         "l.blk.11.attn_o.weight", "l.blk.5.ffn_gate_exps.weight", "l.blk.5.ffn_up_exps.weight",
         "l.blk.5.ffn_down_exps.weight", "l.blk.5.shared_exp.ffn_down.weight", "l.blk.3.attn_dash.q_proj.weight"]


def resolve(rules, name, base):
    out = base
    for rule in rules:
        pattern, _, kind = rule.rpartition("=")
        if fnmatchcase(name, pattern):
            out = kind
    return out


def cpp_policies():
    src = (ROOT / "tools/quantize.cpp").read_text()
    table = src[src.index("unlimited_ocr2_policies() {"):src.index("static const UnlimitedOcr2Policy * g_unlimited")]
    out = {}
    heads = list(re.finditer(r'\{\s*"(unlimited_ocr2_\w+)",\s*GGML_FTYPE_MOSTLY_(\w+),', table))
    for i, m in enumerate(heads):
        body = table[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(table)]
        rules = [f"{p}={t.lower()}" for p, t in re.findall(r'\{\s*"([^"]+)",\s*GGML_TYPE_(\w+)\s*\}', body)]
        out[m.group(1)] = (m.group(2).lower(), rules)
    return out


def main():
    recipes = json.loads((ROOT / "tools/unlimited_ocr2_quant_recipes.json").read_text())
    cpp = cpp_policies()
    assert set(cpp) == {"unlimited_ocr2_q4k_v1", "unlimited_ocr2_q4k_shared8_v1"}, cpp
    for name, (base, rules) in cpp.items():
        entry = recipes[name]
        assert entry["env"] == {"CRISPEMBED_QUANTIZE_UNLIMITED_OCR2_POLICY": name}, entry
        assert entry["type"] == base and entry["rules"] == rules, (name, base, rules, entry)
        measured = recipes[entry["equivalent_to"]]
        assert measured["type"] == base
        for tensor in NAMES:
            assert resolve(rules, tensor, base) == resolve(measured["rules"], tensor, base), (name, tensor)
    # Cosine scoring: identical logits -> 1, a perturbed row -> its own cosine, truncated -> incomplete.
    import numpy as np
    helper = ROOT / "tools/unlimited_ocr2_q4k.py"
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        rng = np.random.default_rng(0)
        a = rng.standard_normal((3, 129280)).astype(np.float32)
        b = a.copy()
        b[1] += 0.5 * rng.standard_normal(129280).astype(np.float32)
        a.tofile(t / "a.f32")
        b.tofile(t / "b.f32")
        subprocess.run([sys.executable, helper, "cosine", t / "a.f32", t / "b.f32", "2", t / "c.json"], check=True,
                       capture_output=True)
        rec = json.loads((t / "c.json").read_text())
        want = float((a[1].astype(np.float64) @ b[1]) / np.linalg.norm(a[1]) / np.linalg.norm(b[1].astype(np.float64)))
        assert rec["complete"] and rec["worst_position"] == 1 and abs(rec["final_cos_min"] - want) < 1e-9, rec
        a[:2].tofile(t / "short.f32")
        r = subprocess.run([sys.executable, helper, "cosine", t / "a.f32", t / "short.f32", "2", t / "d.json"],
                           capture_output=True)
        assert r.returncode == 1 and not json.loads((t / "d.json").read_text())["complete"]
    print("PASS: unlimited-ocr2 quantizer policy equals its measured recipe; cosine scoring checks")


if __name__ == "__main__":
    main()
