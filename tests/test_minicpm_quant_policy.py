#!/usr/bin/env python3
"""The named MiniCPM-OCR quantizer policies equal their measured recipes (no weights).

tools/quantize.cpp (CRISPEMBED_QUANTIZE_MINICPM_POLICY) and tools/minicpm_quant_recipes.json
must agree: the C++ rule table equals the JSON `policy_rules`, and those rules resolve every
tensor class to the same type as the measured recipe they stand for (`equivalent_to`).
"""
from fnmatch import fnmatchcase
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["l.embed_tokens.weight", "l.layers.0.mlp.gate_proj.weight", "l.layers.23.mlp.down_proj.weight",
         "l.layers.0.linear_attn.in_proj_qkv.weight", "l.layers.0.linear_attn.out_proj.weight",
         "l.layers.3.self_attn.q_proj.weight", "l.layers.3.self_attn.o_proj.weight",
         "v.emb.patch_embedding.weight", "v.emb.position_embedding.weight",
         "v.blk.0.self_attn.q_proj.weight", "v.blk.26.self_attn.out_proj.weight", "v.blk.0.mlp.fc1.weight",
         "v.blk.26.mlp.fc2.weight", "v.vmerge.linear_1.weight", "v.vmerge.linear_2.weight",
         "v.vmerge.self_attn.k_proj.weight", "m.0.linear_1.weight", "m.0.linear_2.weight"]


def resolve(rules, name, base):
    out = base
    for rule in rules:
        pattern, _, kind = rule.rpartition("=")
        if fnmatchcase(name, pattern):
            out = kind
    return out


def cpp_policies():
    src = (ROOT / "tools/quantize.cpp").read_text()
    table = src[src.index("minicpm_policies() {"):src.index("static const MinicpmPolicy * g_minicpm_policy")]
    out = {}
    heads = list(re.finditer(r'\{\s*"(minicpm_\w+)",', table))
    for i, m in enumerate(heads):
        body = table[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(table)]
        out[m.group(1)] = [f"{p}={t.lower()}" for p, t in re.findall(r'\{\s*"([^"]+)",\s*GGML_TYPE_(\w+)\s*\}', body)]
    return out


class MinicpmPolicy(unittest.TestCase):
    def test_policies_equal_measured_recipes(self):
        recipes = json.loads((ROOT / "tools/minicpm_quant_recipes.json").read_text())["recipes"]
        cpp = cpp_policies()
        self.assertEqual(set(cpp), {"minicpm_q4k_v1", "minicpm_q4k_min_v1"})
        entries = {r["env"]["CRISPEMBED_QUANTIZE_MINICPM_POLICY"]: r for r in recipes.values() if "env" in r}
        self.assertEqual(set(entries), set(cpp))
        for name, rules in cpp.items():
            entry = entries[name]
            self.assertEqual(entry["policy_rules"], rules)
            measured = recipes[entry["equivalent_to"]]
            self.assertEqual((measured["type"], entry["type"], measured["source"], entry["source"]),
                             ("q8_0", "q8_0", "f16", "f16"))
            for tensor in NAMES:
                self.assertEqual(resolve(rules, tensor, "q8_0"), resolve(measured["tensor_type"], tensor, "q8_0"),
                                 (name, tensor))

    def test_publication_recipe_uses_the_policy(self):
        cfg = json.loads((ROOT / "docs/minicpm-ocr-publication-v1.json").read_text())
        rec = cfg["recipes"]["Q4_K_MIX"]
        self.assertEqual(rec["quantize_env"], {"CRISPEMBED_QUANTIZE_MINICPM_POLICY": "minicpm_q4k_v1"})
        self.assertEqual((rec["from"], rec["type"], rec["precision_gate"], rec["precision_tier"]),
                         ("F16", "q8_0", "cosine", "q4_k"))
        self.assertNotIn("Q4_K_MIX", cfg["default_variants"])


if __name__ == "__main__":
    unittest.main()
