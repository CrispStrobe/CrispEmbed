#!/usr/bin/env python3
"""The named qwen2vl quantizer policies equal their measured recipes (no weights).

tools/quantize.cpp (CRISPEMBED_QUANTIZE_QWEN2VL_POLICY) and
tools/qwen2vl_quant_recipes.json must agree: the C++ rule table equals the JSON
`policy_rules`, and those rules resolve every tensor class to the same type as
the measured recipe they stand for (`equivalent_to`), last match wins.
"""
from fnmatch import fnmatchcase
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

NAMES = {
    "teleocr": ["v.patch_embed.weight", "v.merger.fc1.weight", "v.blk.0.attn_qkv.weight", "v.blk.31.attn_proj.weight",
                "v.blk.7.ffn_gate.weight", "v.blk.7.ffn_up.weight", "v.blk.7.ffn_down.weight",
                "l.embed_tokens.weight", "l.blk.0.attn_q.weight", "l.blk.0.attn_k.weight", "l.blk.27.attn_v.weight",
                "l.blk.13.attn_o.weight", "l.blk.13.ffn_gate.weight", "l.blk.13.ffn_up.weight",
                "l.blk.13.ffn_down.weight"],
    "mineru": ["v.patch_embed.weight", "v.merger.fc2.weight", "v.blk.0.attn_qkv.weight", "v.blk.0.attn_proj.weight",
               "v.blk.0.ffn_fc1.weight", "v.blk.31.ffn_fc2.weight", "l.embed_tokens.weight",
               "l.blk.0.attn_q.weight", "l.blk.0.attn_k.weight", "l.blk.0.attn_v.weight", "l.blk.23.attn_o.weight",
               "l.blk.23.ffn_gate.weight", "l.blk.23.ffn_up.weight", "l.blk.23.ffn_down.weight"],
}


def resolve(rules, name, base):
    out = base
    for rule in rules:
        pattern, _, kind = rule.rpartition("=")
        if fnmatchcase(name, pattern):  # tensor names contain no [ ] so fnmatch == the quantizer glob
            out = kind
    return out


def cpp_policies():
    src = (ROOT / "tools/quantize.cpp").read_text()
    table = src[src.index("qwen2vl_policies() {"):src.index("static const Qwen2vlPolicy * g_qwen2vl_policy")]
    out = {}
    heads = list(re.finditer(r'\{\s*"(\w+_q4k_v\d+)",\s*"(\w+)",', table))
    for i, m in enumerate(heads):
        body = table[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(table)]
        rules = [f"{p}={t.lower()}" for p, t in re.findall(r'\{\s*"([^"]+)",\s*GGML_TYPE_(\w+)\s*\}', body)]
        out[m.group(1)] = (m.group(2), rules)
    return out


def main():
    recipes = json.loads((ROOT / "tools/qwen2vl_quant_recipes.json").read_text())
    cpp = cpp_policies()
    assert set(cpp) == {"teleocr_q4k_v1", "mineru_q4k_v1"}, cpp
    for name, (model, rules) in cpp.items():
        entry = recipes[model][name]
        assert entry["env"] == {"CRISPEMBED_QUANTIZE_QWEN2VL_POLICY": name}, entry
        assert entry["policy_rules"] == rules, (name, rules, entry["policy_rules"])
        measured = recipes[model][entry["equivalent_to"]]
        assert measured["type"] == entry["type"] == "q8_0"
        for tensor in NAMES[model]:
            a = resolve(rules, tensor, "q8_0")
            b = resolve(measured["tensor_type"], tensor, "q8_0")
            assert a == b, (name, tensor, a, b)
    print("PASS: qwen2vl quantizer policies equal their measured recipes")


if __name__ == "__main__":
    main()
