#!/usr/bin/env python3
"""The named Josev quantizer policies equal their measured sweep recipes (no weights).

tools/quantize.cpp (CRISPEMBED_QUANTIZE_JOSEV_POLICY) rule tables, resolved with
the quantizer's last-match-wins glob semantics over the base type q4_k, must give
every one of the 442 Josev tensors the same storage as the measured recipe in
tools/josev2b_q4k_recipes.json, resolved by tools/ci-heavy/josev2b_q4k_sweep.py.
"""
from fnmatch import fnmatchcase
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
TYPES = {'F32': 0, 'F16': 1, 'Q8_0': 8, 'Q4_K': 12}


def cpp_policies():
    src = (ROOT / 'tools/quantize.cpp').read_text()
    table = src[src.index('josev_policies() {'):src.index('static const JosevPolicy * g_josev_policy')]
    heads = list(re.finditer(r'\{\s*"(josev_\w+)",\s*"([\w-]+)",', table))
    out = {}
    for i, m in enumerate(heads):
        body = table[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(table)]
        out[m.group(1)] = (m.group(2), re.findall(r'\{\s*"([^"]+)",\s*GGML_TYPE_(\w+)\s*\}', body))
    return out


def main():
    spec = importlib.util.spec_from_file_location('sweep', ROOT / 'tools/ci-heavy/josev2b_q4k_sweep.py')
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    rows = sweep.prot.schema()
    policies = cpp_policies()
    assert policies, 'no Josev policy found'
    for name, (recipe, rules) in policies.items():
        assert recipe in sweep.RECIPES, (name, recipe)
        measured = sweep.plan(rows, recipe)
        assert any(v == 12 for v in measured.values()), 'a Josev Q4_K policy must contain Q4_K'
        for tensor, want in measured.items():
            got = 12  # base q4_k
            for pattern, kind in rules:
                if fnmatchcase(tensor, pattern):
                    got = TYPES[kind]
            assert got == want, (name, tensor, got, want)
        print(f'{name} == {recipe}: {len(measured)} tensors')


if __name__ == '__main__':
    main()
