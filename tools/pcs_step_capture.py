#!/usr/bin/env python3
"""Build-only observer of actual PCS graph operations, plus archive packing.

Production source is never edited. Wrappers retain the exact existing operand
order. OFF adds no graph outputs. ON must pass decoded/head neutrality checks.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from pcs_steps import write_archive


def overlay(source):
    def replace(anchor, value):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError('Missing or ambiguous step anchor: ' + anchor)
        source = source.replace(anchor, value)
    declaration = '    ggml_tensor * tok_emb = ggml_get_rows(ctx0, ctx.tok_emb_w, inp_ids);'
    replace(declaration, '    pcs_step_trace steps(ctx0, N);\n' + declaration)
    for variable in ['tok_emb', 'pos_emb', 'type_emb']:
        key = {'tok_emb': 'token', 'pos_emb': 'position', 'type_emb': 'type'}[variable]
        end = f'    ggml_tensor * {variable} = '
        line = next(line for line in source.splitlines() if line.startswith(end))
        replace(line, line + f'\n    {variable} = steps.record("emb.{key}", {variable});')
    replace('    ggml_tensor * emb = ggml_add(ctx0, ggml_add(ctx0, tok_emb, pos_emb), type_emb);',
            '    ggml_tensor * emb = steps.record("emb.sum", ggml_add(ctx0, steps.record("native.emb.token_position", ggml_add(ctx0, tok_emb, pos_emb)), type_emb));')
    replace('    emb = ggml_norm(ctx0, emb, 1e-5f);', '    emb = steps.record("native.emb.normalized", ggml_norm(ctx0, emb, 1e-5f));')
    replace('    emb = ggml_add(ctx0, ggml_mul(ctx0, emb, ctx.emb_ln_w), ctx.emb_ln_b);',
            '    emb = steps.record("emb.output", ggml_add(ctx0, steps.record("native.emb.scaled", ggml_mul(ctx0, emb, ctx.emb_ln_w)), ctx.emb_ln_b));')
    replace('        const auto & L = ctx.layers[i];', '        const auto & L = ctx.layers[i];\n        const std::string prefix = "layer." + std::to_string(i) + ".";')
    for letter in ['Q', 'K', 'V']:
        lower = letter.lower()
        line = f'        ggml_tensor * {letter} = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_{lower}_w, cur), L.attn_{lower}_b);'
        replace(line, f'        ggml_tensor * {letter} = steps.record(prefix + "{lower}.bias", ggml_add(ctx0, steps.record(prefix + "{lower}.matmul", ggml_mul_mat(ctx0, L.attn_{lower}_w, cur)), L.attn_{lower}_b));')
    replace('            ggml_tensor * scores = ggml_mul_mat(ctx0, Kc, Qc);', '            ggml_tensor * scores = steps.record(prefix + "attention.qk", ggml_mul_mat(ctx0, Kc, Qc));')
    replace('            scores = ggml_soft_max_ext(ctx0, scores, nullptr, scale, 0.0f);', '            scores = steps.record(prefix + "attention.probabilities", ggml_soft_max_ext(ctx0, scores, nullptr, scale, 0.0f));')
    replace('        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, L.attn_out_w, KQV), L.attn_out_b);',
            '        KQV = steps.record(prefix + "attention.context", KQV);\n        cur = steps.record(prefix + "attention.bias", ggml_add(ctx0, steps.record(prefix + "attention.matmul", ggml_mul_mat(ctx0, L.attn_out_w, KQV)), L.attn_out_b));')
    # The same native expression occurs once in each sublayer; do not merge them.
    block = '        cur = ggml_add(ctx0, cur, residual);\n        cur = ggml_norm(ctx0, cur, 1e-5f);'
    if source.count(block) != 1:  # FFN uses ffn rather than cur.
        raise ValueError('Attention residual anchor changed')
    replace(block, '        cur = steps.record(prefix + "attention.residual", ggml_add(ctx0, cur, residual));\n        cur = ggml_norm(ctx0, cur, 1e-5f);')
    replace('        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, L.attn_ln_w), L.attn_ln_b);', '        cur = steps.record(prefix + "attention.output", ggml_add(ctx0, ggml_mul(ctx0, cur, L.attn_ln_w), L.attn_ln_b));')
    replace('        ggml_tensor * ffn = ggml_add(ctx0, ggml_mul_mat(ctx0, L.ffn_up_w, cur), L.ffn_up_b);', '        ggml_tensor * ffn = steps.record(prefix + "ffn.up.bias", ggml_add(ctx0, steps.record(prefix + "ffn.up.matmul", ggml_mul_mat(ctx0, L.ffn_up_w, cur)), L.ffn_up_b));')
    replace('        ffn = ggml_gelu_erf(ctx0, ffn);', '        ffn = steps.record(prefix + "ffn.gelu", ggml_gelu_erf(ctx0, ffn));')
    replace('        ffn = ggml_add(ctx0, ggml_mul_mat(ctx0, L.ffn_down_w, ffn), L.ffn_down_b);', '        ffn = steps.record(prefix + "ffn.down.bias", ggml_add(ctx0, steps.record(prefix + "ffn.down.matmul", ggml_mul_mat(ctx0, L.ffn_down_w, ffn)), L.ffn_down_b));')
    replace('        cur = ggml_add(ctx0, ffn, residual);', '        cur = steps.record(prefix + "ffn.residual", ggml_add(ctx0, ffn, residual));')
    replace('        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, L.ffn_ln_w), L.ffn_ln_b);', '        cur = steps.record(prefix + "output", ggml_add(ctx0, ggml_mul(ctx0, cur, L.ffn_ln_w), L.ffn_ln_b));')
    for head in ['post', 'pre']:
        for key, line, variable in [
            ('fc1.matmul', f'    ggml_tensor * {head}_h = ggml_mul_mat(ctx0, ctx.{head}_fc1_w, hidden);', f'{head}_h'),
            ('fc1.bias', f'    if (ctx.{head}_fc1_b) {head}_h = ggml_add(ctx0, {head}_h, ctx.{head}_fc1_b);', f'{head}_h'),
            ('relu', f'    {head}_h = ggml_relu(ctx0, {head}_h);', f'{head}_h'),
            ('fc2.matmul', f'    ggml_tensor * {head}_logits = ggml_mul_mat(ctx0, ctx.{head}_fc2_w, {head}_h);', f'{head}_logits'),
            ('output', f'    if (ctx.{head}_fc2_b) {head}_logits = ggml_add(ctx0, {head}_logits, ctx.{head}_fc2_b);', f'{head}_logits'),
        ]:
            replace(line, line + f'\n    {variable} = steps.record("head.{head}.{key}", {variable});')
    replace('            // fc1 + ReLU\n', '            inp = steps.cpu("head.sbd.input", std::move(inp), t);\n            // fc1 + ReLU\n')
    replace('                mid[j] = sum > 0 ? sum : 0; // ReLU', '                sum = steps.scalar("head.sbd.fc1.bias", t, j, sbd_mid, sum);\n                mid[j] = sum > 0 ? sum : 0; // ReLU')
    replace('            // fc2\n', '            mid = steps.cpu("head.sbd.relu", std::move(mid), t);\n            // fc2\n')
    replace('            // The ONNX model derives two different quantities from these seg logits:',
            '            logit0 = steps.scalar("head.sbd.output", t, 0, 2, logit0);\n            logit1 = steps.scalar("head.sbd.output", t, 1, 2, logit1);\n            // The ONNX model derives two different quantities from these seg logits:')
    replace('            inp[d] = (t == 0 || seg_argmax[t - 1]) ? 1.0f : 0.0f;',
            '            inp[d] = (t == 0 || seg_argmax[t - 1]) ? 1.0f : 0.0f;\n            inp = steps.cpu("head.cap.input", std::move(inp), t);')
    replace('                mid[j] = sum > 0 ? sum : 0;\n', '                sum = steps.scalar("head.cap.fc1.bias", t, j, tc_mid, sum);\n                mid[j] = sum > 0 ? sum : 0;\n')
    replace('            result.cap_preds[t].resize(tc_out);', '            mid = steps.cpu("head.cap.relu", std::move(mid), t);\n            result.cap_preds[t].resize(tc_out);')
    replace('                result.cap_preds[t][c] = logit > 0;', '                logit = steps.scalar("head.cap.output", t, c, tc_out, logit);\n                result.cap_preds[t][c] = logit > 0;')
    replace('    ggml_backend_sched_reset(ctx.sched);\n    if (!ggml_backend_sched_alloc_graph(ctx.sched, gf))', '    steps.record("input.ids", inp_ids);\n    steps.record("input.positions", pos_ids);\n    steps.record("input.types", type_ids);\n    steps.build(ctx0, gf);\n    ggml_backend_sched_reset(ctx.sched);\n    if (!ggml_backend_sched_alloc_graph(ctx.sched, gf))')
    replace('    ggml_backend_sched_graph_compute(ctx.sched, gf);', '    steps.set_inputs();\n    ggml_backend_sched_graph_compute(ctx.sched, gf);')
    replace('    ggml_free(ctx0);\n    return result;', '    steps.dump();\n    ggml_free(ctx0);\n    return result;')
    # Declare the observer only after the source includes ggml and env helpers.
    replace('#include "core/env_gate.h"', '#include "core/env_gate.h"\n#include "pcs_step_trace.h"')
    return source


def pack(directory, output):
    import numpy as np
    tensors = {}
    for line in (directory / 'manifest.tsv').read_text().splitlines():
        name, dtype, *dims = line.split('\t')
        if name in tensors or '/' in name or '..' in name:
            raise ValueError('Duplicate or unsafe tensor name')
        shape = list(reversed(list(map(int, dims))))
        while len(shape) > 1 and shape[0] == 1:
            shape.pop(0)
        array = np.fromfile(directory / (name + '.bin'), dtype='<i4' if dtype == 'i32' else '<f4')
        if dtype not in ('i32', 'f32') or array.size != np.prod(shape):
            raise ValueError('Tensor dtype/shape/bytes mismatch')
        tensors[name] = array.reshape(shape)
    write_archive(output, tensors, {'pcs.source': 'native observed graph'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'pack'])
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == 'prepare':
        args.output.write_text(overlay(args.source.read_text()))
    else:
        pack(args.source, args.output)


if __name__ == '__main__':
    main()
