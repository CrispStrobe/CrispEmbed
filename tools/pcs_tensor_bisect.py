#!/usr/bin/env python3
"""Attribute the PCS row37 casing logit to stored tensor groups; reports only.

Runs the pinned original ONNX graph (1-800-BAD-CODE/xlm-roberta_punctuation_
fullstop_truecase, revision d1769a59) with a SUBSET of its initializers replaced
by dequantized tensors from a stored GGUF (mapping: tools/pcs_weight_replay.py
map_stored_weights). Every other tensor keeps its original F32 value, so the
change in the `o` logit of `▁ok` (frozen case 3, token 4 incl. BOS, bit 1) is
the effect of quantizing exactly that group, in exact ONNX Runtime arithmetic,
independent of the native runtime.

    python tools/pcs_tensor_bisect.py SNAPSHOT_DIR STORED.gguf GROUP [GROUP ...]

GROUP is a comma-separated list of native tensor-name prefixes (`enc.0.`,
`head.tc.`, `emb.,enc.,head.` for everything) or NONE. With
`--alt ALT.gguf --alt-prefixes P1,P2` those prefixes are taken from ALT instead
(mixed-precision what-if). One graph session per group; run one group per
process on hosts with < 8 GB RAM.
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import sentencepiece as spm
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tests'))
from pcs_parity import parse_ref  # noqa: E402
from pcs_weight_replay import map_stored_weights  # noqa: E402

CAP_OUT = '/_decoder/_cap_head/_linears.1/Add_output_0'


def snapshot(graph):
    copy = {}
    for init in graph.graph.initializer:
        copy[init.name] = TensorProto()
        copy[init.name].CopyFrom(init)
    return copy


def names_for(manifest, prefixes):
    return {m['original'] for m in manifest if any(m['native'].startswith(p) for p in prefixes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('snapshot', type=Path)
    ap.add_argument('stored', type=Path)
    ap.add_argument('groups', nargs='+')
    ap.add_argument('--alt', type=Path)
    ap.add_argument('--alt-prefixes', default='')
    ap.add_argument('--work', type=Path, default=Path('pcs-tensor-bisect'))
    args = ap.parse_args()
    args.work.mkdir(exist_ok=True)

    graph = onnx.load(str(args.snapshot / 'model.onnx'))
    original = snapshot(graph)
    alt = None
    if args.alt:
        alt_manifest = map_stored_weights(graph, args.alt, args.work)
        alt = (snapshot(graph), names_for(alt_manifest, args.alt_prefixes.split(',')))
    manifest = map_stored_weights(graph, args.stored, args.work)
    stored = snapshot(graph)
    graph.graph.output.append(helper.make_tensor_value_info(CAP_OUT, TensorProto.FLOAT, None))

    sp = spm.SentencePieceProcessor(model_file=str(args.snapshot / 'sp.model'))
    refs, _ = parse_ref(ROOT / 'tests/regression/pcs/blueprint_ref.txt')
    ids = sp.encode(refs[3]['text'])
    if ids != refs[3]['ids']:
        raise ValueError('Original tokenizer IDs differ from the frozen reference')
    feed = {'input_ids': np.array([[sp.bos_id()] + ids + [sp.eos_id()]], dtype=np.int64)}

    for group in args.groups:
        keep = names_for(manifest, [] if group == 'NONE' else group.split(','))
        for init in graph.graph.initializer:
            src = stored if init.name in keep else original
            if alt and init.name in keep and init.name in alt[1]:
                src = alt[0]
            init.CopyFrom(src[init.name])
        session = ort.InferenceSession(graph.SerializeToString(), providers=['CPUExecutionProvider'])
        logits = session.run([CAP_OUT], feed)[0]
        del session
        gc.collect()
        print(f'{group}\trow37_o_logit={float(logits[0, 4, 1]):+.9f}', flush=True)


if __name__ == '__main__':
    main()
