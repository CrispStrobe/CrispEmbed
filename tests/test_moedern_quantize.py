#!/usr/bin/env python3
"""Cheap synthetic check of tools/moedern-colbert/quantize_gguf.py (no model, no inference):
only eligible blk.* matmul weights with 32-aligned rows are quantized, everything else
(embeddings, router, dense head, norms, unaligned rows) is copied as F32 byte-for-byte,
metadata survives, and dequantized weights stay close."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/moedern-colbert'))
import numpy as np  # noqa: E402
import gguf  # noqa: E402
from quantize_gguf import quantize  # noqa: E402

ARCH = 'modernbert_hash_moe_v10'
QUANTIZED = ('blk.0.attn_q.weight', 'blk.0.ffn_up_exps.weight')


def main():
    rng = np.random.default_rng(0)
    tensors = {
        'token_embd.hash.weight': rng.standard_normal((40, 64), dtype=np.float32),
        'token_embd.proj.weight': rng.standard_normal((64, 80), dtype=np.float32),
        'blk.0.attn_norm.weight': rng.standard_normal((64,), dtype=np.float32),
        'blk.0.attn_q.weight': rng.standard_normal((64, 64), dtype=np.float32),
        'blk.0.ffn_gate_inp.weight': rng.standard_normal((8, 64), dtype=np.float32),
        'blk.0.ffn_up_exps.weight': rng.standard_normal((8, 96, 64), dtype=np.float32),
        'blk.0.conv_dw.weight': rng.standard_normal((3, 64), dtype=np.float32),
        'blk.0.ffn_down_shexp.weight': rng.standard_normal((64, 48), dtype=np.float32),  # 48 % 32 != 0
        'dense_head.weight': rng.standard_normal((128, 64), dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / 'f32.gguf', Path(d) / 'q.gguf'
        w = gguf.GGUFWriter(str(src), arch=ARCH)
        w.add_uint32(ARCH + '.hidden_size', 64)
        w.add_string(ARCH + '.source_revision', 'abc')
        for n, a in tensors.items():
            w.add_tensor(n, a, raw_dtype=gguf.GGMLQuantizationType.F32)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        for kind, tol in (('q8_0', 0.05), ('q4_0', 0.6)):
            rep = quantize(str(src), str(dst), kind)
            assert rep['quantized'] == 2 and rep['kept_f32'] == 7, rep
            r = gguf.GGUFReader(str(dst))
            assert r.fields[ARCH + '.hidden_size'].contents() == 64
            assert r.fields[ARCH + '.source_revision'].contents() == 'abc'
            assert r.fields[ARCH + '.crispembed_quantization'].contents() == kind
            qtype = gguf.GGMLQuantizationType[kind.upper()]
            for t in r.tensors:
                ref = tensors[t.name]
                if t.name in QUANTIZED:
                    assert t.tensor_type == qtype, t.name
                    deq = gguf.quants.dequantize(np.asarray(t.data), qtype).reshape(ref.shape)
                    assert np.max(np.abs(deq - ref)) < tol, (t.name, kind)
                else:
                    assert t.tensor_type == gguf.GGMLQuantizationType.F32, t.name
                    assert np.array_equal(np.asarray(t.data).reshape(ref.shape), ref), t.name
    print('ok')


if __name__ == '__main__':
    main()
