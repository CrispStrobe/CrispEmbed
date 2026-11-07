"""Operator-only scalar dtype ablations; accepts native exported Q/K/V."""
import ctypes
import ctypes.util
from pathlib import Path
import numpy as np

_libm = ctypes.CDLL(ctypes.util.find_library('m'))
_libm.expf.argtypes = [ctypes.c_float]
_libm.expf.restype = ctypes.c_float


def expf(x):
    return np.float32(_libm.expf(float(x)))


def attention(q, k, v, gate, cache_f16=False, query_f16=False, probability_f16=False):
    n = q.shape[0]
    if q.shape != (n, 16, 256) or k.shape != (n, 4, 256) or v.shape != k.shape or gate.shape != q.shape:
        raise ValueError('geometry')
    if n not in (4, 16) or not all(np.isfinite(x).all() for x in (q, k, v, gate)):
        raise ValueError('scope/nonfinite')
    if cache_f16:
        k, v = (x.astype(np.float16).astype(np.float32) for x in (k, v))
    if query_f16:
        q = q.astype(np.float16).astype(np.float32)
    logits = np.zeros((16, n, 256), dtype='<f4')
    probs = np.zeros_like(logits)
    context = np.zeros_like(q)
    gated = np.zeros_like(q)
    for t in range(n):
        for h in range(16):
            scores = []
            for s in range(t + 1):
                dot = np.float32(0)
                for i in range(256):
                    dot = np.float32(dot + np.float32(q[t, h, i] * k[s, h // 4, i]))
                scores.append(np.float32(dot * np.float32(.0625)))
            logits[h, t, :t + 1] = scores
            maximum = max(scores)
            values = [expf(np.float32(x - maximum)) for x in scores]
            total = np.float32(0)
            for x in values:
                total = np.float32(total + x)
            for s, x in enumerate(values):
                p = np.float32(x / total)
                probs[h, t, s] = np.float32(np.float16(p)) if probability_f16 else p
            for i in range(256):
                value = np.float32(0)
                for s in range(t + 1):
                    value = np.float32(value + np.float32(probs[h, t, s] * v[s, h // 4, i]))
                context[t, h, i] = value
                gated[t, h, i] = np.float32(value / np.float32(np.float32(1) + expf(-gate[t, h, i])))
    return {'logits': logits, 'probs': probs, 'context': context, 'gated': gated}


def run(root, output):
    root, output = Path(root), Path(output)
    arrays = {name: np.fromfile(root / (name + '.f32'), dtype='<f4').reshape(16, heads, 256)
              for name, heads in [('q', 16), ('k', 4), ('v', 4), ('gate', 16)]}
    original = np.fromfile(root / 'pooled.f32', dtype='<f4').reshape(16, 16, 256)
    for n in (4, 16):
        for mode, flags in [('f32', (False, False, False)), ('kv_f16', (True, False, False)),
                            ('kvq_f16', (True, True, False)), ('kvqp_f16', (True, True, True))]:
            stages = attention(*(arrays[k][:n] for k in ('q', 'k', 'v', 'gate')), *flags)
            if mode == 'f32' and stages['gated'].tobytes() != original[:n].tobytes():
                error = np.max(np.abs(stages['gated'].astype('f8') - original[:n]))
                raise ValueError(f'actual scalar context byte closure FAILED n={n} max={error}')
            dest = output / str(n) / mode
            dest.mkdir(parents=True, exist_ok=True)
            for name, data in stages.items():
                data.tofile(dest / (name + '.f32'))
    return True
