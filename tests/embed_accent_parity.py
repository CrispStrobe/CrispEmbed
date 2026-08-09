"""End-to-end decoded-output gate for the accented-Latin tokenizer fix.

Per-stage / token-id parity is not the acceptance test (dev guide HARD RULE 3):
this runs the REAL crispembed CLI on a REAL GGUF in both arms and compares the
produced EMBEDDING against an independent reference.

Reference = ONNX Runtime on sentence-transformers/all-MiniLM-L6-v2's own onnx
export (miniconda torch is broken for BERT forwards on this Mac, so ORT is the
parity reference), mean-pooled over the attention mask and L2-normalized, which
is what the sentence-transformers module stack does for this model.

    python tests/embed_accent_parity.py ./build/crispembed all-MiniLM-L6-v2-f32.gguf

Measured 2026-08-09 (Apple M1, f32 GGUF from cstr/all-MiniLM-L6-v2-GGUF):

    ASCII     old==new bit-identical, cos vs reference 1.000000 in both arms
    ACCENTED  mean cos vs reference 0.646574 -> 1.000000
              (fr 0.487 / es 0.566 / pt 0.574 / de 0.731 / no 0.875 -> 1.000000)
"""
import json
import os
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ASCII = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models encode text into dense vectors",
    "Prices went up 15% in Q3 2024, according to the report.",
]
ACCENTED = [
    "Die Bäckerei an der Straße verkauft süße Brötchen",
    "Le garçon a déjà mangé son déjeuner à l'hôtel",
    "El niño pequeño compró una piñata en el mercado",
    "A informação está disponível na página três",
    "Øystein bor i Tromsø og går på fjellet",
]


def ref_embed(sess, tok, texts):
    out = []
    for t in texts:
        enc = tok(t, return_tensors="np")
        feed = {i.name: enc[i.name].astype(np.int64) for i in sess.get_inputs() if i.name in enc}
        hidden = sess.run(None, feed)[0]  # (1, T, H)
        mask = enc["attention_mask"].astype(np.float32)[..., None]
        v = (hidden * mask).sum(1) / np.maximum(mask.sum(1), 1e-9)
        v = v[0] / (np.linalg.norm(v[0]) + 1e-12)
        out.append(v)
    return out


def crisp_embed(binary, gguf, texts, gate):
    env = dict(os.environ, CRISPEMBED_WORDPIECE_HF_NORM=gate)
    out = []
    for t in texts:
        r = subprocess.run([binary, "-m", gguf, "--json", t],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise SystemExit(f"crispembed rc={r.returncode} on {t!r}\n{r.stderr[-1500:]}")
        # `--json` emits a JSON array: [ {"text": ..., "embedding": [...]} ]
        blob = r.stdout[r.stdout.index("["):r.stdout.rindex("]") + 1]
        v = np.array(json.loads(blob)[0]["embedding"], dtype=np.float32)
        out.append(v / (np.linalg.norm(v) + 1e-12))
    return out


def main():
    binary, gguf = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(MODEL)
    sess = ort.InferenceSession(hf_hub_download(MODEL, "onnx/model.onnx"),
                                providers=["CPUExecutionProvider"])

    fails = 0
    for name, texts in (("ASCII", ASCII), ("ACCENTED", ACCENTED)):
        ref = ref_embed(sess, tok, texts)
        old = crisp_embed(binary, gguf, texts, "0")
        new = crisp_embed(binary, gguf, texts, "1")
        print(f"\n=== {name} ===")
        print(f"{'cos(old,ref)':>13} {'cos(new,ref)':>13} {'cos(old,new)':>13}   text")
        for i, t in enumerate(texts):
            c_old = float(np.dot(old[i], ref[i]))
            c_new = float(np.dot(new[i], ref[i]))
            c_on = float(np.dot(old[i], new[i]))
            print(f"{c_old:13.6f} {c_new:13.6f} {c_on:13.6f}   {t[:46]}")
        m_old = float(np.mean([np.dot(old[i], ref[i]) for i in range(len(texts))]))
        m_new = float(np.mean([np.dot(new[i], ref[i]) for i in range(len(texts))]))
        print(f"{m_old:13.6f} {m_new:13.6f} {'':>13}   MEAN")

        if name == "ASCII":
            worst = min(float(np.dot(old[i], new[i])) for i in range(len(texts)))
            bitexact = all(np.array_equal(old[i], new[i]) for i in range(len(texts)))
            print(f"  ASCII gate: bit-identical old vs new = {bitexact}, worst cos = {worst:.8f}")
            if not bitexact:
                print("  GATE FAIL: the fix moved an ASCII embedding")
                fails += 1
        else:
            print(f"  ACCENTED gate: mean cos vs reference {m_old:.6f} -> {m_new:.6f}")
            if m_new <= m_old:
                print("  GATE FAIL: hf-norm is not closer to the reference")
                fails += 1
    return fails


if __name__ == "__main__":
    sys.exit(main())
