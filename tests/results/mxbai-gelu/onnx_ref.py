#!/usr/bin/env python
"""ONNX Runtime reference scores for mixedbread-ai/mxbai-rerank-{xsmall,base}-v1.

Reference = the official repo's own onnx/model.onnx export (CPU EP).
Tokenization uses the repo tokenizer.json via the `tokenizers` library only
(no torch forward anywhere -- local miniconda torch mis-executes BERT-class
models on this box, see G7c).
"""
import glob
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path.insert(0, os.environ.get("MXGELU_WORK", "."))
from pairs import QUERIES  # noqa: E402

CACHE = "os.environ.get("HF_HOME", "~/.cache/huggingface") + "/models--mixedbread-ai--mxbai-rerank-%s-v1/snapshots/*/""


def run(model):
    snap = glob.glob(CACHE % model)[0]
    tok = Tokenizer.from_file(snap + "tokenizer.json")
    tok.enable_truncation(max_length=512)
    sess = ort.InferenceSession(snap + "onnx/model.onnx", providers=["CPUExecutionProvider"])
    out = {}
    for qi, (q, docs) in enumerate(QUERIES):
        scores = []
        toklens = []
        for d in docs:
            enc = tok.encode(q, d)
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
            scores.append(float(logits[0][0]))
            toklens.append(len(enc.ids))
        out["q%d" % qi] = {"query": q, "scores": scores, "toklen": toklens,
                           "ids_doc0": tok.encode(q, docs[0]).ids}
    return out


if __name__ == "__main__":
    res = {}
    for m in sys.argv[1:] or ["xsmall", "base"]:
        res[m] = run(m)
        for k, v in res[m].items():
            print(m, k, " ".join("[%d] %.6f" % (i, s) for i, s in enumerate(v["scores"])))
    json.dump(res, open(os.environ.get("MXGELU_WORK", ".") + "/onnx_ref.json", "w"), indent=1)
