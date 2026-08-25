#!/usr/bin/env python
"""Dump a PCS reference from the model's own driving code.

PCS (punctuation + capitalisation + sentence boundary) is the third punctuation
engine and had no ground truth: four registry entries, four classification
heads, and nothing checking any of them. The other two engines each turned up a
real bug the day they got a reference.

The blueprint is NOT reimplemented here. `1-800-BAD-CODE/…_punctuation_fullstop_truecase`
ships `pipeline.py`, which delegates to `punctuators.models.PunctCapSegModelONNX`,
so that package IS the driving code and this runs it (dev guide HARD RULE 13:
read and run the code that actually runs the model, don't reconstruct how it
"could" be used). Install with `pip install punctuators`.

Two details taken from that code rather than guessed:

  1. The ONNX graph exports ARGMAXED predictions, not logits — `pre_preds`,
     `post_preds` (int64) and `cap_preds`, `seg_preds` (bool). So there is no
     cosine to compute here; the comparison is on discrete decisions, which is
     the stronger acceptance gate anyway since they are what produce the text.
  2. BOS/EOS are stripped by the caller — `input_ids[i, 1 : length - 1]` — so
     the per-token arrays line up with the real tokens only after that slice.
     Dumping the padded arrays and letting a comparator guess is how an
     off-by-one gets reported as a first divergence.

`cap_preds` is [T, 16]: one bool per character position within the token, not a
single per-token flag. It is emitted as a bitstring per token so nothing has to
be inferred about its shape downstream.

Usage:
    PYTHONPATH=<punctuators> python tools/dump_pcs_reference.py \\
        --model-dir /path/to/pcs-onnx --corpus corpus.txt --output ref.txt
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="dir holding model.onnx, sp.model, config.yaml")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    import sentencepiece as spm
    import yaml

    cfg = yaml.safe_load(open(f"{a.model_dir}/config.yaml"))
    pre_labels, post_labels = cfg["pre_labels"], cfg["post_labels"]
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{a.model_dir}/sp.model")
    sess = ort.InferenceSession(f"{a.model_dir}/model.onnx",
                                providers=["CPUExecutionProvider"])
    print(f"post_labels={post_labels}", file=sys.stderr)

    # The blueprint's own text -> the package, so the decoded line is the real
    # one rather than a re-derivation of the collector's rules.
    produce = None
    try:
        from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX, PunctCapSegModelONNX
        m = PunctCapSegModelONNX(PunctCapSegConfigONNX(
            directory=a.model_dir, spe_filename="sp.model",
            model_filename="model.onnx", config_filename="config.yaml"))
        produce = lambda t: m.infer([t])[0]  # noqa: E731
        print("blueprint reconstruction: punctuators package", file=sys.stderr)
    except Exception as e:
        print(f"blueprint reconstruction unavailable ({type(e).__name__}); "
              f"dumping per-token predictions only", file=sys.stderr)

    with open(a.corpus) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    with open(a.output, "w") as out:
        out.write("#POST_LABELS " + "\t".join(post_labels) + "\n")
        out.write("#PRE_LABELS " + "\t".join(pre_labels) + "\n")
        for i, text in enumerate(lines):
            ids = sp.EncodeAsIds(text)
            # bos/eos exactly as the dataset builder adds them.
            full = [sp.bos_id()] + ids + [sp.eos_id()]
            arr = np.array([full], dtype=np.int64)
            pre_p, post_p, cap_p, seg_p = sess.run(None, {"input_ids": arr})
            sl = slice(1, len(full) - 1)  # strip BOS/EOS, as the blueprint does
            pre_s = pre_p[0][sl].tolist()
            post_s = post_p[0][sl].tolist()
            cap_s = cap_p[0][sl].tolist()
            seg_s = seg_p[0][sl].tolist()

            out.write(f"#LINE {i}\n#TEXT {text}\n")
            out.write("#TOKENS " + " ".join(sp.IdToPiece(t) for t in ids) + "\n")
            out.write("#IDS " + " ".join(str(t) for t in ids) + "\n")
            out.write("#PRE " + " ".join(str(x) for x in pre_s) + "\n")
            out.write("#POST " + " ".join(str(x) for x in post_s) + "\n")
            out.write("#SEG " + " ".join("1" if x else "0" for x in seg_s) + "\n")
            out.write("#CAP " + " ".join("".join("1" if b else "0" for b in row)
                                         for row in cap_s) + "\n")
            if produce is not None:
                sents = produce(text)
                out.write("#PUNC " + " ".join(sents) + "\n")
                print(f"[{i}] {' '.join(sents)}", file=sys.stderr)
    print(f"wrote {a.output} ({len(lines)} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
