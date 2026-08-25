#!/usr/bin/env python
"""Dump an XLM-R punctuation reference from the HuggingFace blueprint.

The BERT half of this engine (FireRedPunc) got ground truth first, and it
immediately turned up a wrong forward pass — a stray [SEP] worth f16 cos_min
0.931. The XLM-R half had none at all. Its bug was found the same day
(SentencePiece models returned EMPTY output on the default path) and was
verified only by "the two arms now agree", which proves consistency, not
correctness: both arms could be wrong together and nothing would say so.

This closes that. The blueprint is plain `transformers`, since unlike FireRedPunc
there is no vendor harness: `AutoModelForTokenClassification` + argmax, with the
tokenizer's own special tokens. Two details are worth stating because a port can
get them wrong while every shape still fits:

  1. XLM-R DOES take `<s> … </s>`. That is the opposite of FireRedPunc, whose
     blueprint prepends [CLS] and appends nothing — which is exactly why the
     [SEP] removal had to be scoped to the BERT path rather than applied to the
     whole file. Here the tokenizer adds both, and the classifier sees them.
  2. XLM-R position ids start at `padding_idx + 1` = 2, not 0. A port that uses
     BERT's 0-based convention gets a plausible-looking but wrong result.

⚠ The reference emits logits for EVERY position including `<s>` and `</s>`.
The comparison script drops those, because the runtime only ever reports the
real tokens — comparing padded against unpadded is how a harness invents a
"first divergence" that is really an off-by-one.

Usage:
    python tools/dump_xlmr_punc_reference.py --model kredor/punctuate-all \\
        --corpus corpus.txt --output ref.txt
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    cfg = AutoConfig.from_pretrained(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForTokenClassification.from_pretrained(a.model, dtype=torch.float32)
    model.eval()
    id2label = {int(k): v for k, v in cfg.id2label.items()}
    print(f"{a.model}: L={cfg.num_hidden_layers} d={cfg.hidden_size} "
          f"vocab={cfg.vocab_size} labels={id2label}", file=sys.stderr)

    with open(a.corpus) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    with open(a.output, "w") as out:
        out.write(f"#MODEL {a.model}\n")
        out.write("#LABELS " + " ".join(f"{k}={v}" for k, v in sorted(id2label.items())) + "\n")
        for i, text in enumerate(lines):
            enc = tok(text, return_tensors="pt")
            ids = enc["input_ids"][0].tolist()
            toks = tok.convert_ids_to_tokens(ids)
            with torch.no_grad():
                logits = model(**enc).logits[0]
            preds = logits.argmax(dim=-1).tolist()

            # Mark which positions are special, so the comparator can drop them
            # rather than guessing from the count.
            special = tok.get_special_tokens_mask(ids, already_has_special_tokens=True)

            out.write(f"#LINE {i}\n#TEXT {text}\n")
            out.write("#TOKENS " + " ".join(toks) + "\n")
            out.write("#IDS " + " ".join(str(x) for x in ids) + "\n")
            out.write("#SPECIAL " + " ".join(str(x) for x in special) + "\n")
            for row in logits.tolist():
                out.write("#LOGITS " + " ".join(f"{v:.7g}" for v in row) + "\n")
            out.write("#PREDS " + " ".join(str(p) for p in preds) + "\n")
            # Reconstruction: the mark belongs to the WORD, and a word's label
            # is its LAST subtoken's — the model card's convention, and the one
            # the runtime uses. Labelling every subtoken instead turns
            # "hello world" into "hell.o. world." and "s'il vous plaît" into
            # "s'il vous pla.ît.", which looks like a broken model rather than a
            # broken reader. ▁ marks a word start in SentencePiece.
            words, labs = [], []
            for t, p in zip(toks, preds):
                if t in (tok.cls_token, tok.sep_token, tok.pad_token):
                    continue
                if t.startswith("▁") or not words:
                    words.append(t.replace("▁", ""))
                    labs.append(p)
                else:
                    words[-1] += t
                    labs[-1] = p  # last subtoken wins
            txt = " ".join(w + ("" if id2label[l] == "0" else id2label[l])
                           for w, l in zip(words, labs))
            out.write(f"#PUNC {txt.strip()}\n")
            print(f"[{i}] {txt.strip()}", file=sys.stderr)
    print(f"wrote {a.output} ({len(lines)} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
