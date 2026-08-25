#!/usr/bin/env python
"""Dump a FireRedPunc reference from the Python blueprint.

Neither CrispEmbed nor CrispASR had ground truth for this engine: the tokenizer
was checked against HuggingFace (tests/firered_tokenizer_parity.py), but nothing
checked the MODEL. A tokenizer can be exact and the port still wrong.

Blueprint (read line by line, per the dev guide's HARD RULE 13, not summarised):
  github.com/FireRedTeam/FireRedASR2S
    fireredasr2s/fireredpunc/models/fireredpunc_bert.py :: FireRedPuncBert._forward
    fireredasr2s/fireredpunc/punc.py                    :: ModelIO, get_punc_pred

The three details that the SHAPES still allow, i.e. the ones worth stating:

  1. `_forward` prepends [CLS] and NO [SEP]:
         padded_inputs, lengths = self.add_cls(padded_inputs, lengths)
     then DROPS the [CLS] output before the classifier:
         sequence_output = outputs[0][:, 1:]
     so predictions are 1:1 with the input tokens. A port that also appends
     [SEP] gives every token one extra position to attend to — BERT is
     bidirectional, so that is not a no-op, and it is invisible in any shape.
  2. `token_type_ids` are all zero and position ids start at 0 (plain BERT,
     not the RoBERTa padding_idx+1 convention).
  3. The default `sentence_max_length <= 0` path is a plain argmax over the
     logits — no Viterbi, no length-limited segmentation.

Output is a flat text file, one record per input line, so the comparison script
needs no torch:

    #LINE <i>
    #TEXT <original>
    #TOKENS <tok> <tok> ...
    #IDS <id> <id> ...
    #LOGITS <c0> <c1> <c2> <c3> <c4>      (one line per token)
    #PREDS <p> <p> ...
    #PUNC <blueprint punctuated text>

Usage:
    python tools/dump_fireredpunc_reference.py --model-dir /path/to/FireRedPunc \
        --corpus corpus.txt --output ref.txt
"""
import argparse
import os
import re
import sys


def build_model(model_dir):
    import torch
    import transformers

    pkg = torch.load(os.path.join(model_dir, "model.pth.tar"),
                     map_location="cpu", weights_only=False)
    sd = pkg["model_state_dict"]
    args = pkg["args"]

    # Derive the config from the checkpoint's own tensor shapes rather than
    # trusting a config.json: the state dict is complete (199 keys covering the
    # whole encoder plus the classifier), so it is the authority, and this also
    # avoids pulling the 411 MB chinese-lert-base backbone just to be
    # immediately overwritten by load_state_dict.
    vocab, hidden = sd["bert.embeddings.word_embeddings.weight"].shape
    n_pos = sd["bert.embeddings.position_embeddings.weight"].shape[0]
    n_type = sd["bert.embeddings.token_type_embeddings.weight"].shape[0]
    n_layer = max(int(k.split(".")[3]) for k in sd
                  if k.startswith("bert.encoder.layer.")) + 1
    d_ffn = sd["bert.encoder.layer.0.intermediate.dense.weight"].shape[0]
    odim = sd["classifier.weight"].shape[0]
    cfg = transformers.BertConfig(
        vocab_size=vocab, hidden_size=hidden, num_hidden_layers=n_layer,
        num_attention_heads=hidden // 64, intermediate_size=d_ffn,
        max_position_embeddings=n_pos, type_vocab_size=n_type,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    print(f"config from checkpoint: vocab={vocab} d={hidden} L={n_layer} "
          f"ffn={d_ffn} n_pos={n_pos} odim={odim} cls_id={getattr(args,'cls_id',101)}",
          file=sys.stderr)

    bert = transformers.BertModel(cfg, add_pooling_layer=False)
    classifier = torch.nn.Linear(hidden, odim)
    missing, unexpected = bert.load_state_dict(
        {k[len("bert."):]: v for k, v in sd.items() if k.startswith("bert.")},
        strict=False)
    # position_ids is a non-persistent buffer in recent transformers; anything
    # else missing means the checkpoint and this config disagree, which would
    # silently leave randomly-initialised weights in the graph.
    real_missing = [m for m in missing if "position_ids" not in m]
    assert not real_missing, f"missing bert weights: {real_missing[:6]}"
    assert not [u for u in unexpected if "position_ids" not in u], unexpected[:6]
    classifier.load_state_dict({"weight": sd["classifier.weight"],
                                "bias": sd["classifier.bias"]})
    bert.eval()
    classifier.eval()
    return bert, classifier, int(getattr(args, "cls_id", 101))


class RuleBasedTxtFix:
    """Verbatim port of punc.py :: RuleBaedTxtFix.fix (upstream spelling kept).

    Note it LOWERCASES the whole string first and then selectively
    recapitalises. CrispEmbed deliberately does not do that — it emits the
    user's original words — so decoded-text differences in CASE are expected
    and are not a port bug. The logits and preds are the parity gate.
    """

    @classmethod
    def fix(cls, txt_ori, capitalize_first=True):
        txt = txt_ori.lower()
        for mark, ascii_ in (("，", ","), ("。", "."), ("？", "?"), ("！", "!")):
            txt = re.sub(rf"([a-z]){mark}([a-z])", rf"\1{ascii_} \2", txt)
            txt = re.sub(rf"^([a-z]+){mark}", rf"\1{ascii_}", txt)
            txt = re.sub(rf"( [a-zA-Z']+){mark}$", rf"\1{ascii_}", txt)
        for pat, rep in (("^i ", "I "), ("^i'm ", "I'm "), ("^i'd ", "I'd "),
                         ("^i've ", "I've "), ("^i'll ", "I'll "),
                         (" i ", " I "), (" i'm ", " I'm "), (" i'd ", " I'd "),
                         (" i've ", " I've "), (" i'll ", " I'll ")):
            txt = re.sub(pat, rep, txt)
        if capitalize_first and len(txt) > 0 and re.match("[a-z]", txt[0]):
            txt = txt[0].upper() + txt[1:]
        txt = re.sub(r"([.!?。？！])\s+([a-z])",
                     lambda m: f"{m.group(1)} {m.group(2).upper()}", txt)
        return txt


def add_punc_to_txt(tokens, preds, out_dict):
    """Verbatim port of punc.py :: ModelIO.add_punc_to_txt."""
    DEFAULT_OUT = " "
    txt = ""
    for i, token in enumerate(tokens):
        tag = out_dict[preds[i]]
        if token.startswith("##"):
            token = token.replace("##", "")
        elif re.search("[a-zA-Z0-9#]+", token) and i > 0 and \
                re.search("[a-zA-Z0-9#]+", tokens[i - 1]):
            if out_dict[preds[i - 1]] == DEFAULT_OUT:
                token = " " + token
        txt += token if tag == DEFAULT_OUT else token + tag
    return txt.replace("  ", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    import torch
    from transformers import BertTokenizer

    tok_dir = os.path.join(a.model_dir, "chinese-lert-base")
    tokenizer = BertTokenizer.from_pretrained(tok_dir)
    # out_dict: "<space> 0 / ， 1 / 。 2 / ？ 3 / ！ 4"; <space> means literal " ".
    out_dict = {}
    with open(os.path.join(a.model_dir, "out_dict")) as f:
        for line in f:
            if not line.strip():
                continue
            sym, idx = line.rstrip("\n").rsplit(" ", 1)
            out_dict[int(idx)] = " " if sym == "<space>" else sym

    bert, classifier, cls_id = build_model(a.model_dir)

    with open(a.corpus) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    with open(a.output, "w") as out:
        for i, text in enumerate(lines):
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            # _forward: [CLS] + tokens, no [SEP]; all-ones mask; drop CLS output.
            inp = torch.tensor([[cls_id] + ids])
            mask = torch.ones_like(inp)
            with torch.no_grad():
                seq = bert(inp, attention_mask=mask)[0][:, 1:]
                logits = classifier(seq)[0]
            preds = logits.argmax(dim=-1).tolist()
            punc = RuleBasedTxtFix.fix(add_punc_to_txt(tokens, preds, out_dict))

            out.write(f"#LINE {i}\n#TEXT {text}\n")
            out.write("#TOKENS " + " ".join(tokens) + "\n")
            out.write("#IDS " + " ".join(str(x) for x in ids) + "\n")
            for row in logits.tolist():
                out.write("#LOGITS " + " ".join(f"{v:.7g}" for v in row) + "\n")
            out.write("#PREDS " + " ".join(str(p) for p in preds) + "\n")
            out.write(f"#PUNC {punc}\n")
            print(f"[{i}] {punc}", file=sys.stderr)
    print(f"wrote {a.output} ({len(lines)} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
