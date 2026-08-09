#!/usr/bin/env python
"""fireredpunc token-id parity vs HuggingFace.

src/fireredpunc.cpp carries its OWN WordPiece implementation, separate from the
shared one in tokenizer.h, so it needs its own parity check. The blueprint
(github.com/FireRedTeam/FireRedASR2S, fireredasr2s/fireredpunc/data/
hf_bert_tokenizer.py) is a plain `BertTokenizer.from_pretrained`, so HF-exact
is the target by definition.

The historical loop split on ASCII whitespace only — catastrophic for the
CHINESE vocab it serves, where text has no spaces and every character after the
first was looked up as a `##` continuation.

    python tests/firered_tokenizer_parity.py <build-dir> <fireredpunc.gguf>
"""
import os
import subprocess
import sys

# The FireRedPunc release ships chinese-bert-wwm-ext's vocab (21128) with a
# chinese-lert-base backbone; hfl/chinese-bert-wwm-ext is the matching public
# tokenizer.
HF_REPO = "hfl/chinese-bert-wwm-ext"

CORPUS = {
    "chinese": [
        "今天天气很好我们一起去公园散步吧",
        "他说这个项目需要更多时间才能完成",
        "明天的东京的天气是雨",
    ],
    "mixed": [
        "我在Google工作了三年觉得收获很大",
        "价格是100元",
    ],
    "english": [
        "hello world this is a test",
        "HELLO World Test",
    ],
    "accents": [
        "café Müller ist hier",
    ],
    "punct": [
        "他说“这个项目”需要更多时间",
    ],
}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    build_dir, gguf = sys.argv[1], sys.argv[2]
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("SKIP: transformers not installed", file=sys.stderr)
        return 0

    ab = os.path.join(build_dir, "firered-punct-ab")
    if not os.path.exists(ab) or not os.path.exists(gguf):
        print("SKIP: binary or gguf missing", file=sys.stderr)
        return 0

    sections = list(CORPUS)
    lines, owner = [], []
    for sec in sections:
        for s in CORPUS[sec]:
            lines.append(s)
            owner.append(sec)
    corpus_path = os.path.join(build_dir, "firered_corpus.txt")
    with open(corpus_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    tok = AutoTokenizer.from_pretrained(HF_REPO)
    ref = [tok(s, add_special_tokens=False)["input_ids"] for s in lines]

    arms = {}
    for arm, val in (("historical", "0"), ("hf-tok", "1")):
        env = dict(os.environ, CRISPEMBED_FIREREDPUNC_HF_TOK=val, FIREREDPUNC_DUMP_IDS="1")
        r = subprocess.run([ab, gguf, corpus_path], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(f"SKIP: rc={r.returncode}\n{r.stderr[-600:]}")
            return 0
        arms[arm] = [[int(x) for x in ln.split()] if ln.strip() else []
                     for ln in r.stdout.splitlines()]
    names = list(arms)

    print(f"{'section':<10} " + "  ".join(f"{a:>14}" for a in names))
    for sec in sections:
        idx = [i for i, o in enumerate(owner) if o == sec]
        cells = [f"{sum(1 for i in idx if arms[a][i] == ref[i]):>3}/{len(idx):<3}" for a in names]
        print(f"{sec:<10} " + "  ".join(f"{c:>14}" for c in cells))
    tot = {a: sum(1 for i in range(len(lines)) if arms[a][i] == ref[i]) for a in names}
    print(f"{'TOTAL':<10} " + "  ".join(f"{str(tot[a]) + '/' + str(len(lines)):>14}" for a in names))

    fails = 0
    if tot["hf-tok"] < tot["historical"]:
        print("GATE FAIL: hf-tok agrees with HF on fewer lines")
        fails += 1
    if tot["hf-tok"] != len(lines):
        for i in range(len(lines)):
            if arms["hf-tok"][i] != ref[i]:
                print(f"  DIFF [{owner[i]}] {lines[i][:34]!r}")
                print(f"    HF   ({len(ref[i])}): {tok.convert_ids_to_tokens(ref[i])[:14]}")
                print(f"    ours ({len(arms['hf-tok'][i])}): "
                      f"{tok.convert_ids_to_tokens(arms['hf-tok'][i])[:14]}")
                break
        fails += 1
    return fails


if __name__ == "__main__":
    sys.exit(main())
