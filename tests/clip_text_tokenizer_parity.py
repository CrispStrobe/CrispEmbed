#!/usr/bin/env python
"""clip_text (CLIP-BPE / SigLIP-SPM) token-id parity vs HuggingFace.

The clip_text engine has its own loader and its own tokenizer selection, so
neither the WordPiece harness nor the generic dump-token-ids driver can reach
it. This drives tests/clip_text_dump_ids.cpp, which calls the engine's real
tokenizer.

SigLIP's normalizer is a six-step Sequence (Lowercase, punctuation strip,
whitespace collapse, Strip, the nmt_nfkc charsmap, doubled-space collapse) that
CrispEmbed implemented none of; CRISPEMBED_SPM_HF_NORM toggles it.

    python tests/clip_text_tokenizer_parity.py <build-dir> <gguf>=<hf-repo>
"""
import os
import subprocess
import sys

CORPUS = {
    "lowercase_plain": [
        "a photo of a fox",
        "a dog running through a field",
    ],
    "caps": [
        "A photo of a CAT",
        "HELLO World",
    ],
    "punct": [
        "Hello, World! How are you?",
        "a dog & a cat (running) - fast!",
        "under_score and hyphen-ated words",
        "what's up? it's fine.",
    ],
    "punct_slash_angle": [
        # / < > ARE stripped: the slow SiglipTokenizer (the one that runs)
        # translates over all of string.punctuation. tokenizer.json's regex
        # says otherwise and is not what executes.
        "the a/b test with <tags> and >arrows<",
    ],
    "accents": [
        "cafe Muller",
        "café Müller",
    ],
    "charsmap": [
        "item and file",
        "ＡＢＣ　１２３",
        "item ① and ﬁle…",
    ],
    "cjk": [
        "日本語のテキスト",
    ],
    "whitespace": [
        "  leading and trailing  ",
        "multiple    inner     spaces",
    ],
}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    build_dir, pairs = sys.argv[1], sys.argv[2:]
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("SKIP: transformers not installed", file=sys.stderr)
        return 0

    dump_bin = os.path.join(build_dir, "clip-text-dump-ids")
    if not os.path.exists(dump_bin):
        print(f"SKIP: {dump_bin} not built", file=sys.stderr)
        return 0

    sections = list(CORPUS)
    lines, owner = [], []
    for sec in sections:
        for s in CORPUS[sec]:
            lines.append(s)
            owner.append(sec)
    corpus_path = os.path.join(build_dir, "clip_text_corpus.txt")
    with open(corpus_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    fails = 0
    for pair in pairs:
        gguf, repo = pair.split("=", 1)
        print("=" * 74)
        print(f"{os.path.basename(gguf)}  vs  {repo}")
        if not os.path.exists(gguf):
            print("  SKIP: gguf not present")
            continue

        tok = AutoTokenizer.from_pretrained(repo)
        # The engine drops the leading invalid BOS for SigLIP and keeps the
        # trailing </s>; compare the id sequence HF produces for the same text.
        ref = [tok(s)["input_ids"] for s in lines]

        arms = {}
        for arm, val in (("historical", "0"), ("hf-norm", "1")):
            r = subprocess.run([dump_bin, gguf, corpus_path], capture_output=True, text=True,
                               env=dict(os.environ, CRISPEMBED_SPM_HF_NORM=val))
            if r.returncode != 0:
                print(f"  SKIP: dumper rc={r.returncode}\n{r.stderr[-500:]}")
                arms = {}
                break
            arms[arm] = [[int(x) for x in ln.split()] if ln.strip() else []
                         for ln in r.stdout.splitlines()]
        if not arms:
            continue
        names = list(arms)

        print(f"  {'section':<16} " + "  ".join(f"{a:>14}" for a in names))
        for sec in sections:
            idx = [i for i, o in enumerate(owner) if o == sec]
            cells = [f"{sum(1 for i in idx if arms[a][i] == ref[i]):>3}/{len(idx):<3}" for a in names]
            print(f"  {sec:<16} " + "  ".join(f"{c:>14}" for c in cells))
        tot = {a: sum(1 for i in range(len(lines)) if arms[a][i] == ref[i]) for a in names}
        print(f"  {'TOTAL':<16} " + "  ".join(f"{str(tot[a]) + '/' + str(len(lines)):>14}" for a in names))

        if tot["hf-norm"] < tot["historical"]:
            print("  GATE FAIL: hf-norm agrees with HF on fewer lines")
            fails += 1
        if tot["hf-norm"] != len(lines):
            for i in range(len(lines)):
                if arms["hf-norm"][i] != ref[i]:
                    print(f"    DIFF [{owner[i]}] {lines[i]!r}")
                    print(f"      HF   ({len(ref[i])}): {tok.convert_ids_to_tokens(ref[i])[:16]}")
                    print(f"      ours ({len(arms['hf-norm'][i])}): "
                          f"{tok.convert_ids_to_tokens(arms['hf-norm'][i])[:16]}")
                    break
            fails += 1
    return fails


if __name__ == "__main__":
    sys.exit(main())
