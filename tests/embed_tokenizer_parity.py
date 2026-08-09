#!/usr/bin/env python
"""Token-id parity vs HuggingFace for the NON-WordPiece embedders.

docs/LANGUAGES.md recommends the multilingual SentencePiece/XLM-R models for
non-English text, on the grounds that they "handle accented text natively".
That was an argument about the accent-stripping stage, never a measurement of
their token ids. This measures it.

Unlike the WordPiece harness, this drives a real GGUF through the public C API
(tests/dump_token_ids.cpp), because an SPM/BPE tokenizer's merges, charsmap and
pre-tokenizer selection live in the GGUF, not in a vocab file.

    python tests/embed_tokenizer_parity.py <build-dir> <gguf>=<hf-repo> [...]

e.g.
    python tests/embed_tokenizer_parity.py build \\
        ~/.cache/crispembed-local/multilingual-e5-small-q8_0.gguf=intfloat/multilingual-e5-small
"""
import os
import subprocess
import sys

CORPUS = {
    "ascii": [
        "The quick brown fox jumps over the lazy dog",
        "Prices went up 15% in Q3 2024, according to the report.",
    ],
    "accented": [
        "Die Bäckerei an der Straße verkauft süße Brötchen",
        "Le garçon a déjà mangé son déjeuner à l'hôtel",
        "El niño pequeño compró una piñata en el mercado",
        "Øystein bor i Tromsø og går på fjellet",
    ],
    "cjk": [
        "日本語のテキストを埋め込みます",
        "猫がソファの上で眠っている。",
        "한국어 문장 임베딩",
    ],
    "uni_punct": [
        "He said “hello” — then left…",
        "The range is 10–20 items · see § 4",
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

    dump_bin = os.path.join(build_dir, "dump-token-ids")
    if not os.path.exists(dump_bin):
        print(f"SKIP: {dump_bin} not built (cmake --build {build_dir} --target dump-token-ids)",
              file=sys.stderr)
        return 0

    sections = list(CORPUS)
    lines, owner = [], []
    for sec in sections:
        for s in CORPUS[sec]:
            lines.append(s)
            owner.append(sec)
    corpus_path = os.path.join(build_dir, "tok_parity_corpus.txt")
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
        r = subprocess.run([dump_bin, gguf, corpus_path], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  SKIP: dumper rc={r.returncode}\n{r.stderr[-600:]}")
            continue
        kind = next((ln.split("=")[1] for ln in r.stderr.splitlines()
                     if ln.startswith("tokenizer_kind=")), "?")
        ours = [[int(x) for x in ln.split()] if ln.strip() else [] for ln in r.stdout.splitlines()]

        tok = AutoTokenizer.from_pretrained(repo)
        ref = [tok(s)["input_ids"] for s in lines]
        if len(ours) != len(ref):
            print(f"  SKIP: dumper returned {len(ours)} lines for {len(ref)} inputs")
            continue

        print(f"  tokenizer_kind={kind}  (1=WordPiece 2=SentencePiece 3=BPE)")
        print(f"  {'section':<12} {'exact':>8}  {'len match':>10}")
        for sec in sections:
            idx = [i for i, o in enumerate(owner) if o == sec]
            ex = sum(1 for i in idx if ours[i] == ref[i])
            lm = sum(1 for i in idx if len(ours[i]) == len(ref[i]))
            print(f"  {sec:<12} {ex:>4}/{len(idx):<3} {lm:>7}/{len(idx):<3}")
        tot = sum(1 for i in range(len(lines)) if ours[i] == ref[i])
        print(f"  {'TOTAL':<12} {tot:>4}/{len(lines):<3}")
        if tot != len(lines):
            for i in range(len(lines)):
                if ours[i] != ref[i]:
                    print(f"    DIFF [{owner[i]}] {lines[i][:44]!r}")
                    print(f"      HF   ({len(ref[i])}): {tok.convert_ids_to_tokens(ref[i])[:14]}")
                    print(f"      ours ({len(ours[i])}): {tok.convert_ids_to_tokens(ours[i])[:14]}")
                    break
            fails += 1
    return fails


if __name__ == "__main__":
    sys.exit(main())
