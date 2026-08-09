#!/usr/bin/env python
"""WordPiece token-id parity vs HuggingFace.

The ground-truth A/B for the three WordPiece parity fixes. It drives the
SHIPPING C++ tokenizer (tests/wordpiece_dump_ids.cpp links src/tokenizer.cpp)
against HF's fast tokenizer on the same real vocab, turning on one fix per arm
so each column is attributable to a single change:

    CRISPEMBED_WORDPIECE_HF_NORM     BertNormalizer strip_accents + lowercase
    CRISPEMBED_WORDPIECE_HF_PRETOK   BertPreTokenizer split (CJK, Unicode punct)
    CRISPEMBED_WORDPIECE_HF_UNK      whole-word [UNK] + max_input_chars_per_word

Gates: the ASCII section must be byte-identical in EVERY arm (all three fixes
are required to be no-ops on printable ASCII) and HF-exact, and the arms must
be monotone in HF agreement.

Measured 2026-08-09 (35 sentences: ascii, 5 European groups, cjk, unicode
punctuation, mixed):

    all-MiniLM-L6-v2    4 -> 25 -> 34 -> 35 / 35 exact
    all-mpnet-base-v2   4 -> 25 -> 34 -> 35 / 35 exact
    LaBSE (cased)      25 -> 25 -> 35 -> 35 / 35 exact

LaBSE is the control: it declares `lowercase: false`, so the accent arm must
and does leave it completely unchanged.

Usage:
    python tests/wordpiece_hf_parity.py [--build-dir build]
"""
import argparse
import os
import subprocess
import sys

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    # CASED control. LaBSE declares `lowercase: false`, so it must NOT strip
    # accents — it is the model the accent fix would break if the fix were
    # unconditional, and it shares every line of code the fixes touch.
    "sentence-transformers/LaBSE",
]

# Section -> sentences. "ascii" is the invariance gate; the rest is the bug.
CORPUS = {
    "ascii": [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models encode text into dense vectors",
        "Prices went up 15% in Q3 2024, according to the report.",
        "ALL CAPS SHOUTING AND some MiXeD case words",
    ],
    "german": [
        "Die Bäckerei an der Straße verkauft süße Brötchen",
        "Müller fährt über die Brücke nach Köln",
        "Der Fluß führt Hochwasser, die Anwohner sind besorgt",
        "Österreich und die Schweiz grenzen an Süddeutschland",
    ],
    "french": [
        "Le garçon a déjà mangé son déjeuner à l'hôtel",
        "Les élèves étudient la littérature française",
        "François préfère le café très chaud le matin",
        "Une forêt naïve près de la rivière où nous étions",
    ],
    "spanish": [
        "El niño pequeño compró una piñata en el mercado",
        "La señora está aquí con su compañero de trabajo",
        "Mañana iré a la reunión más importante del año",
        "Ángel y María viven en una región montañosa",
    ],
    "portuguese": [
        "A informação está disponível na página três",
        "Não é possível concluir a operação сem permissão",
        "O irmão dela mora em São Paulo há muitos anos",
        "As crianças estão à espera da avó",
    ],
    "nordic_slavic": [
        "Øystein bor i Tromsø og går på fjellet",
        "Łódź jest miastem w środkowej Polsce",
        "Đà Nẵng là một thành phố ven biển",
        "Þórr og Óðinn eru guðir í norrænni goðafræði",
    ],
    # The sections below are the SPLIT stage, not the accent stage. HF's
    # BertPreTokenizer gives every CJK ideograph its own word and isolates
    # Unicode punctuation; the historical per-byte loop does neither.
    "cjk": [
        "日本語のテキストを埋め込みます",
        "中文文本的向量表示",
        "猫がソファの上で眠っている。",
        "한국어 문장 임베딩",
    ],
    "uni_punct": [
        "He said “hello” — then left…",
        "The range is 10–20 items · see § 4",
        "¿Qué pasa? ¡Vamos! «salut»",
        "em—dash and en–dash and ellipsis…",
    ],
    "mixed": [
        "Tokyo東京 is 100€ per night",
        "café、パン、and bread",
        "Straße 12·A, München — 80331",
    ],
}


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("SKIP: transformers / huggingface_hub not installed", file=sys.stderr)
        return 0

    os.makedirs(args.build_dir, exist_ok=True)
    dump_bin = os.path.join(args.build_dir, "wordpiece-dump-ids")
    src = "tests/wordpiece_dump_ids.cpp"
    if not os.path.exists(dump_bin) or os.path.getmtime(src) > os.path.getmtime(dump_bin):
        print("building the id dumper ...", file=sys.stderr)
        run(["c++", "-std=c++17", "-O1", "-Isrc", src, "src/tokenizer.cpp", "-o", dump_bin])

    sections = list(CORPUS)
    lines, owner = [], []
    for sec in sections:
        for s in CORPUS[sec]:
            lines.append(s)
            owner.append(sec)
    corpus_path = os.path.join(args.build_dir, "wp_parity_corpus.txt")
    with open(corpus_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    overall_fail = 0
    for model in MODELS:
        print("=" * 74)
        print(model)
        tok = AutoTokenizer.from_pretrained(model)
        unk_id = tok.unk_token_id
        ref = [tok(s)["input_ids"] for s in lines]

        vocab_path = hf_hub_download(model, "vocab.txt")
        # MPNet wraps with <s>/</s>, BERT with [CLS]/[SEP]. Take the wrapper
        # ids from HF rather than guessing, so the comparison measures
        # tokenization and not the harness's choice of special tokens.
        special = [str(tok.cls_token_id), str(tok.sep_token_id),
                   str(tok.unk_token_id), str(tok.pad_token_id)]
        # Four arms, each turning on exactly ONE more fix, so every column is
        # attributable to a single change. "historical" sets all three gates to
        # 0 and is bit-exact with what shipped before this work.
        ARMS = (
            ("historical", {"HF_NORM": "0", "HF_PRETOK": "0", "HF_UNK": "0"}),
            ("+accents", {"HF_NORM": "1", "HF_PRETOK": "0", "HF_UNK": "0"}),
            ("+split", {"HF_NORM": "1", "HF_PRETOK": "1", "HF_UNK": "0"}),
            ("+wholeUNK", {"HF_NORM": "1", "HF_PRETOK": "1", "HF_UNK": "1"}),
        )
        ARMS = tuple((n, {f"CRISPEMBED_WORDPIECE_{k}": v for k, v in e.items()}) for n, e in ARMS)
        arms = {}
        for arm, envvars in ARMS:
            out = run([dump_bin, vocab_path, corpus_path] + special,
                      env=dict(os.environ, **envvars)).stdout.splitlines()
            arms[arm] = [[int(x) for x in ln.split()] if ln.strip() else [] for ln in out]
        names = [a for a, _ in ARMS]

        print(f"{'section':<14} {'n':>3}  " + "  ".join(f"{a:>20}" for a in names))
        print(f"{'':<14} {'':>3}  " + "  ".join(f"{'match  UNK vs HF':>20}" for _ in names))
        for sec in sections:
            idx = [i for i, o in enumerate(owner) if o == sec]
            ref_unk = sum(ref[i].count(unk_id) for i in idx)
            cells = []
            for arm in names:
                exact = sum(1 for i in idx if arms[arm][i] == ref[i])
                unk = sum(arms[arm][i].count(unk_id) for i in idx)
                cells.append(f"{exact:>2}/{len(idx):<2}  {unk:>3} vs {ref_unk:<3}")
            print(f"{sec:<14} {len(idx):>3}  " + "  ".join(f"{c:>20}" for c in cells))

        tot = {a: sum(1 for i in range(len(lines)) if arms[a][i] == ref[i]) for a in names}
        print(f"{'TOTAL':<14} {len(lines):>3}  "
              + "  ".join(f"{str(tot[a]) + '/' + str(len(lines)) + ' exact':>20}" for a in names))

        # --- Gates -----------------------------------------------------
        # The ASCII section must be byte-identical across ALL arms: both fixes
        # are required to be no-ops on printable ASCII.
        ascii_idx = [i for i, o in enumerate(owner) if o == "ascii"]
        ascii_same = all(arms["historical"][i] == arms[a][i] for i in ascii_idx for a in names)
        ascii_hf = all(arms["+wholeUNK"][i] == ref[i] for i in ascii_idx)
        if not ascii_same:
            print("  GATE FAIL: a fix changed ASCII tokenization")
            overall_fail = 1
        if not ascii_hf:
            print("  GATE FAIL: ASCII does not match HF")
            overall_fail = 1
        # Each arm must be a monotone improvement on the one before it.
        if not all(tot[names[k]] <= tot[names[k + 1]] for k in range(len(names) - 1)):
            print("  GATE FAIL: the arms are not monotone in HF agreement")
            overall_fail = 1
        if tot["+wholeUNK"] != len(lines):
            missing = [lines[i] for i in range(len(lines)) if arms["+wholeUNK"][i] != ref[i]]
            print(f"  NOTE: {len(missing)} sentence(s) still differ from HF:")
            for s in missing[:5]:
                print(f"    {s!r}")
        if ascii_same and ascii_hf:
            print(f"  gates OK: ASCII byte-identical in every arm and HF-exact; total "
                  + " -> ".join(str(tot[a]) for a in names) + f" / {len(lines)}")

    return overall_fail


if __name__ == "__main__":
    sys.exit(main())
