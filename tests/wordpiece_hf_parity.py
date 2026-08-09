#!/usr/bin/env python
"""WordPiece token-id parity vs HuggingFace, for uncased 30k models.

The ground-truth A/B for the accented-Latin tokenizer fix. It drives the
SHIPPING C++ tokenizer (tests/wordpiece_dump_ids.cpp links src/tokenizer.cpp)
against HF's fast tokenizer on the same real vocab, in both arms:

    CRISPEMBED_WORDPIECE_HF_NORM=0   historical per-byte lowercase (pre-fix)
    CRISPEMBED_WORDPIECE_HF_NORM=1   HF BertNormalizer strip_accents+lowercase

and reports, per corpus section, exact sequence match and [UNK] count against
HF. The ASCII section is the safety gate: both arms must be 100%, because the
fix is required to be a no-op on ASCII.

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
        arms = {}
        for arm, val in (("historical", "0"), ("hf-norm", "1")):
            env = dict(os.environ, CRISPEMBED_WORDPIECE_HF_NORM=val)
            out = run([dump_bin, vocab_path, corpus_path] + special, env=env).stdout.splitlines()
            arms[arm] = [[int(x) for x in ln.split()] if ln.strip() else [] for ln in out]

        print(f"{'section':<14} {'n':>3}  {'historical':>22}  {'hf-norm':>22}")
        print(f"{'':<14} {'':>3}  {'match   UNK vs HF UNK':>22}  {'match   UNK vs HF UNK':>22}")
        for sec in sections:
            idx = [i for i, o in enumerate(owner) if o == sec]
            cells = []
            for arm in ("historical", "hf-norm"):
                exact = sum(1 for i in idx if arms[arm][i] == ref[i])
                unk = sum(arms[arm][i].count(unk_id) for i in idx)
                ref_unk = sum(ref[i].count(unk_id) for i in idx)
                cells.append(f"{exact:>2}/{len(idx):<2}   {unk:>3} vs {ref_unk:<3}")
            print(f"{sec:<14} {len(idx):>3}  {cells[0]:>22}  {cells[1]:>22}")

        tot = {a: sum(1 for i in range(len(lines)) if arms[a][i] == ref[i]) for a in arms}
        print(f"{'TOTAL':<14} {len(lines):>3}  {tot['historical']:>2}/{len(lines):<2} exact"
              f"{'':<14}{tot['hf-norm']:>2}/{len(lines):<2} exact")

        # --- Gates -----------------------------------------------------
        ascii_idx = [i for i, o in enumerate(owner) if o == "ascii"]
        ascii_same = all(arms["historical"][i] == arms["hf-norm"][i] for i in ascii_idx)
        ascii_hf = all(arms["hf-norm"][i] == ref[i] for i in ascii_idx)
        if not ascii_same:
            print("  GATE FAIL: the fix changed ASCII tokenization")
            overall_fail = 1
        if not ascii_hf:
            print("  GATE FAIL: ASCII does not match HF")
            overall_fail = 1
        if tot["hf-norm"] < tot["historical"]:
            print("  GATE FAIL: hf-norm matches HF on FEWER sentences than the historical path")
            overall_fail = 1
        if ascii_same and ascii_hf and tot["hf-norm"] >= tot["historical"]:
            print(f"  gates OK: ASCII unchanged and HF-exact; "
                  f"non-ASCII exact match {tot['historical']}/{len(lines)} -> {tot['hf-norm']}/{len(lines)}")

    return overall_fail


if __name__ == "__main__":
    sys.exit(main())
