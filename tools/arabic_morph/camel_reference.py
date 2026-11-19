#!/usr/bin/env python3
"""camel_tools reference and exactness comparison for the native Camel Morph analyzer.

Developer tooling only (evaluation; nothing here ships). Requires `camel-tools`
(MIT) in the running interpreter. The analyzer is camel_tools'
`Analyzer(MorphologyDB(db, 'a'), backoff=...)`, its documented defaults
(backoff NONE, no cache, strict_digit False) unless --backoff is given.

    python tools/arabic_morph/camel_reference.py words --conllu ar_pud-ud-test.conllu --out words.jsonl
    python tools/arabic_morph/camel_reference.py analyze --db full/camel_morph_msa_v1.0.db \
        --words words.jsonl --backoff NONE --out ref.jsonl
    python tools/arabic_morph/camel_reference.py compare --ref ref.jsonl --native native.jsonl
    python tools/arabic_morph/camel_reference.py rank --full ref-full.jsonl --noprob ref-noprob.jsonl \
        --conllu ar_pud-ud-test.conllu

Word and result files are JSON lines; each word line is one JSON string.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

# Hand-picked stress inputs: diacritics, tatweel, hamza/alef variants,
# clitic-heavy forms, Latin, digits, punctuation, whitespace and empty input.
STRESS = [
    "", " ", "\t", " ", " كتاب ", "　كتاب",
    "كَتَبَ", "وَالكِتَابُ", "مُدَرِّسَةٌ", "كِتابٌ", "كتاباً", "كتابًا", "شُكْراً", "قَرَأَتْ", "ٱلْكِتَابُ",
    "كـتـاب", "كتابـ", "ـكتاب", "ـ", "ـــ", "َ", "ًا", "ّ", "ٰ",
    "أحمد", "احمد", "إحمد", "آمن", "امن", "ٱلكتاب", "مسؤول", "مسئول", "مسءول", "سؤال", "سءال", "شيء", "شيئا",
    "على", "علي", "مدرسة", "مدرسه", "إلى", "الى", "أن", "ان", "إن", "قرأ", "قرا", "يقرؤون", "يقرءون",
    "وبكتابهم", "فسيكتبونها", "أفبالكتاب", "وللمدرسين", "ولكنهم", "فأسقيناكموه", "وسيعلمونهم", "بالمدرستين",
    "لِلْمُدَرِّسِينَ", "وَبِكِتَابِهِمْ", "كالكتاب", "فالكتاب", "أبالكتاب", "ولمعلماتهن", "سيكتبونه",
    "hello", "CrispEmbed", "café", "NOAN", "noan", "Straße", "ΑΘΗΝΑ", "москва", "東京",
    "2024", "٢٠٢٤", "abc123", "كتاب2", "3.14", "١٢٫٥", "۱۲۳", "٣كتب", "0",
    ".", "،", "؟", "!!", "«", "»", "...", "—", "(", ")", "\"", "%", "$", "😀", "+", "٪",
    "كتاب.", "الكتاب،", "«الكتاب»", "U.S.A", "e-mail", "a\\b", "ك\\ت", "ك\\\\ت", "ك\\nت", "ك\\gت",
    "ك\\0ت", "ك\\1ت", "\\", "ك-ت", "و-", "-و", "پيتزا", "چاي", "ڤيديو", "گرام", "كتاب‍ي",
    "كتاب‌", "a\u0000b", "كتاب\nكتاب", "1\n2", "كتاب كتاب", "ﻻ", "لا", "ﷲ", "الله", "لله", "ﻛﺘﺎﺏ",
    "ي", "و", "ف", "ب", "ل", "ك", "س", "ه", "ها", "هم", "كتب", "يكتب", "اكتب", "مكتوب", "مكاتب",
    "ءءءءءءءءءءءءءءءءءءءء", "والللللللللللللللللللللللل", "وفبلكالكتابهما",
]


def load_words(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_words(args):
    from camel_tools.tokenizers.word import simple_word_tokenize

    seen = {}
    texts = []
    with open(args.conllu, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# text = "):
                texts.append(line[len("# text = "):].rstrip("\n"))
    pud = []
    for text in texts:
        # Both camel_tools' standard tokenization and raw whitespace tokens
        # (the latter keep attached punctuation and exercise _has_punc).
        for w in simple_word_tokenize(text) + text.split():
            if w not in seen:
                seen[w] = True
                pud.append(w)
    stress = [w for w in STRESS if w not in seen]
    with open(args.out, "w", encoding="utf-8") as f:
        for w in pud + stress:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    print(json.dumps({"texts": len(texts), "pud_unique_words": len(pud), "stress_words": len(stress),
                      "total": len(pud) + len(stress)}))


def to_pairs(analysis):
    out = []
    for k, v in analysis.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("non-finite value")
        out.append([k, v])
    return out


def cmd_analyze(args):
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.morphology.database import MorphologyDB

    t0 = time.perf_counter()
    db = MorphologyDB(args.db, "a")
    analyzer = Analyzer(db, backoff=args.backoff)
    load_s = time.perf_counter() - t0
    words = load_words(args.words)
    t1 = time.perf_counter()
    n_an = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, w in enumerate(words):
            try:
                rec = {"i": i, "ok": True, "analyses": [to_pairs(a) for a in analyzer.analyze(w)]}
                n_an += len(rec["analyses"])
            except Exception as e:  # camel_tools raises for some inputs; record the type
                rec = {"i": i, "ok": False, "error": type(e).__name__ + ": " + str(e)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    secs = time.perf_counter() - t1
    print(json.dumps({"load_s": round(load_s, 3), "words": len(words), "analyses": n_an,
                      "analyze_s": round(secs, 3), "words_per_s": round(len(words) / secs, 1)}))


# merge_features appends these keys in frozenset iteration order, which depends
# on PYTHONHASHSEED; camel_tools defines no order inside a group.
SET_GROUPS = [
    {"gloss", "bw"},
    {"diac", "pattern", "caphi", "catib6", "ud"},
    {"d3tok", "d3seg", "atbseg", "d2seg", "d1seg", "d1tok", "d2tok", "atbtok", "bwtok"},
    {"pos_logprob", "lex_logprob", "pos_lex_logprob"},
    {"gloss", "atbtok", "atbseg", "d1tok", "d1seg", "d2tok", "d2seg", "d3tok", "d3seg", "bwtok"},
    {"root", "pattern", "caphi"},
]


def canonical_keys(keys):
    """Sort each maximal run of keys that belongs to one frozenset group."""
    out = list(keys)
    i = 0
    while i < len(out):
        best = None
        for g in SET_GROUPS:
            j = i
            while j < len(out) and out[j] in g:
                j += 1
            if j - i > 1 and (best is None or j > best):
                best = j
        if best is None:
            i += 1
        else:
            out[i:best] = sorted(out[i:best])
            i = best
    return out


def same_value(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, float) and isinstance(b, float) and a == b
    return a == b


def cmd_compare(args):
    ref = [json.loads(line) for line in open(args.ref, encoding="utf-8")]
    nat = [json.loads(line) for line in open(args.native, encoding="utf-8")]
    stats = {"words": len(ref), "word_exact": 0, "word_exact_ordered": 0, "errors_both": 0,
             "analyses": 0, "analyses_equal": 0, "features": 0, "set_group_reordered_analyses": 0,
             "mismatch_words": 0}
    shown = 0
    if len(ref) != len(nat):
        sys.exit(f"length mismatch {len(ref)} vs {len(nat)}")
    for r, n in zip(ref, nat):
        if not r["ok"]:
            ok = not n["ok"]
            stats["errors_both"] += ok
            stats["word_exact"] += ok
            stats["word_exact_ordered"] += ok
            if not ok:
                stats["mismatch_words"] += 1
            continue
        if not n["ok"]:
            stats["mismatch_words"] += 1
            continue
        ra, na = r["analyses"], n["analyses"]
        stats["analyses"] += len(ra)
        equal = len(ra) == len(na)
        exact_order = equal
        for x, y in zip(ra, na):
            dx, dy = dict(x), dict(y)
            stats["features"] += len(dx)
            same = (dx.keys() == dy.keys() and all(same_value(dx[k], dy[k]) for k in dx)
                    and len(x) == len(dx) and len(y) == len(dy))
            kx, ky = [k for k, _ in x], [k for k, _ in y]
            same = same and canonical_keys(kx) == canonical_keys(ky)
            if kx != ky:
                exact_order = False
                stats["set_group_reordered_analyses"] += 1
            stats["analyses_equal"] += same
            equal = equal and same
        stats["word_exact"] += equal
        stats["word_exact_ordered"] += equal and exact_order
        if not equal:
            stats["mismatch_words"] += 1
            if shown < args.show:
                shown += 1
                print("MISMATCH", r["i"], len(ra), len(na), file=sys.stderr)
                for x, y in zip(ra, na):
                    if x != y:
                        dx, dy = dict(x), dict(y)
                        diff = {k: (dx.get(k), dy.get(k)) for k in set(dx) | set(dy)
                                if not same_value(dx.get(k), dy.get(k))}
                        print("  first differing analysis:", diff,
                              canonical_keys([k for k, _ in x]) == canonical_keys([k for k, _ in y]),
                              file=sys.stderr)
                        break
    print(json.dumps(stats))
    if stats["mismatch_words"]:
        sys.exit(1)


UPOS = {"noun": "NOUN", "noun_prop": "PROPN", "noun_num": "NUM", "noun_quant": "NOUN", "adj": "ADJ",
        "adj_comp": "ADJ", "adj_num": "ADJ", "verb": "VERB", "verb_pseudo": "VERB", "adv": "ADV",
        "adv_interrog": "ADV", "adv_rel": "ADV", "prep": "ADP", "conj": "CCONJ", "conj_sub": "SCONJ",
        "pron": "PRON", "pron_dem": "DET", "pron_rel": "PRON", "pron_interrog": "PRON", "part_neg": "PART",
        "part": "PART", "part_det": "DET", "part_fut": "AUX", "part_verb": "PART", "digit": "NUM",
        "punc": "PUNCT", "foreign": "X", "abbrev": "X", "interj": "INTJ"}
PROB = {"pos_logprob", "lex_logprob", "pos_lex_logprob"}


def ranked(analyses):
    """camel_tools.disambig.mle out-of-context ordering (its _scored_analyses
    fallback): score 10**pos_lex_logprob / max, ties by pos_lex_logprob,
    lex_logprob, diac (ScoredAnalysis.__lt__)."""
    from camel_tools.disambig.common import ScoredAnalysis

    if not analyses:
        return []
    ds = [dict(a) for a in analyses]
    probs = [10 ** (d.get("pos_lex_logprob") if d.get("pos_lex_logprob") is not None else -99.0) for d in ds]
    mx = max(probs)
    scored = [ScoredAnalysis(p / mx, d, d["diac"], d.get("pos_lex_logprob", -99), d.get("lex_logprob", -99))
              for d, p in zip(ds, probs)]
    scored.sort()
    return [s.analysis for s in scored]


def norm_lemma(s):
    from camel_tools.morphology.analyzer import DEFAULT_NORMALIZE_MAP
    from camel_tools.utils.dediac import dediac_ar

    return DEFAULT_NORMALIZE_MAP.map_string(dediac_ar(s))


def cmd_rank(args):
    words = load_words(args.words)
    full = [json.loads(line) for line in open(args.full, encoding="utf-8")]
    nop = [json.loads(line) for line in open(args.noprob, encoding="utf-8")]
    gold = {}
    if args.conllu:
        # Gold (UPOS, lemma) for whitespace tokens that UD did not split into
        # syntactic words, when the token has one unambiguous annotation.
        multi, seen = set(), {}
        with open(args.conllu, encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) != 10:
                    continue
                if "-" in cols[0]:
                    multi.add(cols[1])
                elif cols[0].isdigit():
                    seen.setdefault(cols[1], set()).add((cols[3], norm_lemma(cols[2])))
        gold = {w: next(iter(v)) for w, v in seen.items() if w not in multi and len(v) == 1}
    s = {"words": 0, "analyses_full": 0, "analyses_noprob": 0, "same_analysis_sets": 0,
         "words_with_nondefault_prob": 0, "ambiguous_words": 0, "top1_same": 0, "top1_same_lex_pos": 0,
         "full_order_same": 0, "gold_words": 0, "gold_upos_top1_full": 0, "gold_upos_top1_noprob": 0,
         "gold_lemma_top1_full": 0, "gold_lemma_top1_noprob": 0, "gold_upos_lemma_top1_full": 0,
         "gold_upos_lemma_top1_noprob": 0, "gold_upos_lemma_any": 0}
    key = lambda d: json.dumps(sorted((k, v) for k, v in d.items() if k not in PROB), ensure_ascii=False)  # noqa: E731
    for w, f, n in zip(words, full, nop):
        if not f.get("ok") or not n.get("ok") or not f["analyses"]:
            continue
        s["words"] += 1
        fa, na = [dict(x) for x in f["analyses"]], [dict(x) for x in n["analyses"]]
        s["analyses_full"] += len(fa)
        s["analyses_noprob"] += len(na)
        s["same_analysis_sets"] += [key(x) for x in fa] == [key(x) for x in na]
        s["words_with_nondefault_prob"] += any(x.get("pos_lex_logprob", -99.0) != -99.0 for x in fa)
        rf, rn = ranked(f["analyses"]), ranked(n["analyses"])
        if len(fa) >= 2:
            s["ambiguous_words"] += 1
            s["top1_same"] += key(rf[0]) == key(rn[0])
            s["top1_same_lex_pos"] += (rf[0]["lex"], rf[0]["pos"]) == (rn[0]["lex"], rn[0]["pos"])
            s["full_order_same"] += [key(x) for x in rf] == [key(x) for x in rn]
        if w in gold:
            upos, lemma = gold[w]
            s["gold_words"] += 1
            for name, top in (("full", rf[0]), ("noprob", rn[0])):
                u = UPOS.get(top["pos"]) == upos
                lm = norm_lemma(top["lex"]) == lemma
                s[f"gold_upos_top1_{name}"] += u
                s[f"gold_lemma_top1_{name}"] += lm
                s[f"gold_upos_lemma_top1_{name}"] += u and lm
            s["gold_upos_lemma_any"] += any(UPOS.get(x["pos"]) == upos and norm_lemma(x["lex"]) == lemma for x in fa)
    print(json.dumps(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("words")
    p.add_argument("--conllu", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--db", required=True)
    p.add_argument("--words", required=True)
    p.add_argument("--backoff", default="NONE")
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--ref", required=True)
    p.add_argument("--native", required=True)
    p.add_argument("--show", type=int, default=5)
    p = sub.add_parser("rank")
    p.add_argument("--words", required=True)
    p.add_argument("--full", required=True)
    p.add_argument("--noprob", required=True)
    p.add_argument("--conllu")
    args = ap.parse_args()
    {"words": cmd_words, "analyze": cmd_analyze, "compare": cmd_compare, "rank": cmd_rank}[args.cmd](args)


if __name__ == "__main__":
    main()
