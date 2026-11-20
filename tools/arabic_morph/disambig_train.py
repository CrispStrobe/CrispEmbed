#!/usr/bin/env python3
"""Offline trainer and evaluator for the Camel Morph context disambiguator.

Developer tooling only; nothing here ships. The model is a bigram hidden
Markov model whose hidden states are the analyses the native analyzer
returns for each token, grouped into candidates (tag, lex, pos). It is
trained without annotations by expectation maximization on raw text
(Arabic Wikipedia, CC BY-SA 4.0): the analyzer restricts each token to its
own analyses, and EM estimates the tag transitions and the lexeme
frequencies P(lex, pos | pos). The model format is
docs/arabic-morph-disambig-format.md. Requires numpy and pyarrow.

    python disambig_train.py sentences --parquet wiki-2.parquet ... --exclude eval-ids.txt \
        --max-tokens 20000000 --out train.txt
    python disambig_train.py eval-sample --parquet wiki-4.parquet --articles 1600 --seed 20261120 \
        --out wiki-eval.txt --ids eval-ids.txt
    python disambig_train.py analyze --cli crispembed-ar-morph --manifest proofing-bin/manifest.json \
        --sentences train.txt [--sentences more.txt] --out analyses.jsonl
    python disambig_train.py train --em ar-morph-disambig-em --analyses analyses.jsonl \
        --sentences train.txt --tags pos,prc0,prc1,prc2,prc3,enc0:flag --out model.cmdis
    python disambig_train.py evaluate --model model.cmdis --analyses analyses.jsonl \
        --conllu ar_pud-ud-test.conllu [--full-analyses full.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from collections import Counter, defaultdict

# Token classes, in priority order. The Dart host uses the same expression
# (packages/quds_arabic_morph, ArabicMorphTokenizer): an Arabic word (letters,
# then letters, harakat and tatweel), a number, a run of other letters, or
# any other single non-space character.
AR_LETTERS = "ء-غف-يٱ-ۓ"
AR_MARKS = "ً-ٰٟـ"
TOKEN_RE = re.compile(
    f"[{AR_LETTERS}][{AR_LETTERS}{AR_MARKS}]*"
    r"|[0-9٠-٩۰-۹]+(?:[.,٫٬][0-9٠-٩۰-۹]+)*"
    r"|[^\W\d_]+"
    r"|\S")
SENTENCE_END = {".", "!", "?", "؟", "…"}
AR_WORD_RE = re.compile(f"^[{AR_LETTERS}]")

# Features kept per analysis for candidate keys (diac, root and source are
# dropped: they never take part in a tag).
KEY_FEATURES = ("lex", "pos", "prc3", "prc2", "prc1", "prc1.5", "prc0", "per", "asp", "vox", "mod", "gen", "num",
                "stt", "cas", "enc0", "enc1", "enc2", "form_gen", "form_num", "rat")
FEATURE_INDEX = {k: i for i, k in enumerate(KEY_FEATURES)}


def tokenize(text):
    return TOKEN_RE.findall(text)


def split_sentences(text, min_tokens=3, max_tokens=120):
    """Lines, then sentence-final punctuation; the end mark stays in its sentence."""
    out = []
    for line in text.split("\n"):
        cur = []
        for tok in tokenize(line):
            cur.append(tok)
            if tok in SENTENCE_END:
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
    keep = []
    for s in out:
        if not (min_tokens <= len(s) <= max_tokens):
            continue
        ar = sum(1 for t in s if AR_WORD_RE.match(t))
        if ar * 2 < len(s):
            continue
        keep.append(s)
    return keep


def iter_articles(paths):
    import pyarrow.parquet as pq

    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["id", "title", "text"], batch_size=2000):
            for row in batch.to_pylist():
                yield row


def cmd_eval_sample(args):
    rows = list(iter_articles(args.parquet))
    rng = random.Random(args.seed)
    picked = rng.sample(range(len(rows)), args.articles)
    ids, n = [], 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in sorted(picked):
            ids.append(rows[i]["id"])
            for s in split_sentences(rows[i]["text"]):
                # Evaluation lines keep the original spacing of the line text.
                f.write(" ".join(s) + "\n")
                n += 1
    with open(args.ids, "w") as f:
        f.write("\n".join(ids) + "\n")
    print(json.dumps({"articles": len(ids), "sentences": n}))


def cmd_sentences(args):
    exclude = set()
    for path in args.exclude or []:
        exclude |= {line.strip() for line in open(path) if line.strip()}
    tokens = sents = skipped = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in iter_articles(args.parquet):
            if row["id"] in exclude:
                skipped += 1
                continue
            for s in split_sentences(row["text"]):
                f.write("\t".join(s) + "\n")
                tokens += len(s)
                sents += 1
            if tokens >= args.max_tokens:
                break
    print(json.dumps({"sentences": sents, "tokens": tokens, "excluded_articles": skipped}))


def read_sentences(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line.split("\t")


def cmd_analyze(args):
    """Distinct tokens through the native CLI (compact ABI, dedup); one line
    per word: {"w": word, "a": [[feature values in KEY_FEATURES order], ...]}
    with analyses projected onto KEY_FEATURES and deduplicated."""
    words = Counter()
    for path in args.sentences:
        for s in read_sentences(path):
            words.update(s)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["w"] for line in f}
    todo = [w for w in words if w not in done]
    print(json.dumps({"distinct": len(words), "todo": len(todo)}), file=sys.stderr)
    sha = hashlib.sha256(open(args.manifest, "rb").read()).hexdigest()
    with tempfile.TemporaryDirectory() as tmp, open(args.out, "a", encoding="utf-8") as out:
        for start in range(0, len(todo), args.chunk):
            chunk = todo[start:start + args.chunk]
            wpath, opath = os.path.join(tmp, "w.jsonl"), os.path.join(tmp, "o.jsonl")
            with open(wpath, "w", encoding="utf-8") as f:
                for w in chunk:
                    f.write(json.dumps(w, ensure_ascii=False) + "\n")
            cmd = [args.cli, "--manifest", args.manifest, "--sha256", sha, "--jsonl", wpath, "--out", opath]
            if not args.text:
                cmd += ["--compact", "--dedup"]
            subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
            with open(opath, encoding="utf-8") as f:
                for w, line in zip(chunk, f):
                    rec = json.loads(line)
                    seen, keys = set(), []
                    for an in rec.get("analyses") or []:
                        d = dict(an)
                        key = tuple(d.get(k) for k in KEY_FEATURES)
                        if args.keep_prob:
                            key = key + (d.get("pos_lex_logprob"), d.get("lex_logprob"), d.get("diac"))
                        if key not in seen:
                            seen.add(key)
                            keys.append(key)
                    out.write(json.dumps({"w": w, "a": keys}, ensure_ascii=False) + "\n")
            print(json.dumps({"analyzed": min(start + args.chunk, len(todo))}), file=sys.stderr)


def load_analyses(path, words=None, convert=None):
    """word -> analyses (or convert(analyses)), optionally only for `words`."""
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if words is not None:
                w = json.loads(line[:line.index(', "a": ')] + "}")["w"]
                if w not in words:
                    continue
            rec = json.loads(line)
            table[rec["w"]] = convert(rec["a"]) if convert else rec["a"]
    return table


# ---------------------------------------------------------------- tag spec

def parse_tags(spec):
    """'pos,prc0,enc0:flag' -> [(feature, mode)], mode 'value' or 'flag'."""
    out = []
    for item in spec.split(","):
        name, _, mode = item.partition(":")
        mode = mode or "value"
        if name not in FEATURE_INDEX or mode not in ("value", "flag"):
            raise SystemExit(f"bad tag feature {item}")
        out.append((name, mode))
    if not out or out[0] != ("pos", "value"):
        raise SystemExit("the tag spec must start with pos")
    return out


def tag_of(key, spec):
    parts = []
    for name, mode in spec:
        v = key[FEATURE_INDEX[name]]
        if mode == "flag":
            v = "0" if v in (None, "0", "na", "") else "1"
        elif v is None:
            v = "-"
        parts.append(v)
    return "|".join(parts)


OOV = ("<oov>", "<oov>")  # (lex, pos) of a token without analyses
OOV_TAG = "<oov>"


def candidates(analyses, spec):
    """Distinct (tag, lex, pos) candidates, first-seen order; and per
    analysis its candidate index."""
    if not analyses:
        return [(OOV_TAG, OOV[0], OOV[1])], []
    index, cands, of = {}, [], []
    for key in analyses:
        c = (tag_of(key, spec), key[0], key[1])
        if c not in index:
            index[c] = len(cands)
            cands.append(c)
        of.append(index[c])
    return cands, of


# ---------------------------------------------------------------- training

def cmd_train(args):
    spec = parse_tags(args.tags)
    vocab = set()
    for path in args.sentences:
        for s in read_sentences(path):
            vocab.update(s)
    table = load_analyses(args.analyses, vocab, lambda a: candidates(a, spec)[0])
    tags = {"<s>": 0}
    lexpos = {}
    poses = {}
    type_ids = {}
    types = []  # per type: list of (tag id, lexpos id)

    def tid(t):
        return tags.setdefault(t, len(tags))

    def lid(lp):
        if lp not in lexpos:
            lexpos[lp] = len(lexpos)
            poses.setdefault(lp[1], len(poses))
        return lexpos[lp]

    sentences = []
    for path in args.sentences:
        for s in read_sentences(path):
            ids = []
            for w in s:
                t = type_ids.get(w)
                if t is None:
                    cands = table.get(w) or candidates([], spec)[0]
                    t = type_ids[w] = len(types)
                    types.append([(tid(c[0]), lid((c[1], c[2]))) for c in cands])
                ids.append(t)
            sentences.append(ids)
    lp_list = sorted(lexpos, key=lexpos.get)
    pos_list = sorted(poses, key=poses.get)
    with tempfile.TemporaryDirectory() as tmp:
        cpath, opath = os.path.join(tmp, "corpus.bin"), os.path.join(tmp, "counts.bin")
        with open(cpath, "wb") as f:
            f.write(b"AMEM" + struct.pack("<III", len(tags), len(lexpos), len(poses)))
            f.write(struct.pack(f"<{len(lp_list)}I", *[poses[lp[1]] for lp in lp_list]))
            f.write(struct.pack("<I", len(types)))
            for cands in types:
                f.write(struct.pack("<I", len(cands)))
                f.write(struct.pack(f"<{2 * len(cands)}I", *[x for c in cands for x in c]))
            f.write(struct.pack("<I", len(sentences)))
            for ids in sentences:
                f.write(struct.pack(f"<I{len(ids)}I", len(ids), *ids))
        print(json.dumps({"tags": len(tags), "lexpos": len(lexpos), "pos": len(poses), "types": len(types),
                          "sentences": len(sentences), "tokens": sum(map(len, sentences))}), file=sys.stderr)
        cmd = [args.em, cpath, opath, str(args.iterations), str(args.alpha), str(args.kappa), str(args.beta),
               str(args.threads), args.init]
        if args.init_discount:
            # Multiplies the initial weight of every tag whose pos is listed.
            disc = {k: float(v) for k, v in (x.split("=") for x in args.init_discount.split(","))}
            wpath = os.path.join(tmp, "init.f64")
            with open(wpath, "wb") as f:
                for t in sorted(tags, key=tags.get):
                    f.write(struct.pack("<d", disc.get(t.split("|")[0], 1.0)))
            cmd.append(wpath)
        subprocess.run(cmd, check=True)
        with open(opath, "rb") as f:
            nt, nl = struct.unpack("<II", f.read(8))
            import numpy as np

            trans = np.frombuffer(f.read(8 * nt * nt), dtype="<f8").reshape(nt, nt)
            lexc = np.frombuffer(f.read(8 * nl), dtype="<f8")
    tag_list = sorted(tags, key=tags.get)
    write_model(args.out, spec, tag_list, trans, lp_list, lexc, args, {
        "sentences": len(sentences), "tokens": int(sum(map(len, sentences))), "types": len(types)})


# ---------------------------------------------------------------- model file

MAGIC = b"CEMDISAM"


def write_model(path, spec, tag_list, trans, lp_list, lexc, args, corpus):
    """Smooths the expected counts of EM into log-probabilities, as the EM
    M step does: T(cur | prev) = (C(prev, cur) + kappa U(cur)) / (C(prev) +
    kappa) with the add-alpha tag unigram U; transitions with an expected
    count below min_trans are left to the backoff kappa U(cur) / (C(prev) +
    kappa). Lexemes: add-beta per pos (an unlisted lexeme of a pos gets
    beta / total); lexemes with an expected count below min_count are left
    out."""
    import numpy as np

    nt = len(tag_list)
    col = trans.sum(axis=0)
    uni = (col + args.alpha) / (col.sum() + args.alpha * nt)
    row = trans.sum(axis=1)
    backoff = np.log(args.kappa / (row + args.kappa))
    entries = []
    for p_, c_ in zip(*np.nonzero(trans >= args.min_trans)):
        entries.append((int(p_), int(c_), math.log((trans[p_, c_] + args.kappa * uni[c_]) / (row[p_] + args.kappa))))
    by_pos = defaultdict(list)
    for i, lp in enumerate(lp_list):
        by_pos[lp[1]].append(i)
    lexp, unseen = [], {}
    for pos, idx in by_pos.items():
        total = float(sum(lexc[i] for i in idx)) + args.beta * (len(idx) + 1)
        unseen[pos] = math.log(args.beta / total)
        for i in idx:
            if lexc[i] >= args.min_count:
                lexp.append((pos, lp_list[i][0], math.log((lexc[i] + args.beta) / total)))
    meta = {"format": "crispembed-camel-morph-disambig-v1", "tags": [f"{n}:{m}" for n, m in spec],
            "corpus": corpus, "iterations": args.iterations, "alpha": args.alpha, "kappa": args.kappa,
            "beta": args.beta, "min_count": args.min_count, "min_trans": args.min_trans, "init": args.init,
            "init_discount": args.init_discount}
    if getattr(args, "meta", None):
        meta.update(json.loads(open(args.meta, encoding="utf-8").read()))
    write_model_logp(path, spec, tag_list, np.log(uni), backoff, entries, lexp, unseen, meta)


def write_model_logp(path, spec, tag_list, uni_logp, backoff_logw, entries, lexp, unseen, meta):
    """Writes docs/arabic-morph-disambig-format.md. tag_list[0] is "<s>".
    Natural-log values: T(cur | prev) is the entry (prev, cur, logp) when
    listed, else exp(backoff_logw[prev] + uni_logp[cur]); lexp holds
    (pos, lex, logp) and unseen {pos: logp}."""
    import numpy as np

    nt = len(tag_list)
    if tag_list[0] != "<s>" or len(set(tag_list)) != nt:
        raise SystemExit("tag 0 must be <s> and tags unique")
    entries = sorted(entries)
    keep = sorted(lexp, key=lambda x: (x[0].encode(), x[1].encode()))
    pos_list = sorted(unseen, key=lambda p: p.encode())
    strings = sorted(set(tag_list) | set(pos_list) | {p for p, _, _ in keep} | {lx for _, lx, _ in keep}
                     | {n for n, _ in spec}, key=lambda s: s.encode())
    sid = {s: i for i, s in enumerate(strings)}
    sections = []
    blob = b"".join(s.encode() for s in strings)
    offs, o = [0], 0
    for s in strings:
        o += len(s.encode())
        offs.append(o)
    sections.append((b"STRS", struct.pack(f"<I{len(offs)}I", len(strings), *offs) + blob))
    sections.append((b"SPEC", struct.pack("<I", len(spec)) + b"".join(
        struct.pack("<II", sid[n], 0 if m == "value" else 1) for n, m in spec)))
    sections.append((b"TAGS", struct.pack(f"<I{nt}I", nt, *[sid[x] for x in tag_list])))
    sections.append((b"TRAN", struct.pack("<I", nt) + np.asarray(uni_logp, dtype="<f4").tobytes()
                     + np.asarray(backoff_logw, dtype="<f4").tobytes() + struct.pack("<I", len(entries))
                     + b"".join(struct.pack("<IIf", a, b, v) for a, b, v in entries)))
    sections.append((b"UNSN", struct.pack("<I", len(pos_list)) + b"".join(
        struct.pack("<If", sid[p], unseen[p]) for p in pos_list)))
    sections.append((b"LEXP", struct.pack("<I", len(keep)) + b"".join(
        struct.pack("<IIf", sid[p], sid[lx], lg) for p, lx, lg in keep)))
    sections.append((b"META", json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()))
    write_sections(path, sections)


def write_sections(path, sections):
    def pad(b):
        return b + b"\0" * (-len(b) % 8)

    header_size, n = 64, len(sections)
    offset = header_size + 24 * n
    offset += -offset % 8
    directory, body = b"", b""
    for tag, data in sections:
        directory += tag + struct.pack("<IQQ", zlib.crc32(data), offset + len(body), len(data))
        body += pad(data)
    dir_bytes = pad(directory)
    size = header_size + len(dir_bytes) + len(body)
    head = MAGIC + struct.pack("<IIIIQII", 1, header_size, n, 0, size, zlib.crc32(directory), 0) + b"\0" * 24
    head = head[:36] + struct.pack("<I", zlib.crc32(head)) + head[40:]
    with open(path, "wb") as f:
        f.write(head + dir_bytes + body)


def read_model(path):
    data = open(path, "rb").read()
    assert data[:8] == MAGIC
    n = struct.unpack_from("<I", data, 16)[0]
    secs = {}
    for i in range(n):
        tag, _crc, off, size = struct.unpack_from("<4sIQQ", data, 64 + 24 * i)
        secs[tag] = data[off:off + size]
    import numpy as np

    s = secs[b"STRS"]
    ns = struct.unpack_from("<I", s)[0]
    offs = struct.unpack_from(f"<{ns + 1}I", s, 4)
    base = 4 + 4 * (ns + 1)
    strings = [s[base + offs[i]:base + offs[i + 1]].decode() for i in range(ns)]
    sp = secs[b"SPEC"]
    spec = [(strings[a], "value" if b == 0 else "flag")
            for a, b in struct.iter_unpack("<II", sp[4:4 + 8 * struct.unpack_from("<I", sp)[0]])]
    tg = secs[b"TAGS"]
    nt = struct.unpack_from("<I", tg)[0]
    tags = [strings[x] for x in struct.unpack_from(f"<{nt}I", tg, 4)]
    tr = secs[b"TRAN"]
    uni = np.frombuffer(tr[4:4 + 4 * nt], dtype="<f4").astype(np.float64)
    backoff = np.frombuffer(tr[4 + 4 * nt:4 + 8 * nt], dtype="<f4").astype(np.float64)
    trans = backoff[:, None] + uni[None, :]
    ne = struct.unpack_from("<I", tr, 4 + 8 * nt)[0]
    for a, b, v in struct.iter_unpack("<IIf", tr[8 + 8 * nt:8 + 8 * nt + 12 * ne]):
        trans[a, b] = v
    un = secs[b"UNSN"]
    unseen = {strings[a]: b for a, b in struct.iter_unpack("<If", un[4:])}
    lx = secs[b"LEXP"]
    lexp = {(strings[b], strings[a]): c for a, b, c in struct.iter_unpack("<IIf", lx[4:])}
    return {"spec": spec, "tags": tags, "tag_id": {t: i for i, t in enumerate(tags)}, "trans": trans,
            "unseen": unseen, "lexp": lexp, "meta": json.loads(secs[b"META"])}


# ---------------------------------------------------------------- inference

def posteriors(model, sentence_analyses):
    """Forward-backward over one sentence; returns per token (cands, of, post)."""
    import numpy as np

    spec, tid, trans = model["spec"], model["tag_id"], model["trans"]
    unk = model.get("unk_tag")
    toks = []
    for analyses in sentence_analyses:
        cands, of = candidates(analyses, spec)
        tg = np.array([tid.get(c[0], -1) for c in cands])
        em = np.array([model["lexp"].get((c[1], c[2]), model["unseen"].get(c[2], -30.0)) for c in cands],
                      dtype=np.float64)
        toks.append((cands, of, tg, em))
    T = np.exp(trans.astype(np.float64))
    floor = 1e-12

    def tr(prev, cur):  # prev, cur: arrays of tag ids (-1 = unknown tag)
        m = np.full((len(prev), len(cur)), floor)
        pv, cv = prev >= 0, cur >= 0
        m[np.ix_(pv, cv)] = T[np.ix_(prev[pv], cur[cv])]
        return m

    n = len(toks)
    alphas, scales = [], []
    prev_tags, prev_a = np.array([0]), np.array([1.0])
    for cands, of, tg, em in toks:
        a = (prev_a @ tr(prev_tags, tg)) * np.exp(em)
        s = a.sum()
        a = a / s
        alphas.append(a)
        scales.append(s)
        prev_tags, prev_a = tg, a
    end = tr(prev_tags, np.array([0]))[:, 0]
    s_end = float(prev_a @ end)
    b = end / s_end
    out = [None] * n
    for i in range(n - 1, -1, -1):
        cands, of, tg, em = toks[i]
        g = alphas[i] * b
        g = g / g.sum()
        out[i] = (cands, of, g)
        if i:
            b = tr(toks[i - 1][2], tg) @ (np.exp(em) * b) / scales[i]
    return out


# ---------------------------------------------------------------- evaluation

UPOS = {"noun": "NOUN", "noun_prop": "PROPN", "noun_num": "NUM", "noun_quant": "NOUN", "adj": "ADJ",
        "adj_comp": "ADJ", "adj_num": "ADJ", "verb": "VERB", "verb_pseudo": "VERB", "adv": "ADV",
        "adv_interrog": "ADV", "adv_rel": "ADV", "prep": "ADP", "conj": "CCONJ", "conj_sub": "SCONJ",
        "pron": "PRON", "pron_dem": "DET", "pron_rel": "PRON", "pron_interrog": "PRON", "part_neg": "PART",
        "part": "PART", "part_det": "DET", "part_fut": "AUX", "part_verb": "PART", "digit": "NUM",
        "punc": "PUNCT", "foreign": "X", "abbrev": "X", "interj": "INTJ"}
NORM = str.maketrans({"إ": "ا", "أ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ـ": None})
HARAKAT = re.compile("[ً-ْٰ]")


def norm_lemma(s):
    return HARAKAT.sub("", s or "").translate(NORM)


def read_conllu(path):
    """Sentences as host tokens: (form, gold or None). The text is rebuilt
    from the UD words and their SpaceAfter=No marks and tokenized like the
    host does; a token has gold (UPOS, lemma) when it is exactly one UD word
    (so clitic combinations that UD splits, such as وفي, have none)."""
    sents, words = [], []

    def flush():
        if not words:
            return
        text, spans = "", {}
        for form, gold, space in words:
            spans[(len(text), len(text) + len(form))] = gold
            text += form + (" " if space else "")
        sents.append([(m.group(0), spans.get(m.span())) for m in TOKEN_RE.finditer(text)])
        words.clear()

    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            flush()
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if not cols[0].isdigit():
            continue  # PUD has no multiword token ranges or empty nodes
        words.append((cols[1], (cols[3], norm_lemma(cols[2])), "SpaceAfter=No" not in cols[9]))
    flush()
    return sents


def noprob_top(analyses):
    """camel_tools' MLE order without probabilities: every score ties, so the
    lexicographically smallest diacritization comes first."""
    return min(analyses, key=lambda a: a[-1]) if analyses else None


def full_top(analyses):
    """camel_tools.disambig.mle out-of-context top-1 over the full database
    (score 10**pos_lex_logprob, ties by pos_lex_logprob, lex_logprob, diac)."""
    if not analyses:
        return None
    return min(analyses, key=lambda a: (-(a[-3] if a[-3] is not None else -99.0),
                                        -(a[-2] if a[-2] is not None else -99.0), a[-1]))


def cmd_evaluate(args):
    model = read_model(args.model)
    sents = read_conllu(args.conllu)
    lo, _, hi = (args.range or ":").partition(":")
    first = int(lo or 0)
    sents = sents[first:int(hi) if hi else None]
    vocab = {w for s in sents for w, _ in s}
    table = load_analyses(args.analyses, vocab)
    full = load_analyses(args.full_analyses, vocab) if args.full_analyses else None
    stats = Counter()
    margins = []
    native = None
    if args.native:
        with open(args.native, encoding="utf-8") as f:
            native = [json.loads(line)["tokens"] for line in f][first:first + len(sents)]
        if len(native) != len(sents):
            raise SystemExit("native output does not match the sentences")
    for si, s in enumerate(sents):
        sa = [table.get(form, []) for form, _ in s]
        post = posteriors(model, sa)
        if native is not None:
            # The native candidates' best analysis (rank 0) must be the
            # reference's best candidate, with the same score.
            for tok, (cands, of, g) in zip(native[si], post):
                if not tok["analyses"]:
                    continue
                best = tok["rank"].index(0)
                d = dict(tok["analyses"][best])
                b = int(g.argmax())
                stats["native_tokens"] += 1
                stats["native_same_top"] += (d.get("lex"), d.get("pos")) == (cands[b][1], cands[b][2])
                stats["native_max_score_diff"] = max(stats["native_max_score_diff"],
                                                     abs(tok["score"][best] - float(g[b])))
        for (form, gold), analyses, (cands, of, g) in zip(s, sa, post):
            if gold is None or not analyses:
                continue
            upos, lemma = gold
            stats["tokens"] += 1
            ok = lambda pos, lex: (UPOS.get(pos) == upos, norm_lemma(lex) == lemma)  # noqa: E731
            stats["oracle_both"] += any(all(ok(c[1], c[0])) for c in analyses)
            best = int(g.argmax())
            u, lm = ok(cands[best][2], cands[best][1])
            stats["model_upos"] += u
            stats["model_lemma"] += lm
            stats["model_both"] += u and lm
            srt = sorted(g, reverse=True)
            margin = srt[0] - (srt[1] if len(srt) > 1 else 0.0)
            for th in (0.5, 0.8, 0.9, 0.95):
                if g[best] >= th:
                    stats[f"conf{th}_n"] += 1
                    stats[f"conf{th}_both"] += u and lm
                    stats[f"conf{th}_upos"] += u
            margins.append(margin)
            # Baseline from the noprob analyses themselves needs diac: use full if given.
            if full is not None:
                fa = full.get(form, [])
                for name, top in (("noprob", noprob_top(fa)), ("full", full_top(fa))):
                    if top is None:
                        continue
                    u2, l2 = ok(top[1], top[0])
                    stats[f"{name}_upos"] += u2
                    stats[f"{name}_lemma"] += l2
                    stats[f"{name}_both"] += u2 and l2
    n = stats["tokens"]
    report = {k: v for k, v in sorted(stats.items())}
    report["pct"] = {k: round(100 * v / n, 1) for k, v in sorted(stats.items())
                     if k != "tokens" and not k.endswith("_n") and not k.startswith(("conf", "native"))}
    for th in (0.5, 0.8, 0.9, 0.95):
        c = stats[f"conf{th}_n"]
        if c:
            report["pct"][f"conf{th}"] = {"coverage": round(100 * c / n, 1),
                                          "both": round(100 * stats[f"conf{th}_both"] / c, 1),
                                          "upos": round(100 * stats[f"conf{th}_upos"] / c, 1)}
    report["model"] = model["meta"]
    print(json.dumps(report, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- bundle

MODEL_NAME = "camel_morph_msa_v1.0.disambig.cmdis"
BUNDLE_LICENSE = """\
The Camel Morph MSA context disambiguation model in this directory is
licensed under the Creative Commons Attribution-ShareAlike 4.0 International
License (CC BY-SA 4.0): https://creativecommons.org/licenses/by-sa/4.0/

It is derived from Arabic Wikipedia text (CC BY-SA 4.0) and Camel Morph MSA
v1.0 (CC BY 4.0). See NOTICE for attribution and the changes made.
"""
BUNDLE_NOTICE = """\
Camel Morph MSA context disambiguation model ({model})

A bigram hidden Markov model over the analyses of the Camel Morph MSA v1.0
noprob database, trained without annotations by expectation maximization on
raw Arabic Wikipedia text. It holds tag transition and lexeme probabilities
only; it contains no sentence of the training text.

Licence: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
Produced with CrispEmbed (MIT): tools/arabic_morph/disambig_train.py.

Attribution:
- Training text: Arabic Wikipedia, "{dump}" dump, by Wikipedia contributors,
  CC BY-SA 4.0 (https://ar.wikipedia.org/, licence
  https://creativecommons.org/licenses/by-sa/4.0/). Retrieved through the
  Hugging Face dataset wikimedia/wikipedia ({dump}); {tokens} tokens of
  {sentences} sentences were used. {held_out}
- Lexemes, parts of speech and the analyses the tags are built from: Camel
  Morph MSA v1.0 (c) 2024 by New York University Abu Dhabi, CC BY 4.0
  (https://github.com/CAMeL-Lab/camel_morph), in the noprob proofing-bin
  variant (database SHA-256 {db_sha256}); no Penn Arabic Treebank
  probability was used.

Changes (CC BY-SA 4.0 section 3(b), CC BY 4.0 section 3(a)(1)(B)): the text
was tokenized, analyzed with the database and reduced to expected counts of
tag bigrams and lexemes, which were smoothed and stored as log-probabilities.
"""
CARD = """\
---
license: cc-by-sa-4.0
language:
- ar
tags:
- arabic
- morphology
- part-of-speech
- disambiguation
- camel-morph
---

# Camel Morph MSA context disambiguation model

A small model that ranks the analyses of the
[Camel Morph MSA v1.0](https://github.com/CAMeL-Lab/camel_morph) analyzer in
their sentence. It is the context disambiguator of CrispEmbed's native Camel
Morph analyzer (`crispembed_ar_morph_disambiguate`, C ABI version 3) and is
used by Quds Office for Arabic grammar proofing.

- `{directory}/`: the bundle (`manifest.json`, `{model}`, `NOTICE`, `LICENSE`).
  Format: [crispembed-camel-morph-disambig-v1](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/arabic-morph-disambig-format.md).
- It is meant for the `proofing-bin` database of
  [cstr/camel-morph-msa-v1.0](https://huggingface.co/cstr/camel-morph-msa-v1.0)
  (SHA-256 `{db_sha256}`), the `noprob` variant without Penn Arabic Treebank
  probabilities.

## Model

A bigram hidden Markov model whose states are the analyses the database
returns for each token, grouped by tag (`{tags}`), lemma and part of speech.
It was trained without annotated data: expectation maximization over
{tokens} tokens of Arabic Wikipedia ({dump}), {iterations} iterations. The
first E step weights each candidate by the frequency of its tag among
unambiguous tokens{discount}. Transitions back off to the tag unigram;
lexeme probabilities are add-beta smoothed per part of speech.

## Evaluation

UD Arabic-PUD test (GitHub master, 20 November 2026; CC BY-SA 3.0; evaluation
only, never trained on),
top-1 analysis in context against gold UPOS and lemma, {eval_tokens} tokens
that are one UD word. The noprob and full columns rank each word out of
context as camel_tools' MLE disambiguator does, without and with the Penn
Arabic Treebank probabilities of the `full` database (not used here).

{eval_table}

## Licence and attribution

CC BY-SA 4.0. The model is derived from Arabic Wikipedia text by Wikipedia
contributors (CC BY-SA 4.0) and from Camel Morph MSA v1.0, (c) 2024 New York
University Abu Dhabi (CC BY 4.0). See `{directory}/NOTICE`. No LDC, Penn
Arabic Treebank or non-commercial resource was used.
"""


def cmd_bundle(args):
    """Writes a model bundle (manifest, model, NOTICE, LICENSE) and the model card."""
    from pathlib import Path

    model = Path(args.model).read_bytes()
    meta = read_model(args.model)["meta"]
    db_manifest = json.loads(Path(args.db_manifest).read_text(encoding="utf-8"))
    db_entry = next(f for f in db_manifest["files"] if f["path"] == db_manifest["db"])
    out = Path(args.out)
    root = out / args.directory
    root.mkdir(parents=True, exist_ok=True)
    corpus = meta.get("corpus", {})
    held_out = (f"The {args.held_out} articles drawn for evaluation were excluded." if args.held_out else "")
    files = {
        MODEL_NAME: model,
        "NOTICE": BUNDLE_NOTICE.format(model=MODEL_NAME, dump=args.dump, tokens=f"{corpus.get('tokens', 0):,}",
                                       sentences=f"{corpus.get('sentences', 0):,}", held_out=held_out,
                                       db_sha256=db_entry["sha256"]).encode(),
        "LICENSE": BUNDLE_LICENSE.encode(),
    }
    for name, data in files.items():
        (root / name).write_bytes(data)
    manifest = {
        "format": "crispembed-camel-morph-disambig-bundle",
        "format_version": 1,
        "model": MODEL_NAME,
        "model_format": meta["format"],
        "license": "CC-BY-SA-4.0",
        "attribution": ("Camel Morph MSA context disambiguation model, CC BY-SA 4.0; trained on Arabic Wikipedia "
                        "(Wikipedia contributors, CC BY-SA 4.0) over Camel Morph MSA v1.0 (c) 2024 New York "
                        "University Abu Dhabi (CC BY 4.0)"),
        "database": {"variant": db_manifest["variant"], "db": db_manifest["db"], "sha256": db_entry["sha256"],
                     "manifest_sha256": hashlib.sha256(Path(args.db_manifest).read_bytes()).hexdigest()},
        "training": meta,
        "files": [{"path": n, "size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for n, d in sorted(files.items())],
    }
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (root / "manifest.json").write_text(text, encoding="utf-8")
    table = Path(args.eval_table).read_text(encoding="utf-8").strip() if args.eval_table else "(not recorded)"
    discount = (f", with the weight of {meta['init_discount']} tags scaled down" if meta.get("init_discount") else "")
    (out / "README.md").write_text(CARD.format(
        directory=args.directory, model=MODEL_NAME, db_sha256=db_entry["sha256"], tags=", ".join(meta["tags"]),
        tokens=f"{corpus.get('tokens', 0):,}", dump=args.dump, iterations=meta.get("iterations"),
        discount=discount, eval_tokens=args.eval_tokens, eval_table=table), encoding="utf-8")
    print(json.dumps({"manifest_sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "model_sha256": hashlib.sha256(model).hexdigest(), "model_bytes": len(model),
                      "files": manifest["files"]}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval-sample")
    p.add_argument("--parquet", nargs="+", required=True)
    p.add_argument("--articles", type=int, default=1600)
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--out", required=True)
    p.add_argument("--ids", required=True)
    p = sub.add_parser("sentences")
    p.add_argument("--parquet", nargs="+", required=True)
    p.add_argument("--exclude", action="append")
    p.add_argument("--max-tokens", type=int, default=20_000_000)
    p.add_argument("--out", required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--cli", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--sentences", action="append", required=True)
    p.add_argument("--chunk", type=int, default=50000)
    p.add_argument("--text", action="store_true", help="text bundle: no compact output")
    p.add_argument("--keep-prob", action="store_true", help="append pos_lex_logprob, lex_logprob, diac")
    p.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--em", required=True)
    p.add_argument("--analyses", required=True)
    p.add_argument("--sentences", action="append", required=True)
    p.add_argument("--tags", default="pos")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--min-trans", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--min-count", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--init", choices=("uniform", "unambiguous"), default="unambiguous")
    p.add_argument("--meta", help="JSON object merged into the model's META (sources, licences)")
    p.add_argument("--init-discount", help="pos=factor,... multiplies the first E step's weights of those tags")
    p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--model", required=True)
    p.add_argument("--analyses", required=True)
    p.add_argument("--full-analyses")
    p.add_argument("--native", help="crispembed-ar-morph --sentences output for the same CoNLL-U tokens")
    p.add_argument("--range", help="sentences first:end (Python slice) of the CoNLL-U file, e.g. 0:500 (dev)")
    p.add_argument("--conllu", required=True)
    p = sub.add_parser("bundle")
    p.add_argument("--model", required=True)
    p.add_argument("--db-manifest", required=True, help="the proofing-bin manifest.json the model was trained with")
    p.add_argument("--out", required=True)
    p.add_argument("--directory", default="bigram-v1")
    p.add_argument("--dump", default="20231101.ar")
    p.add_argument("--held-out", type=int, default=0)
    p.add_argument("--eval-table", help="Markdown table for the card")
    p.add_argument("--eval-tokens", default="")
    args = ap.parse_args()
    {"bundle": cmd_bundle, "eval-sample": cmd_eval_sample, "sentences": cmd_sentences, "analyze": cmd_analyze, "train": cmd_train,
     "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
