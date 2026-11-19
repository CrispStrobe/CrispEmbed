#!/usr/bin/env python3
"""Build the Camel Morph MSA v1.0 database bundles for the native analyzer.

Developer tooling only; nothing here runs in an app. The database is
CC BY 4.0 data published by the CAMeL Lab (New York University Abu Dhabi); it
is pinned here by git commit, git blob id, size and SHA-256 and is never
modified in the ``full`` variant.

    python models/convert-camel-morph-db.py --output out/          # downloads the pinned DB
    python models/convert-camel-morph-db.py --source camel_morph_msa_v1.0.db --output out/

Output layout (one Hugging Face repository, three self-contained bundles):

    out/README.md                 Hugging Face card (license: cc-by-4.0)
    out/NOTICE, out/LICENSE       attribution and licence pointer
    out/full/                     the database byte-for-byte as released
    out/noprob/                   the same database without the PATB-derived
                                  probability fields (see PROB_FIELDS)
    out/proofing-bin/             noprob reduced to PROOFING_PROFILE in the binary
                                  format of docs/arabic-morph-binary-format.md

Each bundle holds ``manifest.json`` (variant, source pin, every file's size and
SHA-256), the database, ``NOTICE`` and ``LICENSE``. The native loader
(src/arabic_morph.cpp) authenticates manifest.json with a caller-supplied
SHA-256 and every listed file with the manifest's pins before parsing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import textwrap
import urllib.request
import zlib

SOURCE_REPO = "https://github.com/CAMeL-Lab/camel_morph"
SOURCE_COMMIT = "aaa39653c2a53ebacf087f0b98e7f7cfb55cca81"
SOURCE_PATH = "official_releases/lrec-coling2024_release/databases/camel-morph-msa/camel_morph_msa_v1.0.db"
SOURCE_SHA256 = "110e03baa282874062d076e23d9671e985b6f76520b891d58cba4e55de0b8f60"
SOURCE_GIT_BLOB = "19f528666fda8b1cd67ef838089166a95eecc591"
SOURCE_SIZE = 84959404
SOURCE_LAST_CHANGE = "e818de087015a4419fe58aeaa563147b23bf976b"  # last commit touching SOURCE_PATH
DB_NAME = "camel_morph_msa_v1.0.db"
NOPROB_DB_NAME = "camel_morph_msa_v1.0.noprob.db"

# Fields whose values come from the Penn Arabic Treebank (PATB/ATB) training
# split: camel_morph's db_maker.py writes `lex_logprob` and `pos_lex_logprob`
# from a pickle produced by sandbox/get_logprob_atb.py over ATB MAGOLD data.
# `pos_logprob` is DEFINEd but no entry carries a value. camel_tools fills an
# absent value with -99.0, so the noprob analyzer output keeps every key.
PROB_FIELDS = ("pos_logprob", "lex_logprob", "pos_lex_logprob")

NOTICE = """Camel Morph MSA v1.0 (c) 2024 by New York University Abu Dhabi is licensed under CC BY 4.0.

  Camel Morph MSA v1.0: https://github.com/CAMeL-Lab/camel_morph
  New York University Abu Dhabi: https://nyuad.nyu.edu
  CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

Source: {repo} at commit {commit}, file {path}
(SHA-256 {sha256}).

Citation: Christian Khairallah, Salam Khalifa, Reham Marzouk, Mayar Nassar and
Nizar Habash. 2024. Camel Morph MSA: A Large-Scale Open-Source Morphological
Analyzer for Modern Standard Arabic. LREC-COLING 2024.
https://aclanthology.org/2024.lrec-main.240

Provenance caveat (from the paper and camel_morph's db_maker.py): the lexicon
was seeded from CALIMA-Star/SAMA and reworked by hand, and the lex_logprob and
pos_lex_logprob values are estimated from the Penn Arabic Treebank training
split. The noprob variant removes those values.
{changes}"""

NOPROB_CHANGES = """
Changes in this variant (CC BY 4.0 section 3(a)(1)(B)): every
pos_logprob/lex_logprob/pos_lex_logprob value was removed from the PREFIXES,
SUFFIXES and STEMS sections. All other bytes are unchanged; DEFINE, DEFAULT and
ORDER lines still declare the fields, so an analyzer reports -99.0 for them.
"""

LICENSE = """The database in this directory is licensed under the Creative Commons
Attribution 4.0 International License (CC BY 4.0).

Full legal code: https://creativecommons.org/licenses/by/4.0/legalcode
Summary:         https://creativecommons.org/licenses/by/4.0/

See NOTICE for the required attribution and the list of changes, if any.
The CrispEmbed analyzer code that reads it is MIT-licensed and distributed
separately; no code is included here.
"""

CARD = """---
license: cc-by-4.0
language:
- ar
tags:
- arabic
- morphology
- morphological-analysis
- camel-morph
pretty_name: Camel Morph MSA v1.0 (CrispEmbed bundle)
---

# Camel Morph MSA v1.0 — CrispEmbed bundle

[Camel Morph MSA v1.0](https://github.com/CAMeL-Lab/camel_morph) © 2024 by
[New York University Abu Dhabi](https://nyuad.nyu.edu) is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This repository redistributes the Camel Morph MSA v1.0 ALMOR database as data
for CrispEmbed's native MIT-licensed port of the
[camel_tools](https://github.com/CAMeL-Lab/camel_tools) analyzer
(`crispembed_ar_morph_*` C ABI). No code is included.

| Folder | Content |
| --- | --- |
| `full/` | `{db}` byte-identical to the upstream release |
| `noprob/` | `{noprob_db}`: the same database without the `pos_logprob`, `lex_logprob` and `pos_lex_logprob` values |
| `proofing-bin/` | `{bin_db}`: the noprob database in CrispEmbed's memory-mappable binary format, reduced to the features proofing uses ({profile}) |

Each folder is a self-contained bundle: `manifest.json` pins every file by
size and SHA-256 and records the upstream source:

- repository: {repo}
- commit: `{commit}`
- path: `{path}`
- SHA-256: `{sha256}`

## Provenance

According to the LREC-COLING 2024 paper, the lexicon was seeded from
CALIMA-Star/SAMA and reworked by hand. camel_morph's `db_maker.py` writes
`lex_logprob` and `pos_lex_logprob` from log probabilities estimated on the
Penn Arabic Treebank training split (`pos_logprob` is declared but carries no
values). The `noprob/` variant drops those values for users who want no
treebank-derived statistics; analysis sets and all other features are
unchanged, while camel_tools reports the constant -99.0 for the dropped
fields, so likelihood-based ranking falls back to its tie-breakers.

## Citation

Christian Khairallah, Salam Khalifa, Reham Marzouk, Mayar Nassar and Nizar
Habash. 2024. *Camel Morph MSA: A Large-Scale Open-Source Morphological
Analyzer for Modern Standard Arabic.* LREC-COLING 2024.
https://aclanthology.org/2024.lrec-main.240
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def load_source(path: str | None) -> bytes:
    if path:
        data = Path(path).read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/CAMeL-Lab/camel_morph/{SOURCE_COMMIT}/{SOURCE_PATH}"
        with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 (pinned https URL)
            data = response.read(SOURCE_SIZE + 1)
    if len(data) != SOURCE_SIZE or sha256_bytes(data) != SOURCE_SHA256 or git_blob(data) != SOURCE_GIT_BLOB:
        sys.exit("source database does not match the pinned size/SHA-256/git blob")
    return data


def drop_prob_fields(data: bytes) -> tuple[bytes, int]:
    """Remove PROB_FIELDS tokens from PREFIXES/SUFFIXES/STEMS entry lines only."""
    out = []
    section = None
    removed = 0
    entry_sections = {"###PREFIXES###", "###SUFFIXES###", "###STEMS###"}
    for line in data.decode("utf-8").splitlines(keepends=True):
        body = line.rstrip("\n")
        if body.startswith("###"):
            section = body.strip()
            out.append(line)
            continue
        if section in entry_sections:
            parts = body.split("\t")
            if len(parts) != 3:
                sys.exit(f"unexpected entry line in {section}: {body[:80]!r}")
            toks = parts[2].split(" ")
            kept = [t for t in toks if t.split(":", 1)[0] not in PROB_FIELDS]
            removed += len(toks) - len(kept)
            parts[2] = " ".join(kept)
            line = "\t".join(parts) + line[len(body):]
        out.append(line)
    return "".join(out).encode("utf-8"), removed


def write_bundle(root: Path, variant: str, db_name: str, db_bytes: bytes, removed: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    changes = NOPROB_CHANGES if variant == "noprob" else ""
    files = {
        db_name: db_bytes,
        "NOTICE": NOTICE.format(repo=SOURCE_REPO, commit=SOURCE_COMMIT, path=SOURCE_PATH, sha256=SOURCE_SHA256,
                                changes=changes).encode(),
        "LICENSE": LICENSE.encode(),
    }
    for name, payload in files.items():
        (root / name).write_bytes(payload)
    manifest = {
        "format": "crispembed-camel-morph-bundle",
        "format_version": 1,
        "variant": variant,
        "db": db_name,
        "license": "CC-BY-4.0",
        "attribution": "Camel Morph MSA v1.0 (c) 2024 by New York University Abu Dhabi is licensed under CC BY 4.0",
        "source": {
            "repository": SOURCE_REPO,
            "commit": SOURCE_COMMIT,
            "last_change_commit": SOURCE_LAST_CHANGE,
            "path": SOURCE_PATH,
            "size": SOURCE_SIZE,
            "sha256": SOURCE_SHA256,
            "git_blob": SOURCE_GIT_BLOB,
        },
        "dropped_fields": list(PROB_FIELDS) if variant == "noprob" else [],
        "dropped_values": removed,
        "analyzer": {"reference": "camel_tools 1.6.0 Analyzer", "default_backoff": "NONE"},
        "files": [{"path": name, "size": len(payload), "sha256": sha256_bytes(payload)}
                  for name, payload in sorted(files.items())],
    }
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (root / "manifest.json").write_text(text, encoding="utf-8")
    return {"variant": variant, "manifest_sha256": sha256_bytes(text.encode()), "manifest": manifest}


# ---------------------------------------------------------------------------
# Compact binary proofing database (docs/arabic-morph-binary-format.md).
#
# The binary file holds the analyzer's parsed MorphologyDB for a feature
# profile: a sorted string table, fixed-width rows of per-column value ids,
# sorted form indexes and sorted category-compatibility pairs. It is derived
# from the noprob text database by a parser that mirrors the native text
# loader (src/arabic_morph.cpp, itself checked against camel_tools), and it
# is byte-identical on every run.

BIN_DB_NAME = "camel_morph_msa_v1.0.proofing.cmorph"
BIN_MAGIC = b"CEMORPHB"
BIN_FORMAT_VERSION = 1
BIN_HEADER_BYTES = 64
BIN_DB_FORMAT = "crispembed-camel-morph-bin-v1"

# Features the proofing variant keeps. Quds Office's Arabic grammar rules read
# pos, lex, cas, stt, num, gen, per, asp, mod, form_gen, form_num, prc0-prc3
# and enc0; diac, root, vox, rat, prc1.5, enc1, enc2 and source (whose
# computed value `spvar` marks spelling-variant readings) are kept as the
# remaining inflectional/clitic and lemma-identity features. Every other
# feature (gloss, bw, caphi, pattern*, cm_*, tokenizations, catib6/ud, stem*,
# *_logprob, correctness, mid_root_diac) is dropped.
PROOFING_PROFILE = ("diac", "lex", "root", "pos", "prc3", "prc2", "prc1", "prc1.5", "prc0", "per", "asp", "vox",
                    "mod", "form_gen", "gen", "form_num", "num", "stt", "cas", "enc0", "enc1", "enc2", "rat",
                    "source")
# Stored but never output: the backoff modes test `NOUN_PROP in bw` on the
# stems of the NOAN form and rewrite their bw, so bw is kept on those rows
# only (absent everywhere else).
AUX_STEM_FEATURES = {"bw": "NOAN"}
# camel_tools raises KeyError when a NOAN stem lacks one of these; the binary
# analyzer skips the dropped ones, so the converter requires them present.
NOAN_REQUIRED = ("diac", "lex", "bw", "gloss")
KEY_DEFINED, KEY_OUTPUT = 1, 2
CAT_COLUMN = 0xFFFFFFFF
NONE_VALUE = 0xFFFFFFFF


def _fail(what: str):
    sys.exit(f"invalid database: {what}")


class TextDb:
    """Mirror of the native loader (db_parser in src/arabic_morph.cpp)."""

    def __init__(self, data: bytes):
        text = data.decode("utf-8")
        text = text.replace("\r\n", "\n").replace("\r", "\n")  # universal newlines
        parts = text.split("\n")
        self.lines = [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
        self.defines: list[str] = []
        self.defaults: dict[str, list[tuple[str, str | None]]] = {}
        self.order: list[str] = []
        self.pattern_in_order = False
        self.stem_backoffs: dict[str, list[str]] = {}
        self.tables = {"prefix": [], "suffix": [], "stem": []}  # (form, cat, [(key, value)])
        self.compat = {"AB": [], "BC": [], "AC": []}
        self._parse()

    def _line(self, i: int, section: str) -> str:
        if i >= len(self.lines):
            _fail(f"missing section after {section}")
        return self.lines[i]

    @staticmethod
    def _set(feats: list[tuple[str, object]], key: str, value) -> None:
        for n, (k, _) in enumerate(feats):
            if k == key:
                feats[n] = (key, value)
                return
        feats.append((key, value))

    def _entry(self, toks: list[str], is_stem: bool) -> list[tuple[str, str]]:
        feats: list[tuple[str, str]] = []
        for tok in toks:
            if not tok:
                continue
            if ":" not in tok:
                _fail("invalid key value pair")
            key, value = tok.split(":", 1)
            if key in PROB_FIELDS:
                _fail("probability field in a proofing source")
            self._set(feats, key, value)
        if is_stem:
            n = next((n for n, (k, _) in enumerate(feats) if k == "lex"), None)
            if n is None:
                _fail("STEMS entry without lex")
            lex = feats[n][1]
            cut = min([p for p in (lex.find("_"), lex.find("-")) if p >= 0], default=-1)
            if cut >= 0:
                feats[n] = ("lex", lex[:cut])
        return feats

    def _parse(self) -> None:
        i = 0
        defined = set()
        while True:  # DEFINES
            line = self._line(i, "DEFINES").strip()
            i += 1
            if line == "###DEFINES###":
                continue
            if line == "###DEFAULTS###":
                break
            toks = line.split(" ")
            if len(toks) < 3 or toks[0] != "DEFINE":
                _fail("DEFINES line")
            if toks[1] not in defined:
                defined.add(toks[1])
                self.defines.append(toks[1])
        while True:  # DEFAULTS
            line = self._line(i, "DEFAULTS").strip()
            i += 1
            if line == "###ORDER###":
                break
            toks = line.split(" ")
            if len(toks) < 2 or toks[0] != "DEFAULT":
                _fail("DEFAULTS line")
            parsed: list[tuple[str, str | None]] = []
            for tok in toks[1:]:
                if ":" not in tok:
                    _fail("DEFAULTS value")
                key, value = tok.split(":", 1)
                self._set(parsed, key, None if value == "*" else value)
            pos = dict(parsed).get("pos", "missing")
            if pos == "missing":
                _fail("DEFAULTS line without pos")
            if pos is not None:
                self.defaults[pos] = parsed
        while True:  # ORDER
            line = self._line(i, "ORDER").strip()
            i += 1
            if line == "###TOKENIZATIONS###":
                break
            toks = line.split(" ")
            if len(toks) < 2 or toks[1] not in defined:
                _fail("ORDER line")
            self.order = toks[1:]
            self.pattern_in_order = "pattern" in toks[1:]
        if not self.order:
            _fail("missing ORDER")
        while True:  # TOKENIZATIONS
            line = self._line(i, "TOKENIZATIONS").strip()
            i += 1
            if line == "###STEMBACKOFF###":
                break
        while True:  # STEMBACKOFF
            line = self._line(i, "STEMBACKOFF").strip()
            i += 1
            if line == "###PREFIXES###":
                break
            toks = line.split(" ")
            if len(toks) < 3 or toks[0] != "STEMBACKOFF":
                _fail("STEMBACKOFF line")
            self.stem_backoffs[toks[1]] = toks[2:]
        for table, end in (("prefix", "###SUFFIXES###"), ("suffix", "###STEMS###")):
            while True:
                line = self._line(i, table)
                i += 1
                parts = line.split("\t")
                if len(parts) != 3:
                    if line.strip() == end:
                        break
                    _fail(f"invalid {table} line")
                self.tables[table].append((parts[0].strip(), parts[1],
                                           self._entry(parts[2].strip().split(" "), False)))
        while True:
            line = self._line(i, "STEMS").strip()
            i += 1
            if line == "###TABLE AB###":
                break
            parts = line.split("\t")
            if len(parts) != 3:
                _fail("invalid STEMS line")
            self.tables["stem"].append((parts[0], parts[1], self._entry(parts[2].split(" "), True)))
        for name, end in (("AB", "###TABLE BC###"), ("BC", "###TABLE AC###"), ("AC", None)):
            while True:
                if i >= len(self.lines):
                    if end:
                        _fail(f"missing marker after TABLE {name}")
                    break
                line = self.lines[i].strip()
                i += 1
                if end and line == end:
                    break
                toks = line.split()
                if len(toks) != 2:
                    _fail(f"invalid TABLE {name} line")
                self.compat[name].append((toks[0], toks[1]))


def _crc(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _pad8(buf: bytearray) -> None:
    buf.extend(b"\0" * (-len(buf) % 8))


def _width(count: int) -> int:
    # The all-ones value of each width marks an absent feature.
    return 1 if count < 0xFF else 2 if count < 0xFFFF else 4


def build_binary(text_db: bytes, profile=PROOFING_PROFILE) -> tuple[bytes, dict]:
    db = TextDb(text_db)
    profile = list(profile)
    for key in profile:
        if key not in db.defines:
            _fail(f"profile feature {key} is not DEFINEd")
    # fix_form copies form_gen/form_num into a '-' gen/num, so those values
    # depend on the form features.
    for key, form_key in (("gen", "form_gen"), ("num", "form_num")):
        if key in profile and form_key in db.defines and form_key not in profile:
            _fail(f"profile keeps {key} without {form_key}")
    # Columns follow an order consistent with every stem's own key order (the
    # output order); ties follow ORDER. Stored keys outside ORDER come last.
    names = list(dict.fromkeys([k for k in db.order if k in profile or k in AUX_STEM_FEATURES] +
                               profile + list(AUX_STEM_FEATURES)))
    rank = {k: n for n, k in enumerate(names)}
    after: dict[str, set[str]] = {k: set() for k in names}
    for form, _, feats in db.tables["stem"]:
        seq = [k for k, _ in feats if k in profile]
        for a, b in zip(seq, seq[1:]):
            after[a].add(b)
        if form == "NOAN":
            have = {k for k, _ in feats}
            for k in NOAN_REQUIRED:
                if k not in have:
                    _fail(f"NOAN stem without {k}")
    before = {k: sum(k in v for v in after.values()) for k in names}
    stored = []
    ready = sorted((k for k in names if before[k] == 0), key=rank.get)
    while ready:
        k = ready.pop(0)
        stored.append(k)
        for b in after[k]:
            before[b] -= 1
            if before[b] == 0:
                ready.append(b)
        ready.sort(key=rank.get)
    if len(stored) != len(names):
        _fail("stem key orders are inconsistent")
    key_index = {k: n for n, k in enumerate(stored)}
    column_rank = key_index
    for form, _, feats in db.tables["stem"]:
        seq = [column_rank[k] for k, _ in feats if k in profile]
        if seq != sorted(seq):
            _fail(f"stem key order differs from the column order for {form!r}")

    strings: set[str] = set(stored)
    cats: list[str] = []
    cat_id: dict[str, int] = {}

    def cat(name: str) -> int:
        if name not in cat_id:
            cat_id[name] = len(cats)
            cats.append(name)
        return cat_id[name]

    tables = {}
    for table in ("prefix", "suffix", "stem"):
        rows = db.tables[table]
        kept = []
        for form, category, feats in rows:
            vals = {}
            for k, v in feats:
                if k in profile or (table == "stem" and AUX_STEM_FEATURES.get(k) == form):
                    vals[k] = v
            kept.append((form, cat(category), vals))
        tables[table] = kept
    compat = {name: sorted({(cat(a), cat(b)) for a, b in pairs}) for name, pairs in db.compat.items()}
    for table, rows in tables.items():
        for form, _, vals in rows:
            strings.add(form)
            strings.update(vals.values())
    strings.update(cats)
    for pos, feats in db.defaults.items():
        strings.add(pos)
        strings.update(v for k, v in feats if k in profile and v is not None)
    for name, names in db.stem_backoffs.items():
        strings.add(name)
        strings.update(names)
    ordered = sorted(strings, key=lambda s: s.encode("utf-8"))
    sid = {s: n for n, s in enumerate(ordered)}

    sections: list[tuple[bytes, bytes]] = []
    # STRS
    buf = bytearray(struct.pack("<I", len(ordered)))
    blob = bytearray()
    offsets = [0]
    for s in ordered:
        blob += s.encode("utf-8")
        offsets.append(len(blob))
    buf += struct.pack(f"<{len(offsets)}I", *offsets)
    buf += blob
    sections.append((b"STRS", bytes(buf)))
    # KEYS
    buf = bytearray(struct.pack("<I", len(stored)))
    for k in stored:
        flags = (KEY_DEFINED if k in db.defines else 0) | (KEY_OUTPUT if k in profile else 0)
        buf += struct.pack("<II", sid[k], flags)
    sections.append((b"KEYS", bytes(buf)))
    # CATS
    sections.append((b"CATS", struct.pack(f"<I{len(cats)}I", len(cats), *[sid[c] for c in cats])))
    # DFLT (output features only, in the DEFAULT line's order)
    buf = bytearray(struct.pack("<I", len(db.defaults)))
    for pos in sorted(db.defaults, key=lambda s: s.encode("utf-8")):
        feats = [(k, v) for k, v in db.defaults[pos] if k in profile]
        buf += struct.pack("<II", sid[pos], len(feats))
        for k, v in feats:
            buf += struct.pack("<II", key_index[k], NONE_VALUE if v is None else sid[v])
    sections.append((b"DFLT", bytes(buf)))
    # BKOF
    buf = bytearray(struct.pack("<I", len(db.stem_backoffs)))
    for name in sorted(db.stem_backoffs, key=lambda s: s.encode("utf-8")):
        names = db.stem_backoffs[name]
        buf += struct.pack(f"<II{len(names)}I", sid[name], len(names), *[sid[c] for c in names])
    sections.append((b"BKOF", bytes(buf)))
    # Entry tables
    stats = {}
    for tag, table in ((b"PREF", "prefix"), (b"SUFF", "suffix"), (b"STEM", "stem")):
        rows = tables[table]
        groups: dict[str, list[int]] = {}
        for n, (form, _, _) in enumerate(rows):
            groups.setdefault(form, []).append(n)  # file order inside a form
        forms = sorted(groups, key=lambda s: s.encode("utf-8"))
        order = [n for f in forms for n in groups[f]]
        present = [k for k in stored if any(k in vals for _, _, vals in rows)]
        columns = [(CAT_COLUMN, sorted({c for _, c, _ in rows}))]
        for k in present:
            columns.append((key_index[k], sorted({sid[vals[k]] for _, _, vals in rows if k in vals})))
        layout = []
        offset = 0
        for key, values in columns:
            w = _width(len(values))
            layout.append((key, w, offset, values, {v: n for n, v in enumerate(values)}))
            offset += w
        row_bytes = offset
        buf = bytearray(struct.pack("<IIII", len(rows), len(forms), len(layout), row_bytes))
        for key, w, off, values, _ in layout:
            buf += struct.pack("<IIII", key, w, off, len(values))
        for _, _, _, values, _ in layout:
            buf += struct.pack(f"<{len(values)}I", *values)
        _pad8(buf)
        first = 0
        for f in forms:
            buf += struct.pack("<III", sid[f], first, len(groups[f]))
            first += len(groups[f])
        _pad8(buf)
        fmt = {1: "B", 2: "H", 4: "I"}
        for n in order:
            _, c, vals = rows[n]
            for key, w, _, _, local in layout:
                if key == CAT_COLUMN:
                    v = local[c]
                else:
                    name = stored[key]
                    v = local[sid[vals[name]]] if name in vals else (1 << (8 * w)) - 1
                buf += struct.pack("<" + fmt[w], v)
        sections.append((tag, bytes(buf)))
        stats[table] = {"rows": len(rows), "forms": len(forms), "row_bytes": row_bytes,
                        "columns": {("category" if k == CAT_COLUMN else stored[k]): w for k, w, _, _, _ in layout}}
    for tag, name in ((b"TBAB", "AB"), (b"TBBC", "BC"), (b"TBAC", "AC")):
        pairs = compat[name]
        sections.append((tag, struct.pack(f"<II{len(pairs)}Q", len(pairs), 0, *[(a << 32) | b for a, b in pairs])))

    directory = bytearray()
    body = bytearray()
    offset = BIN_HEADER_BYTES + 24 * len(sections)
    offset += -offset % 8
    for tag, payload in sections:
        directory += struct.pack("<4sIQQ", tag, _crc(payload), offset + len(body), len(payload))
        body += payload
        _pad8(body)
    file_bytes = offset + len(body)
    flags = 1 if db.pattern_in_order and "pattern" in profile else 0
    header = bytearray(struct.pack("<8sIIIIQII", BIN_MAGIC, BIN_FORMAT_VERSION, BIN_HEADER_BYTES, len(sections),
                                   flags, file_bytes, _crc(bytes(directory)), 0))
    header += b"\0" * (BIN_HEADER_BYTES - len(header))
    struct.pack_into("<I", header, 36, _crc(bytes(header)))
    out = bytes(header) + bytes(directory) + b"\0" * (-(BIN_HEADER_BYTES + len(directory)) % 8) + bytes(body)
    assert len(out) == file_bytes
    stats.update({"strings": len(ordered), "string_bytes": len(blob), "categories": len(cats),
                  "keys": stored, "profile": profile,
                  "compat_pairs": {k: len(v) for k, v in compat.items()}, "file_bytes": file_bytes})
    return out, stats


PROOFING_CHANGES = (
    "Changes in this variant (CC BY 4.0 section 3(a)(1)(B)): derived from the noprob variant (no "
    "pos_logprob/lex_logprob/pos_lex_logprob values) and converted to CrispEmbed's binary analyzer format "
    "({fmt}). Only these features are kept: {profile}. bw is kept on the NOAN backoff stems only. All other "
    "features (gloss, bw, caphi, pattern, pattern_abstract, cm_*, catib6, ud, the tokenization features, "
    "correctness, mid_root_diac) were removed. Prefixes, suffixes, stems, categories, compatibility tables, "
    "DEFAULT and STEMBACKOFF entries are otherwise unchanged."
)


def write_bin_bundle(root: Path, noprob: bytes) -> dict:
    data, stats = build_binary(noprob)
    again, _ = build_binary(noprob)
    if again != data:
        sys.exit("binary database is not reproducible")
    root.mkdir(parents=True, exist_ok=True)
    changes = "\n" + textwrap.fill(PROOFING_CHANGES.format(fmt=BIN_DB_FORMAT, profile=", ".join(PROOFING_PROFILE)),
                                   width=79, break_on_hyphens=False) + "\n"
    files = {
        BIN_DB_NAME: data,
        "NOTICE": NOTICE.format(repo=SOURCE_REPO, commit=SOURCE_COMMIT, path=SOURCE_PATH, sha256=SOURCE_SHA256,
                                changes=changes).encode(),
        "LICENSE": LICENSE.encode(),
    }
    for name, payload in files.items():
        (root / name).write_bytes(payload)
    manifest = {
        "format": "crispembed-camel-morph-bundle",
        "format_version": 2,
        "variant": "proofing-bin",
        "db": BIN_DB_NAME,
        "db_format": BIN_DB_FORMAT,
        "license": "CC-BY-4.0",
        "attribution": "Camel Morph MSA v1.0 (c) 2024 by New York University Abu Dhabi is licensed under CC BY 4.0",
        "source": {
            "repository": SOURCE_REPO,
            "commit": SOURCE_COMMIT,
            "last_change_commit": SOURCE_LAST_CHANGE,
            "path": SOURCE_PATH,
            "size": SOURCE_SIZE,
            "sha256": SOURCE_SHA256,
            "git_blob": SOURCE_GIT_BLOB,
        },
        "derived_from": {"variant": "noprob", "db": NOPROB_DB_NAME, "size": len(noprob),
                         "sha256": sha256_bytes(noprob)},
        "profile": list(PROOFING_PROFILE),
        "dropped_fields": list(PROB_FIELDS),
        "analyzer": {"reference": "camel_tools 1.6.0 Analyzer", "default_backoff": "NONE",
                     "output": "the profile features of every camel_tools analysis, in camel_tools' order"},
        "files": [{"path": name, "size": len(payload), "sha256": sha256_bytes(payload)}
                  for name, payload in sorted(files.items())],
    }
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (root / "manifest.json").write_text(text, encoding="utf-8")
    return {"variant": "proofing-bin", "manifest_sha256": sha256_bytes(text.encode()), "manifest": manifest,
            "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="local copy of the pinned database (default: download the pinned commit)")
    ap.add_argument("--output", required=True, help="output directory (Hugging Face repository layout)")
    args = ap.parse_args()

    data = load_source(args.source)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    noprob, removed = drop_prob_fields(data)
    results = [write_bundle(out / "full", "full", DB_NAME, data, 0),
               write_bundle(out / "noprob", "noprob", NOPROB_DB_NAME, noprob, removed),
               write_bin_bundle(out / "proofing-bin", noprob)]
    fmt = dict(db=DB_NAME, noprob_db=NOPROB_DB_NAME, bin_db=BIN_DB_NAME, profile=", ".join(PROOFING_PROFILE), repo=SOURCE_REPO, commit=SOURCE_COMMIT, path=SOURCE_PATH,
               sha256=SOURCE_SHA256)
    (out / "README.md").write_text(CARD.format(**fmt), encoding="utf-8")
    (out / "NOTICE").write_text(NOTICE.format(repo=SOURCE_REPO, commit=SOURCE_COMMIT, path=SOURCE_PATH,
                                              sha256=SOURCE_SHA256, changes=NOPROB_CHANGES), encoding="utf-8")
    (out / "LICENSE").write_text(LICENSE, encoding="utf-8")
    summary = {r["variant"]: {"manifest_sha256": r["manifest_sha256"],
                              "db_sha256": next(f["sha256"] for f in r["manifest"]["files"]
                                                if f["path"] == r["manifest"]["db"]),
                              "dropped_values": r["manifest"].get("dropped_values")} for r in results}
    summary["proofing-bin"]["stats"] = results[2]["stats"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
