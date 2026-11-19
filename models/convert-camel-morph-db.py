#!/usr/bin/env python3
"""Build the Camel Morph MSA v1.0 database bundles for the native analyzer.

Developer tooling only; nothing here runs in an app. The database is
CC BY 4.0 data published by the CAMeL Lab (New York University Abu Dhabi); it
is pinned here by git commit, git blob id, size and SHA-256 and is never
modified in the ``full`` variant.

    python models/convert-camel-morph-db.py --output out/          # downloads the pinned DB
    python models/convert-camel-morph-db.py --source camel_morph_msa_v1.0.db --output out/

Output layout (one Hugging Face repository, two self-contained bundles):

    out/README.md                 Hugging Face card (license: cc-by-4.0)
    out/NOTICE, out/LICENSE       attribution and licence pointer
    out/full/                     the database byte-for-byte as released
    out/noprob/                   the same database without the PATB-derived
                                  probability fields (see PROB_FIELDS)

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
import sys
import urllib.request

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
               write_bundle(out / "noprob", "noprob", NOPROB_DB_NAME, noprob, removed)]
    fmt = dict(db=DB_NAME, noprob_db=NOPROB_DB_NAME, repo=SOURCE_REPO, commit=SOURCE_COMMIT, path=SOURCE_PATH,
               sha256=SOURCE_SHA256)
    (out / "README.md").write_text(CARD.format(**fmt), encoding="utf-8")
    (out / "NOTICE").write_text(NOTICE.format(repo=SOURCE_REPO, commit=SOURCE_COMMIT, path=SOURCE_PATH,
                                              sha256=SOURCE_SHA256, changes=NOPROB_CHANGES), encoding="utf-8")
    (out / "LICENSE").write_text(LICENSE, encoding="utf-8")
    summary = {r["variant"]: {"manifest_sha256": r["manifest_sha256"],
                              "db_sha256": next(f["sha256"] for f in r["manifest"]["files"]
                                                if f["path"] == r["manifest"]["db"]),
                              "dropped_values": r["manifest"]["dropped_values"]} for r in results}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
