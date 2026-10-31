#!/usr/bin/env python3
"""Publish an experimental OCR model's GGUFs to Hugging Face from a reviewed protocol.

Driven by .github/workflows/ocr-gguf-publish.yml. Everything model-specific lives
in the protocol JSON (destination, upstream pins, converter, recipes, receipts,
card text): docs/mineru25-pro-publication-v1.json, docs/teleocr-publication-v1.json.
Every gate below the `publish` subcommand's Hub import is stdlib-only and runs
before any credential, package or weight access:

  plan PROTOCOL VARIANTS        protocol/approval, variant recipes, parity receipt and
                                per-variant precision receipts; prints the build plan
  notices PROTOCOL DIR          fetch the upstream LICENSE text and model card at their
                                pins; verify bytes + SHA256
  publish PROTOCOL VARIANTS GGUF_DIR NOTICES_DIR OUT
                                hash every file, render card + manifest, upload to the
                                allowlisted destination, verify immutable LFS identity
                                and an empty-cache readback of every file

Receipts share one schema (tools/mineru25_receipts.py, tools/teleocr_receipts.py):
`cases[].reference_ids` and `cases[].arms[arm] = {exit, ids_exact, ids}`; precision
receipts list `arms[] = {dtype, cases[] = {id, ids_exact}}`.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "crispembed-ocr-gguf-publication-v1"
FIXED_FILES = ("README.md", "LICENSE", "UPSTREAM_README.md", "manifest.json", ".gitattributes")
ALLOWED_DESTINATIONS = {"cstr/MinerU2.5-Pro-2604-1.2B-GGUF", "cstr/TeleOCR-GGUF"}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def load(protocol, root=ROOT):
    path = root / protocol
    cfg = json.loads(path.read_text())
    require(cfg.get("schema") == SCHEMA, "unexpected protocol schema")
    require(cfg.get("approved_by_owner") is True, "publication not approved by the owner; nothing accessed")
    require(cfg.get("destination") in ALLOWED_DESTINATIONS, "unreviewed publication destination")
    up = cfg["upstream"]
    require(re.fullmatch("[0-9a-f]{40}", up["revision"]) and up["license"] == "apache-2.0",
            "upstream pin or license malformed")
    lf = up["license_file"]
    require(re.fullmatch("[0-9a-f]{64}", lf["sha256"]) and lf["url"].startswith("https://"), "LICENSE pin malformed")
    require(re.fullmatch("[0-9a-f]{64}", up["readme"]["sha256"]), "README pin malformed")
    conv = cfg["converter"]
    require((root / conv["script"]).is_file() and "{src}" in conv["args"] and "{out}" in conv["args"],
            "converter invocation malformed")
    for name, rec in cfg["recipes"].items():
        require(rec["from"] in ("convert", *cfg["recipes"]) and rec["from"] != name, f"recipe {name} source invalid")
    cfg["_protocol"] = protocol
    return cfg


def parse_variants(text, cfg):
    names = [v.strip().upper() for v in re.split(r"[,\s]+", text or "") if v.strip()]
    if not names:
        names = list(cfg["default_variants"])
    require(len(set(names)) == len(names), "duplicate variant")
    unknown = [n for n in names if n not in cfg["recipes"]]
    require(not unknown, "unknown variant(s): " + ",".join(unknown) + "; allowed: " + ",".join(cfg["recipes"]))
    return names


def receipts(cfg, root=ROOT):
    """Parity summary from the committed receipt; refuses unless every case/arm is token-ID exact."""
    path = root / cfg["receipts"]["parity"]
    r = json.loads(path.read_text())
    require(r.get("completed") is True, "parity receipt incomplete")
    require(r["reference_revision"] == cfg["upstream"]["revision"], "parity receipt revision differs")
    require(re.fullmatch("[0-9a-f]{40}", r.get("commit", "")), "parity receipt commit malformed")
    require([c["id"] for c in r["cases"]] == cfg["parity_cases"], "parity receipt case set differs")
    tokens = {}
    for c in r["cases"]:
        for arm in cfg["parity_arms"]:
            a = c["arms"].get(arm)
            require(a and a["exit"] == 0 and a["ids_exact"] is True and a["ids"] == c["reference_ids"],
                    f"parity receipt not token-ID exact: {c['id']}/{arm}")
        tokens[c["id"]] = len(c["reference_ids"])
    return {"status": "token-id-exact", "arms": list(cfg["parity_arms"]), "cases": tokens, "commit": r["commit"],
            "reference": {"revision": r["reference_revision"], "versions": r["reference_versions"],
                          "max_new_tokens": r["max_new_tokens"]},
            "receipt_sha256": digest(path)}


def precision(cfg, variant, root=ROOT):
    """A recipe with a precision_receipt is publishable only if that receipt shows it token-ID exact."""
    rec = cfg["recipes"][variant]
    if "precision_receipt" not in rec:
        return None
    path = root / rec["precision_receipt"]
    require(path.is_file(), f"{variant} has no precision receipt yet ({rec['precision_receipt']})")
    r = json.loads(path.read_text())
    arms = [a for a in r.get("arms", []) if a.get("dtype") == rec["precision_dtype"]]
    require(arms, f"{variant} precision receipt has no {rec['precision_dtype']} arm")
    for a in arms:
        ids = sorted(c["id"] for c in a.get("cases", []))
        require(ids == sorted(cfg["parity_cases"]), f"{variant} precision arm {a.get('name')} misses cases")
        require(all(c.get("ids_exact") is True for c in a["cases"]),
                f"{variant} is not token-ID exact in {rec['precision_receipt']} ({a.get('name')})")
    return {"receipt": rec["precision_receipt"], "receipt_sha256": digest(path), "commit": r.get("commit"),
            "arms": [a.get("name") for a in arms], "ids_exact": True}


def plan(cfg, variants):
    """Ordered build steps: conversions first (including every quantization source), then quantizations.

    Returns (steps, drop) where drop lists intermediate sources built only to quantize from.
    """
    needed, order = set(variants), []
    for v in variants:
        src = cfg["recipes"][v]["from"]
        while src != "convert":
            needed.add(src)
            src = cfg["recipes"][src]["from"]
    steps = []
    for v in cfg["recipes"]:
        if v not in needed:
            continue
        rec = cfg["recipes"][v]
        if rec["from"] == "convert":
            steps.append(("convert", v, rec["filename"], rec["convert_args"]))
        else:
            tt = [a for o in rec.get("tensor_type", []) for a in ("--tensor-type", o)]
            steps.append(("quantize", v, rec["filename"],
                          [cfg["recipes"][rec["from"]]["filename"], rec["type"], *tt, *rec.get("extra_args", [])]))
    steps.sort(key=lambda s: s[0] != "convert")
    drop = [cfg["recipes"][v]["filename"] for v in cfg["recipes"] if v in needed and v not in variants]
    return steps, drop


def fetch(url, expected, opener=urllib.request.urlopen):
    with opener(url, timeout=60) as resp:
        data = resp.read(expected["bytes"] + 1)
    require(len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"],
            "pinned upstream notice differs: " + url)
    return data


def notices(cfg, out_dir, opener=urllib.request.urlopen):
    up = cfg["upstream"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "LICENSE").write_bytes(fetch(up["license_file"]["url"], up["license_file"], opener))
    readme = f"https://huggingface.co/{up['repo']}/resolve/{up['revision']}/README.md"
    (out / "UPSTREAM_README.md").write_bytes(fetch(readme, up["readme"], opener))


def card(cfg, files, parity, prec, source_commit):
    up, c = cfg["upstream"], cfg["card"]
    rows = "\n".join(f"| `{f['filename']}` | {f['variant']} | {f['bytes']:,} | `{f['sha256']}` |" for f in files)
    cases = ", ".join(f"`{k}` ({n})" for k, n in parity["cases"].items())
    names = [f["variant"] for f in files]
    exact = [v for v in names if prec.get(v)]
    reduced = ("" if not exact else "\n- " + "; ".join(
        f"**{v}** is token-ID exact on the same {len(parity['cases'])} cases (receipt `{prec[v]['receipt']}`)"
        for v in exact) + ".")
    quantized = any(cfg["recipes"][v].get("type") for v in names)
    recipes = "\n".join(f"- **{v}**: {cfg['recipes'][v].get('note', 'see manifest.json')}" for v in names)
    first = files[0]["filename"]
    return f"""---
license: apache-2.0
license_link: LICENSE
base_model: {up['repo']}
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags: [{', '.join(['gguf', 'ocr', 'crispembed', *c['tags']])}]
---
# {c['title']} GGUF (CrispEmbed)

GGUF conversions of [{up['repo']}](https://huggingface.co/{up['repo']}/tree/{up['revision']})
by {c['authors']}, for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)'s native ggml runtime.
All credit for the model belongs to the original authors; these files only change the storage format.
{c['runtime_note']}

## License and attribution

- The original model is released under the **Apache License 2.0** ({up['copyright']}).
  These derived files are distributed under the same license.
- `LICENSE`: {c['license_note']} (SHA256 `{up['license_file']['sha256']}`). Upstream ships no NOTICE file,
  so none is included.
- `UPSTREAM_README.md` is the unmodified upstream model card at revision `{up['revision']}`
  (SHA256 `{up['readme']['sha256']}`).

## Statement of changes (Apache-2.0 section 4(b))

These files were modified from the original by CrispEmbed: the safetensors weights of revision
`{up['revision']}` were **format-converted** to GGUF (tensor renaming, casting, tokenizer and
preprocessing metadata embedded){' and, where noted, **quantized**' if quantized else ''}.
No weights were trained, fine-tuned, merged or otherwise altered.

## Files

| File | Precision | Bytes | SHA256 |
|---|---|---:|---|
{rows}

{recipes}

## How the files were produced

1. Downloaded `{up['repo']}` at the pinned revision `{up['revision']}`; the converter
   (`{cfg['converter']['script']}`) checks the source files, tensor names and shapes.
2. Quantized variants were produced with CrispEmbed's own `crispembed-quantize` from the listed source GGUF.
3. Built and uploaded by CrispEmbed commit `{source_commit}` (workflow `ocr-gguf-publish.yml`,
   protocol `{cfg['_protocol']}`); `manifest.json` lists every recipe, input pin, receipt and file hash.

## Parity status (CrispEmbed native vs. the original Python model, CPU, greedy)

- The reference arms ({', '.join(parity['arms'])}) reproduce the original model's generated token IDs
  **exactly** on all {len(parity['cases'])} frozen requests ({cases} tokens), CrispEmbed commit
  `{parity['commit'][:8]}`, reference revision `{parity['reference']['revision'][:12]}`.{reduced}
- {c['not_verified']}

## Usage

{c['usage'].replace('{first}', first)}
"""


def manifest(cfg, files, parity, prec, source_commit, card_sha):
    return {"schema": "crispembed-ocr-gguf-publication-manifest-v1", "destination": cfg["destination"],
            "protocol": cfg["_protocol"], "source_commit": source_commit, "upstream": cfg["upstream"],
            "recipes": {f["variant"]: cfg["recipes"][f["variant"]] for f in files},
            "files": files, "readme_sha256": card_sha, "parity": parity,
            "precision": {v: p for v, p in prec.items() if p}, "protocol_sha256": digest(ROOT / cfg["_protocol"])}


def publish(cfg, variants, gguf_dir, notice_dir, out, source_commit):
    require(re.fullmatch("[0-9a-f]{40}", source_commit or ""), "exact source commit required")
    parity = receipts(cfg)
    prec = {v: precision(cfg, v) for v in variants}
    files = []
    for v in variants:
        name = cfg["recipes"][v]["filename"]
        path = Path(gguf_dir) / name
        require(path.is_file() and path.stat().st_size > 3 * 10**8, "missing or truncated GGUF: " + name)
        files.append({"variant": v, "filename": name, "bytes": path.stat().st_size, "sha256": digest(path)})
    lic, readme = Path(notice_dir) / "LICENSE", Path(notice_dir) / "UPSTREAM_README.md"
    up = cfg["upstream"]
    require(digest(lic) == up["license_file"]["sha256"] and digest(readme) == up["readme"]["sha256"],
            "upstream notices differ")
    card_bytes = card(cfg, files, parity, prec, source_commit).encode()
    card_sha = hashlib.sha256(card_bytes).hexdigest()
    man_bytes = (json.dumps(manifest(cfg, files, parity, prec, source_commit, card_sha), indent=2) + "\n").encode()
    small = {"README.md": card_bytes, "manifest.json": man_bytes, "LICENSE": lic.read_bytes(),
             "UPSTREAM_README.md": readme.read_bytes()}
    (out / "README.md").write_bytes(card_bytes)
    (out / "manifest.json").write_bytes(man_bytes)
    expected = {f["filename"]: f["sha256"] for f in files}
    expected.update({n: hashlib.sha256(b).hexdigest() for n, b in small.items()})

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    require(os.environ.get("HF_TOKEN"), "publication credential unavailable")
    api = HfApi(token=os.environ["HF_TOKEN"])
    allowed = {*FIXED_FILES, *(r["filename"] for r in cfg["recipes"].values())}
    if api.repo_exists(cfg["destination"]):
        current = {s.rfilename for s in api.model_info(cfg["destination"]).siblings}
        require(current <= allowed, "destination holds files outside the allowlist")
    else:
        api.create_repo(cfg["destination"], private=False, exist_ok=False)
    ops = [CommitOperationAdd(path_in_repo=f["filename"], path_or_fileobj=str(Path(gguf_dir) / f["filename"]))
           for f in files]
    ops += [CommitOperationAdd(path_in_repo=n, path_or_fileobj=b) for n, b in small.items()]
    commit = api.create_commit(repo_id=cfg["destination"], operations=ops,
                               commit_message=f"Publish {cfg['card']['title']} GGUF: " + ", ".join(variants))
    info = api.model_info(cfg["destination"], revision=commit.oid, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    require(set(remote) <= allowed, "remote file set outside the allowlist")
    for f in files:
        s = remote[f["filename"]]
        require(s.size == f["bytes"] and s.lfs and s.lfs.sha256 == f["sha256"], "remote LFS identity differs")
    cache = Path(os.environ.get("HEAVY_SCRATCH", out)) / "fresh-hf-cache"
    cache.mkdir(parents=True, exist_ok=False)
    for name, sha in expected.items():
        p = Path(hf_hub_download(cfg["destination"], name, revision=commit.oid, cache_dir=cache,
                                 force_download=True, token=False))
        require(digest(p) == sha, "empty-cache readback differs: " + name)
        if name.endswith(".gguf"):
            p.unlink()
    return {"repo": cfg["destination"], "commit": commit.oid, "files": files,
            "small_files_sha256": {n: expected[n] for n in small},
            "urls": {f["filename"]: f"https://huggingface.co/{cfg['destination']}/resolve/{commit.oid}/{f['filename']}"
                     for f in files}}


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    require(len(argv) > 2, "usage: plan PROTOCOL VARIANTS | notices PROTOCOL DIR | "
                           "publish PROTOCOL VARIANTS GGUF_DIR NOTICES_DIR OUT")
    cfg = load(argv[2])
    if cmd == "plan":
        variants = parse_variants(argv[3] if len(argv) > 3 else "", cfg)
        receipts(cfg)
        for v in variants:
            precision(cfg, v)
        steps, drop = plan(cfg, variants)
        for kind, v, name, args in steps:
            print(kind, v, name, *args)
        for name in drop:
            print("drop", "-", name)
        return
    if cmd == "notices":
        notices(cfg, argv[3])
        return
    require(cmd == "publish" and len(argv) == 7, "usage: publish PROTOCOL VARIANTS GGUF_DIR NOTICES_DIR OUT")
    out = Path(argv[6])
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "crispembed-ocr-gguf-publication-report-v1", "protocol": argv[2], "status": "started"}
    save = lambda: (out / "publication.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        variants = parse_variants(argv[3], cfg)
        report.update(variants=variants)
        save()
        report["remote"] = publish(cfg, variants, argv[4], argv[5], out, os.environ.get("SOURCE_COMMIT"))
        report["status"] = "published and re-verified"
        save()
    except BaseException as e:
        report.update(status="failed or rejected", error=f"{type(e).__name__}: {e}")
        save()
        raise


if __name__ == "__main__":
    main(sys.argv)
