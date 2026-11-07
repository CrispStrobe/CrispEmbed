#!/usr/bin/env python3
"""Publish the owner-approved UltraBERT-Embed-150M Q4_K GGUF to Hugging Face.

Driven by .github/workflows/ultrabert-publish.yml; protocol docs/ultrabert-publication-v1.json.
The published bytes are exactly the GGUF measured by the committed receipts (retained payload
artifact of run 40843704011, scored on all 5,483 SciFact rows by run 40871150002); nothing is
re-quantized. Every gate before the `publish` Hub import is stdlib-only:

  plan VARIANT                    protocol/approval and receipt joins (model SHA, recipe, quality)
  extract VARIANT ZIP DIR         pick the member whose bytes/SHA256 equal the receipt's model
  notices DIR                     fetch pinned LICENSE text and upstream card; verify bytes + SHA256
  publish VARIANT GGUF_DIR NOTICES_DIR OUT
                                  card + manifest, upload, LFS identity, empty-cache readback
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "docs/ultrabert-publication-v1.json"
SCHEMA = "crispembed-ultrabert-publication-v1"
DESTINATION = "cstr/UltraBERT-Embed-150M-CrispEmbed-GGUF"
FIXED_FILES = ("README.md", "LICENSE", "UPSTREAM_README.md", "manifest.json", ".gitattributes")
Q4K_BAR = 0.94  # docs/precision-acceptance.md


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def load(root=ROOT):
    cfg = json.loads((root / PROTOCOL).read_text())
    require(cfg.get("schema") == SCHEMA, "unexpected protocol schema")
    require(cfg.get("approved_by_owner") is True, "publication not approved by the owner; nothing accessed")
    require(cfg.get("destination") == DESTINATION, "unreviewed publication destination")
    up = cfg["upstream"]
    require(re.fullmatch("[0-9a-f]{40}", up["revision"]) and up["license"] == "apache-2.0", "upstream pin malformed")
    for pin in (up["readme"], up["license_file"]):
        require(re.fullmatch("[0-9a-f]{64}", pin["sha256"]) and pin["bytes"] > 0, "notice pin malformed")
    return cfg


def evidence(cfg, variant, root=ROOT):
    """Join the protocol's model pin to the model receipt and the full-panel quality receipt."""
    require(variant in cfg["variants"], "unknown variant: " + variant)
    v = cfg["variants"][variant]
    m = json.loads((root / v["receipts"]["model"]).read_text())
    require(m["model"] == v["model"] and m["original_f32"] == v["original_f32"] and m["recipe"] == v["recipe"],
            "model receipt does not match the protocol pin")
    require(m["payload_artifact"]["id"] == v["source_artifact"]["id"] and
            m["payload_artifact"]["digest"] == v["source_artifact"]["digest"], "payload artifact pin differs")
    q = json.loads((root / v["receipts"]["quality"]).read_text())
    a = q["aggregate"]
    require(a["completed"] is True and a["recipe"] == v["recipe"] and a["model"] == v["model"],
            "quality receipt is not the complete panel for this model")
    cos = a["normalized"]["min_row_cosine"]
    require(cos > Q4K_BAR, f"min embedding cosine {cos} is not above the Q4_K bar {Q4K_BAR}")
    return {"model_receipt": v["receipts"]["model"], "model_receipt_sha256": digest(root / v["receipts"]["model"]),
            "quality_receipt": v["receipts"]["quality"], "quality_receipt_sha256": digest(root / v["receipts"]["quality"]),
            "quality_run": q["run"], "runtime_source": q["runtime_source"], "build_mode": q["build_mode"],
            "rows": a["rows"], "queries": a["queries"], "raw": a["raw"], "normalized": a["normalized"],
            "quality": a["quality"], "stored_types": m["actual_stored_types"], "bar": Q4K_BAR}


def extract(cfg, variant, archive, out_dir):
    want = cfg["variants"][variant]["model"]
    out = Path(out_dir) / cfg["variants"][variant]["filename"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.file_size != want["bytes"]:
                continue
            with z.open(info) as src, out.open("wb") as dst:
                for block in iter(lambda: src.read(8 * 2**20), b""):
                    dst.write(block)
            if digest(out) == want["sha256"]:
                return info.filename
            out.unlink()
    raise ValueError("no artifact member matches the receipt's model bytes and SHA256")


def fetch(url, expected, opener=urllib.request.urlopen):
    with opener(url, timeout=60) as resp:
        data = resp.read(expected["bytes"] + 1)
    require(len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"],
            "pinned upstream notice differs: " + url)
    return data


def notices(cfg, out_dir, opener=urllib.request.urlopen):
    up, out = cfg["upstream"], Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "LICENSE").write_bytes(fetch(up["license_file"]["url"], up["license_file"], opener))
    readme = f"https://huggingface.co/{up['repo']}/resolve/{up['revision']}/README.md"
    (out / "UPSTREAM_README.md").write_bytes(fetch(readme, up["readme"], opener))


def card(cfg, variant, f, ev, source_commit):
    up, v = cfg["upstream"], cfg["variants"][variant]
    q, o, n = ev["quality"], ev["quality"]["original"], ev["quality"]["native"]
    gates = " ".join(cfg["runtime"]["gates"])
    mib = lambda b: f"{b / 2**20:,.1f} MiB"
    return f"""---
license: apache-2.0
license_link: LICENSE
base_model: {up['repo']}
base_model_relation: quantized
pipeline_tag: sentence-similarity
library_name: gguf
tags: [gguf, crispembed, embeddings, ultrabert, q4_k, experimental]
---
# UltraBERT-Embed-150M Q4_K GGUF (CrispEmbed, EXPERIMENTAL)

A quantized GGUF of [{up['repo']}](https://huggingface.co/{up['repo']}/tree/{up['revision']})
by {up['authors']}, for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)'s default-off native UltraBERT
text-embedding path. All credit for the model belongs to the original authors.

**Status: experimental.** It clears CrispEmbed's Q4_K precision bar (final-embedding cosine > {ev['bar']}) and
retrieval quality is close to the original, but it fails the stricter F32 vector and top-1 agreement gates (below).

> **Do not mix Q4_K vectors into an index built with F32 (or the original model) vectors.** Per-row cosine to the
> original can drop to {ev['normalized']['min_row_cosine']:.4f} and top-1 agreement is {q['top1_agreement']:.3f};
> re-embed the whole corpus with the same file you query with.

## Quality (SciFact, {ev['queries']} queries over {ev['rows']:,} rows, vs the original F32 PyTorch model)

| Precision | Size | Min embedding cosine | Mean embedding cosine | nDCG@10 | MRR@10 | Recall@100 | Top-1 agreement | Strict F32 vector gate | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Original (F32) | {mib(v['original_f32']['bytes'])} | 1 | 1 | {o['ndcg_at_10']:.4f} | {o['mrr_at_10']:.4f} | {o['recall_at_100']:.4f} | reference | reference | reference |
| Q4_K `{v['recipe']}` (this file) | {mib(f['bytes'])} | {ev['normalized']['min_row_cosine']:.4f} | not measured | {n['ndcg_at_10']:.4f} ({n['ndcg_at_10'] - o['ndcg_at_10']:+.4f}) | {n['mrr_at_10']:.4f} ({n['mrr_at_10'] - o['mrr_at_10']:+.4f}) | {n['recall_at_100']:.4f} ({n['recall_at_100'] - o['recall_at_100']:+.4f}) | {q['top1_agreement']:.3f} | fail (max abs normalized {ev['normalized']['max_absolute']:.4f}) | experimental |

- Measured by CrispEmbed run {ev['quality_run']} (runtime commit `{ev['runtime_source'][:12]}`, {ev['build_mode'].split(' (')[0]} build,
  hosted x86-64 CPU) on exactly these bytes. The protocol's own strict gates (F32 vector budget, top-1 >= 0.99,
  task degradation <= 0.001) fail and stay recorded; publication follows the owner's precision policy.
- Not measured: GPU backends, ARM/Apple, other corpora/languages, long-context behaviour, latency.

## Files

| File | Precision | Bytes | SHA256 |
|---|---|---:|---|
| `{f['filename']}` | {variant} | {f['bytes']:,} | `{f['sha256']}` |

- {v['note']} Stored types: {', '.join(f'{k} {c}' for k, c in ev['stored_types'].items())}.

## License and attribution

- The original model is released under the **Apache License 2.0**; this derived file is distributed under the same
  license. The model repository declares `license: apache-2.0` but ships no LICENSE or NOTICE file, so `LICENSE` is
  the verbatim canonical Apache-2.0 text (SHA256 `{up['license_file']['sha256']}`) and no NOTICE is included.
- `UPSTREAM_README.md` is the unmodified upstream model card at revision `{up['revision']}`
  (SHA256 `{up['readme']['sha256']}`).

## Statement of changes (Apache-2.0 section 4(b))

Modified from the original by CrispEmbed: the `model.safetensors` weights of revision `{up['revision']}` were
**format-converted** to GGUF (`models/convert-ultrabert-to-gguf.py`, F32, tokenizer metadata embedded; F32 SHA256
`{v['original_f32']['sha256']}`) and then **quantized** with CrispEmbed's `crispembed-quantize` (recipe above).
No weights were trained, fine-tuned, merged or otherwise altered.

## How the file was produced

1. Converted and quantized by CrispEmbed run {v['source_artifact']['run']}; the file is copied unchanged from that
   run's retained artifact and checked against the receipt's size and SHA256.
2. Uploaded by CrispEmbed commit `{source_commit}` (workflow `ultrabert-publish.yml`, protocol `{PROTOCOL}`);
   `manifest.json` lists the pins, receipts and file hashes.

## Usage

UltraBERT support is CPU-only, experimental and default-off:

```bash
{gates} \\
  crispembed -m {f['filename']} "your text"
```
"""


def publish(cfg, variant, gguf_dir, notice_dir, out, source_commit):
    require(re.fullmatch("[0-9a-f]{40}", source_commit or ""), "exact source commit required")
    ev = evidence(cfg, variant)
    v, up = cfg["variants"][variant], cfg["upstream"]
    path = Path(gguf_dir) / v["filename"]
    require(path.is_file() and path.stat().st_size == v["model"]["bytes"] and digest(path) == v["model"]["sha256"],
            "GGUF differs from the measured model")
    f = {"variant": variant, "filename": v["filename"], "bytes": v["model"]["bytes"], "sha256": v["model"]["sha256"]}
    lic, readme = Path(notice_dir) / "LICENSE", Path(notice_dir) / "UPSTREAM_README.md"
    require(digest(lic) == up["license_file"]["sha256"] and digest(readme) == up["readme"]["sha256"], "notices differ")
    card_bytes = card(cfg, variant, f, ev, source_commit).encode()
    man = {"schema": "crispembed-ultrabert-publication-manifest-v1", "destination": DESTINATION,
           "source_commit": source_commit, "protocol_sha256": digest(ROOT / PROTOCOL), "upstream": up,
           "variant": {variant: v}, "files": [f], "evidence": ev,
           "readme_sha256": hashlib.sha256(card_bytes).hexdigest()}
    man_bytes = (json.dumps(man, indent=2) + "\n").encode()
    small = {"README.md": card_bytes, "manifest.json": man_bytes, "LICENSE": lic.read_bytes(),
             "UPSTREAM_README.md": readme.read_bytes()}
    for name, data in small.items():
        (out / name).write_bytes(data)
    expected = {f["filename"]: f["sha256"], **{n: hashlib.sha256(b).hexdigest() for n, b in small.items()}}

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    require(os.environ.get("HF_TOKEN"), "publication credential unavailable")
    api = HfApi(token=os.environ["HF_TOKEN"])
    allowed = {*FIXED_FILES, *(x["filename"] for x in cfg["variants"].values())}
    if api.repo_exists(DESTINATION):
        require({s.rfilename for s in api.model_info(DESTINATION).siblings} <= allowed,
                "destination holds files outside the allowlist")
    else:
        api.create_repo(DESTINATION, private=False, exist_ok=False)
    ops = [CommitOperationAdd(path_in_repo=f["filename"], path_or_fileobj=str(path))]
    ops += [CommitOperationAdd(path_in_repo=n, path_or_fileobj=b) for n, b in small.items()]
    commit = api.create_commit(repo_id=DESTINATION, operations=ops,
                               commit_message=f"Publish UltraBERT-Embed-150M {variant} GGUF (experimental)")
    info = api.model_info(DESTINATION, revision=commit.oid, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    require(set(remote) <= allowed, "remote file set outside the allowlist")
    s = remote[f["filename"]]
    require(s.size == f["bytes"] and s.lfs and s.lfs.sha256 == f["sha256"], "remote LFS identity differs")
    cache = Path(os.environ.get("HEAVY_SCRATCH", out)) / "fresh-hf-cache"
    cache.mkdir(parents=True, exist_ok=False)
    for name, sha in expected.items():
        p = Path(hf_hub_download(DESTINATION, name, revision=commit.oid, cache_dir=cache, force_download=True,
                                 token=False))
        require(digest(p) == sha, "empty-cache readback differs: " + name)
    return {"repo": DESTINATION, "commit": commit.oid, "files": [f],
            "small_files_sha256": {n: expected[n] for n in small},
            "urls": {f["filename"]: f"https://huggingface.co/{DESTINATION}/resolve/{commit.oid}/{f['filename']}"}}


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    cfg = load()
    if cmd == "plan" and len(argv) == 3:
        print(json.dumps(evidence(cfg, argv[2]), indent=1))
        return
    if cmd == "extract" and len(argv) == 5:
        print(extract(cfg, argv[2], argv[3], argv[4]))
        return
    if cmd == "notices" and len(argv) == 3:
        notices(cfg, argv[2])
        return
    require(cmd == "publish" and len(argv) == 6, "usage: plan V | extract V ZIP DIR | notices DIR | publish V GGUF NOTICES OUT")
    out = Path(argv[5])
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "crispembed-ultrabert-publication-report-v1", "variant": argv[2], "status": "started"}
    save = lambda: (out / "publication.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        save()
        report["remote"] = publish(cfg, argv[2], argv[3], argv[4], out, os.environ.get("SOURCE_COMMIT"))
        report["status"] = "published and re-verified"
        save()
    except BaseException as e:
        report.update(status="failed or rejected", error=f"{type(e).__name__}: {e}")
        save()
        raise


if __name__ == "__main__":
    main(sys.argv)
