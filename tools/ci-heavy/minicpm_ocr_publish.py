#!/usr/bin/env python3
"""Publish MiniCPM-OCR-1.0 GGUFs (openbmb/MiniCPM-OCR-1.0, Apache-2.0) to Hugging Face.

Driven by .github/workflows/minicpm-ocr-publish.yml. Every gate below the
`publish` subcommand's Hub import is stdlib-only and runs before any credential,
package or weight access:

  plan VARIANTS                 protocol/approval, variant recipes, parity receipt and
                                per-variant precision receipts; prints the build plan
  notices DIR                   fetch the upstream LICENSE (pinned code-repo commit) and
                                README (pinned model revision); verify bytes + SHA256
  publish VARIANTS GGUF_DIR NOTICES_DIR OUT
                                hash every file, render card + manifest, upload to the
                                allowlisted destination, verify immutable LFS identity
                                and an empty-cache readback of every file

Docs: docs/minicpm-ocr-publication-v1.json (protocol), docs/minicpm-ocr-source.json
(upstream pin), tests/results/minicpm-ocr/native-parity.json (parity receipt).
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "docs/minicpm-ocr-publication-v1.json"
SCHEMA = "crispembed-minicpm-ocr-publication-v1"
DESTINATION = "cstr/MiniCPM-OCR-1.0-GGUF"
FIXED_FILES = ("README.md", "LICENSE", "UPSTREAM_README.md", "manifest.json", ".gitattributes")


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
    source = json.loads((root / cfg["source_contract"]).read_text())
    up = cfg["upstream"]
    require(source["repository"] == up["repo"] and source["revision"] == up["revision"],
            "protocol upstream pin differs from the source contract")
    require(re.fullmatch("[0-9a-f]{40}", up["revision"]) and up["license"] == "apache-2.0",
            "upstream pin or license malformed")
    require(source["files"]["README.md"]["sha256"] == up["readme"]["sha256"]
            and source["files"]["README.md"]["bytes"] == up["readme"]["bytes"], "README pin differs from the source contract")
    lf = up["license_file"]
    require(re.fullmatch("[0-9a-f]{40}", lf["commit"]) and lf["commit"] in lf["url"], "LICENSE pin malformed")
    return cfg, source


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
    require(path.is_file(), f"{variant} has no precision receipt yet ({rec['precision_receipt']}); publish F16 only")
    r = json.loads(path.read_text())
    require(r.get("completed") is True and r.get("schema") == "crispembed-minicpm-ocr-bench-v1",
            f"{variant} benchmark is not a completed declared producer; publish F16 only")
    require(r.get("reference_revision") == cfg["upstream"]["revision"] and
            re.fullmatch("[0-9a-f]{40}", r.get("commit", "")), "benchmark source/original revision differs")
    frozen = json.loads((root / cfg["receipts"]["parity"]).read_text())
    require(r.get("max_new_tokens") == frozen["max_new_tokens"], "benchmark output cap differs from frozen parity")
    require([c["id"] for c in frozen["cases"]] == cfg["parity_cases"], "frozen original case coverage differs")
    counts = {c["id"]: len(c["reference_ids"]) for c in frozen["cases"]}
    arms = [a for a in r.get("arms", []) if a.get("dtype") == rec["precision_dtype"]]
    names = [a.get("name") for a in arms]
    required = rec.get("precision_required_arms", [])
    require(required and len(required) == len(set(required)) and len(names) == len(set(names)) and
            set(names) == set(required), f"{variant} precision selected arm coverage differs: {names}")
    for a in arms:
        ids = sorted(c["id"] for c in a.get("cases", []))
        require(ids == sorted(cfg["parity_cases"]), f"{variant} precision arm {a.get('name')} misses cases")
        require(all(c.get("exit") == 0 and c.get("ids_exact") is True and c.get("text_exact") is True and
                    c.get("cer") == 0 and c.get("first_id_difference") is None and
                    c.get("n_ref_tokens") == counts[c["id"]] and c.get("n_tokens") == counts[c["id"]]
                    for c in a["cases"]),
                f"{variant} is not a successful token-ID/text-exact panel in {rec['precision_receipt']} "
                f"({a.get('name')}); publish F16 only")
    return {"receipt": rec["precision_receipt"], "receipt_sha256": digest(path), "commit": r.get("commit"),
            "arms": [a.get("name") for a in arms], "ids_exact": True,
            "completed_benchmark": True, "text_exact": True,
            "exactness_evidence": "Legacy producer compared actual generated IDs but retained only predicates/counts; "
                                  "this publication guard does not rederive missing ID vectors."}


def plan(cfg, variants):
    """Ordered build steps: F32 is converted whenever any variant quantizes from it."""
    need_f32 = "F32" in variants or any(cfg["recipes"][v]["from"] == "F32" for v in variants)
    steps = []
    if need_f32:
        steps.append(("convert", "F32", cfg["recipes"]["F32"]["filename"], cfg["recipes"]["F32"]["convert_args"]))
    for v in variants:
        rec = cfg["recipes"][v]
        if v == "F32":
            continue
        if rec["from"] == "convert":
            steps.append(("convert", v, rec["filename"], rec["convert_args"]))
        else:
            tt = [a for o in rec.get("tensor_type", []) for a in ("--tensor-type", o)]
            steps.append(("quantize", v, rec["filename"], [rec["type"], *tt]))
    steps.sort(key=lambda s: s[0] != "convert")
    return steps, need_f32 and "F32" not in variants


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
    up = cfg["upstream"]
    rows = "\n".join(f"| `{f['filename']}` | {f['variant']} | {f['bytes']:,} | `{f['sha256']}` |" for f in files)
    lf = up["license_file"]
    cases = ", ".join(f"`{k}` ({n})" for k, n in parity["cases"].items())
    names = [f["variant"] for f in files]
    quant = ("" if not any(prec.get(v) for v in names) else
             "\n- " + "; ".join(f"**{v}** is token-ID exact on the same five cases (receipt `{prec[v]['receipt']}`)"
                                 for v in names if prec.get(v)) + ".")
    only_f16 = "" if any(prec.get(v) for v in names) else (
        "\n- Only unquantized precisions are published: no quantized variant has a token-ID-exact precision receipt yet.")
    return f"""---
license: apache-2.0
license_link: LICENSE
base_model: {up['repo']}
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags: [gguf, ocr, crispembed, minicpm-ocr, qwen3.5, siglip]
---
# MiniCPM-OCR-1.0 GGUF (CrispEmbed)

GGUF conversions of [{up['repo']}](https://huggingface.co/{up['repo']}/tree/{up['revision']})
by OpenBMB, for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)'s native ggml runtime.
All credit for the model belongs to the original authors (OpenBMB); these files only change the storage format.
They are **not** the llama.cpp files from `openbmb/MiniCPM-OCR-1.0-gguf` and are not loadable by llama.cpp.

## License and attribution

- The original model is released under the **Apache License 2.0**, {up['copyright']}.
  These derived files are distributed under the same license.
- `LICENSE` is the verbatim upstream Apache-2.0 text. The model repository declares `license: apache-2.0` but ships no
  LICENSE file, so it is taken from the upstream code repository
  [{lf['code_repo'].removeprefix('https://github.com/')}@{lf['commit'][:12]}]({lf['code_repo']}/blob/{lf['commit']}/LICENSE)
  (SHA256 `{lf['sha256']}`). Upstream ships no NOTICE file, so none is included.
- `UPSTREAM_README.md` is the unmodified upstream model card at revision `{up['revision']}`
  (SHA256 `{up['readme']['sha256']}`).

## Statement of changes (Apache-2.0 section 4(b))

These files were modified from the original by CrispEmbed: the BF16 safetensors weights of revision
`{up['revision']}` were **format-converted** to GGUF (tensor renaming, BF16 to F32/F16 casting, tokenizer and
preprocessing metadata embedded){' and, where noted, **quantized**' if any(f['variant'] not in ('F32', 'F16') for f in files) else ''}.
No weights were trained, fine-tuned, merged or otherwise altered.

## Files

| File | Precision | Bytes | SHA256 |
|---|---|---:|---|
{rows}

## How the files were produced

1. Downloaded `{up['repo']}` at the pinned revision `{up['revision']}`; every file was checked against the size and
   SHA256 pins in CrispEmbed's `docs/minicpm-ocr-source.json`.
2. Converted with `models/convert-minicpm-ocr-to-gguf.py` (`--dtype f32` lossless; `--dtype f16`; norms, biases and
   other 1-D tensors stay F32).
3. Quantized variants, if any, were produced from the F32 GGUF with CrispEmbed's own `crispembed-quantize`.
4. Built and uploaded by CrispEmbed commit `{source_commit}` (workflow `minicpm-ocr-publish.yml`);
   `manifest.json` lists every recipe, input pin, receipt and file hash.

## Parity status (CrispEmbed native vs. the original Transformers model, CPU, greedy)

- The **F32 and F16** conversions reproduce the original model's generated token IDs **exactly** on all
  {len(parity['cases'])} reference cases ({cases} tokens; Transformers {parity['reference']['versions'].get('transformers')},
  up to {parity['reference']['max_new_tokens']} new tokens), CrispEmbed commit `{parity['commit'][:8]}`.{quant}{only_f16}
- Not verified: GPU backends, multi-page documents, task-level quality benchmarks, prompts other than the reference tasks.

## Usage

MiniCPM-OCR support is experimental and default-off in CrispEmbed; it reuses the Qwen3.5 decoder gate:

```bash
CRISPEMBED_MINICPM_OCR=1 CRISPEMBED_QWEN35_OCR=1 \\
  crispembed -m minicpm-ocr-1.0-f16.gguf --ocr page.png --ocr-prompt "Text Recognition:"
```
"""


def manifest(cfg, files, parity, prec, source_commit, card_sha):
    return {"schema": "crispembed-minicpm-ocr-publication-manifest-v1", "destination": cfg["destination"],
            "source_commit": source_commit, "upstream": cfg["upstream"],
            "recipes": {f["variant"]: cfg["recipes"][f["variant"]] for f in files},
            "files": files, "readme_sha256": card_sha, "parity": parity,
            "precision": {v: p for v, p in prec.items() if p}, "protocol_sha256": digest(ROOT / PROTOCOL)}


def publish(cfg, variants, gguf_dir, notice_dir, out, source_commit):
    require(re.fullmatch("[0-9a-f]{40}", source_commit or ""), "exact source commit required")
    parity = receipts(cfg)
    prec = {v: precision(cfg, v) for v in variants}
    files = []
    for v in variants:
        name = cfg["recipes"][v]["filename"]
        path = Path(gguf_dir) / name
        require(path.is_file() and path.stat().st_size > 5 * 10**8, "missing or truncated GGUF: " + name)
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
                               commit_message="Publish MiniCPM-OCR-1.0 GGUF: " + ", ".join(variants))
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
    cfg, _ = load()
    if cmd == "plan":
        variants = parse_variants(argv[2] if len(argv) > 2 else "", cfg)
        receipts(cfg)
        for v in variants:
            precision(cfg, v)
        steps, drop_f32 = plan(cfg, variants)
        for kind, v, name, args in steps:
            print(kind, v, name, *args)
        print("drop-f32" if drop_f32 else "keep-f32")
        return
    if cmd == "notices":
        notices(cfg, argv[2])
        return
    require(cmd == "publish" and len(argv) == 6, "usage: plan VARIANTS | notices DIR | publish VARIANTS GGUF_DIR NOTICES_DIR OUT")
    out = Path(argv[5])
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "crispembed-minicpm-ocr-publication-report-v1", "status": "started"}
    save = lambda: (out / "publication.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        variants = parse_variants(argv[2], cfg)
        report.update(variants=variants)
        save()
        report["remote"] = publish(cfg, variants, argv[3], argv[4], out, os.environ.get("SOURCE_COMMIT"))
        report["status"] = "published and re-verified"
        save()
    except BaseException as e:
        report.update(status="failed or rejected", error=f"{type(e).__name__}: {e}")
        save()
        raise


if __name__ == "__main__":
    main(sys.argv)
