#!/usr/bin/env python3
"""Publish Unlimited-OCR-2 GGUFs (baidu/Unlimited-OCR-2, MIT) to Hugging Face.

Driven by .github/workflows/unlimited-ocr2-publish.yml. Every gate below the
`publish` subcommand's Hub import is stdlib-only and runs before any credential,
package or weight access:

  plan VARIANTS                 protocol/approval, variant recipes and parity receipts;
                                prints the shell-friendly build plan (no network)
  notices DIR                   fetch the upstream LICENSE (pinned code-repo commit) and
                                README (pinned model revision); verify bytes + SHA256
  publish VARIANTS GGUF_DIR NOTICES_DIR OUT
                                hash every file, render card + manifest, upload to the
                                allowlisted destination, verify immutable LFS identity
                                and an empty-cache readback of every file

Docs: docs/unlimited-ocr2-publication-v1.json (protocol), docs/unlimited-ocr2-source.json
(upstream pin), tests/results/unlimited-ocr2/*.json (parity receipts).
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "docs/unlimited-ocr2-publication-v1.json"
SCHEMA = "crispembed-unlimited-ocr2-publication-v1"
DESTINATION = "cstr/Unlimited-OCR-2-GGUF"
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
    require(re.fullmatch("[0-9a-f]{40}", up["revision"]) and up["license"] == "mit", "upstream pin or license malformed")
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
    """Parity summary from the committed receipts; refuses if they do not show the claimed state."""
    r = {k: json.loads((root / p).read_text()) for k, p in cfg["receipts"].items()}
    lim = cfg["parity_limits"]["tiled_max_box_coordinate_difference"]
    require(r["original"]["revision"] == cfg["upstream"]["revision"], "reference receipt revision differs")
    require(r["untiled"]["ids_exact"] is True, "untiled receipt is not ID-exact")
    tiled = [c for c in r["tiled"]["cases"] if c["tiled"]]
    untiled = [c for c in r["tiled"]["cases"] if not c["tiled"]]
    require(tiled and all(c["ids_exact"] for c in untiled), "tiling receipt untiled case not exact")
    require(all(c["ids_exact"] for c in tiled), "tiled receipt is not token-ID exact")
    require(all(c["fast_ids_equal_exact"] for c in r["fast_decoder"]["cases"]), "fast decoder differs from exact")
    return {
        "untiled": {"case": r["untiled"]["case"], "ids_exact": True, "tokens": r["untiled"]["tokens"],
                    "commit": r["untiled"]["commit"]},
        "tiled": {"status": "token-id-exact", "cases": [c["id"] for c in tiled], "ids_exact": True,
                  "commit": r["tiled"]["commit"]},
        "fast_decoder": {"ids_equal_exact": True, "cases": len(r["fast_decoder"]["cases"]),
                         "commit": r["fast_decoder"]["commit"]},
        "reference": {"revision": r["original"]["revision"], "versions": r["original"]["versions"]},
        "receipt_sha256": {k: digest(root / p) for k, p in cfg["receipts"].items()},
    }


def plan(cfg, variants):
    """Ordered build steps: F32 is converted whenever any variant quantizes from it."""
    need_f32 = "F32" in variants or any(cfg["recipes"][v]["from"] == "F32" for v in variants)
    steps = []
    if need_f32:
        steps.append(("convert", "F32", cfg["recipes"]["F32"]["filename"], []))
    for v in variants:
        rec = cfg["recipes"][v]
        if v == "F32":
            continue
        if rec["from"] == "convert":
            steps.append(("convert", v, rec["filename"], rec["convert_args"]))
        else:
            tt = [a for o in rec.get("tensor_type", []) for a in ("--tensor-type", o)]
            steps.append(("quantize", v, rec["filename"], [rec["type"], *tt]))
    # Every conversion before the first quantization: the workflow frees the source shards then.
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


def card(cfg, files, parity, source_commit):
    up = cfg["upstream"]
    rows = "\n".join(f"| `{f['filename']}` | {f['variant']} | {f['bytes']:,} | `{f['sha256']}` |" for f in files)
    lf = up["license_file"]
    t = parity["tiled"]
    text_proof = parity.get("trial_text")
    text_status = ""
    if text_proof:
        text_status = f"""
- **Raw decoded output:** original IDs and complete raw UTF-8 text are byte-exact on all four
  synthetic tiled F16 pages at runtime `{text_proof['source'][:8]}`. Default tiling was used.
  Published GGUFs contain only the128000 base pieces; this runtime supplies the pinned831
  missing added pieces, including `<PAGE>`, `<BLOCK>`, `<FORK>` and table `<tr>/<td>` markup.
  The converter now writes added-token metadata for future conversions; these published weights
  are unchanged and continue to require that runtime fallback. `OUTPUT=legacy` preserves the old
  marker-dropping output; `OUTPUT=clean` is a separate layout-stripped rendering retaining table HTML.
"""
    return f"""---
license: mit
license_link: LICENSE
base_model: {up['repo']}
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
tags: [gguf, ocr, crispembed, deepseek-v2, moe]
---
# Unlimited-OCR-2 GGUF (CrispEmbed)

GGUF conversions of [{up['repo']}](https://huggingface.co/{up['repo']}/tree/{up['revision']})
by Baidu, for [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed)'s native ggml runtime.
All credit for the model belongs to the original authors; these files only change the storage format.

## License and attribution

- The original model is released under the **MIT License**, Copyright (c) 2026 Baidu.
- `LICENSE` is the verbatim upstream MIT text. The model repository declares `license: mit` but ships no
  LICENSE file, so it is taken from the upstream code repository
  [{lf['code_repo'].removeprefix('https://github.com/')}@{lf['commit'][:12]}]({lf['code_repo']}/blob/{lf['commit']}/LICENSE)
  (SHA256 `{lf['sha256']}`).
- `UPSTREAM_README.md` is the unmodified upstream model card at revision `{up['revision']}`
  (SHA256 `{up['readme']['sha256']}`).
- These GGUFs are a modified (converted and, where noted, quantized) form of the original weights.

## Files

| File | Precision | Bytes | SHA256 |
|---|---|---:|---|
{rows}

## How the files were produced

1. Downloaded `{up['repo']}` at the pinned revision `{up['revision']}`; both safetensors shards were
   checked against the SHA256 values in CrispEmbed's `docs/unlimited-ocr2-source.json`.
2. Converted with `models/convert-unlimited-ocr-to-gguf.py` (F32 lossless; F16 with `--fp16`).
3. Quantized variants were produced from the F32 GGUF with CrispEmbed's own `crispembed-quantize`
   (`tools/quantize.cpp`; 1-D tensors keep their source type, embeddings/router/lm_head/vision keep a Q8_0 floor).
4. Built and uploaded by CrispEmbed commit `{source_commit}` (workflow `unlimited-ocr2-publish.yml`);
   `manifest.json` lists every recipe, input pin, receipt and file hash.

## Parity status (CrispEmbed native vs. the original PyTorch model, CPU, greedy)

Receipts were measured on the **F32** conversion. A CPU precision sweep (receipt
`tests/results/unlimited-ocr2/precision-sweep.json` in CrispEmbed) found the **F16 and Q8_0**
conversions also token-ID exact on the untiled page; Q4_K and mixed Q4 variants were not, and are not published.

- **Untiled pages:** generated token IDs are exactly identical to the original
  ({parity['untiled']['tokens']} tokens on `{parity['untiled']['case']}`).
- **Tiled pages:** generated token IDs are exactly identical to the original on {len(t['cases'])} tiled
  pages (4x3 tiles, 1503 vision tokens), from CrispEmbed commit `{t['commit'][:8]}` on. Earlier runtimes
  emitted 256 instead of 100 vision tokens per 640px tile and matched the text only; the weights were never affected.
- The opt-in fast decoder reproduces the exact decoder token-for-token on all {parity['fast_decoder']['cases']} pages.
- **Published files, user-style trial** (Kaggle CPU, receipt `tests/results/unlimited-ocr2/trial.json`
  in CrispEmbed; four tiled pages compared with the original): **F16 is token-ID exact on all 4**
  (paragraphs, table, receipt, German), with the fast decoder equal to the exact decoder everywhere.
  **Q8_0 is exact on the untiled panel page but diverges on 2 of the 4 tiled trial pages** (table at
  generated token 7, receipt at tokens 131 and 258). Each divergence is one bounding-box coordinate
  token off by 1-3 (for example `102` vs `99`); sequence length and recognized text are unchanged.
  **Use F16 when exact agreement with the original matters**; Q8_0 is smaller and faster.
{text_status}- Not verified: GPU backends, long multi-page documents, task-level quality benchmarks.

## Usage

Unlimited-OCR-2 support is default-off in CrispEmbed and is enabled with an environment gate:

```bash
CRISPEMBED_UNLIMITED_OCR2=1 crispembed -m unlimited-ocr-2-f16.gguf --ocr page.png
# pages larger than 640 px are tiled like the original (crop_mode=None);
# CRISPEMBED_UNLIMITED_OCR2_TILES=0 forces the faster untiled view.
# Output is the original's raw decode (layout markup kept);
# CRISPEMBED_UNLIMITED_OCR2_OUTPUT=clean strips <PAGE>/<BLOCK>label[box]<FORK> and keeps tables.
```
"""


def manifest(cfg, files, parity, source_commit, card_sha):
    return {"schema": "crispembed-unlimited-ocr2-publication-manifest-v1", "destination": cfg["destination"],
            "source_commit": source_commit, "upstream": cfg["upstream"],
            "recipes": {f["variant"]: cfg["recipes"][f["variant"]] for f in files},
            "files": files, "readme_sha256": card_sha, "parity": parity,
            "protocol_sha256": digest(ROOT / PROTOCOL)}



def trial_text_receipt(root=ROOT):
    """Original bytes/IDs, not self-reported parity flags; runs before Hub access."""
    path = root / "tests/results/unlimited-ocr2/verify-text-v2.json"
    verified = json.loads(path.read_text())
    trial = json.loads((root / "tests/results/unlimited-ocr2/trial.json").read_text())
    expected_source = "8ca9d3de2a8160e6f5235156d3550c9719583744"
    require(verified.get("schema") == "crispembed-unlimited-ocr2-verify-v1" and
            verified.get("commit") == expected_source and verified.get("scope") == "text",
            "reviewed exact text verification source/scope required")
    require(set(verified["tests"]) == {"test-unlimited-ocr2-text", "test-unlimited-ocr2-tiles"} and
            all(row["exit"] == 0 for row in verified["tests"].values()), "compiled text/tiling controls failed")
    require(verified["page_sha_matches_trial"] and all(verified["page_sha_matches_trial"].values()),
            "verification page bytes differ from original trial")
    references = {row["id"]: row for row in trial["reference"]["cases"]}
    cases = [row for row in verified["runs"] if row["tag"] == "after" and row["alias"] == "unlimited-ocr2"]
    names = {"table", "receipt", "paragraphs", "german"}
    require(len(cases) == 4 and {row["page"] for row in cases} == names, "complete four-page text verification required")
    for row in cases:
        original = references[row["page"]]
        ids = original["ids"] if isinstance(original["ids"], list) else json.loads(original["ids"])
        require(row["exit"] == 0 and row["ids"] == ids and
                row["text"].encode("utf-8") == original["text"].encode("utf-8") and
                "CRISPEMBED_UNLIMITED_OCR2_TILES" not in row["env"] and
                "CRISPEMBED_UNLIMITED_OCR2_OUTPUT" not in row["env"],
                "default F16 original IDs/raw text differ: " + row["page"])
    return {"source": expected_source, "scope": "four synthetic tiled pages, published F16",
            "default_ids_exact": True, "raw_text_bytes_exact": True,
            "cases": sorted(names), "receipt": str(path.relative_to(root)), "receipt_sha256": digest(path),
            "published_tokenizer_uses_runtime_added_token_fallback": True}


def refresh_payload(cfg, previous, parity, verification):
    """Pure card-only render. Weight identities and producing source stay frozen."""
    require(previous.get("schema") == "crispembed-unlimited-ocr2-publication-manifest-v1" and
            previous.get("destination") == cfg["destination"], "previous manifest schema/destination differs")
    files, original_source = previous["files"], previous["source_commit"]
    require(re.fullmatch("[0-9a-f]{40}", original_source or ""), "previous producing source malformed")
    require(files and all(f["variant"] in cfg["recipes"] and f["filename"] == cfg["recipes"][f["variant"]]["filename"] and
                         type(f["bytes"]) is int and f["bytes"] > 10**9 and re.fullmatch("[0-9a-f]{64}", f["sha256"])
                         for f in files), "previous weight identities malformed")
    updated = dict(parity, trial_text=verification)
    readme = card(cfg, files, updated, original_source).encode("utf-8")
    data = manifest(cfg, files, updated, original_source, hashlib.sha256(readme).hexdigest())
    require(data["files"] == previous["files"] and data["source_commit"] == previous["source_commit"],
            "card refresh changed weight/source identities")
    return {"README.md": readme, "manifest.json": (json.dumps(data, indent=2) + "\n").encode()}


def refresh_card(cfg, out, root=ROOT):
    verification = trial_text_receipt(root)  # BEFORE credentials, client import or any network.
    publication = json.loads((root / "tests/results/unlimited-ocr2/publication.json").read_text())
    old = publication["card_update"]
    parent = old["hub_commit"]
    require(re.fullmatch("[0-9a-f]{40}", parent), "exact existing Hub parent required")
    def small(name, revision, expected=None):
        url = f"https://huggingface.co/{cfg['destination']}/resolve/{revision}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(1024 * 1024 + 1)
        require(len(data) <= 1024 * 1024, "card/notice metadata exceeds1MiB")
        if expected: require(hashlib.sha256(data).hexdigest() == expected, "immutable small file SHA differs: " + name)
        return data
    previous_bytes = small("manifest.json", parent, old["manifest.json"])
    previous = json.loads(previous_bytes)
    require(previous["files"] == publication["remote"]["files"], "previous manifest/weight receipt differs")
    small("README.md", parent, old["README.md"])
    notices = {"LICENSE": cfg["upstream"]["license_file"]["sha256"],
               "UPSTREAM_README.md": cfg["upstream"]["readme"]["sha256"]}
    for name, expected in notices.items(): small(name, parent, expected)
    payload = refresh_payload(cfg, previous, receipts(cfg, root), verification)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items(): (out / name).write_bytes(data)
    state = {"status": "prepared", "previous_hub_commit": parent, "verification": verification,
             "changed_files": sorted(payload), "weights_unchanged": True}
    def save_state():
        (out / "card-refresh-state.json").write_text(json.dumps(state, indent=2) + "\n")
    save_state()
    from huggingface_hub import CommitOperationAdd, HfApi
    require(os.environ.get("HF_TOKEN"), "card publication credential unavailable")
    api = HfApi(token=os.environ["HF_TOKEN"])
    current = api.model_info(cfg["destination"], revision="main", files_metadata=True)
    require(current.sha == parent, "Hub parent advanced; review before a new card refresh")
    def weights(info):
        remote = {row.rfilename: row for row in info.siblings}
        for row in previous["files"]:
            actual = remote[row["filename"]]
            require(actual.size == row["bytes"] and actual.lfs and actual.lfs.sha256 == row["sha256"],
                    "weight identity changed during card-only refresh")
    weights(current)
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=data) for name, data in payload.items()]
    require(len(operations) == 2 and set(payload) == {"README.md", "manifest.json"}, "only card/manifest may change")
    commit = api.create_commit(repo_id=cfg["destination"], operations=operations,
                               parent_commit=parent, commit_message="Clarify original raw decode, tiling and F16/Q8 trial limits")
    state.update(status="uploaded; immutable readback pending", hub_commit=commit.oid)
    save_state()
    weights(api.model_info(cfg["destination"], revision=commit.oid, files_metadata=True))
    result = {"hub_commit": commit.oid, "previous_hub_commit": parent, "changed_files": sorted(payload),
              "weights_unchanged": True, "notices_unchanged": True, "source_commit": previous["source_commit"],
              "verification": verification}
    for name, data in payload.items():
        expected = hashlib.sha256(data).hexdigest(); small(name, commit.oid, expected); result[name] = expected
    for name, expected in notices.items(): small(name, commit.oid, expected)
    state.update(status="uploaded and immutable readback verified", notices_unchanged=True)
    save_state()
    return result


def publish(cfg, variants, gguf_dir, notice_dir, out, source_commit):
    require(re.fullmatch("[0-9a-f]{40}", source_commit or ""), "exact source commit required")
    parity = receipts(cfg)
    files = []
    for v in variants:
        name = cfg["recipes"][v]["filename"]
        path = Path(gguf_dir) / name
        require(path.is_file() and path.stat().st_size > 10**9, "missing or truncated GGUF: " + name)
        files.append({"variant": v, "filename": name, "bytes": path.stat().st_size, "sha256": digest(path)})
    lic, readme = Path(notice_dir) / "LICENSE", Path(notice_dir) / "UPSTREAM_README.md"
    up = cfg["upstream"]
    require(digest(lic) == up["license_file"]["sha256"] and digest(readme) == up["readme"]["sha256"],
            "upstream notices differ")
    card_bytes = card(cfg, files, parity, source_commit).encode()
    card_sha = hashlib.sha256(card_bytes).hexdigest()
    man_bytes = (json.dumps(manifest(cfg, files, parity, source_commit, card_sha), indent=2) + "\n").encode()
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
                               commit_message="Publish Unlimited-OCR-2 GGUF: " + ", ".join(variants))
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
            p.unlink()  # keep runner disk free between large readbacks
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
        steps, drop_f32 = plan(cfg, variants)
        for kind, v, name, args in steps:
            print(kind, v, name, *args)
        print("drop-f32" if drop_f32 else "keep-f32")
        return
    if cmd == "notices":
        notices(cfg, argv[2])
        return
    if cmd == "refresh-plan":
        require(len(argv) == 2, "usage: refresh-plan")
        print(json.dumps(trial_text_receipt(), indent=2))
        return
    if cmd == "refresh-card":
        require(len(argv) == 3, "usage: refresh-card OUT")
        out = Path(argv[2]); out.mkdir(parents=True, exist_ok=True)
        try:
            result = refresh_card(cfg, out)
            (out / "card-update.json").write_text(json.dumps(result, indent=2) + "\n")
        except BaseException as error:
            (out / "card-update.json").write_text(json.dumps({"status": "failed or rejected",
                "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
            raise
        return
    require(cmd == "publish" and len(argv) == 6, "usage: plan VARIANTS | notices DIR | publish VARIANTS GGUF_DIR NOTICES_DIR OUT")
    out = Path(argv[5])
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "crispembed-unlimited-ocr2-publication-report-v1", "status": "started"}
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
