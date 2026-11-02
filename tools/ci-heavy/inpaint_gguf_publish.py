#!/usr/bin/env python3
"""Publish owner-approved inpainting GGUFs (docs/inpaint-publication-v1.json).

The files are copied byte for byte from the private preservation repositories at
the pinned revision; nothing is converted or requantized here. Every source file
is checked against the protocol SHA-256 before upload, and every uploaded file is
read back from an empty cache at the new immutable revision.

  plan     PROTOCOL REPO                 print what would be published (no network)
  card     PROTOCOL REPO OUT_DIR         render README.md/manifest preview (no network)
  publish  PROTOCOL REPO WORK_DIR OUT_DIR

Credentials: HF_SOURCE_TOKEN reads the private repositories (HF_TOKEN_2 secret);
HF_TOKEN writes the public destination (HF_PUBLISH_TOKEN secret). The private
repositories are only read.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "crispembed-inpaint-gguf-publication-v1"
BARS = {"F32": 0.99, "F16": 0.98, "Q8_0": 0.97, "Q4_K": 0.94, "Q4_K-mix": 0.94}
SHA = re.compile("[0-9a-f]{64}")
REV = re.compile("[0-9a-f]{40}")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(protocol, repo, root=ROOT):
    path = Path(protocol)
    require(path.as_posix() == "docs/inpaint-publication-v1.json", "unknown publication protocol")
    cfg = json.loads((root / path).read_text(encoding="utf-8"))
    require(cfg.get("schema") == SCHEMA, "protocol schema differs")
    require(cfg.get("approved_by_owner") is True and cfg.get("status") == "approved",
            "protocol is not owner-approved")
    require(repo in cfg["repos"], "destination is not in the protocol: " + repo)
    entry = cfg["repos"][repo]
    require(re.fullmatch(r"cstr/[A-Za-z0-9._-]+-GGUF", repo), "destination name outside the allowlist pattern")
    for name, v in entry["variants"].items():
        require(SHA.fullmatch(v["sha256"]) and REV.fullmatch(v["source_revision"]), "bad pin for " + name)
        require(v["source_repo"].endswith("-private"), "variants must come from a private preservation repo")
        require(v["filename"].endswith(".gguf") and "/" not in v["filename"], "bad filename " + v["filename"])
    for lf in entry["license_files"]:
        src = lf.get("from", {})
        require(SHA.fullmatch(lf.get("sha256") or src.get("sha256") or ""), "licence file without pin: " + lf["path"])
    return cfg, entry


def license_sha(lf):
    return lf.get("sha256") or lf["from"]["sha256"]


def fmt(x):
    return f"{x:.6f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def card(cfg, repo, entry, source_commit):
    model = entry["model"]
    up = entry["upstream"]
    titles = {"migan512": "MI-GAN 512 (Places2)", "moebius": "Moebius inpainting",
              "pixelhacker": "PixelHacker inpainting", "sdxl_vae": "SDXL VAE (inpainting companion)"}
    licenses = {"migan512": "license: mit", "pixelhacker": "license: mit", "sdxl_vae": "license: mit",
                "moebius": "license: other\nlicense_name: moebius-upstream-records\nlicense_link: LICENSE.md"}
    lines = ["---", licenses[model], "library_name: crispembed", "tags:", "- gguf", "- inpainting",
             "- image-inpainting", "- crispembed", "---", "", f"# {titles[model]} — GGUF for CrispEmbed", "",
             "**Experimental.** These files run only in CrispEmbed's native inpainting engine behind default-off "
             "gates (" + ", ".join(f"`{g}`" for g in entry.get("runtime_gates", [])) + "). They are exact copies "
             "of the qualified files; nothing was requantized for publication.", "", "## Licence", ""]
    if model == "moebius":
        lines += ["- Code (upstream Moebius): Apache-2.0.",
                  "- Weights: the official Moebius README states **Apache-2.0**; the author's Hugging Face model card "
                  "states **MIT**. Both records are reproduced verbatim in `licenses/`. No dual licence is claimed; "
                  "the conversion only reproduces the upstream weights and their licence records."]
    elif model == "pixelhacker":
        lines += ["- Weights: MIT (author weight repository).",
                  "- Code (upstream PixelHacker): Apache-2.0. Native ports also study flash-linear-attention (MIT)."]
    elif model == "migan512":
        lines += ["- Code: MIT. Weights: MIT (`LICENSE-WEIGHTS`), Copyright (c) 2024 Picsart AI Research (PAIR)."]
    else:
        lines += ["- SDXL VAE weights: MIT (stabilityai/sdxl-vae), as redistributed in the PixelHacker weight "
                  "repository."]
    lines += ["", "Licence and notice files: " + ", ".join(f"`{lf['path']}`" for lf in entry["license_files"]) + ".",
              "", "## Lineage", ""]
    for k, v in up.items():
        lines.append(f"- {k.replace('_', ' ')}: {v}")
    lines.append(f"- Converter: `{cfg['converters'][model]}`; quantizer: `crispembed-quantize` (inpaint-v1 policy).")
    lines.append(f"- Publication source: private preservation repository at an exact revision (see `manifest.json`); "
                 f"CrispEmbed source {source_commit}.")
    lines += ["", "## Files and precision against the acceptance bars", "",
              "Bars (final-output cosine, minimum over cases): F32 > 0.99, F16 > 0.98, Q8_0 > 0.97, Q4_K > 0.94. "
              "Synthetic-panel values compare with the original PyTorch model; real-photo values compare the masked "
              "region with native F32 (no original output exists for those photos).", "",
              "| Variant | File | Bytes | SHA-256 | Min cosine | Bar | Notes |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, v in entry["variants"].items():
        cos = v.get("min_cosine_vs_original", v.get("min_cosine"))
        notes = "; ".join(x for x in (v.get("vae") and f"pair with {v['vae']}", v.get("recipe"), v.get("caveat")) if x)
        lines.append(f"| {name} | `{v['filename']}` | {v['bytes']} | `{v['sha256']}` | "
                     f"{fmt(cos) if cos is not None else 'paired (see notes)'} | {BARS.get(name, '-')} | {notes} |")
    if entry.get("excluded"):
        lines += ["", "Not published: " + "; ".join(f"{k} ({v})" for k, v in entry["excluded"].items()) + "."]
    if model == "migan512":
        lines += ["", "**Caveat:** Q4_K is 0.9019 on the street crop with a DBNet text mask (masked-region cosine "
                  "against F32), below the 0.94 Q4_K bar on that crop; every other case clears it."]
    if entry.get("vae_repo"):
        lines += ["", f"The SDXL VAE companion files are in [{entry['vae_repo']}](https://huggingface.co/"
                  f"{entry['vae_repo']})."]
    if model == "sdxl_vae":
        lines += ["", entry.get("note", "")]
    lines += ["", "## Use", "", "See CrispEmbed `docs/inpainting-cli.md` and `docs/inpainting-precision.md`. "
              "Non-F32 files additionally need `CRISPEMBED_INPAINT_QUANT=1`.", ""]
    return "\n".join(lines)


def moebius_license_md(entry):
    return ("# Moebius licence records\n\nThe upstream Moebius code is Apache-2.0. For the weights, the official "
            "README states Apache-2.0 and the author's Hugging Face model card states MIT. Both records are kept "
            "verbatim in `licenses/`; no dual licence is claimed.\n")


def plan(cfg, repo, entry):
    return [(name, v["source_repo"], v["source_revision"], v["source_path"], v["filename"], v["sha256"])
            for name, v in entry["variants"].items()]


def small_files(cfg, repo, entry, source_commit, root=ROOT, fetch=None):
    files = {}
    for lf in entry["license_files"]:
        if "from_repo_path" in lf:
            data = (root / lf["from_repo_path"]).read_bytes()
        else:
            require(fetch is not None, "private licence file needs a fetcher")
            data = fetch(lf["from"])
        require(hashlib.sha256(data).hexdigest() == license_sha(lf), "pinned licence text differs: " + lf["path"])
        files[lf["path"]] = data
    if entry["model"] == "moebius":
        files["LICENSE.md"] = moebius_license_md(entry).encode()
    readme = card(cfg, repo, entry, source_commit).encode()
    files["README.md"] = readme
    manifest = {"schema": "crispembed-inpaint-gguf-publication-manifest-v1", "protocol": "docs/inpaint-publication-v1.json",
                "repo": repo, "source_commit": source_commit, "approval": cfg["approval"],
                "variants": entry["variants"], "upstream": entry["upstream"],
                "license_files": {p: hashlib.sha256(b).hexdigest() for p, b in files.items() if p != "README.md"},
                "readme_sha256": hashlib.sha256(readme).hexdigest()}
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return files


def publish(cfg, repo, entry, work, out, source_commit):
    require(REV.fullmatch(source_commit or ""), "exact source commit required")
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    src_token, dst_token = os.environ.get("HF_SOURCE_TOKEN"), os.environ.get("HF_TOKEN")
    require(src_token and dst_token, "source and publication credentials required")
    work = Path(work)

    def fetch_private(src):
        p = hf_hub_download(src["source_repo"], src["source_path"], revision=src["source_revision"],
                            local_dir=work / "src" / src["source_repo"].split("/")[1], token=src_token)
        require(digest(p) == src["sha256"] and Path(p).stat().st_size == src["bytes"],
                "private source differs from the protocol pin: " + src["source_path"])
        return Path(p)

    ggufs = {}
    for name, v in entry["variants"].items():
        ggufs[v["filename"]] = fetch_private(v)
    small = small_files(cfg, repo, entry, source_commit, fetch=lambda s: fetch_private(s).read_bytes())
    (out / repo.split("/")[1]).mkdir(parents=True, exist_ok=True)
    for n, b in small.items():
        target = out / repo.split("/")[1] / n
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b)

    expected = {v["filename"]: v["sha256"] for v in entry["variants"].values()}
    expected.update({n: hashlib.sha256(b).hexdigest() for n, b in small.items()})
    api = HfApi(token=dst_token)
    allowed = set(expected) | {".gitattributes"}
    if api.repo_exists(repo):
        info = api.model_info(repo)
        require(not info.private, "destination exists and is private")
        require({s.rfilename for s in info.siblings} <= allowed, "destination holds files outside the allowlist")
    else:
        api.create_repo(repo, private=False, exist_ok=False)
    ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p)) for n, p in ggufs.items()]
    ops += [CommitOperationAdd(path_in_repo=n, path_or_fileobj=b) for n, b in small.items()]
    commit = api.create_commit(repo_id=repo, operations=ops, commit_message="Publish " + ", ".join(entry["variants"])
                               + " (docs/inpaint-publication-v1.json, CrispEmbed " + source_commit[:12] + ")")
    info = api.model_info(repo, revision=commit.oid, files_metadata=True)
    require(not info.private, "published repository is private")
    remote = {s.rfilename: s for s in info.siblings}
    require(set(remote) <= allowed and set(expected) <= set(remote), "remote file set differs")
    for v in entry["variants"].values():
        s = remote[v["filename"]]
        require(s.size == v["bytes"] and s.lfs and s.lfs.sha256 == v["sha256"], "remote LFS identity differs")
    for p in ggufs.values():
        p.unlink()
    cache = work / "fresh-hf-cache"
    cache.mkdir(parents=True, exist_ok=False)
    readback = {}
    for name, sha in expected.items():
        p = Path(hf_hub_download(repo, name, revision=commit.oid, cache_dir=cache, force_download=True, token=False))
        got = digest(p)
        require(got == sha, "empty-cache readback differs: " + name)
        readback[name] = got
        if name.endswith(".gguf"):
            p.unlink()
    return {"repo": repo, "revision": commit.oid, "anonymous_readback_sha256": readback,
            "urls": {v["filename"]: f"https://huggingface.co/{repo}/resolve/{commit.oid}/{v['filename']}"
                     for v in entry["variants"].values()}}


def main(argv):
    require(len(argv) >= 4, __doc__)
    cmd, protocol, repo = argv[1], argv[2], argv[3]
    cfg, entry = load(protocol, repo)
    if cmd == "plan":
        for row in plan(cfg, repo, entry):
            print(*row)
        return
    if cmd == "card":
        out = Path(argv[4])
        out.mkdir(parents=True, exist_ok=True)
        (out / "README.md").write_text(card(cfg, repo, entry, "0" * 40), encoding="utf-8")
        return
    require(cmd == "publish" and len(argv) == 6, __doc__)
    out = Path(argv[5])
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "crispembed-inpaint-gguf-publication-report-v1", "protocol": protocol, "repo": repo,
              "status": "started"}
    save = lambda: (out / f"publication-{repo.split('/')[1]}.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        report["remote"] = publish(cfg, repo, entry, argv[4], out, os.environ.get("SOURCE_COMMIT"))
        report["status"] = "published and re-verified"
    except BaseException as e:
        report.update(status="failed or rejected", error=f"{type(e).__name__}: {e}")
        save()
        raise
    save()


if __name__ == "__main__":
    main(sys.argv)
