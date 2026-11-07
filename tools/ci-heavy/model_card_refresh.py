#!/usr/bin/env python3
"""Card-only refresh of an already-published CrispEmbed model repository on Hugging Face.

Driven by .github/workflows/model-card-refresh.yml from a reviewed spec
(docs/model-card-refresh-v1.json). Only small text files may change (README.md,
manifest.json, NOTICE, UPSTREAM_README.md); GGUF bytes are never uploaded.

  snapshot REPO                 print the current Hub head and every file's size/LFS SHA (anonymous)
  plan SPEC ID OUT              stdlib only, anonymous: verify the pinned parent commit's README,
                                weights, license/notice pins; render the new small files into OUT
  publish SPEC ID OUT           plan, then commit the small files with parent_commit pinned, verify
                                every GGUF's LFS identity is unchanged, and read back every
                                changed small file from the new immutable commit (empty cache)

Card text comes from files committed beside the spec (docs/model-cards/); quality numbers in
those files cite committed receipts.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "crispembed-model-card-refresh-v1"
SMALL = ("README.md", "manifest.json", "NOTICE", "UPSTREAM_README.md")
PINNED = ("LICENSE", "NOTICE", "UPSTREAM_README.md")
MARK_OPEN, MARK_CLOSE = "<!-- crispembed-quality-status -->", "<!-- /crispembed-quality-status -->"
ALLOWED = {"cstr/got-ocr2-crispembed-GGUF", "cstr/OXR-1.0-GGUF", "cstr/TeleOCR-GGUF",
           "cstr/MinerU2.5-Pro-2604-1.2B-GGUF", "cstr/MI-GAN-512-GGUF"}
HEX40, HEX64 = re.compile("[0-9a-f]{40}"), re.compile("[0-9a-f]{64}")
LIMIT = 1024 * 1024


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def http(url, opener=urllib.request.urlopen):
    with opener(url, timeout=60) as resp:
        data = resp.read(LIMIT + 1)
    require(len(data) <= LIMIT, "small file exceeds 1 MiB: " + url)
    return data


def hub_file(repo, rev, name, opener=urllib.request.urlopen):
    return http(f"https://huggingface.co/{repo}/resolve/{rev}/{name}", opener)


def hub_tree(repo, rev, opener=urllib.request.urlopen):
    rows = json.loads(http(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true", opener))
    return {r["path"]: {"bytes": r["size"], "lfs_sha256": (r.get("lfs") or {}).get("oid")}
            for r in rows if r["type"] == "file"}


def load(spec_path, ident, root=ROOT):
    spec = json.loads((root / spec_path).read_text())
    require(spec.get("schema") == SCHEMA, "unexpected spec schema")
    require(spec.get("approved_by_owner") is True, "card refresh not approved by the owner")
    require(ident in spec["entries"], "unknown entry: " + ident)
    e = dict(spec["entries"][ident], _id=ident, _spec=spec_path)
    require(e["repo"] in ALLOWED, "unreviewed destination")
    require(HEX40.fullmatch(e["parent"]) and HEX64.fullmatch(e["base_readme_sha256"]), "parent/README pin malformed")
    require(e["mode"] in ("insert", "replace"), "mode must be insert or replace")
    require(e["weights"] and all(n.endswith(".gguf") and HEX64.fullmatch(w["sha256"]) and type(w["bytes"]) is int
                                 for n, w in e["weights"].items()), "weight pins malformed")
    for name, p in e.get("pins", {}).items():
        require(name in PINNED and HEX64.fullmatch(p["sha256"]), "notice pin malformed: " + name)
        require("url" not in p or p["url"].startswith("https://"), "notice pin url must be https: " + name)
    require((root / e["card_file"]).is_file(), "card file missing: " + e["card_file"])
    return e


def render_readme(entry, old, root=ROOT):
    text = (root / entry["card_file"]).read_text()
    if entry["mode"] == "replace":
        return text.encode()
    old = old.decode()
    block = f"{MARK_OPEN}\n{text.rstrip()}\n{MARK_CLOSE}\n"
    if MARK_OPEN in old:
        start, end = old.index(MARK_OPEN), old.index(MARK_CLOSE) + len(MARK_CLOSE)
        new = old[:start] + block.rstrip("\n") + old[end:]
    else:
        anchor = "\n" + entry["anchor"] + "\n"
        require(old.count(anchor) == 1, "anchor heading not found exactly once: " + entry["anchor"])
        new = old.replace(anchor, "\n" + block + "\n" + entry["anchor"] + "\n")
    return new.encode()


def check_weights(entry, tree):
    ggufs = {n: w for n, w in tree.items() if n.endswith(".gguf")}
    require(set(ggufs) == set(entry["weights"]), "GGUF file set differs from the spec")
    for n, w in entry["weights"].items():
        require(ggufs[n]["bytes"] == w["bytes"] and ggufs[n]["lfs_sha256"] == w["sha256"], "weight identity differs: " + n)


def plan(entry, out, source_commit, opener=urllib.request.urlopen, root=ROOT):
    repo, parent = entry["repo"], entry["parent"]
    tree = hub_tree(repo, parent, opener)
    check_weights(entry, tree)
    old_readme = hub_file(repo, parent, "README.md", opener)
    require(sha(old_readme) == entry["base_readme_sha256"], "parent README differs from the reviewed base")
    payload = {"README.md": render_readme(entry, old_readme, root)}
    for name, p in entry.get("pins", {}).items():
        if name in tree:
            data = hub_file(repo, parent, name, opener)
            require(sha(data) == p["sha256"], "Hub file differs from its pin: " + name)
        else:
            require("url" in p, name + " absent on the Hub and has no source url")
            data = http(p["url"], opener)
            require(sha(data) == p["sha256"] and len(data) == p["bytes"], "pinned upstream differs: " + p["url"])
            payload[name] = data
    if "LICENSE" in entry.get("pins", {}):
        require("LICENSE" in tree, "LICENSE missing on the Hub")
    if "manifest.json" in tree and entry.get("update_manifest", True):
        man = json.loads(hub_file(repo, parent, "manifest.json", opener))
        before = json.dumps(man.get("files"), sort_keys=True)
        hist = man.setdefault("card_refresh", [])
        hist.append({"date": entry["date"], "spec": entry["_spec"], "entry": entry["_id"], "parent": parent,
                     "previous_readme_sha256": entry["base_readme_sha256"], "readme_sha256": sha(payload["README.md"]),
                     "source_commit": source_commit, "weights_unchanged": True})
        if "readme_sha256" in man:
            man["readme_sha256"] = sha(payload["README.md"])
        require(json.dumps(man.get("files"), sort_keys=True) == before, "manifest weight rows changed")
        payload["manifest.json"] = (json.dumps(man, indent=2) + "\n").encode()
    require(set(payload) <= set(SMALL), "only small text files may change")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        (out / name).write_bytes(data)
    return payload, tree


def publish(entry, out, source_commit):
    require(HEX40.fullmatch(source_commit or ""), "exact source commit required")
    payload, _ = plan(entry, out, source_commit)
    from huggingface_hub import CommitOperationAdd, HfApi
    require(os.environ.get("HF_TOKEN"), "publication credential unavailable")
    api = HfApi(token=os.environ["HF_TOKEN"])
    head = api.model_info(entry["repo"], revision="main")
    require(head.sha == entry["parent"], f"Hub head {head.sha} is not the reviewed parent; re-review")
    ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=d) for n, d in sorted(payload.items())]
    commit = api.create_commit(repo_id=entry["repo"], operations=ops, parent_commit=entry["parent"],
                               commit_message=entry["commit_message"])
    tree = hub_tree(entry["repo"], commit.oid)
    check_weights(entry, tree)
    readback = {}
    for name, data in payload.items():
        got = hub_file(entry["repo"], commit.oid, name)  # anonymous, no local cache
        require(sha(got) == sha(data), "readback differs: " + name)
        readback[name] = {"bytes": len(got), "sha256": sha(got)}
    for name, p in entry.get("pins", {}).items():
        require(sha(hub_file(entry["repo"], commit.oid, name)) == p["sha256"], "pinned notice changed: " + name)
    return {"repo": entry["repo"], "parent": entry["parent"], "commit": commit.oid, "changed_files": readback,
            "weights_unchanged": entry["weights"], "pins_verified": sorted(entry.get("pins", {}))}


def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "snapshot":
        repo = argv[2]
        head = json.loads(http(f"https://huggingface.co/api/models/{repo}"))["sha"]
        print(json.dumps({"repo": repo, "head": head, "files": hub_tree(repo, head)}, indent=1))
        return
    require(cmd in ("plan", "publish") and len(argv) == 5, "usage: snapshot REPO | plan|publish SPEC ID OUT")
    entry = load(argv[2], argv[3])
    out = Path(argv[4])
    if cmd == "plan":
        payload, _ = plan(entry, out, os.environ.get("SOURCE_COMMIT", "unset"))
        for n, d in sorted(payload.items()):
            print(n, len(d), sha(d))
        return
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "crispembed-model-card-refresh-report-v1", "spec": argv[2], "entry": argv[3],
              "source_commit": os.environ.get("SOURCE_COMMIT"), "status": "started"}
    save = lambda: (out / "card-refresh.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        report["remote"] = publish(entry, out / "files", os.environ.get("SOURCE_COMMIT"))
        report["status"] = "published and re-verified"
    except BaseException as e:
        report.update(status="failed or rejected", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main(sys.argv)
