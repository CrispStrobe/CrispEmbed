#!/usr/bin/env python3
"""Fetch pinned original sources/checkpoints and a Kaggle original capture.

usage:
  latent_fetch.py sources MODEL DEST          # pinned GitHub files, verified
  latent_fetch.py checkpoint NAME DEST_FILE   # moebius|pixelhacker|sdxl_vae, HF, verified
  latent_fetch.py capture MODEL KOUT DEST     # verify Kaggle gzip parts, extract
"""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from latent_inpaint_contract import load_contract, model_record, verify_checkpoint, verify_source  # noqa: E402


def download(url, path, size, digest):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "CrispEmbed-latent-fetch/1"})
    h, count = hashlib.sha256(), 0
    with urllib.request.urlopen(request, timeout=120) as response, path.open("xb") as out:
        for block in iter(lambda: response.read(1 << 20), b""):
            count += len(block)
            h.update(block)
            out.write(block)
    if count != size or h.hexdigest() != digest:
        raise SystemExit(f"pinned download differs: {url}")


def sources(model, dest):
    contract = load_contract()
    record = contract["source"][model]
    repository = urllib.parse.urlsplit(record["repository"]).path
    for relative, expected in record["files"].items():
        url = ("https://raw.githubusercontent.com" + repository + "/" + record["commit"] + "/" +
               urllib.parse.quote(relative, safe="/"))
        download(url, dest / relative, expected["bytes"], expected["sha256"])
    verify_source(dest, model, contract)
    print("sources verified", model)


def checkpoint(name, dest):
    record = model_record(name, load_contract())
    row = record["checkpoint"]
    download(row["url"], dest, row["bytes"], row["sha256"])
    verify_checkpoint(dest, record)
    print("checkpoint verified", name, row["sha256"])


def capture(model, kout, dest):
    manifest = json.loads((kout / "capture-manifest.json").read_text())
    if manifest["model"] != model or not manifest["complete"]:
        raise SystemExit("capture manifest model/completeness differs")
    whole = hashlib.sha256()
    archive = dest.parent / manifest["archive"]
    with archive.open("xb") as out:
        for part in manifest["parts"]:
            data = (kout / part["file"]).read_bytes()
            if len(data) != part["bytes"] or hashlib.sha256(data).hexdigest() != part["sha256"]:
                raise SystemExit("capture part differs: " + part["file"])
            whole.update(data)
            out.write(data)
    if whole.hexdigest() != manifest["sha256"]:
        raise SystemExit("capture archive digest differs")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest.parent, filter="data")
    archive.unlink()
    (dest.parent / f"{model}-original-capture").rename(dest)
    report = json.loads((dest / "report.json").read_text())
    if hashlib.sha256((dest / "report.json").read_bytes()).hexdigest() != manifest["report_sha256"]:
        raise SystemExit("capture report digest differs")
    print("capture verified", model, manifest["sha256"], [c["id"] for c in report["cases"]])


def main(argv):
    if len(argv) == 4 and argv[1] == "sources":
        sources(argv[2], Path(argv[3]))
    elif len(argv) == 4 and argv[1] == "checkpoint":
        checkpoint(argv[2], Path(argv[3]))
    elif len(argv) == 5 and argv[1] == "capture":
        capture(argv[2], Path(argv[3]), Path(argv[4]))
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
