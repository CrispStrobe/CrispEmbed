#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch only the immutable assets declared by the Marker OCR/retrieval panel.

No inference, corpus mutation, unverified fallback, model publication or remote
code execution. A failed page remains a failed page in the receipt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
UA = {"User-Agent": "CrispEmbed-frozen-OCR-validation/1.0 (https://github.com/CrispStrobe/CrispEmbed)"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def blob_sha(path):
    path = Path(path)
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as response:
        return json.load(response)


def download(url, target, maximum, expected_sha=None, expected_blob=None, expected_size=None):
    target = Path(target)
    row = {"url": url, "expected_sha256": expected_sha, "expected_git_blob": expected_blob}
    target.parent.mkdir(parents=True, exist_ok=True)

    def valid(path):
        row.update(bytes=path.stat().st_size, sha256=sha(path))
        if expected_blob:
            row["git_blob"] = blob_sha(path)
        return ((expected_sha is None or row["sha256"] == expected_sha) and
                (expected_blob is None or row["git_blob"] == expected_blob) and
                (expected_size is None or row["bytes"] == expected_size))

    if target.is_file():
        if not valid(target):
            raise ValueError("Existing task cache does not match its immutable pin: " + target.name)
        return dict(row, passed=True, cached=True)
    partial = target.with_name(target.name + ".partial")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as response, partial.open("wb") as output:
            count = 0
            for part in iter(lambda: response.read(1024 * 1024), b""):
                count += len(part)
                if count > maximum:
                    raise ValueError("Download exceeds predeclared file budget")
                output.write(part)
        if not valid(partial):
            return dict(row, passed=False, error="immutable byte pin mismatch")
        partial.replace(target)
        return dict(row, passed=True, cached=False)
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError):
            row["http_status"] = exc.code
        return dict(row, passed=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("Keep the earlier receipt and choose another --report")
    protocol = json.loads(args.protocol.read_text())
    corpus_path = ROOT / protocol["corpus"]["path"]
    if sha(corpus_path) != protocol["corpus"]["sha256"]:
        raise ValueError("Frozen corpus changed")
    corpus = json.loads(corpus_path.read_text())
    if [p["id"] for p in corpus["pages"]] != protocol["corpus"]["page_ids"]:
        raise ValueError("Frozen page selection changed")
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    maximum = protocol["resources"]["max_download_bytes_per_file"]
    report = {"kind": "marker-gold-ab-asset-fetch", "protocol_sha256": sha(args.protocol),
              "corpus_sha256": sha(corpus_path), "script_sha256": sha(__file__),
              "assets": {}, "pages": {}, "candidate_outputs_examined": False}
    started = time.monotonic()

    def checkpoint():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    revision = protocol["ocr_provider"]["tessdata_revision"]
    tree = get_json(f"https://api.github.com/repos/tesseract-ocr/tessdata_fast/git/trees/{revision}")
    files = {row["path"]: row for row in tree["tree"]}
    for name in ("eng.traineddata", "fra.traineddata", "deu.traineddata", "frk.traineddata", "LICENSE"):
        resolved = name
        entry = files[resolved]
        if entry["mode"] == "120000":
            # The pinned frk alias names deu_latf.traineddata. Resolve only an
            # in-tree basename, and independently check the symlink blob.
            url = f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/{revision}/{name}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as response:
                link = response.read(1024)
            h = hashlib.sha1(f"blob {len(link)}\0".encode() + link).hexdigest()
            resolved = link.decode().strip()
            if h != entry["sha"] or Path(resolved).name != resolved or resolved not in files:
                raise ValueError("Unsafe or changed tessdata alias")
            entry = files[resolved]
        row = download(f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/{revision}/{resolved}",
                       cache / "tessdata" / name, maximum, expected_blob=entry["sha"], expected_size=entry["size"])
        report["assets"][name] = dict(row, resolved_path=resolved, repository_revision=revision)
        print(name, row["passed"], flush=True)
        checkpoint()
    # The installed Tesseract font is engine support data, not a learned model.
    font = Path("/usr/share/tesseract-ocr/5/tessdata/pdf.ttf")
    if not font.is_file():
        raise FileNotFoundError("Installed Tesseract PDF support font is unavailable")
    font_revision = "3decf1c8252ba6dbeef0bf908f4b0aab7f18d113"
    font_source = f"https://raw.githubusercontent.com/tesseract-ocr/tessconfigs/{font_revision}/"
    font_digest = "c7845420925a23d88ed830a63957b8af85a66a8daf8d9fc90e843673b2ef1a59"
    if sha(font) != font_digest:
        raise ValueError("Installed PDF support font differs from its verified upstream source")
    shutil.copyfile(font, cache / "tessdata/pdf.ttf")
    report["assets"]["pdf.ttf"] = {"passed": True, "sha256": sha(font), "bytes": font.stat().st_size,
                                    "source": "installed Tesseract 5.5.0 support font",
                                    "source_url": font_source + "pdf.ttf", "repository_revision": font_revision,
                                    "license": "Apache-2.0", "license_url": font_source + "LICENSE"}
    model = protocol["retrieval"]
    report["assets"]["embedding_model"] = download(
        f"https://huggingface.co/{model['model_repository']}/resolve/{model['revision']}/{model['filename']}",
        cache / model["filename"], maximum, expected_sha=model["sha256"], expected_size=model["bytes"])
    checkpoint()

    def image_page(page):
        info = page["image"]
        target = cache / "images" / (page["id"] + ".image")
        attempts = []
        for round_number in range(4):
            for url in [info["url"]] + info.get("fallback_urls", []):
                attempt = download(url, target, maximum, expected_sha=info["sha256"])
                attempts.append(dict(attempt, round=round_number + 1))
                if attempt["passed"]:
                    return page["id"], {"passed": True, "sha256": info["sha256"], "attempts": attempts}
            if all(a.get("http_status") in (400, 401, 403, 404, 410) for a in attempts[-(1 + len(info.get("fallback_urls", []))):]):
                break
            time.sleep(0.25)
        return page["id"], {"passed": False, "expected_sha256": info["sha256"], "attempts": attempts}

    wiki = [page for page in corpus["pages"] if page["source"] == "wikisource"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for page_id, row in pool.map(image_page, wiki):
            report["pages"][page_id] = row
            print(page_id, row["passed"], flush=True)
            checkpoint()
    cord = [page for page in corpus["pages"] if page["source"] == "cord-v2"]
    if cord:
        pin = cord[0]["image"]["hf_parquet"]
        path = cache / "cord-test.parquet"
        downloaded = download(pin["url"], path, maximum, expected_sha=pin["sha256"])
        report["assets"]["cord_parquet"] = downloaded
        if downloaded["passed"]:
            import pyarrow.parquet as pq
            images = pq.read_table(path, columns=["image"]).column("image")
            for page in cord:
                info = page["image"]
                if info["hf_parquet"]["url"] != pin["url"] or info["hf_parquet"]["sha256"] != pin["sha256"]:
                    raise ValueError("CORD pages do not share the pinned parquet")
                data = images[info["hf_parquet"]["row"]].as_py()["bytes"]
                actual = hashlib.sha256(data).hexdigest()
                ok = actual == info["sha256"]
                if ok:
                    (cache / "images" / (page["id"] + ".image")).write_bytes(data)
                report["pages"][page["id"]] = {"passed": ok, "sha256": actual,
                    "expected_sha256": info["sha256"], "bytes": len(data), "parquet_row": info["hf_parquet"]["row"]}
        else:
            for page in cord:
                report["pages"][page["id"]] = {"passed": False, "error": "pinned CORD parquet unavailable"}
    report["elapsed_seconds"] = time.monotonic() - started
    report["passed_pages"] = sum(row["passed"] for row in report["pages"].values())
    report["total_pages"] = len(corpus["pages"])
    report["passed"] = report["passed_pages"] == report["total_pages"] and all(row["passed"] for row in report["assets"].values())
    checkpoint()
    print(f"{report['passed_pages']}/{report['total_pages']} exact image pins; assets passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
