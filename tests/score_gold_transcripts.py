#!/usr/bin/env python3
"""Score a directory of pre-computed transcripts with the parity harness's metrics.

Some parity arms cannot run in-process next to the others: the olmOCR toolkit
needs a GPU inference server and consumes PDFs rather than images, so its
transcripts are produced on a remote host and brought back as files.  Scoring
them still has to use the *same* normalisation and the *same* CER/WER/
wer_unordered definitions as `ocr_external_parity.py`, otherwise the row cannot
be read against the rows next to it — so this imports those functions rather
than restating them, and emits a document in the same shape, which means
`summarize` and `render_markdown` also come from there.

The timing columns are deliberately left empty unless a `pages.json` supplies
them, and even then they carry the producing host's clock, not this one's.

    python tests/score_gold_transcripts.py \
        --images ~/crispembed-ocr-synth \
        --gold tests/regression/gold/olmocr/synth \
        --engine olmocr-toolkit --output /tmp/olmocr_synth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_external_parity import (  # noqa: E402
    load_fixtures, normalize, render_markdown, score, summarize)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path,
                    help="directory of <fixture stem>.txt transcripts")
    ap.add_argument("--engine", default="olmocr-toolkit")
    ap.add_argument("--kind", default="external")
    ap.add_argument("--suffix", default=".txt")
    ap.add_argument("--require-truth", action="store_true", default=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    manifest_path = args.gold / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    pages_path = args.gold / "pages.json"
    pages = {}
    if pages_path.exists():
        pages = {p["fixture"]: p for p in json.loads(pages_path.read_text())}
    elif manifest.get("pages"):
        pages = {p["fixture"]: p for p in manifest["pages"]}

    fixtures = [f for f in load_fixtures(args.images) if f["truth"] or not args.require_truth]
    rows, missing = [], []
    for fx in fixtures:
        # Arms disagree on whether a transcript is named after the fixture or
        # after its stem; accept both rather than make the caller care.
        for cand in (Path(fx["name"]).stem + args.suffix, fx["name"] + args.suffix):
            t = args.gold / cand
            if t.exists():
                break
        if not t.exists():
            missing.append(fx["name"])
            continue
        text = t.read_text()
        page = pages.get(fx["name"], {})
        entry = {
            "kind": args.kind,
            "text": text,
            # The producing host's clock.  Kept because a page that took two
            # minutes is worth knowing about, dropped from any comparison
            # because it was not measured here.
            "proc_ms": round(page["gen_s"] * 1000.0, 1) if page.get("gen_s") else None,
            "engine_ms": None,
            "remote_timing": True,
            "attempts": page.get("n_attempts"),
            "final_temperature": page.get("final_temperature"),
            "deterministic": page.get("deterministic"),
        }
        if fx["truth"]:
            entry.update(score(text, fx["truth"]))
        rows.append({"fixture": fx["name"], "truth_chars": len(normalize(fx["truth"] or "")),
                     "engines": {args.engine: entry}})

    result = {
        "version": 1,
        "images": str(args.images),
        "repeats": 1,
        "reference_engine": "",
        "skipped": {},
        "fixtures": rows,
        "gold_dir": str(args.gold),
        "gold_manifest": {k: v for k, v in manifest.items() if k != "pages"},
        "missing_transcripts": missing,
    }
    result["summary"] = summarize(result)
    md = render_markdown(result)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(md)
    print(md)
    if missing:
        print(f"missing transcripts: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
