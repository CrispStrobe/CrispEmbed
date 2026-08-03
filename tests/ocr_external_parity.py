#!/usr/bin/env python3
"""Head-to-head OCR parity: CrispEmbed vs Tesseract / EasyOCR / PaddleOCR.

Existing harnesses in this repo compare CrispEmbed against per-stage tensor
references.  That proves a graph is faithful; it says nothing about whether the
shipped pipeline reads a page as well or as fast as the engine it ports.  This
runs the external engines and the native ones over the same images and reports
the two numbers a user actually feels: transcription error and latency.

Two quality scores are produced because they answer different questions:

  * ``cer``/``wer`` vs ground truth  — absolute quality.  Only available for the
    synthetic corpus (``tests/ocr_synth_corpus.py``), which knows its own text.
  * ``cer`` vs a reference *engine*  — port fidelity.  A CrispEmbed engine that
    disagrees with the upstream engine it ports is a runtime bug to bisect with
    ``crispembed-diff``, independent of how hard the page is.

Timing is deliberately reported in two columns, because the arms are not
symmetric (dev-guide rule 4a): ``proc_ms`` is wall time for a whole invocation
including model load — what a CLI user pays — while ``engine_ms`` excludes model
load (parsed from native stage-bench stderr, or measured directly for the
in-process Python engines).  Never mix the two columns in a speed claim.

Usage:
  python tests/ocr_synth_corpus.py --output /tmp/ocr-synth
  python tests/ocr_external_parity.py --images /tmp/ocr-synth \
      --model-dir /Volumes/backups/ai/crispembed-gguf --repeats 3 \
      --output /tmp/ocr-parity.json --markdown /tmp/ocr-parity.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESSDATA = "/opt/homebrew/share/tessdata"

# Native stage-bench lines (stderr, opt-in via env) carry the load-excluded cost.
STAGE_BENCH = {
    "ppocrv6": re.compile(r"\[ppocrv6-stage-bench\].*?total=([0-9.]+) ms"),
    # easyocr's total= includes model load; detect+recognize= is the
    # load-excluded figure this column claims to be.
    "easyocr": re.compile(r"\[easyocr-stage-bench\].*?detect\+recognize=([0-9.]+) ms"),
    "tesseract": re.compile(r"\[tesseract-stage-bench\].*?total=([0-9.]+) ms"),
}
REGIONS_RE = re.compile(r"^regions=(\d+)\s+mean_conf=([0-9.]+)", re.M)


# ---------------------------------------------------------------- text metrics

def _edit(a: list | str, b: list | str) -> int:
    try:
        import Levenshtein

        if isinstance(a, str):
            return Levenshtein.distance(a, b)
    except ImportError:
        pass
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize(s: str) -> str:
    """Collapse layout differences so CER measures recognition, not line breaks.

    Engines disagree on where a line ends and whether regions are joined by a
    space or a newline; scoring that as character error would drown the signal
    we care about.  Case and punctuation are preserved — they are real errors.
    """
    s = s.replace("­", "")  # soft hyphen: never in ground truth
    return re.sub(r"\s+", " ", s).strip()


def score(hyp: str, ref: str) -> dict:
    h, r = normalize(hyp), normalize(ref)
    if not r:
        return {"cer": None, "wer": None, "ref_chars": 0}
    cer = _edit(h, r) / len(r)
    hw, rw = h.split(), r.split()
    wer = _edit(hw, rw) / max(1, len(rw))
    return {"cer": round(cer, 5), "wer": round(wer, 5), "ref_chars": len(r),
            "exact": h == r}


# ------------------------------------------------------------- engine adapters

class Engine:
    """One OCR arm.  ``run`` returns (text, proc_ms, engine_ms, extra)."""

    name = "?"
    kind = "?"  # external | crispembed

    def available(self) -> str:
        return ""

    def run(self, image: Path, repeats: int) -> tuple[str, float, float | None, dict]:
        raise NotImplementedError


class TesseractCLI(Engine):
    kind = "external"

    def __init__(self, lang: str = "eng", psm: int = 6):
        self.lang, self.psm = lang, psm
        self.name = f"tesseract-cli:{lang}"

    def available(self) -> str:
        if not shutil.which("tesseract"):
            return "tesseract not on PATH"
        return ""

    def run(self, image: Path, repeats: int):
        env = os.environ.copy()
        env.pop("TESSDATA_PREFIX", None)  # CLAUDE.md: never inherit an ambiguous one
        cmd = ["tesseract", str(image), "stdout", "-l", self.lang,
               "--psm", str(self.psm)]
        if Path(TESSDATA).is_dir():
            cmd += ["--tessdata-dir", TESSDATA]
        times, out = [], ""
        for _ in range(repeats):
            t = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            times.append((time.perf_counter() - t) * 1000)
            out = p.stdout
        # Tesseract loads its model per invocation; it has no separable
        # "engine only" cost from the CLI, so engine_ms stays None rather than
        # being faked from the wall clock.
        return out, statistics.median(times), None, {}


class EasyOCRPy(Engine):
    kind = "external"
    name = "easyocr-py"

    def __init__(self, langs=("en",), gpu=False):
        self.langs, self.gpu, self._reader = list(langs), gpu, None

    def available(self) -> str:
        try:
            import easyocr  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment probe
            return f"import easyocr failed: {exc}"
        return ""

    def _ensure(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self.langs, gpu=self.gpu, verbose=False)
        return self._reader

    def run(self, image: Path, repeats: int):
        reader = self._ensure()
        times, lines = [], []
        for _ in range(repeats):
            t = time.perf_counter()
            lines = reader.readtext(str(image), detail=0, paragraph=False)
            times.append((time.perf_counter() - t) * 1000)
        med = statistics.median(times)
        # In-process: the model is already resident, so wall == engine cost.
        return "\n".join(lines), med, med, {"regions": len(lines)}


class PaddleOCRPy(Engine):
    kind = "external"
    name = "paddleocr-py"

    def __init__(self, lang: str = "en"):
        self.lang, self._ocr = lang, None

    def available(self) -> str:
        try:
            import paddleocr  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment probe
            return f"import paddleocr failed: {exc}"
        return ""

    def _ensure(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(lang=self.lang, use_angle_cls=True, show_log=False)
        return self._ocr

    @staticmethod
    def _texts(result) -> list[str]:
        # PaddleOCR 2.x returns [[ [box, (text, conf)], ... ]]; 3.x returns a
        # dict-like record.  Accept both rather than pinning a version.
        out = []
        if not result:
            return out
        page = result[0]
        if isinstance(page, dict):
            return list(page.get("rec_texts", []))
        for item in page or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                payload = item[1]
                out.append(payload[0] if isinstance(payload, (list, tuple)) else str(payload))
        return out

    def run(self, image: Path, repeats: int):
        ocr = self._ensure()
        times, texts = [], []
        for _ in range(repeats):
            t = time.perf_counter()
            try:
                res = ocr.ocr(str(image), cls=True)
            except TypeError:
                res = ocr.ocr(str(image))
            times.append((time.perf_counter() - t) * 1000)
            texts = self._texts(res)
        med = statistics.median(times)
        return "\n".join(texts), med, med, {"regions": len(texts)}


class CrispEmbedCLI(Engine):
    kind = "crispembed"

    def __init__(self, name: str, engine: str, binary: Path, det: Path | None,
                 rec: Path | None, bench_env: dict[str, str] | None = None,
                 stage_key: str | None = None, extra_args: list[str] | None = None):
        self.name = name
        self.engine = engine
        self.binary = binary
        self.det, self.rec = det, rec
        self.bench_env = bench_env or {}
        self.stage_key = stage_key
        self.extra_args = extra_args or []

    def available(self) -> str:
        if not self.binary.exists():
            return f"missing binary {self.binary}"
        for m in (self.det, self.rec):
            if m is not None and not m.exists():
                return f"missing model {m}"
        return ""

    def run(self, image: Path, repeats: int):
        env = os.environ.copy()
        env.update(self.bench_env)
        cmd = [str(self.binary), "--ocr-pipeline", str(image), "--ocr-engine", self.engine]
        if self.det:
            cmd += ["--ocr-det", str(self.det)]
        if self.rec:
            cmd += ["--ocr-rec", str(self.rec)]
        cmd += self.extra_args
        times, stages, out, err, rc = [], [], "", "", 0
        for _ in range(repeats):
            t = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                               cwd=ROOT, timeout=1800)
            times.append((time.perf_counter() - t) * 1000)
            out, err, rc = p.stdout, p.stderr, p.returncode
            if self.stage_key and self.stage_key in STAGE_BENCH:
                m = STAGE_BENCH[self.stage_key].search(err)
                if m:
                    stages.append(float(m.group(1)))
        extra = {"returncode": rc}
        m = REGIONS_RE.search(out)
        if m:
            extra["regions"] = int(m.group(1))
            extra["mean_conf"] = float(m.group(2))
        text = REGIONS_RE.sub("", out).strip()
        if rc != 0:
            # A non-zero exit never gets timed as a win (dev-guide rule 4a).
            extra["stderr_tail"] = err[-600:]
            return "", statistics.median(times), None, extra
        return text, statistics.median(times), (statistics.median(stages) if stages else None), extra


# ------------------------------------------------------------------- the sweep

def load_fixtures(images: Path) -> list[dict]:
    gt_path = images / "ground_truth.json"
    truth = {}
    if gt_path.exists():
        for rec in json.loads(gt_path.read_text())["records"]:
            truth[rec["file"]] = rec["text"]
    fixtures = []
    for p in sorted(images.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            continue
        fixtures.append({"name": p.name, "path": p, "truth": truth.get(p.name)})
    return fixtures


def build_engines(args) -> list[Engine]:
    binary = ROOT / args.build_dir / "crispembed"
    md = Path(args.model_dir) if args.model_dir else None
    engines: list[Engine] = [
        TesseractCLI(args.tesseract_lang, args.psm),
        EasyOCRPy(),
        PaddleOCRPy(),
    ]

    def model(name: str) -> Path | None:
        return (md / name) if md else None

    if md:
        engines += [
            CrispEmbedCLI("crispembed-tesseract", "tesseract", binary,
                          model(args.dbnet), model(args.tess_rec),
                          {"CRISPEMBED_TESSERACT_BENCH": "1"}, "tesseract"),
            CrispEmbedCLI("crispembed-easyocr", "easyocr", binary,
                          model(args.dbnet), model(args.easyocr_rec),
                          {"CRISPEMBED_EASYOCR_BENCH": "1"}, "easyocr"),
            CrispEmbedCLI("crispembed-ppocrv6", "ppocrv6", binary,
                          model(args.ppocr_det), model(args.ppocr_rec),
                          {"CRISPEMBED_PPOCRV6_BENCH": "1"}, "ppocrv6"),
        ]
    return [e for e in engines if e.name not in args.skip]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path,
                    help="directory of fixtures; ground_truth.json is used when present")
    ap.add_argument("--model-dir", default=os.environ.get("CRISPEMBED_GGUF_DIR", ""))
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tesseract-lang", default="eng")
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--dbnet", default="dbnet-ic15-q8_0.gguf")
    ap.add_argument("--tess-rec", default="tesseract-eng-q8_0-seeded.gguf")
    ap.add_argument("--easyocr-rec", default="easyocr-english-g2-f16.gguf")
    ap.add_argument("--ppocr-det", default="PP-OCRv6_small_det-f16.gguf")
    ap.add_argument("--ppocr-rec", default="PP-OCRv6_small_rec-f16.gguf")
    ap.add_argument("--reference", default="tesseract-cli:eng",
                    help="engine whose output port-fidelity CER is measured against")
    ap.add_argument("--skip", action="append", default=[])
    ap.add_argument("--output", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    fixtures = load_fixtures(args.images)
    if args.limit:
        fixtures = fixtures[: args.limit]
    if not fixtures:
        print(f"no fixtures in {args.images}", file=sys.stderr)
        return 2

    engines = build_engines(args)
    skipped = {}
    active = []
    for e in engines:
        why = e.available()
        if why:
            skipped[e.name] = why
            print(f"SKIP {e.name}: {why}", flush=True)
        else:
            active.append(e)

    rows = []
    for fx in fixtures:
        record = {"fixture": fx["name"], "has_truth": fx["truth"] is not None, "engines": {}}
        for e in active:
            text, proc_ms, engine_ms, extra = e.run(fx["path"], args.repeats)
            entry = {
                "kind": e.kind,
                "text": text,
                "proc_ms": round(proc_ms, 1),
                "engine_ms": round(engine_ms, 1) if engine_ms is not None else None,
                **extra,
            }
            if fx["truth"]:
                entry.update(score(text, fx["truth"]))
            record["engines"][e.name] = entry
            tag = f"cer={entry.get('cer')}" if fx["truth"] else f"chars={len(normalize(text))}"
            print(f"  {fx['name']:28} {e.name:22} {proc_ms:8.1f} ms  {tag}", flush=True)

        # Port fidelity: every arm scored against the chosen reference engine's
        # own transcription of this same image.
        ref = record["engines"].get(args.reference, {}).get("text")
        if ref:
            for name, entry in record["engines"].items():
                if name == args.reference:
                    continue
                entry["vs_reference"] = score(entry["text"], ref)
        rows.append(record)

    result = {
        "version": 1,
        "images": str(args.images),
        "repeats": args.repeats,
        "reference_engine": args.reference,
        "skipped": skipped,
        "fixtures": rows,
    }
    summary = summarize(result)
    result["summary"] = summary
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    md = render_markdown(result)
    if args.markdown:
        args.markdown.write_text(md)
    print()
    print(md)
    return 0


def summarize(result: dict) -> dict:
    per_engine: dict[str, dict] = {}
    for fx in result["fixtures"]:
        for name, entry in fx["engines"].items():
            agg = per_engine.setdefault(name, {
                "kind": entry["kind"], "cer": [], "wer": [], "ref_cer": [],
                "proc_ms": [], "engine_ms": [], "failures": 0, "n": 0,
            })
            agg["n"] += 1
            if entry.get("returncode", 0) != 0 or not entry["text"].strip():
                agg["failures"] += 1
            if entry.get("cer") is not None:
                agg["cer"].append(entry["cer"])
                agg["wer"].append(entry["wer"])
            if entry.get("vs_reference", {}).get("cer") is not None:
                agg["ref_cer"].append(entry["vs_reference"]["cer"])
            agg["proc_ms"].append(entry["proc_ms"])
            if entry["engine_ms"] is not None:
                agg["engine_ms"].append(entry["engine_ms"])
    out = {}
    for name, agg in per_engine.items():
        mean = lambda xs: round(statistics.fmean(xs), 5) if xs else None  # noqa: E731
        med = lambda xs: round(statistics.median(xs), 1) if xs else None  # noqa: E731
        out[name] = {
            "kind": agg["kind"],
            "n": agg["n"],
            "failures": agg["failures"],
            "mean_cer": mean(agg["cer"]),
            "mean_wer": mean(agg["wer"]),
            "mean_cer_vs_reference": mean(agg["ref_cer"]),
            "median_proc_ms": med(agg["proc_ms"]),
            "median_engine_ms": med(agg["engine_ms"]),
        }
    return out


def render_markdown(result: dict) -> str:
    lines = [
        f"### OCR head-to-head ({result['images']}, repeats={result['repeats']})",
        "",
        f"Port-fidelity reference engine: `{result['reference_engine']}`.",
        "`proc_ms` includes model load (one CLI invocation); `engine_ms` excludes it.",
        "The two columns are not comparable to each other.",
        "",
        "| engine | kind | n | fail | CER↓ | WER↓ | CER vs ref | proc ms | engine ms |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    fmt = lambda v: "—" if v is None else f"{v}"  # noqa: E731
    for name, s in sorted(result["summary"].items(), key=lambda kv: (kv[1]["kind"], kv[0])):
        lines.append(
            f"| `{name}` | {s['kind']} | {s['n']} | {s['failures']} | {fmt(s['mean_cer'])} | "
            f"{fmt(s['mean_wer'])} | {fmt(s['mean_cer_vs_reference'])} | "
            f"{fmt(s['median_proc_ms'])} | {fmt(s['median_engine_ms'])} |"
        )
    if result["skipped"]:
        lines += ["", "Skipped: " + ", ".join(f"`{k}` ({v})" for k, v in result["skipped"].items())]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
