#!/usr/bin/env python3
"""CrispEmbed OCR portfolio regression driver — run ONE model.

For a model named in ``manifest.json`` this:

1. Downloads the GGUF under test, pinned to an HF revision SHA (so an
   upstream re-quantise can't silently regress us).
2. Runs ``crispembed -m <gguf> --ocr <image>`` and captures stdout.
3. Asserts the text matches the pinned ``expected_text`` under a
   *lenient* comparison — normalised (case/punct/whitespace-insensitive)
   character error rate (CER) at or below ``match.max_cer``. OCR is not
   byte-exact across builds; small punctuation/spacing drift is fine, a
   scrambled or empty transcript is not.
4. Runs a repetition / no-garbage guard: the ``colorcolorcolor…``
   degeneration signature (one token or short n-gram repeated far beyond
   any real text) fails even if CER somehow slips through.
5. OPTIONALLY, if the model declares a ``diff`` block AND its reference
   GGUF exists on HF, downloads ``<model>-ref.gguf`` and runs
   ``test-<model>-diff <gguf> <ref>``, asserting every stage's
   ``cos_min`` is at or above its pinned threshold. If the ref is absent
   on HF the diff step is skipped (not failed) — diff testing is opt-in
   by the mere presence of the ref.

This is the per-model unit. The Kaggle portfolio harness and the CI
workflow both call ``regression_for()`` in a loop over the manifest.

Env:
  BUILD_DIR    build dir containing crispembed + test-*-diff (default: ./build)
  CRISPEMBED_BIN / DIFF_BIN_DIR  override binary locations
  HF_TOKEN     for private/gated repos (optional for public)
  GOT_OCR_FORCE_CPU=1 etc. pass through to the binary

Usage:
  python tests/regression/run_one.py --name got-ocr2
  python tests/regression/run_one.py --all
  python tests/regression/run_one.py --name got-ocr2 --rebake   # print captured text
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_MANIFEST = HERE / "manifest.json"


# ── infra ────────────────────────────────────────────────────────────
def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_model(manifest: dict, name: str) -> dict:
    for m in manifest["models"]:
        if m["name"] == name:
            return m
    die(f"model '{name}' not in manifest")


def build_dir() -> Path:
    return Path(os.environ.get("BUILD_DIR", str(REPO_ROOT / "build"))).resolve()


def crispembed_bin() -> Path:
    if os.environ.get("CRISPEMBED_BIN"):
        return Path(os.environ["CRISPEMBED_BIN"])
    b = build_dir() / "crispembed"
    if not b.exists():
        b = build_dir() / "bin" / "crispembed"
    return b


def diff_bin(binary_name: str) -> Path:
    d = Path(os.environ.get("DIFF_BIN_DIR", str(build_dir())))
    p = d / binary_name
    if not p.exists():
        p = d / "bin" / binary_name
    return p


def hf_download(repo: str, file_in_repo: str, revision: str, dest_dir: Path,
                optional: bool = False):
    """Download one file at a pinned revision. Return Path, or None if
    `optional` and the file/repo is absent (404)."""
    # The regression runner may execute in a read-only sandbox where the
    # process-wide /tmp is unavailable. Keep hub staging beside the external
    # artifact cache instead.
    dest_dir.mkdir(parents=True, exist_ok=True)
    # The HF/Xet client concatenates TMPDIR with a generated filename in one
    # path. Keep the separator explicit; without it a sandbox path such as
    # /tmp/crispembed-regression becomes /tmpcrispembed-regressionXXXX.
    tmp_root = str(dest_dir) + os.sep
    os.environ["TMPDIR"] = tmp_root
    import tempfile
    tempfile.tempdir = tmp_root
    # Override inherited cache locations as well. CI runners can export a
    # read-only /tmp cache, and huggingface_hub's symlink probe then creates
    # malformed paths such as /tmpXXXX.
    os.environ["HF_HOME"] = str(dest_dir / ".hf")
    os.environ["HF_HUB_CACHE"] = str(dest_dir / ".hf" / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(dest_dir / ".hf" / "hub")
    os.environ["HF_XET_CACHE"] = str(dest_dir / ".hf" / "xet")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    token = os.environ.get("HF_TOKEN")
    try:
        # Download into the hub cache and copy the resolved blob into the
        # regression directory.  Using ``local_dir`` makes huggingface_hub
        # probe symlink support against the common parent (often /tmp), which
        # is read-only in some CI/sandbox environments.
        cached = Path(hf_hub_download(
            repo_id=repo, filename=file_in_repo, revision=revision,
            cache_dir=str(dest_dir / ".hf" / "hub"), token=token))
        target = dest_dir / file_in_repo
        target.parent.mkdir(parents=True, exist_ok=True)
        if cached.resolve() != target.resolve():
            shutil.copyfile(cached, target)
        return target
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        if optional:
            print(f"  (optional artifact absent on HF: {repo}/{file_in_repo}) — skipping")
            return None
        die(f"required HF artifact missing: {repo}/{file_in_repo}@{revision}: {e}")
    except Exception as e:  # noqa: BLE001
        # 404s sometimes surface as generic HfHubHTTPError
        if optional and ("404" in str(e) or "not found" in str(e).lower()):
            print(f"  (optional artifact absent on HF: {repo}/{file_in_repo}) — skipping")
            return None
        die(f"HF download failed: {repo}/{file_in_repo}@{revision}: {e}")


def resolve_sample_hf(spec: dict, dest_dir: Path) -> Path:
    """Extract a fixture image embedded in an HF-dataset parquet at a pinned
    revision, and write it to dest_dir as a PNG.

    This keeps license-restricted source images (e.g. CROHME = CC-BY-NC-SA)
    OUT of this MIT/Apache repo: they are fetched from the original dataset at
    test time (like the GGUFs already are), never committed here. `spec` keys:
    dataset, file (parquet path in the dataset repo), revision (commit sha —
    pin it so the row is stable), row (integer index), image_column, and
    optional label_column/expect_label which gate against the dataset shifting
    under the pinned row."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    pq = hf_hub_download(repo_id=spec["dataset"], filename=spec["file"],
                         revision=spec["revision"], repo_type="dataset",
                         local_dir=str(dest_dir), token=os.environ.get("HF_TOKEN"))
    df = pd.read_parquet(pq)
    idx = int(spec["row"])
    if idx >= len(df):
        die(f"sample_hf row {idx} out of range (dataset has {len(df)} rows)")
    row = df.iloc[idx]
    lbl_col, expect = spec.get("label_column"), spec.get("expect_label")
    if lbl_col and expect is not None and str(row[lbl_col]) != expect:
        die(f"sample_hf row {idx} label mismatch: got {str(row[lbl_col])!r}, "
            f"expected {expect!r} — the pinned dataset revision may have changed")
    img = row[spec["image_column"]]
    data = img["bytes"] if isinstance(img, dict) else bytes(img)
    out = dest_dir / f"sample_hf_{spec['dataset'].replace('/', '_')}_{idx}.png"
    out.write_bytes(data)
    return out


# ── text comparison (lenient) ────────────────────────────────────────
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(s: str) -> str:
    """Lower-case, drop punctuation, collapse whitespace. Lenient OCR
    comparison ignores punctuation/spacing/case drift between builds."""
    s = s.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_error_rate(expected: str, actual: str) -> float:
    """Normalised CER. 0.0 == identical (after normalisation)."""
    e, a = normalize_text(expected), normalize_text(actual)
    if not e:
        return 0.0 if not a else 1.0
    return _levenshtein(e, a) / len(e)


# ── repetition / no-garbage guard ────────────────────────────────────
def detect_garbage(text: str, min_len: int = 12) -> str | None:
    """Return a reason string if `text` looks like degenerate repetition
    (the colorcolorcolor… signature), else None.

    Two detectors:
      * word-level: a single word occupies an implausible fraction of a
        multi-word transcript, or repeats many times.
      * char-level: a short substring (2..8 chars) tiles most of a
        space-poor blob (catches 'colorcolorcolor' with no spaces).
    """
    t = text.strip()
    if len(t) < min_len:
        return None  # too short to judge; correctness handled by CER

    words = normalize_text(t).split()
    if len(words) >= 8:
        from collections import Counter
        top, n = Counter(words).most_common(1)[0]
        if n >= max(10, int(0.5 * len(words))):
            return f"word '{top}' repeats {n}/{len(words)} times"

    # char-level tiling: a short unit (1..8 chars) repeated back-to-back 8+
    # times ANYWHERE in the blob (robust to a junk prefix like the real
    # '…雍colorcolorcolor…' output). Whitespace stripped so 'color color …'
    # is caught too.
    blob = re.sub(r"\s+", "", t.lower())
    if len(blob) >= 24:
        m = re.search(r"(.{1,8}?)\1{7,}", blob)
        if m:
            seg = m.group(1)
            span = len(m.group(0))
            return f"substring '{seg}' repeats {span // len(seg)}x ({span} chars)"
    return None


# ── OCR run ──────────────────────────────────────────────────────────
def run_ocr(bin_path: Path, gguf: Path, image: Path, extra_args: list[str],
            timeout: int = 900, detector: Path | None = None) -> str:
    if not bin_path.exists():
        die(f"crispembed binary not found: {bin_path}")
    cmd = [str(bin_path), "-m", str(gguf)]
    if detector is not None:
        cmd += ["--ocr-det", str(detector), "--ocr-rec", str(gguf)]
    cmd += ["--ocr", str(image), *extra_args]
    # OCR may emit arbitrary model-derived bytes for an invalid/untrained
    # character sequence.  The regression guard must inspect that output and
    # report it, rather than crashing while decoding the child pipe as UTF-8.
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                          check=False, timeout=timeout)
    if proc.returncode != 0:
        die(f"crispembed --ocr exited {proc.returncode}\n"
            f"  cmd: {' '.join(cmd)}\n  stderr tail: {proc.stderr[-500:]}")
    return proc.stdout.strip()


# ── diff harness ─────────────────────────────────────────────────────
# CrispEmbed test-*-diff lines look like:
#   llm_layer_0: cos_min=0.999960 max_abs=0.173531 PASS
# Diff binaries print per-stage cosines in several bespoke formats. Match all of
# them, else a numerically-correct engine gets a false "no parseable stage lines"
# FAIL (this masked ~7 correct engines in the 2026-07 Kaggle CUDA run).
# Strip ANSI colour codes before parsing — several diff binaries (lfm2, lilt, …)
# wrap the PASS/FAIL tag in \033[32m…\033[0m, which otherwise sits between the
# leading whitespace and "[PASS]" and defeats the anchored regexes below.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

_DIFF_PATTERNS = [
    # PP-OCRv6 native activation harness: "[ppocrv6-diff] stage cos=…".
    re.compile(r"^\s*\[ppocrv6-diff\]\s+(\S+)\s+cos=([-+0-9.eE]+)"),
    # got-ocr "stage: cos_min=…" AND hat/pan/tbsrn "output   cos_min=…" (the colon
    # is optional — the SR binaries pad the name with spaces, no colon).
    re.compile(r"^\s*([^\s:]+):?\s+cos(?:_min)?=([-+0-9.eE]+)\s+max_abs="),
    # swinir / dat:                  "output   cos=0.998  cos_ch_min=... max_abs=..."
    re.compile(r"^\s*(\S+)\s+cos=([-+0-9.eE]+)\s+cos_ch_min="),
    # lfm2 (assertion form):         "[PASS] layer_15    cos=0.999714  (thr=0.999)"
    re.compile(r"^\s*\[(?:PASS|FAIL)\]\s+(\S+)\s+cos=([-+0-9.eE]+)"),
    # lilt table: "stage <elements int> <cos_min> <max_abs> <status>". REQUIRE a
    # PASS/FAIL/SKIP status at the end so this does NOT match LAYOUT_DEBUG value
    # dumps like "memory[s3,0,:8]: 0.0 0.0 0.0 0.0 …" (which end in a number).
    re.compile(r"^\s*(\S+)\s+\d+\s+([01]\.\d+)\s+[-+0-9.eE]+\s+(?:PASS|FAIL|SKIP)\b"),
    # layout table: "stage <cos_min> <cos_mean> <max_abs> <status>" (status required).
    re.compile(r"^\s*(\S+)\s+([01]\.\d+)\s+[01]\.\d+\s+[-+0-9.eE]+\s+(?:PASS|FAIL|SKIP)\b"),
    # bert_ner (bare columns):       "final_hidden   0.995906   6.60e-02 PASS"
    re.compile(r"^\s*(\S+)\s+([01]\.\d+)\s+[-+0-9.eE]+\s+(?:PASS|FAIL)\s*$"),
]
# granite-vision prints the stage NAME on one line ("C++ stage: projector …") and
# the score on the NEXT ("    cos_min=0.952 max_abs=2.97e+00 FAIL") — no name on
# the score line, so the anchored patterns above miss it. Match the nameless
# score line and give it a synthetic stage id (evaluated against the manifest's
# global "*" threshold).
_DIFF_NAMELESS = re.compile(r"^\s*cos(?:_min)?=([-+0-9.eE]+)\s+max_abs=\S+\s+(?:PASS|FAIL|SKIP)")
# kept for back-compat references
_DIFF_LINE = _DIFF_PATTERNS[0]


def parse_diff_stdout(stdout: str) -> dict[str, float]:
    stages: dict[str, float] = {}
    nameless = 0
    for line in stdout.splitlines():
        line = _ANSI.sub("", line)
        matched = False
        for pat in _DIFF_PATTERNS:
            m = pat.match(line)
            if m:
                try:
                    stages[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
                matched = True
                break
        if not matched:
            m = _DIFF_NAMELESS.match(line)
            if m:
                try:
                    nameless += 1
                    stages[f"stage_{nameless}"] = float(m.group(1))
                except ValueError:
                    pass
    return stages


def evaluate_stage_thresholds(stages: dict[str, float],
                              thresholds: dict[str, float]):
    """thresholds may be {"*": 0.999} for a global floor, and/or per-stage
    overrides. Returns (fails, missing). fails: [(stage, cos, thr)]."""
    fails, missing = [], []
    default = thresholds.get("*")
    checked = set()
    for stage, cos in stages.items():
        thr = thresholds.get(stage, default)
        if thr is None:
            continue
        checked.add(stage)
        if cos < thr:
            fails.append((stage, cos, thr))
    for stage, thr in thresholds.items():
        if stage == "*":
            continue
        if stage not in stages:
            missing.append(stage)
    if default is not None and not checked:
        missing.append("<any stage matching '*'>")
    return fails, missing


def run_diff(diff_binary: Path, gguf: Path, ref: Path,
             extra_args: list[str], timeout: int = 900) -> dict[str, float]:
    if not diff_binary.exists():
        die(f"diff binary not found: {diff_binary}")
    # PP-OCRv6's native diff harness takes MODEL + IMAGE and receives the
    # Python activation fixture through PPOCRV6_REF; other diff binaries use
    # the conventional MODEL + REF argv shape.
    if diff_binary.name == "test-ppocrv6-rec":
        cmd = [str(diff_binary), str(gguf), *extra_args]
    else:
        cmd = [str(diff_binary), str(gguf), str(ref), *extra_args]
    env = dict(os.environ)
    if diff_binary.name == "test-ppocrv6-rec":
        env["PPOCRV6_REF"] = str(ref)
    env.setdefault("DYLD_LIBRARY_PATH", str(diff_binary.parent))
    env.setdefault("LD_LIBRARY_PATH", str(diff_binary.parent))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          check=False, timeout=timeout, env=env)
    # Parse the per-stage cosines BEFORE reacting to the exit code. A
    # teardown/atexit crash (common on CUDA: the GPU device is freed by a C++
    # static dtor at process exit, AFTER the diff already printed every stage —
    # the ggml v0.10 residency assert / SIGABRT class) must not mask a correct
    # forward. If we parsed complete stage lines, judge by them and only WARN on
    # the signal; die only when the process crashed before producing any output.
    stages = parse_diff_stdout(proc.stdout + "\n" + proc.stderr)
    if proc.returncode < 0:
        if stages:
            print(f"[warn] diff harness exited on signal {-proc.returncode} "
                  f"after producing {len(stages)} stage(s) — treating as a "
                  f"teardown crash and judging by the stages")
        else:
            die(f"diff harness died from signal {-proc.returncode} before any "
                f"stage output\n  stderr tail: {proc.stderr[-400:]}")
    if not stages:
        die(f"diff harness produced no parseable stage lines.\n"
            f"  stdout tail: {proc.stdout[-400:]}\n"
            f"  stderr tail: {proc.stderr[-400:]}")
    return stages


# ── per-model regression ─────────────────────────────────────────────
def regression_for(name: str, manifest: dict, work_dir: Path,
                   rebake: bool = False) -> int:
    """Run one model's regression. Return number of failures (0 == pass)."""
    entry = find_model(manifest, name)
    work_dir.mkdir(parents=True, exist_ok=True)

    g = entry["gguf"]
    gguf = hf_download(g["repo"], g["file"], g.get("revision", "main"), work_dir)
    detector = None
    if entry.get("detector"):
        d = entry["detector"]
        detector = hf_download(d["repo"], d["file"], d.get("revision", "main"), work_dir)

    # Restoration/SR engines (restormer, scunet, …) have no --ocr path: their only
    # regression signal is the diff harness (golden output vs a PyTorch reference).
    # Such entries set "diff_only": true and skip the OCR/garbage/text steps.
    if entry.get("diff_only"):
        return _run_diff_block(name, entry, gguf, work_dir, g)

    # run_check: an engine whose per-stage ref is unreliable (e.g. gliner's dumper
    # produces a dead model) is guarded by an independent task-output assertion baked
    # into its test binary — run `<binary> <gguf> [args]` and require the pinned exit.
    rc_spec = entry.get("run_check")
    if rc_spec:
        rc_bin = diff_bin(rc_spec["binary"])
        if not rc_bin.exists():
            # An optional test binary that isn't built in this config is a SKIP,
            # not a FAIL. e.g. test-punct-diff needs crisp_punc (CrispASR), which
            # the Kaggle CUDA kernel doesn't clone (CRISPEMBED_HAS_CRISP_PUNC=OFF)
            # — the punct models (pcs/fireredpunc/fullstop) then can't be checked
            # here. Previously this crashed with FileNotFoundError → false FAIL.
            print(f"[{name}] SKIP run_check ({rc_spec['binary']} not built)")
            return 0
        argv = [str(rc_bin), str(gguf), *rc_spec.get("args", [])]
        print(f"[{name}] run_check: {' '.join(argv)}")
        r = subprocess.run(argv, env={**os.environ, **rc_spec.get("env", {})})
        exp = int(rc_spec.get("expect_exit", 0))
        ok = r.returncode == exp
        print(f"[{name}] {'PASS' if ok else 'FAIL'} (exit {r.returncode}, expect {exp})")
        return 0 if ok else 1

    if entry.get("sample_hf"):
        # License-restricted source image fetched from its dataset at test time
        # (never committed here) — see resolve_sample_hf.
        sample = resolve_sample_hf(entry["sample_hf"], work_dir)
    else:
        sample = REPO_ROOT / entry["sample"]
        if not sample.exists():
            # allow fixtures to live in an HF fixtures repo
            fx = manifest.get("fixtures")
            if fx and "sample" in entry:
                sample = hf_download(fx["repo"], entry["sample"],
                                     fx.get("revision", "main"), work_dir)
            else:
                die(f"sample image missing: {sample}")

    extra = entry.get("ocr_args", [])
    print(f"[{name}] running crispembed --ocr {sample.name} ...")
    text = run_ocr(crispembed_bin(), gguf, sample, extra, detector=detector)

    if rebake:
        print(f"[{name}] CAPTURED OUTPUT:\n{text}\n")
        return 0

    failures = 0

    # 1) no-garbage guard
    reason = detect_garbage(text)
    if reason:
        print(f"[{name}] FAIL garbage-guard: {reason}\n  text: {text[:200]!r}")
        failures += 1
    else:
        print(f"[{name}] PASS garbage-guard")

    # 2) lenient text match
    expected = entry.get("expected_text")
    if expected is None:
        print(f"[{name}] SKIP text-match (no expected_text pinned yet)")
    else:
        max_cer = float(entry.get("match", {}).get("max_cer", 0.10))
        cer = char_error_rate(expected, text)
        verdict = "PASS" if cer <= max_cer else "FAIL"
        if verdict == "FAIL":
            failures += 1
        print(f"[{name}] {verdict} text-match cer={cer:.3f} (max {max_cer:.3f})")
        if verdict == "FAIL":
            print(f"  expected: {expected[:160]!r}\n  actual:   {text[:160]!r}")

    # 3) optional diff harness (opt-in by ref presence on HF)
    failures += _run_diff_block(name, entry, gguf, work_dir, g, count_only=True)

    print(f"[{name}] {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return failures


def _run_diff_block(name: str, entry: dict, gguf: Path, work_dir: Path,
                    g: dict, count_only: bool = False) -> int:
    """Run the optional diff harness. Returns failure count. When count_only is
    False (diff-only entries) it also prints the final PASS/FAIL banner."""
    diff = entry.get("diff")
    failures = 0
    if diff:
        ref_spec = diff["ref"]
        ref = hf_download(ref_spec["repo"], ref_spec["file"],
                          ref_spec.get("revision", g.get("revision", "main")),
                          work_dir, optional=True)
        if ref is None:
            print(f"[{name}] SKIP diff-harness (ref not on HF yet)")
        else:
            stages = run_diff(diff_bin(diff["binary"]), gguf, ref,
                              diff.get("args", []))
            thresholds = diff.get("thresholds", {"*": 0.999})
            fails, missing = evaluate_stage_thresholds(stages, thresholds)
            if fails or missing:
                failures += 1
                for s, c, thr in fails:
                    print(f"[{name}] FAIL diff {s}: cos_min={c:.6f} < {thr}")
                for s in missing:
                    print(f"[{name}] FAIL diff missing stage: {s}")
            else:
                worst = min(stages.values())
                print(f"[{name}] PASS diff-harness ({len(stages)} stages, "
                      f"worst cos_min={worst:.6f})")
    elif not count_only:
        die(f"diff_only entry '{name}' has no diff block")

    if not count_only:
        print(f"[{name}] {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--name", help="single model name from the manifest")
    ap.add_argument("--all", action="store_true", help="run every model")
    ap.add_argument("--work-dir", type=Path,
                    default=Path(os.environ.get("REGRESSION_WORK", "/tmp/crispembed-regression")))
    ap.add_argument("--rebake", action="store_true",
                    help="print captured OCR text instead of asserting (to seed expected_text)")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if args.all:
        names = [m["name"] for m in manifest["models"]]
    elif args.name:
        names = [args.name]
    else:
        die("specify --name NAME or --all")

    total = 0
    for n in names:
        total += regression_for(n, manifest, args.work_dir, rebake=args.rebake)
    if not args.rebake:
        print(f"\n=== {len(names)} model(s), {total} failure(s) ===")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
