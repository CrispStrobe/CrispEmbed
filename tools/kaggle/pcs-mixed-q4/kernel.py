"""PCS mixed-precision 4-bit candidates: quantize, native strict gate, exact-arithmetic replay.

Clones the exact source (__SHA__), builds crispembed-quantize and punct-ab (CPU), downloads the
pinned published PCS files (cstr/pcs-xlmr-base-GGUF revision 27f17edb, SHA-256 checked) and:
  1. re-runs the published q4_k-imatrix recipe (`crispembed-quantize F32 OUT q4_k --imatrix`) as a
     provenance control (byte-identical or not is recorded, not assumed);
  2. builds each candidate with the same recipe plus `--tensor-type GLOB=q8_0` overrides;
  3. gates every file with tests/pcs_parity.py WITHOUT any recorded-loss allowance (all four heads,
     decoded text exact), and replays it in the pinned original ONNX graph
     (tools/pcs_tensor_bisect.py for the row37 `o` logit, tools/pcs_weight_replay.py --model for the
     six decoded cases with the original collector).
Output: summary.json plus every candidate GGUF. Nothing is uploaded from here.
See docs/pcs-quality-root-cause.md.
"""
import hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

SHA = "__SHA__"
SRC = Path("/kaggle/temp/ce")
BUILD = Path("/kaggle/temp/b")
MODELS = Path("/kaggle/temp/models")
OUT = Path("/kaggle/working")
REPO, REV = "cstr/pcs-xlmr-base-GGUF", "27f17edbd7e33949f42a9a4fe4322b5af8172e6b"
PINNED = {
    "pcs-xlmr-base.gguf": "28057dd95378348dd72a374446bdd83d0fa30ab24ecb0e8d01ce44aa8f4459d7",
    "pcs-xlmr-base.imatrix": "4c04cf422c12973320f4b7c27757ed5304d394333ed42e62024765407712acbc",
    "pcs-xlmr-base-q4_k-imatrix.gguf": "faee4bf8469a502362b5967a2cfeb7a1732539fbd999c63b489d00efff3531da",
    "pcs-xlmr-base-q8_0.gguf": "5656dca3883fe391ddf53926cd4af89a1b5b474a5a50eb5884389fed737ca2db",
}
CANDIDATES = {
    "enc0": ["enc.0.*"],
    "tc": ["head.tc.*"],
    "enc0-tc": ["enc.0.*", "head.tc.*"],
    "enc0-tc-enc3": ["enc.0.*", "head.tc.*", "enc.3.*"],
    "enc0-tc-enc2-enc3": ["enc.0.*", "head.tc.*", "enc.2.*", "enc.3.*"],
    "enc0to5-tc": [f"enc.{i}.*" for i in range(6)] + ["head.tc.*"],
}
summary = {"schema": "pcs-mixed-q4-v1", "sha": SHA, "source": {"repository": REPO, "revision": REV},
           "files": {}, "started": time.time()}


def save():
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


def sh(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), text=True, capture_output=True, **kw)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tensor_types(path):
    import gguf
    counts = {}
    for t in gguf.GGUFReader(str(path)).tensors:
        prefix = ".".join(t.name.split(".")[:2])
        counts.setdefault(t.tensor_type.name, {}).setdefault(prefix, 0)
        counts[t.tensor_type.name][prefix] += 1
    return counts


def evaluate(name, path, snapshot):
    rec = summary["files"].setdefault(name, {})
    rec.update(sha256=sha256(path), bytes=path.stat().st_size, types=tensor_types(path))
    r = sh([sys.executable, "tests/pcs_parity.py", BUILD, path, "tests/regression/pcs/blueprint_ref.txt"],
           cwd=SRC, env=dict(os.environ, LD_LIBRARY_PATH=str(BUILD)))
    rec["native_strict"] = {"rc": r.returncode, "stdout": r.stdout[-3000:], "stderr": r.stderr[-1000:],
                            "pass": r.returncode == 0 and r.stdout.strip().endswith("PASS")}
    save()
    r = sh([sys.executable, "tools/pcs_tensor_bisect.py", snapshot, path, "emb.,enc.,head.",
            "--work", f"/kaggle/temp/bisect-{name}"], cwd=SRC)
    m = re.search(r"row37_o_logit=([+-][0-9.]+)", r.stdout)
    rec["replay_row37_o_logit"] = float(m.group(1)) if m else None
    rec["bisect_stderr"] = r.stderr[-800:] if not m else ""
    r = sh([sys.executable, "tools/pcs_weight_replay.py", "--model", path], cwd=SRC)
    rep = SRC / f"pcs-weight-replay-{path.stem}" / "summary.json"
    if rep.exists():
        cases = json.loads(rep.read_text())["cases"]
        rec["replay_decoded_exact"] = sum(c["decoded_exact"] for c in cases)
        rec["replay_cases"] = len(cases)
        rec["replay_decoded"] = [c["decoded"] for c in cases]
    rec["replay_rc"] = r.returncode
    rec["replay_stderr"] = r.stderr[-800:] if r.returncode else ""
    save()


try:
    Path("/kaggle/temp").mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    assert sh(["git", "clone", "-q", "--filter=blob:none", "https://github.com/CrispStrobe/CrispEmbed.git", SRC]).returncode == 0
    assert sh(["git", "checkout", "-q", SHA], cwd=SRC).returncode == 0
    assert sh(["git", "submodule", "update", "-q", "--init", "--depth", "1", "ggml"], cwd=SRC).returncode == 0
    sys.path.insert(0, str(SRC))
    try:
        import kaggle_harness as kh
        kh.ensure_build_cache()
    except Exception as e:  # cache is an optimisation only
        summary["cache_error"] = repr(e)[:300]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf", "onnx", "onnxruntime",
                    "sentencepiece", "pyyaml", "huggingface_hub"], check=True)
    r = sh(["cmake", "-S", SRC, "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF"])
    assert r.returncode == 0, r.stderr[-2000:]
    r = sh(["cmake", "--build", BUILD, "--target", "crispembed-quantize", "punct-ab", "--parallel", "4"])
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    summary["cpu"] = Path("/proc/cpuinfo").read_text().split("model name")[1].split("\n")[0].strip(": \t")
    summary["cpu_flags_avx512f"] = " avx512f " in Path("/proc/cpuinfo").read_text()

    from huggingface_hub import hf_hub_download, snapshot_download
    files = {}
    for name, digest in PINNED.items():
        files[name] = Path(hf_hub_download(REPO, name, revision=REV, local_dir=MODELS))
        assert sha256(files[name]) == digest, f"{name} hash differs"
    snapshot = Path(snapshot_download("1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase",
                                      revision="d1769a597ce8dfaa070d436bc67d4ee761f58884",
                                      allow_patterns=["model.onnx", "sp.model", "config.yaml"]))
    assert sha256(snapshot / "model.onnx") == "c43ca686dabc237c3b06be834b9423c07580fef7e2b1a6c09976f7d60caa5d89"
    save()

    q = BUILD / "crispembed-quantize"
    base = [q, files["pcs-xlmr-base.gguf"]]
    imat = ["--imatrix", files["pcs-xlmr-base.imatrix"]]
    repro = OUT / "repro-q4_k-imatrix.gguf"
    r = sh(base + [repro, "q4_k"] + imat)
    summary["recipe"] = "crispembed-quantize pcs-xlmr-base.gguf OUT q4_k --imatrix pcs-xlmr-base.imatrix"
    summary["repro"] = {"rc": r.returncode, "sha256": sha256(repro) if repro.exists() else None,
                        "byte_identical_to_published": repro.exists() and sha256(repro) == PINNED["pcs-xlmr-base-q4_k-imatrix.gguf"],
                        "log_tail": (r.stdout + r.stderr)[-1500:]}
    save()

    evaluate("published-q8_0", files["pcs-xlmr-base-q8_0.gguf"], snapshot)
    evaluate("published-q4_k-imatrix", files["pcs-xlmr-base-q4_k-imatrix.gguf"], snapshot)
    for name, globs in CANDIDATES.items():
        out = OUT / f"pcs-xlmr-base-q4_k-imatrix-{name}.gguf"
        args = base + [out, "q4_k"] + imat
        for g in globs:
            args += ["--tensor-type", f"{g}=q8_0"]
        r = sh(args)
        summary["files"][name] = {"q8_0_globs": globs, "quantize_rc": r.returncode,
                                  "quantize_log_tail": (r.stdout + r.stderr)[-800:]}
        if r.returncode == 0:
            evaluate(name, out, snapshot)
        save()
except BaseException as e:
    summary["error"] = repr(e)[:1500]
    raise
finally:
    summary["elapsed_s"] = round(time.time() - summary.pop("started", time.time()), 1)
    save()
