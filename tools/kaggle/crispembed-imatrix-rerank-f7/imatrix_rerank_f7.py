#!/usr/bin/env python3
"""F7b for the RERANKERS — re-collect + re-quantize every published cross-encoder
imatrix artifact with an F7-fixed binary (Kaggle, chr1s4).

Why (measured, not assumed — the published .imatrix files were read with
`gguf.GGUFReader` before this kernel was written):

    ms-marco-MiniLM-L-6-v2.imatrix      24 keys:  6 attn_output +  6 ffn_down +  6 ffn_up +  6 leaf_N
    ms-marco-MiniLM-L-12-v2.imatrix     48 keys: 12 + 12 + 12 + 12 leaf_N
    bge-reranker-v2-m3.imatrix          96 keys: 24 attn.o + 48 ffn.fcN + 24 leaf_N
    jina-reranker-v2-...multilingual    48 keys: 12 attn.o + 24 ffn.fcN + 12 leaf_N
    mxbai-rerank-{base,xsmall}-v1       72 keys: attn_q/attn_k/attn_output/ffn_up/ffn_down + 12 leaf_N
    bge-reranker-base.imatrix           72 keys: q/k/v/o + ffn — NO leaf_N (already clean)

`leaf_N` == n_layer in every defective file: that is the runtime's pre-merged
QKV weight, unnamed before F7 (`68033e8d`), so `attn.{q,k,v}.weight` quantized
with NO importance. bge-reranker-base escaped only because its GGUF ships F16
q/k/v and `src/crispembed.cpp` skips the pre-merge for non-F32 weights — it is
this run's no-change CONTROL.

Same reason the ms-marco pair MUST calibrate on the `-g7c` base: those repos
carry BOTH the superseded `<name>.gguf` (F32, 1-layer head, no BertPooler — the
G7c defect) and the corrected `<name>-g7c.gguf` (F16 2-D weights, 2-layer
tanh head). `pick_base_gguf` prefers the exact `<name>.gguf`, so the source is
pinned explicitly via `base_file`. The g7c artifacts are OLLAMA-mode-named
(`blk.N.attn_q.weight`); a `--crisp` conversion renames the encoder and would
match 0 tensors — the quantizer's own "N with imatrix" line is the proof, and
this run now FAILS instead of uploading a mislabeled artifact when it reads 0.

Naming (G3/F7b precedent — new names only, published SHAs stay valid; several
of these q4_k-imatrix/q8_0 files are pinned in examples/cli/model_hashes.h):
    <prefix>-f7.imatrix / <prefix>-f7-imatrix-ab.txt
    <prefix>-q4_k-imatrix-f7.gguf / <prefix>-iq4_xs-f7.gguf
and for ms-marco, composed with the existing correction suffix:
    <prefix>-g7c-f7.imatrix / <prefix>-g7c-f7-imatrix-ab.txt
    <prefix>-q4_k-imatrix-g7c-f7.gguf / <prefix>-iq4_xs-g7c-f7.gguf

Nothing here promotes a default: the registry and model_hashes.h are untouched.
"""
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
if not WORK.exists():
    WORK = Path("/tmp/crisp-imatrix-rerank-f7")
    WORK.mkdir(parents=True, exist_ok=True)

# The pipeline changes this run needs (explicit `base_file`/`prefix`, quantizer
# stdout verification, imatrix coverage digest, raw-score block) live on this
# branch; F7 itself is on main and merged into it.
BRANCH = os.environ.get("CRISP_BRANCH", "feat/imatrix-rerank-f7")
os.environ["CRISP_BRANCH"] = BRANCH
# Smallest bases first so a late OOM/timeout still leaves the priority models done.
# (ms-marco 70/92MB, mxbai-xsmall 286MB, mxbai-base 739MB, bge-base 948MB,
#  jina 1119MB, bge-v2-m3 2277MB — all well under BIG_BYTES, so every model
#  calibrates and quantizes from its full-precision base.)
os.environ["MODELS"] = ("ms-marco-MiniLM-L-6-v2,ms-marco-MiniLM-L-12-v2,"
                        "mxbai-rerank-xsmall-v1,mxbai-rerank-base-v1,"
                        "bge-reranker-base,jina-reranker-v2-base-multilingual,"
                        "bge-reranker-v2-m3")
os.environ["FORCE"] = "1"   # every one of these already has imatrix artifacts

# ── HF token: glob BOTH mount layouts before handing over to the harness ─────
# (t19 run 1 completed the full pipeline and then lost every upload to 401
# because the token dataset mounted only under /kaggle/input/datasets/<a>/<s>/.)
def _find_hf_token():
    import glob
    pats = ["/kaggle/input/*/hf_token.txt", "/kaggle/input/datasets/*/*/hf_token.txt"]
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            try:
                tok = Path(p).read_text().strip()
            except OSError:
                continue
            if tok:
                print(f"[rerank-f7] HF token found at {p} (len {len(tok)})", flush=True)
                return tok
    print(f"[rerank-f7] WARNING: no hf_token.txt under {pats}", flush=True)
    return None


if not os.environ.get("HF_TOKEN"):
    _tok = _find_hf_token()
    if not _tok:
        raise SystemExit("[rerank-f7] FATAL: no HF token — uploads would 401; aborting early")
    os.environ["HF_TOKEN"] = _tok
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _tok
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
repo = WORK / "CrispEmbed"
if not repo.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", BRANCH,
                           REPO_URL, str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "submodule", "update",
                           "--init", "--recursive"])
print("[rerank-f7] cloned {} at {}".format(BRANCH, subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()), flush=True)

# F7 guard: on a pre-F7 tree the collector files QKV statistics under leaf_N
# again and this run would re-publish the exact defect it exists to repair.
for rel in ("src/crispembed.cpp", "tools/quantize.cpp"):
    if "qkv_merged" not in (repo / rel).read_text(encoding="utf-8"):
        raise SystemExit(f"[rerank-f7] FATAL: clone lacks the F7 qkv_merged fix in {rel} "
                         f"(branch {BRANCH}) — refusing to re-publish the leaf_N defect")
print("[rerank-f7] F7 guard OK: qkv_merged present in crispembed.cpp + quantize.cpp", flush=True)

kdir = repo / "tools" / "kaggle" / "crispembed-imatrix-quant"
script = kdir / "imatrix_quant.py"
if not script.exists():
    raise SystemExit(f"[rerank-f7] FATAL: {script} missing in the clone")
for corpus in ("calib_corpus.jsonl", "eval_corpus.jsonl"):
    if not (kdir / corpus).exists():
        raise SystemExit(f"[rerank-f7] FATAL: {kdir / corpus} missing")
src_text = script.read_text()
if "base_file" not in src_text or "with imatrix" not in src_text:
    raise SystemExit("[rerank-f7] FATAL: cloned imatrix_quant.py lacks the base_file pin "
                     "and/or the quantizer-coverage check — wrong branch?")

sys.path.insert(0, str(kdir))
ns = {"__name__": "__rerank_f7__", "__file__": str(script)}
exec(compile(src_text, str(script), "exec"), ns)

# Rerankers already have first-class support in the shared pipeline (no surgery
# needed): MODE[...] == "rerank" routes calibration through 14 (query, [docs])
# pairs on the `--rerank` path — the collector fires on that graph exactly like
# the embed path — and the A/B metric is mean Kendall-tau over 30 eval pairs
# (6 docs each) vs the full-precision gold, with mean|dscore| as tiebreaker.
for _m in ("ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-12-v2", "bge-reranker-base",
           "bge-reranker-v2-m3", "jina-reranker-v2-base-multilingual",
           "mxbai-rerank-base-v1", "mxbai-rerank-xsmall-v1"):
    assert ns["MODE"].get(_m) == "rerank", f"{_m} is not in rerank MODE"

QS_F7 = [
    ("q8_0",   False, None),                              # A/B reference
    ("q4_k",   False, None),                              # baseline, shows the delta
    ("q4_k",   True,  "{prefix}-q4_k-imatrix-f7.gguf"),
    ("iq4_xs", True,  "{prefix}-iq4_xs-f7.gguf"),
]
for _m in ("bge-reranker-base", "bge-reranker-v2-m3",
           "jina-reranker-v2-base-multilingual",
           "mxbai-rerank-base-v1", "mxbai-rerank-xsmall-v1"):
    ns["OVERRIDES"][_m] = {"meta_prefix": f"{_m}-f7", "quants": QS_F7}

QS_G7C_F7 = [
    ("q8_0",   False, None),
    ("q4_k",   False, None),
    ("q4_k",   True,  "{prefix}-q4_k-imatrix-g7c-f7.gguf"),
    ("iq4_xs", True,  "{prefix}-iq4_xs-g7c-f7.gguf"),
]
for _m in ("ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-12-v2"):
    ns["OVERRIDES"][_m] = {
        "base_file": f"{_m}-g7c.gguf",   # NOT <name>.gguf (superseded, no BertPooler)
        "prefix": _m,
        "meta_prefix": f"{_m}-g7c-f7",
        "quants": QS_G7C_F7,
    }

ns["main"]()

# ── Independent spot-check: uploaded ms-marco artifact vs ONNX Runtime ────────
# tau/dscore are relative to the run's own gold, so they cannot see an absolute
# scale collapse. This re-downloads the artifact THAT WAS JUST UPLOADED and
# compares its decoded logits against a faithful HF export (Xenova). Failure is
# reported, never fatal — the quants are already published at this point.
def _spot_check():
    import json
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.environ["HF_TOKEN"])
    cli = repo / "build" / "crispembed"
    query = "How many people live in Berlin?"
    docs = [
        "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "Paris is the capital and most populous city of France.",
        "Berlin is well known for its museums and its metropolitan area of about six million people.",
        "Domestic cats sleep for a large part of the day.",
        "New York City had an estimated population of 8,804,190 in 2020.",
        "The Berlin Wall divided the city from 1961 until 1989.",
    ]
    out = ["===== ms-marco spot-check: uploaded artifact vs ONNX Runtime =====",
           f"query: {query}"]
    for i, d in enumerate(docs):
        out.append(f"  doc[{i}] {d}")

    ref = {}
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               "onnxruntime", "transformers"])
        import onnxruntime as ort
        from transformers import AutoTokenizer
        mdl = hf_hub_download("Xenova/ms-marco-MiniLM-L-6-v2", "onnx/model.onnx")
        tk = AutoTokenizer.from_pretrained("Xenova/ms-marco-MiniLM-L-6-v2")
        sess = ort.InferenceSession(mdl, providers=["CPUExecutionProvider"])
        want = {i.name for i in sess.get_inputs()}
        enc = tk([query] * len(docs), docs, padding=True, truncation=True, return_tensors="np")
        feed = {k: v for k, v in enc.items() if k in want}
        logits = sess.run(None, feed)[0]
        ref = {i: float(logits[i][0]) for i in range(len(docs))}
        out.append("ONNX ref (Xenova/ms-marco-MiniLM-L-6-v2, CPU EP):")
        out.append("  " + " ".join(f"[{i}]{s:+8.3f}" for i, s in sorted(ref.items())))
    except Exception as e:
        out.append(f"ONNX reference UNAVAILABLE: {type(e).__name__}: {e}")

    for fn in ("ms-marco-MiniLM-L-6-v2-g7c.gguf",
               "ms-marco-MiniLM-L-6-v2-q4_k-imatrix-g7c-f7.gguf",
               "ms-marco-MiniLM-L-6-v2-iq4_xs-g7c-f7.gguf"):
        try:
            p = hf_hub_download("cstr/ms-marco-MiniLM-L-6-v2-GGUF", fn,
                                token=api.token, local_dir=str(WORK / "spot"))
            r = subprocess.run([str(cli), "-m", p, "--json", "--rerank", query, *docs],
                               capture_output=True, text=True)
            if r.returncode != 0:
                out.append(f"{fn}: rerank FAILED rc={r.returncode}\n{r.stderr[-800:]}")
                continue
            sc = {x["index"]: x["score"] for x in json.loads(r.stdout)["results"]}
            line = " ".join(f"[{i}]{s:+8.3f}" for i, s in sorted(sc.items()))
            dmax = (max(abs(sc[i] - ref[i]) for i in ref) if ref else float("nan"))
            rng = max(sc.values()) - min(sc.values())
            out.append(f"{fn}:\n  {line}\n  max|delta_vs_onnx|={dmax:.4f}  score_range={rng:.3f}"
                       f"  ({'HF-scale OK' if rng > 5 else 'SCALE COLLAPSE'})")
            Path(p).unlink(missing_ok=True)
        except Exception as e:
            out.append(f"{fn}: spot-check error {type(e).__name__}: {e}")

    text = "\n".join(out) + "\n"
    print("\n" + text, flush=True)
    sp = WORK / "ms-marco-MiniLM-L-6-v2-g7c-f7-spotcheck.txt"
    sp.write_text(text)
    api.upload_file(path_or_fileobj=str(sp), path_in_repo=sp.name,
                    repo_id="cstr/ms-marco-MiniLM-L-6-v2-GGUF", repo_type="model",
                    commit_message="F7b rerank spot-check: uploaded q4_k+imatrix-g7c-f7 vs ONNX ref")


try:
    _spot_check()
except Exception as _e:
    import traceback
    print(f"[rerank-f7] spot-check failed (non-fatal): {_e}\n{traceback.format_exc()[-1500:]}",
          flush=True)
