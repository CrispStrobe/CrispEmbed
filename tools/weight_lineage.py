"""Weight-lineage probe: cosine between same-named tensors of two checkpoints, read with
HTTP range requests from their safetensors (a few MB per tensor, no full download).

Fine-tuning moves weights, but a tensor stays far closer to its own initialization than
to an independently trained checkpoint; deep layers separate siblings that share early
pretraining. Used for license lineage (tests/results/licenses/qwen-vision-lineage.json).
Edit TELE/CANDS/SUFFIXES below for other pairs."""
import json, struct, sys, urllib.request
import numpy as np

def hf(repo, path, rev="main"):
    return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"

def get(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

def index(repo, rev="main"):
    try:
        idx = json.loads(get(hf(repo, "model.safetensors.index.json", rev)))
        return idx["weight_map"]
    except Exception:
        return None

def tensor(repo, name, rev="main"):
    wm = index(repo, rev)
    fname = wm[name] if wm else "model.safetensors"
    url = hf(repo, fname, rev)
    n = struct.unpack("<Q", get(url, 0, 7))[0]
    header = json.loads(get(url, 8, 8 + n - 1))
    meta = header[name]
    a, b = meta["data_offsets"]
    raw = get(url, 8 + n + a, 8 + n + b - 1)
    if meta["dtype"] == "BF16":
        arr = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif meta["dtype"] == "F16":
        arr = np.frombuffer(raw, np.float16).astype(np.float32)
    else:
        arr = np.frombuffer(raw, np.float32)
    return arr.reshape(meta["shape"])

def find(repo, suffix, rev="main"):
    wm = index(repo, rev)
    if wm:
        names = list(wm)
    else:
        url = hf(repo, "model.safetensors", rev)
        n = struct.unpack("<Q", get(url, 0, 7))[0]
        names = list(json.loads(get(url, 8, 8 + n - 1)))
    hits = [x for x in names if x.endswith(suffix)]
    return hits[0]

TELE = ("XingChen-AGI/TeleOCR", "b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d")
CANDS = [("Qwen/Qwen2.5-VL-3B-Instruct", "main"), ("Qwen/Qwen2.5-VL-7B-Instruct", "main"),
         ("Qwen/Qwen2.5-VL-32B-Instruct", "main")]
SUFFIXES = ["visual.patch_embed.proj.weight", "visual.blocks.0.attn.qkv.weight", "visual.blocks.15.mlp.down_proj.weight",
            "visual.blocks.31.attn.proj.weight"]
for suf in SUFFIXES:
    t = tensor(TELE[0], find(TELE[0], suf, TELE[1]), TELE[1]).ravel()
    row = [suf]
    for repo, rev in CANDS:
        try:
            c = tensor(repo, find(repo, suf, rev), rev).ravel()
            cos = float(np.dot(t, c) / (np.linalg.norm(t) * np.linalg.norm(c))) if c.shape == t.shape else float("nan")
            rel = float(np.linalg.norm(t - c) / np.linalg.norm(c)) if c.shape == t.shape else float("nan")
            row.append(f"{repo.split('/')[1]}: cos={cos:.6f} rel_diff={rel:.4f}")
        except Exception as e:
            row.append(f"{repo.split('/')[1]}: error {type(e).__name__}: {e}")
    print(" | ".join(row), flush=True)
