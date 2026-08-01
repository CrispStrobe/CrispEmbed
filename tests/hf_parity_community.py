"""HF (Python) ground-truth parity for the community-GGUF matrix.

The matrix's own checks (rc, shape, garbage guard, cross-conversion) can ALL pass
on a subtly wrong model:
  - rc/shape say nothing about the numbers;
  - the garbage guard only asserts related > unrelated, which survives real
    degradation (the nomic-v1.5 rope.freq_base bug kept cosine 0.99 on short text);
  - cross-conversion compares us against US — two conversions can agree and both
    be wrong.

Only the original Python model is ground truth. This runs the HF/sentence-
transformers model and compares its embedding to ours per manifest entry.

Threshold: min_hf_cos (default 0.95). These GGUFs are q4_k vs HF fp32, so some
loss is the quant floor, NOT a bug (see CLAUDE.md's parity table: q4_k sits
~0.93-0.98 depending on model). A STRUCTURAL bug — wrong pooling, wrong rope
theta, fabricated dims — craters cosine far below the quant floor, which is what
this is sized to catch.

Env:
  CRISPEMBED_BIN, CRISPEMBED_MODELS_DIR  (as run_community_gguf.py)
  HF_HOME                                 point at a writable cache; ~/.cache/huggingface
                                          is a symlink to an often-unmounted volume here

Usage:
  HF_HOME=/tmp/hf python tests/hf_parity_community.py --name all-MiniLM-L6-v2
  HF_HOME=/tmp/hf python tests/hf_parity_community.py --all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_community_gguf as drv  # noqa: E402

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def hf_embed(entry: dict, texts: list[str]):
    """Embed with the ORIGINAL Python model via sentence-transformers.

    sentence-transformers applies the model's own pooling + normalization from
    its config, so this is the real reference rather than a hand-rolled forward
    (which is where a wrong-pooling assumption would silently creep back in).
    """
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(
        entry["hf_repo"],
        trust_remote_code=bool(entry.get("hf_trust_remote_code")),
        device="cpu",
    )
    return m.encode(texts, convert_to_numpy=True, normalize_embeddings=False)


def run(entry: dict) -> bool:
    binary, model = drv._bin(), drv.resolve_model(entry)
    if not binary.is_file() or model is None:
        print(f"  [SKIP] {entry['name']}: binary or GGUF missing")
        return True

    qp, dp = entry.get("query_prefix", ""), entry.get("doc_prefix", "")
    texts = [qp + entry["query"], dp + entry["related"], dp + entry["unrelated"]]

    ours = []
    for t in texts:
        v, _, rc = drv.embed(binary, model, t)
        if rc != 0 or not v:
            print(f"  [FAIL] {entry['name']}: crispembed rc={rc}")
            return False
        ours.append(v)

    ref = hf_embed(entry, texts)
    need = entry.get("min_hf_cos", 0.95)
    cs = [drv.cosine(list(map(float, ref[i])), ours[i]) for i in range(len(texts))]
    worst = min(cs)
    ok = worst >= need
    print(f"  [{'PASS' if ok else 'FAIL'}] {entry['name']:24s} "
          f"cos vs HF: min={worst:.4f} mean={sum(cs)/len(cs):.4f} (need >={need})  "
          f"per-text={[f'{c:.4f}' for c in cs]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="HF ground-truth parity for the community matrix")
    ap.add_argument("--name")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    models = drv.load_manifest()["models"]
    todo = [e for e in models if e["name"] == a.name] if a.name else (models if a.all else [])
    if not todo:
        ap.print_help()
        return 2

    print("HF (Python) ground-truth parity")
    bad = 0
    for e in todo:
        if not e.get("hf_repo"):
            print(f"  [SKIP] {e['name']}: no hf_repo")
            continue
        try:
            if not run(e):
                bad += 1
        except Exception as ex:  # a missing HF dep/model must not look like a pass
            print(f"  [ERROR] {e['name']}: {type(ex).__name__}: {ex}")
            bad += 1
    print("FAILED" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
