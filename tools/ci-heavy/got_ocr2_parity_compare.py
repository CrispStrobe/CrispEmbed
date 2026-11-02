#!/usr/bin/env python3
"""Compare native GOT-OCR2 arms against the original-pipeline capture.

Reads <out>/orig.json (got_ocr2_orig_probe.py) and every native arm directory
<out>/native/<model>.<arm>.<case>/ holding pixels.f32, prompt_ids.txt,
gen_ids.txt and text.txt (from CRISPEMBED_GOT_OCR_DUMP_DIR + the CLI output).
Writes <out>/parity.json and prints the per-factor table.

    python got_ocr2_parity_compare.py <out> [hf_port_compare.json]
"""

import difflib
import json
import os
import sys

import numpy as np

STOPS = (151643, 151645)


def ids_of(path):
    if not os.path.exists(path):
        return None
    t = open(path).read().split()
    return [int(x) for x in t]


def strip_stop(ids):
    return [i for i in ids if i not in STOPS] if ids is not None else None


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    out = sys.argv[1]
    orig = json.load(open(os.path.join(out, "orig.json")))
    hf_ref = {}
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        for c in json.load(open(sys.argv[2]))["cases"]:
            hf_ref[c["id"]] = c["reference_text"].strip()
    o_prompt = orig["prompt_ids"]
    rows = []
    nd = os.path.join(out, "native")
    for name in sorted(os.listdir(nd)):
        model, arm, cid = name.split(".", 2)
        d = os.path.join(nd, name)
        oc = orig["cases"].get(cid)
        if oc is None:
            continue
        ref = oc["runs"]["20"]
        ref_ids = strip_stop(ref["token_ids"])
        gen = strip_stop(ids_of(os.path.join(d, "gen_ids.txt")))
        prm = ids_of(os.path.join(d, "prompt_ids.txt"))
        txt = open(os.path.join(d, "text.txt"), errors="replace").read().strip() if os.path.exists(
            os.path.join(d, "text.txt")) else ""
        row = {"model": model, "arm": arm, "case": cid, "prompt_exact": prm == o_prompt,
               "ids_exact": gen == ref_ids, "n_ids": len(gen or []), "ref_n_ids": len(ref_ids),
               "first_divergence": first_div(gen or [], ref_ids),
               "text_exact": txt == ref["text"].strip(),
               "char_ratio": round(difflib.SequenceMatcher(None, txt, ref["text"].strip()).ratio(), 4),
               "text": txt}
        if prm is not None and prm != o_prompt:
            row["prompt_diff"] = {"native_no_imgpad": [i for i in prm if i != 151859],
                                  "orig_no_imgpad": orig["prompt_ids_no_imgpad"]}
        if cid in hf_ref:
            row["char_ratio_vs_hf_port"] = round(difflib.SequenceMatcher(None, txt, hf_ref[cid]).ratio(), 4)
        pp = os.path.join(d, "pixels.f32")
        op = os.path.join(out, f"orig.{cid}.pixels.npy")
        if os.path.exists(pp) and os.path.exists(op):
            a = np.fromfile(pp, dtype=np.float32)
            b = np.load(op).astype(np.float32).ravel()
            if a.size == b.size:
                dd = np.abs(a - b)
                row["pixels"] = {"max_abs": float(dd.max()), "mean_abs": float(dd.mean()),
                                 "n_diff_gt_1e-5": int((dd > 1e-5).sum()), "n": int(a.size)}
        if os.path.exists(os.path.join(d, "secs.txt")):
            row["secs"] = float(open(os.path.join(d, "secs.txt")).read().strip())
        rows.append(row)
    # original-side controls: ngram off vs ngram 20
    controls = {}
    for cid, oc in orig["cases"].items():
        r20 = oc["runs"].get("20")
        for ng, r in oc["runs"].items():
            if ng == "20":
                continue
            controls[f"{cid}.ngram{ng}"] = {
                "ids_exact_vs_ngram20": strip_stop(r["token_ids"]) == strip_stop(r20["token_ids"]),
                "char_ratio_vs_ngram20": round(difflib.SequenceMatcher(None, r["text"], r20["text"]).ratio(), 4)}
        op = os.path.join(out, f"orig.{cid}.pixels.npy")
        for k in ("default", "slow"):
            hp = os.path.join(out, f"hf.{k}.{cid}.pixels.npy")
            if os.path.exists(hp) and os.path.exists(op):
                a, b = np.load(hp).astype(np.float32), np.load(op).astype(np.float32)
                if a.shape == b.shape:
                    dd = np.abs(a - b)
                    controls[f"{cid}.hf_port_{k}_pixels_vs_orig"] = {"max_abs": float(dd.max()),
                                                                    "mean_abs": float(dd.mean())}
        if cid in hf_ref:
            controls[f"{cid}.orig_vs_hf_port_text"] = round(
                difflib.SequenceMatcher(None, r20["text"], hf_ref[cid]).ratio(), 4)
    rec = {"rows": rows, "original_controls": controls,
           "orig_reference": {cid: {"text": oc["runs"]["20"]["text"], "n_ids": len(strip_stop(oc["runs"]["20"]["token_ids"])),
                                    "seconds": oc["runs"]["20"]["seconds"], "hit_cap": oc["runs"]["20"]["hit_cap"]}
                              for cid, oc in orig["cases"].items()}}
    json.dump(rec, open(os.path.join(out, "parity.json"), "w"), indent=1)
    print(f"{'model':6s} {'arm':8s} {'case':16s} prompt pix_max   ids_exact first_div n/ref   ratio  vsHF")
    for r in rows:
        px = r.get("pixels", {}).get("max_abs")
        print(f"{r['model']:6s} {r['arm']:8s} {r['case']:16s} {str(r['prompt_exact']):6s} "
              f"{(f'{px:.4f}' if px is not None else '-'):8s} {str(r['ids_exact']):9s} {str(r['first_divergence']):9s} "
              f"{r['n_ids']}/{r['ref_n_ids']:<5d} {r['char_ratio']:.4f} {r.get('char_ratio_vs_hf_port', '-')}")
    print("original controls:", json.dumps(controls))


if __name__ == "__main__":
    main()
