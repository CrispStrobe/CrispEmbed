#!/usr/bin/env python3
"""Render the CUDA audit receipts (tools/kaggle/cuda-audit) as Markdown tables.

    python tests/results/gpu-audit/cuda-kaggle/summarize.py RUN_DIR [RUN_DIR ...] > SUMMARY.md

Later run directories override earlier ones for the same job id. The bar of an
artifact is taken from its recipe label in the file name (q8_0, q4_k, iq4_xs, f16,
...), falling back to the dominant GGUF weight type; a "q4_k" recipe whose 384-wide
tensors fell back to q8_0 is still judged as Q4_K. "both arms" in the verdict column
means the CPU arm misses the bar against the same reference too, i.e. the artifact,
not the backend, is below the bar.
"""
import json
import re
import sys
from pathlib import Path

BARS = {"f32": 0.99, "f16": 0.98, "bf16": 0.98, "q8_0": 0.97, "q6_k": 0.97, "q5_k": 0.94, "q4_k": 0.94,
        "q4_0": 0.94, "iq4_xs": 0.94}


def label(name, gguf_prec):
    m = re.search(r"(iq4_xs|q4_k|q4_0|q5_k|q6_k|q8_0|f16|bf16|f32)", name.lower())
    return m.group(1) if m else (gguf_prec or "f32")


def f(x, n=5):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def best(w):
    return min(w) if isinstance(w, list) and w else w


def load(dirs):
    jobs = {}
    for d in dirs:
        for p in sorted(Path(d, "receipts").glob("*.json")):
            jobs[p.stem] = json.loads(p.read_text())
    return jobs


def embed_rows(jobs):
    out = ["| Job | Artifact | Bar | CUDA vs CPU ref | CPU vs CPU ref | CUDA vs CPU same file | CUDA s | CPU s | GPU used | Verdict |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for jid, r in jobs.items():
        if r.get("kind") != "embed":
            continue
        if r.get("error"):
            out.append(f"| {jid} | — | | | | | | | | error: {r['error']} |")
            continue
        for name, e in r["files"].items():
            lab = label(name, e.get("precision"))
            bar = BARS.get(lab)
            c = (e.get("cuda_vs_cpu_reference") or {}).get("cos_min")
            p = (e.get("cpu_vs_cpu_reference") or {}).get("cos_min")
            s = (e.get("cuda_vs_cpu_same_artifact") or {}).get("cos_min")
            arms = e.get("arms", {})
            errs = [a.get("parse_error") or (f"rc={a.get('rc')}" if a.get("rc") else None) for a in arms.values()]
            errs = [x for x in errs if x]
            if c is None:
                verdict = "no result" + (f" ({errs[0][:60]})" if errs else "")
            elif bar is None:
                verdict = "no bar"
            elif c > bar:
                verdict = "**qualifies**"
            else:
                verdict = "below bar, both arms" if p is not None and p <= bar else "**below bar (CUDA only)**"
            out.append(f"| {jid.removeprefix('embed-')} | {name} ({lab}) | {f(bar, 2)} | {f(c)} | {f(p)} | {f(s)} | "
                       f"{f(best((arms.get('cuda') or {}).get('wall_s')), 2)} | "
                       f"{f(best((arms.get('cpu') or {}).get('wall_s')), 2)} | "
                       f"{(arms.get('cuda') or {}).get('cuda_used')} | {verdict} |")
    return out


def cmd_rows(jobs):
    out = ["| Job | Prec. | Arm | rc | best s | GPU used | vs CPU arm | vs higher-precision CPU |",
           "|---|---|---|---:|---:|---|---|---|"]
    for jid, r in jobs.items():
        if r.get("kind") != "cmd":
            continue
        if r.get("error"):
            out.append(f"| {jid} | | | | | | error: {r['error'][:120]} | |")
            continue
        for name, a in r["arms"].items():
            v = a.get("vs_cpu") or {}
            desc = []
            if name == "cpu":
                desc.append("reference")
            elif v:
                if v.get("byte_identical"):
                    desc.append("byte-identical")
                if v.get("ocr"):
                    o = v["ocr"]
                    desc.append(f"boxes {o['n_boxes'][0]}/{o['n_boxes'][1]}, texts equal {o.get('texts_equal')},"
                                f" max Δbox {o.get('box_max_dpx')} px, text ratio {o['joined_text_ratio']:.4f}")
                elif v.get("image"):
                    im = v["image"]
                    if "shape_differs" in im:
                        desc.append(f"shape differs {im['shape_differs']}")
                    else:
                        desc.append(f"cos {f(v.get('final_cosine'), 6)}, max|Δ| {im['max_abs']:.0f}, "
                                    f"PSNR {im['psnr_db']:.1f} dB" +
                                    (f", IoU {im['mask_iou']:.4f}" if "mask_iou" in im else ""))
                elif v.get("json"):
                    desc.append(f"JSON cos {f(v.get('final_cosine'), 6)}, strings equal {v['json']['strings_equal']}")
                elif not v.get("byte_identical"):
                    desc.append(f"text exact {v.get('text_exact')}, char ratio {f(v.get('char_ratio'), 4)}")
            ref = a.get("vs_reference_cpu") or {}
            rdesc = ""
            if ref:
                rdesc = (f"cos {f(ref.get('final_cosine'), 6)}" if ref.get("final_cosine") is not None else
                         f"text exact {ref.get('text_exact')}, ratio {f(ref.get('char_ratio'), 4)}")
            env = ",".join(f"{k}={v}" for k, v in (a.get("env") or {}).items())
            out.append(f"| {jid} | {r.get('precision') or '—'} | {name}{' (' + env + ')' if env else ''} | "
                       f"{a.get('rc')} | {f(best(a.get('wall_s')), 2)} | {a.get('cuda_used')} | {'; '.join(desc)} | "
                       f"{rdesc} |")
    return out


def diff_rows(jobs):
    out = ["| Job | Arm | rc | stages | worst stage | threshold fails | s | GPU used |",
           "|---|---|---:|---:|---|---|---:|---|"]
    for jid, r in jobs.items():
        if r.get("kind") != "diff":
            continue
        if r.get("error"):
            out.append(f"| {jid} | | | | error: {r['error']} | | | |")
            continue
        for name, a in r["arms"].items():
            w = a.get("worst_stage")
            fails = a.get("threshold_fails") or []
            out.append(f"| {jid} | {name} | {a.get('rc')} | {a.get('n_stages')} | "
                       f"{(w[0] + ' ' + f(w[1], 6)) if w else '—'} | "
                       f"{'; '.join(f'{s} {c:.4f}<{t}' for s, c, t in fails) or 'none'} | "
                       f"{f(a.get('wall_s'), 2)} | {a.get('cuda_used')} |")
    return out


def main(argv):
    jobs = load(argv[1:])
    print("# CUDA audit receipts (generated by summarize.py)\n")
    print("## Embedding / rerank / sparse / ColBERT\n")
    print("\n".join(embed_rows(jobs)))
    print("\n## CLI engines (OCR, VLM, SR, NER, segmentation)\n")
    print("\n".join(cmd_rows(jobs)))
    print("\n## Stage diffs against the committed Python references\n")
    print("\n".join(diff_rows(jobs)))


if __name__ == "__main__":
    main(sys.argv)
