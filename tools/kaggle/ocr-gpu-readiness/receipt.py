#!/usr/bin/env python3
"""Build tests/results/ocr-gpu-readiness/readiness.json from the kernels' summary.json files.

    python3 tools/kaggle/ocr-gpu-readiness/receipt.py OUT.json cpu=<summary.json> gpu=<summary.json> [...]
Each input is labelled with the host it ran on (`cpu` = Kaggle CPU kernel, `gpu` = Kaggle T4 kernel, whose runs
carry their own arm `cuda` / `cpu`). Several files per label are merged (e.g. an engine re-run).
Per run: stage seconds parsed from the engines' bench/timing lines, decode tok/s, token counts, output
hashes and confidence statistics; then pairwise output comparisons. Large arrays are not copied.
"""
import hashlib, json, re, sys

CASES = ("inventory-table", "two-column-page", "historical-receipt")


def stages(engine, lines):
    j = "\n".join(lines)
    f = lambda pat: [float(x) for x in re.findall(pat, j)]
    st = {}
    if engine.startswith("lighton"):
        v, pf = f(r"vision_encoder: (\d+)ms"), f(r"\] prefill: (\d+)ms")
        d = re.search(r"decode_total: (\d+)ms \((\d+) steps\)", j)
        if v: st["vision_s"] = (v[0] + sum(f(r"projection: (\d+)ms"))) / 1e3
        if pf: st["prefill_s"] = pf[0] / 1e3
        if d: st["decode_s"], st["decode_steps"] = int(d.group(1)) / 1e3, int(d.group(2))
    elif engine.startswith("deepseek"):
        m = re.search(r"sam=([\d.]+) ms qwen2_enc=([\d.]+) ms projector=([\d.]+) ms prompt=([\d.]+) ms "
                      r"prefill=([\d.]+) ms decode=([\d.]+) ms .*gen_tokens=(\d+)", j)
        if m:
            g = [float(x) for x in m.groups()]
            st.update(vision_s=(g[0] + g[1] + g[2]) / 1e3, prompt_s=g[3] / 1e3, prefill_s=g[4] / 1e3,
                      decode_s=g[5] / 1e3, decode_steps=int(g[6]))
    elif engine.startswith("uocr2"):
        v = sum(f(r"sam_encoder: (\d+)ms")) + sum(f(r"clip_encoder: (\d+)ms")) + sum(f(r"projection\+assembly: (\d+)ms"))
        pf = f(r"prefill done S=\d+ ([\d.]+) ms")
        d = re.search(r"decode done: (\d+) tokens, ([\d.]+) ms total", j)
        st["vision_s"] = v / 1e3
        if pf: st["prefill_s"] = pf[0] / 1e3
        if d and pf: st["decode_s"], st["decode_steps"] = (float(d.group(2)) - pf[0]) / 1e3, int(d.group(1))
    elif engine.startswith("minicpm"):
        st["vision_s"] = sum(f(r"minicpm_ocr\.vision part=\d+ .*? total=([\d.]+)ms")) / 1e3
        pf = f(r"minicpm_ocr\.prefill ([\d.]+)ms")
        d = re.search(r"minicpm_ocr\.decode steps=(\d+) total=([\d.]+)ms", j)
        if pf: st["prefill_s"] = pf[0] / 1e3
        if d: st["decode_s"], st["decode_steps"] = float(d.group(2)) / 1e3, int(d.group(1))
    if st.get("decode_s"):
        st["decode_tok_s"] = round(st["decode_steps"] / st["decode_s"], 2)
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in st.items()}


def main(out, specs):
    runs, meta = [], {}
    for spec in specs:
        host, path = spec.split("=", 1)
        d = json.load(open(path))
        meta.setdefault(host, []).append({k: d.get(k) for k in ("sha", "cpu", "gpu", "build_s", "elapsed_s")})
        meta.setdefault("models", {}).update({k: v for k, v in d["models"].items() if "error" not in v})
        meta.setdefault("sched_probe_t4", {}).update(d.get("sched") or {})
        for r in d["runs"]:
            if r.get("skipped"):
                continue
            conf = r.get("confidences") or []
            row = {"host": "kaggle-cpu" if host == "cpu" else "kaggle-t4", "arm": r["arm"], "engine": r["engine"],
                   "case": r["case"], "rc": r["rc"], "wall_s": r["wall_s"], "init_s": r.get("init_s"),
                   "recognize_s": r.get("recognize_s"), "max_rss_mib": r.get("max_rss_mib"),
                   "gpu_util_mean": r.get("gpu_util_mean"), "gpu_mem_peak_mib": r.get("gpu_mem_peak_mib"),
                   "text_chars": len(r.get("text", "")), "text_sha256": r.get("text_sha256"),
                   "ids_n": len(r["ids"]) if r.get("ids") is not None else None, "ids_sha256": r.get("ids_sha256"),
                   "conf_n": len(conf), "conf_mean": round(sum(conf) / len(conf), 6) if conf else None,
                   **stages(r["engine"], r.get("stage_lines", []))}
            row["_ids"], row["_conf"], row["_text"] = r.get("ids"), conf, r.get("text")
            runs.append(row)
    key = {(r["host"], r["arm"], r["engine"], r["case"]): r for r in runs}

    def compare(a, b):
        c = {"text_exact": a["_text"] == b["_text"], "chars": [a["text_chars"], b["text_chars"]]}
        if a["_ids"] is not None and b["_ids"] is not None:
            x, y = a["_ids"], b["_ids"]
            c["ids_exact"] = x == y
            c["first_id_divergence"] = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]),
                                            None if len(x) == len(y) else min(len(x), len(y)))
        ca, cb = a["_conf"], b["_conf"]
        n = min(len(ca), len(cb))
        if n:
            c["conf_n"] = [len(ca), len(cb)]
            first = next((i for i in range(n) if abs(ca[i] - cb[i]) > 1e-3), None)
            c["conf_first_step_over_1e-3"] = first
            pre = first if first is not None else n
            c["conf_max_abs_before_divergence"] = max((abs(ca[i] - cb[i]) for i in range(pre)), default=0.0)
        return c

    comparisons = []
    engines = sorted({r["engine"] for r in runs})
    for e in engines:
        for cid in CASES:
            ref = key.get(("kaggle-cpu", "cpu", e, cid))
            for other, label in ((("kaggle-t4", "cuda", e, cid), "t4-cuda vs kaggle-cpu"),
                                 (("kaggle-t4", "cpu", e, cid), "t4-host-cpu vs kaggle-cpu")):
                if ref and other in key:
                    comparisons.append({"engine": e, "case": cid, "pair": label, **compare(key[other], ref)})
            a, b = key.get(("kaggle-t4", "cuda", e, cid)), key.get(("kaggle-t4", "cpu", e, cid))
            if a and b:
                comparisons.append({"engine": e, "case": cid, "pair": "t4-cuda vs t4-host-cpu (same binary)", **compare(a, b)})
    for r in runs:
        for k in ("_ids", "_conf", "_text"):
            r.pop(k)
    json.dump({"schema": "crispembed-ocr-gpu-readiness-v1", "meta": meta, "runs": runs, "comparisons": comparisons},
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
