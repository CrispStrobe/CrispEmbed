#!/usr/bin/env python3
"""Qwen3-VL-2B: C++ (crispembed) vs transformers, stage by stage.

qwen3vl_ref_ocr.py showed upstream reads the 1000x420 ALPHA/OMEGA fixture
exactly while the C++ engine scrambles it (fox.png is exact in both). This
finds the first diverging stage.

Reference = the real transformers model with forward hooks (not a
re-implementation), written as a GGUF with the stage names the C++ diff code
compares when CRISPEMBED_QWEN2VL_REF is set:
  vis_patch_embed (N, Dv)   input of vision block 0 (patch embed + pos embed)
  vis_layer_i     (N, Dv)   vision block outputs
  mrope_positions (T, 3)    get_rope_index
  llm_embed       (T, D)    input of decoder layer 0 (image features scattered)
  llm_layer_i     (T, D)    decoder layer outputs (before deepstack add)
  llm_post_ds_i   (T, D)    after the deepstack add = input of layer i+1
  llm_final_norm  (T, D)
Plus input pixels: C++ CRISPEMBED_DUMP_PATCHES vs the processor's pixel_values.

The C++ side runs the F16 GGUF on CPU. fox.png is the control arm: every stage
is expected to pass there. Exit 0 only if both images pass every stage.

    gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/qwen3vl_stage_diff.py \\
        -f pip="torch torchvision transformers>=4.57 accelerate pillow gguf"
"""
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("HEAVY_OUT", "out"))
SCR = Path(os.environ.get("HEAVY_SCRATCH", "scratch"))
OUT.mkdir(parents=True, exist_ok=True)
SCR.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
GGUF_REPO, GGUF_FILE = "cstr/qwen3-vl-2b-crispembed-gguf", "qwen3-vl-2b-f16.gguf"
PROMPT = "Read all the text in this image. Output the exact text content only."  # qwen2vl_ocr.cpp default
IMAGES = {
    "fox": REPO / "tests/regression/images/fox.png",
    "alpha_omega_1000x420": HERE / "fixtures/alpha_omega_1000x420.png",
}
# transformers greedy output on the same PNGs (tools/ci-heavy/qwen3vl_ref_ocr.py,
# run 36533519606). The C++ text must equal these exactly.
EXPECTED_TEXT = {
    "fox": "The quick brown fox jumps\nover the lazy dog. 12345",
    "alpha_omega_1000x420": "ALPHA\nOMEGA",
}
# Stage-by-stage gate only where the INPUT is identical: fox's patches match
# transformers to 2/255, the ALPHA/OMEGA resize differs by up to ~15/255 at
# sharp text edges (interpolation kernel), which alone moves later stages.
STRICT_STAGES = {"fox"}
res = {"images": {}, "errors": []}


def save():
    (OUT / "result.json").write_text(json.dumps(res, indent=1))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_cpp():
    b = SCR / "build"
    subprocess.check_call(["cmake", "-S", str(REPO), "-B", str(b), "-DCMAKE_BUILD_TYPE=Release"],
                          stdout=subprocess.DEVNULL)
    subprocess.check_call(["cmake", "--build", str(b), "--target", "crispembed-cli", f"-j{os.cpu_count() or 4}"],
                          stdout=subprocess.DEVNULL)
    exe = next(p for p in b.rglob("crispembed") if p.is_file() and os.access(p, os.X_OK))
    return exe


def dump_reference(model, proc, img_path, out_gguf):
    import gguf
    import torch
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt")
    cap = {}
    hooks = []
    vis = model.model.visual
    txt = model.model.language_model

    def arr(t):
        t = t[0] if isinstance(t, (tuple, list)) else t
        return t.detach().float().reshape(-1, t.shape[-1]).numpy().copy()

    def first_arg(args, kwargs):
        return args[0] if args else kwargs["hidden_states"]

    hooks.append(vis.patch_embed.register_forward_hook(
        lambda m, a, o: cap.__setitem__("vis_patch_embed_nopos", arr(o))))
    hooks.append(vis.blocks[0].register_forward_pre_hook(
        lambda m, a, k: cap.__setitem__("vis_patch_embed", arr(first_arg(a, k))), with_kwargs=True))
    for i, blk in enumerate(vis.blocks):
        hooks.append(blk.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(f"vis_layer_{i}", arr(o))))
    hooks.append(txt.layers[0].register_forward_pre_hook(
        lambda m, a, k: cap.__setitem__("llm_embed", arr(first_arg(a, k))), with_kwargs=True))
    n_ds = len(getattr(vis, "deepstack_visual_indexes", [5, 11, 17]))
    for i, lyr in enumerate(txt.layers):
        hooks.append(lyr.register_forward_hook(lambda m, a, o, i=i: cap.__setitem__(f"llm_layer_{i}", arr(o))))
        if 1 <= i <= n_ds:  # input of layer i = output of layer i-1 after its deepstack add
            hooks.append(lyr.register_forward_pre_hook(
                lambda m, a, k, i=i: cap.__setitem__(f"llm_post_ds_{i - 1}", arr(first_arg(a, k))), with_kwargs=True))
    hooks.append(txt.norm.register_forward_hook(lambda m, a, o: cap.__setitem__("llm_final_norm", arr(o))))
    # The mRoPE ids the text model actually receives (get_rope_index's
    # signature differs between transformers versions; this does not).
    pos_cap = {}
    hooks.append(txt.register_forward_pre_hook(
        lambda m, a, k: pos_cap.__setitem__("p", k.get("position_ids")), with_kwargs=True))
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    pos = pos_cap.get("p")
    if pos is not None:
        pos = pos[-3:, 0, :]  # (3, T); newer versions prepend a text-position row
        cap["mrope_positions"] = pos.T.float().numpy().copy()  # (T, 3)
    pixels = inputs["pixel_values"].float().numpy()

    w = gguf.GGUFWriter(str(out_gguf), "qwen3vl_ref")
    w.add_string("general.name", "qwen3vl_reference_transformers")
    for name, a in cap.items():
        w.add_tensor(name, np.ascontiguousarray(a, dtype=np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    grid = inputs["image_grid_thw"][0].tolist()
    return {"n_tokens": int(inputs["input_ids"].shape[1]), "grid_thw": grid,
            "input_ids": inputs["input_ids"][0].tolist(),
            "stages": {k: list(v.shape) for k, v in cap.items()}}, pixels, cap


def run_cpp(exe, gguf_path, img, ref, patches_out, dump_dir):
    dump_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CRISPEMBED_QWEN2VL_REF=str(ref), CRISPEMBED_DUMP_PATCHES=str(patches_out),
               CRISPEMBED_DIFF_DUMP_DIR=str(dump_dir))
    r = subprocess.run([str(exe), "-m", str(gguf_path), "--ocr", str(img), "--ocr-max-tokens", "4",
                        "-t", str(os.cpu_count() or 4)], capture_output=True, text=True, env=env, timeout=3600)
    stages = {}
    for m in re.finditer(r"diff (\S+): cos_min=([-\d.]+) max_abs=(\S+) (PASS|FAIL)", r.stderr):
        stages[m.group(1)] = {"cos_min": float(m.group(2)), "max_abs": float(m.group(3)), "pass": m.group(4) == "PASS"}
    mm = re.search(r"mRoPE pos mismatches: (\d+)/(\d+)", r.stderr)
    mism = [l.strip() for l in r.stderr.splitlines() if "MISMATCH tok" in l][:10]
    (OUT / f"cpp_stderr_{Path(img).stem}.txt").write_text(r.stderr)
    return {"rc": r.returncode, "stages": stages,
            "mrope_mismatches": [int(mm.group(1)), int(mm.group(2))] if mm else None, "mrope_examples": mism,
            "stderr_tail": r.stderr[-3000:]}


def block_to_raster(h, w, m=2):
    """perm[i] = raster index of the patch at merge-block position i."""
    perm = []
    for mh in range(h // m):
        for mw in range(w // m):
            for ir in range(m):
                for ic in range(m):
                    perm.append((mh * m + ir) * w + mw * m + ic)
    return np.array(perm)


def row_cos(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(na * nb, 1e-30)


def analyse_rows(dump_dir, cap, grid):
    """Per stage: row counts, bad rows, and whether a raster<->block row
    permutation (vision) or an offset (LLM) explains the mismatch."""
    out = {}
    h, w = grid[1], grid[2]
    for name, ref in cap.items():
        f = dump_dir / f"{name}.f32"
        if not f.exists() or ref.ndim != 2:
            continue
        cpp = np.fromfile(f, dtype=np.float32)
        D = ref.shape[1]
        if cpp.size % D:
            out[name] = {"cpp_elems": int(cpp.size), "ref_shape": list(ref.shape), "note": "not a multiple of D"}
            continue
        cpp = cpp.reshape(-1, D)
        n = min(len(cpp), len(ref))
        cs = row_cos(cpp[:n], ref[:n])
        bad = np.where(cs < 0.999)[0]
        e = {"rows_cpp": len(cpp), "rows_ref": len(ref), "n_bad": int(len(bad)),
             "first_bad": int(bad[0]) if len(bad) else None, "cos_min": float(cs.min()),
             "cos_median": float(np.median(cs))}
        if name.startswith("vis_") and len(cpp) == len(ref) == h * w:
            p = block_to_raster(h, w)
            inv = np.empty_like(p); inv[p] = np.arange(len(p))
            # known-answer checked: C++ rows in raster order -> cpp[p] matches
            e["cos_min_if_cpp_is_raster"] = float(row_cos(cpp[p], ref).min())
            e["cos_min_if_ref_is_raster"] = float(row_cos(cpp[inv], ref).min())
        out[name] = e
    return out


def cpp_ocr_text(exe, gguf_path, img):
    r = subprocess.run([str(exe), "-m", str(gguf_path), "--ocr", str(img), "--ocr-max-tokens", "48",
                        "-t", str(os.cpu_count() or 4)], capture_output=True, text=True, timeout=3600)
    return r.stdout.strip()


def compare_patches(cpp_file, ref_pixels):
    raw = np.fromfile(cpp_file, dtype=np.int32, count=4)
    n, d = int(raw[0]), int(raw[1])
    cpp = np.fromfile(cpp_file, dtype=np.float32, offset=16).reshape(n, d)
    out = {"cpp_shape": [n, d], "ref_shape": list(ref_pixels.shape)}
    if cpp.shape == ref_pixels.shape:
        diff = np.abs(cpp - ref_pixels)
        out.update(max_abs=float(diff.max()), worst_patch=int(diff.max(axis=1).argmax()))
    return out


try:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.set_num_threads(os.cpu_count() or 4)
    log("building crispembed-cli")
    exe = build_cpp()
    log(f"built {exe}")
    gg = hf_hub_download(GGUF_REPO, GGUF_FILE)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    for name, img in IMAGES.items():
        log(f"reference: {name}")
        ref = SCR / f"ref_{name}.gguf"
        meta, pixels, cap = dump_reference(model, proc, img, ref)
        log(f"C++: {name}")
        cpp = run_cpp(exe, gg, img, ref, SCR / f"patches_{name}.bin", SCR / f"dump_{name}")
        cpp["ocr_text"] = cpp_ocr_text(exe, gg, img)
        log(f"C++ OCR text [{name}]: {cpp['ocr_text']!r}")
        cpp["rows"] = analyse_rows(SCR / f"dump_{name}", cap, meta["grid_thw"])
        if name == "fox":  # raw material for offline analysis (~15 MB)
            pe = SCR / f"dump_{name}" / "vis_patch_embed.f32"
            pf = SCR / f"patches_{name}.bin"
            np.savez_compressed(
                OUT / "fox_arrays.npz", ref_pixels=pixels,
                cpp_pixels=np.fromfile(pf, dtype=np.float32, offset=16).reshape(pixels.shape) if pf.exists() else np.zeros(0),
                ref_patch_embed=cap["vis_patch_embed"], ref_patch_embed_nopos=cap["vis_patch_embed_nopos"],
                cpp_patch_embed=np.fromfile(pe, dtype=np.float32) if pe.exists() else np.zeros(0))
        try:
            cpp["patches"] = compare_patches(SCR / f"patches_{name}.bin", pixels)
        except Exception as e:
            cpp["patches"] = {"error": repr(e)}
        res["images"][name] = {"ref": meta, "cpp": cpp}
        save()
        log(json.dumps({k: v for k, v in cpp.items() if k != "stderr_tail"})[:1500])
except Exception:
    res["errors"].append(traceback.format_exc())
    print(res["errors"][-1], file=sys.stderr)
finally:
    save()


def order(k):
    grp = 0 if k.startswith("vis_patch") else 1 if k.startswith("vis_") else 2 if k == "llm_embed" else 3
    n = re.findall(r"\d+", k)
    return (grp, int(n[0]) if n else 0, k)


names = list(res["images"])
all_stages = sorted({s for n in names for s in res["images"][n]["cpp"]["stages"]}, key=order)
lines = ["### Qwen3-VL-2B stage diff: C++ (F16 GGUF, CPU) vs transformers (fp32)\n",
         "| stage | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
for n in names:
    c = res["images"][n]["cpp"]
    lines.insert(1, f"- **{n}** C++ OCR text: `{c.get('ocr_text', '').replace(chr(10), '⏎')}`")
    lines.insert(1, f"- **{n}**: tokens {res['images'][n]['ref']['n_tokens']}, grid {res['images'][n]['ref']['grid_thw']}, "
                    f"patches {c.get('patches')}, mRoPE mismatches {c['mrope_mismatches']} {c['mrope_examples'][:3]}")
for s in all_stages:
    cells = []
    for n in names:
        e = res["images"][n]["cpp"]["stages"].get(s)
        cells.append("—" if not e else f"{'✅' if e['pass'] else '❌'} {e['cos_min']:.5f} / {e['max_abs']:.1e}")
    lines.append(f"| {s} | " + " | ".join(cells) + " |")
for n in names:
    rows = res["images"][n]["cpp"].get("rows", {})
    lines += [f"\n#### {n}: row analysis", "| stage | rows C++/ref | bad rows | first bad | cos median | cos_min if C++ raster / if ref raster |",
              "|---|---|---|---|---|---|"]
    for st in sorted(rows, key=order):
        e = rows[st]
        if "rows_cpp" not in e:
            lines.append(f"| {st} | {e} | | | | |")
            continue
        perm = (f"{e['cos_min_if_cpp_is_raster']:.4f} / {e['cos_min_if_ref_is_raster']:.4f}"
                if "cos_min_if_cpp_is_raster" in e else "")
        lines.append(f"| {st} | {e['rows_cpp']}/{e['rows_ref']} | {e['n_bad']} | {e['first_bad']} | "
                     f"{e['cos_median']:.5f} | {perm} |")
if res["errors"]:
    lines.append("\n**errors:**\n```\n" + res["errors"][-1][-1500:] + "\n```")
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
def image_ok(n):
    c = res["images"][n]["cpp"]
    text_ok = c.get("ocr_text", "").strip() == EXPECTED_TEXT[n]
    # F16 GGUF vs fp32 reference: a single numerically fragile row may dip
    # (fox: 1 of 600 rows to 0.986 in vision blocks 17-22, recovered at 23).
    # Every bug found here broke ~100% of rows, so gate on the row picture:
    # median row >= 0.9999 and at most 0.5% of rows under 0.999, per stage.
    rows = c.get("rows", {})
    stages_ok = bool(rows) and all(
        "rows_cpp" in e and e["rows_cpp"] == e["rows_ref"] and e["cos_median"] >= 0.9999
        and e["n_bad"] <= max(1, int(0.005 * e["rows_ref"]))
        for e in rows.values())
    mrope_ok = (c["mrope_mismatches"] or [1])[0] == 0
    return text_ok and mrope_ok and (stages_ok or n not in STRICT_STAGES)


verdict = {n: image_ok(n) for n in names}
print("verdict:", verdict)
with open(OUT / "summary.md", "a") as f:
    f.write(f"\n**verdict** (exact OCR text on all, mRoPE exact, every stage on {sorted(STRICT_STAGES)}): {verdict}\n")
ok = not res["errors"] and len(names) == len(IMAGES) and all(verdict.values())
sys.exit(0 if ok else 1)
