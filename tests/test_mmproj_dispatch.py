#!/usr/bin/env python3
"""Test the unified merge dispatcher (models/merge-llamacpp-gguf.py).

Verifies clip.projector_type auto-detection routes to the correct per-family
merge script, that unsupported/missing projectors error cleanly, and that a full
dispatch (no --detect) actually produces a valid merged GGUF. Pure Python, no
download. Run:  ~/miniconda3/bin/python tests/test_mmproj_dispatch.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile

from gguf import GGUFWriter, GGUFReader

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.dirname(HERE), "models")
CLI = os.path.join(MODELS, "merge-llamacpp-gguf.py")

# Reuse the InternVL fixture builders for the end-to-end dispatch check.
_spec = importlib.util.spec_from_file_location("t_internvl",
                                               os.path.join(HERE, "test_mmproj_internvl.py"))
t_internvl = importlib.util.module_from_spec(_spec)
sys.path.insert(0, MODELS)
_spec.loader.exec_module(t_internvl)


def _mmproj(path, proj):
    w = GGUFWriter(path, "clip")
    w.add_string("clip.projector_type", proj)
    w.add_bool("clip.has_vision_encoder", True)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def main():
    fails = []

    def check(cond, msg):
        print(("  ok  " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as d:
        # 1. Routing: each projector_type -> the right family label.
        for proj, label in [("qwen2vl_merger", "Qwen2-VL"), ("idefics3", "SmolVLM"),
                            ("internvl", "InternVL")]:
            p = os.path.join(d, f"{proj}.gguf")
            _mmproj(p, proj)
            r = subprocess.run([sys.executable, CLI, "--mmproj", p, "--detect"],
                               capture_output=True, text=True)
            check(r.returncode == 0 and label in r.stdout, f"{proj} routes to {label}")

        # 2. Unsupported projector -> clean non-zero error.
        p = os.path.join(d, "unsup.gguf")
        _mmproj(p, "gemma3")
        r = subprocess.run([sys.executable, CLI, "--mmproj", p, "--detect"],
                           capture_output=True, text=True)
        check(r.returncode != 0 and "unsupported" in (r.stdout + r.stderr),
              "unsupported projector_type errors cleanly")

        # 3. Missing projector_type -> clean non-zero error.
        p = os.path.join(d, "noproj.gguf")
        w = GGUFWriter(p, "clip"); w.add_bool("clip.has_vision_encoder", True)
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        r = subprocess.run([sys.executable, CLI, "--mmproj", p, "--detect"],
                           capture_output=True, text=True)
        check(r.returncode != 0 and "no clip.projector_type" in (r.stdout + r.stderr),
              "missing projector_type errors cleanly")

        # 4. Full dispatch (no --detect) produces a valid internvl2 GGUF.
        llm = os.path.join(d, "llm.gguf")
        mmproj = os.path.join(d, "mmproj.gguf")
        out = os.path.join(d, "crisp.gguf")
        t_internvl.build_llm(llm)
        t_internvl.build_mmproj(mmproj)
        r = subprocess.run([sys.executable, CLI, "--llm", llm, "--mmproj", mmproj, "--output", out],
                           capture_output=True, text=True)
        check(r.returncode == 0, f"full dispatch exits 0 (rc={r.returncode})")
        if r.returncode == 0:
            md = {f.name: f.contents() for f in GGUFReader(out).fields.values()}
            check(md.get("general.architecture") == "internvl2",
                  "dispatched merge produced internvl2 GGUF")
        else:
            print(r.stdout + r.stderr)

    print("\n" + ("ALL PASS" if not fails else f"FAILED ({len(fails)}): " + "; ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
