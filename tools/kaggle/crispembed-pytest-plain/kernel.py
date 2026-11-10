"""Run `pytest tests` on a plain environment and write a small summary JSON.

Plain environment: a fresh venv with only pytest numpy pillow gguf
sentencepiece tokenizers (no torch/transformers/cv2), no model files, no HF
token, no native build. Runs twice at the pinned SHA: a full clone and a
`--depth 1` clone. Output: summary.json (counts, failing ids, grouped skip
reasons). SHA is substituted by the pusher; nothing account-specific here.
"""
import collections, json, os, re, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path

SHA = "__SHA__"
REPO = "https://github.com/CrispStrobe/CrispEmbed.git"
WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


venv = TMP / "venv"
# Kaggle's system python lacks ensurepip and ships a sitecustomize that fails
# outside its own environment (stderr noise in every child). Use a managed
# standalone CPython instead, so the environment really is plain.
sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
sh([sys.executable, "-m", "uv", "venv", "-q", "--managed-python", "--python", "3.13", str(venv)])
py = str(venv / "bin" / "python")
sh([sys.executable, "-m", "uv", "pip", "install", "-q", "--python", py,
    "pytest", "numpy", "pillow", "gguf", "sentencepiece", "tokenizers"])

# Drop tokens/account vars and Kaggle's PYTHONPATH (its sitecustomize prints an
# error inside a foreign venv, which pollutes every child's stderr).
env = {k: v for k, v in os.environ.items()
       if not k.startswith(("HF_", "HUGGING", "KAGGLE_", "CRISPEMBED_"))
       and k not in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE")}
summary = {"sha": SHA}
for mode in ("full", "shallow"):
    d = TMP / f"repo-{mode}"
    if mode == "full":
        sh(["git", "clone", "-q", REPO, str(d)])
        sh(["git", "-C", str(d), "checkout", "-q", SHA])
    else:
        d.mkdir()
        sh(["git", "-C", str(d), "init", "-q"])
        sh(["git", "-C", str(d), "fetch", "-q", "--depth", "1", REPO, SHA])
        sh(["git", "-C", str(d), "checkout", "-q", "FETCH_HEAD"])
    xml = TMP / f"junit-{mode}.xml"
    r = subprocess.run([py, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-rfEs",
                        f"--junitxml={xml}"], cwd=d, env=env, capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1:] or [""]
    (WORK / f"pytest-{mode}-tail.txt").write_text("\n".join(r.stdout.splitlines()[-400:]))
    res = {"exit": r.returncode, "line": tail[0], "failed": [], "errors": [],
           "skip_reasons": collections.Counter(), "counts": collections.Counter()}
    if xml.exists():
        for tc in ET.parse(xml).getroot().iter("testcase"):
            tid = f"{tc.get('classname')}::{tc.get('name')}"
            kind = "passed"
            for child in tc:
                if child.tag in ("failure", "error", "skipped"):
                    kind = child.tag
                    if child.tag == "skipped":
                        msg = re.sub(r"[0-9a-f]{12,40}", "<sha>", child.get("message") or "")
                        res["skip_reasons"][msg[:200]] += 1
                    elif child.tag == "failure":
                        res["failed"].append(f"{tid} | {(child.get('message') or '')[:160]}")
                    else:
                        res["errors"].append(f"{tid} | {(child.get('message') or '')[:160]}")
            res["counts"][kind] += 1
    res["skip_reasons"] = dict(res["skip_reasons"].most_common())
    summary[mode] = res
    print(mode, res["line"], flush=True)

(WORK / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({m: summary[m]["line"] for m in ("full", "shallow")}))
