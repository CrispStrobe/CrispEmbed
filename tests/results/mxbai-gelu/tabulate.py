#!/usr/bin/env python
"""Parse arms.txt + onnx_ref.json into per-doc score tables and deltas."""
import json
import os
import re
import sys

SC = os.environ.get("MXGELU_WORK", ".")
ref = json.load(open(SC + "/onnx_ref.json"))

arms = {}  # (label, qi, spell) -> [score per doc idx]
cur = None
for line in open(SC + "/arms.txt"):
    m = re.match(r"== (\S+) \| q(\d+) \| env=(\S+) ==", line)
    if m:
        cur = (m.group(1), int(m.group(2)), m.group(3))
        arms[cur] = [None] * 6
        continue
    m = re.match(r"\[(\d+)\] (-?\d+\.\d+) ", line)
    if m and cur:
        arms[cur][int(m.group(1))] = float(m.group(2))


def kendall(a, b):
    n = len(a)
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return (c - d) / (c + d) if (c + d) else 1.0


order = ["f16-new/unset", "f16-new/0", "f16-new/1",
         "shipped-q8_0/unset", "shipped-q8_0/1",
         "q8_0-new/unset", "q8_0-new/0", "q8_0-new/1"]

out = []
for model in ("xsmall", "base"):
    for qi in (0, 1):
        R = ref[model]["q%d" % qi]["scores"]
        out.append("### %s — q%d: %r" % (model, qi, ref[model]["q%d" % qi]["query"]))
        out.append("")
        hdr = "| arm | " + " | ".join("d%d" % i for i in range(6)) + " | max abs delta vs ONNX | tau |"
        out.append(hdr)
        out.append("|" + "---|" * 9)
        out.append("| **ONNX ref** | " + " | ".join("%.6f" % s for s in R) + " | — | — |")
        for key in order:
            lbl, spell = key.split("/")
            k = ("%s-%s" % (model, lbl), qi, spell)
            if k not in arms or None in arms[k]:
                continue
            s = arms[k]
            md = max(abs(a - b) for a, b in zip(s, R))
            out.append("| %s env=%s | " % (lbl, spell) + " | ".join("%.6f" % x for x in s) +
                       " | %.6f | %.3f |" % (md, kendall(s, R)))
        out.append("")
        # byte-identity check unset vs 0
        for lbl in ("f16-new", "q8_0-new"):
            a = arms.get(("%s-%s" % (model, lbl), qi, "unset"))
            b = arms.get(("%s-%s" % (model, lbl), qi, "0"))
            c = arms.get(("%s-%s" % (model, lbl), qi, "1"))
            if a and b:
                out.append("gate: %s unset==0 : %s   |  0 vs 1 max delta: %.6f" %
                           (lbl, "IDENTICAL" if a == b else "DIFFER %r %r" % (a, b),
                            max(abs(x - y) for x, y in zip(b, c)) if c else float("nan")))
        out.append("")
print("\n".join(out))
