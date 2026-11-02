"""Differential fuzz: native pyre engine vs CPython re on helper patterns.

usage: python fuzz_regex.py BINARY [--cases N] [--seed S]
BINARY is the test binary built from tests/test_mineru_doc.cpp (mode `re`).
"""
import argparse, random, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from regex_patterns import PATTERNS  # noqa: E402

PIECES = ["\\", "\\\\", "(", ")", "[", "]", "{", "}", "|", "-", "->", "-->", "<-->", "< - - >", "_", "___", " ",
          "  ", "\t", "\n", "\n\n\n\n\n\n", "\r\n", "\u2028", "\x85", "\xa0", "\u3000", "a", "Z", "x", "9", "٣", "中",
          "文", "é", "K", ":", ";", ",", ".", "::", "---", ":---:", "<br>", "<BR/>", "\\n", "\\\\n", '"', "'",
          "\\left(", "\\right)", "\\right.", "\\rightarrow", "\\left\\{", "\\begin{array}{cc}", "\\end{array}", "&",
          "\\tag{1}", "_{a}", "_ {b{c}}", "\\(", "\\)", "\\[", "\\]", "<eq>", "</eq>", "<nl>", "<fcel>", "<ecel>",
          "<lcel>", "```", "```mermaid", "graph", "GRAP", "flowchar", "subgraph", "A", "B1", "node id",
          "<|box_start|>", "<|box_end|>", "<|ref_start|>", "<|ref_end|>", "text", "table", "12", "345",
          "<|rotate_right|>", "txt_contd_tgt", "aaaaaaaaaaaaaaaaaaaaaaa", "AnAoApAq", "\\times", "\\timesX",
          "[AC2X]", "[ AC2X ]", "=", "–", "abcdefghij"]

def rand_text(rng):
    return "".join(rng.choice(PIECES) for _ in range(rng.randint(0, 24)))

def spans(p, s):
    out = ""
    for m in p.finditer(s):
        for g in range(p.groups + 1):
            a, b = m.span(g)
            out += f"{a},{b};"
        out += "|"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lines, expect = [], []
    for pat, flags in PATTERNS:
        p = re.compile(pat, flags)
        nflags = (1 if flags & re.DOTALL else 0) | (2 if flags & re.IGNORECASE else 0)
        for _ in range(a.cases):
            s = rand_text(rng)
            for op in ("spans", "split", "full"):
                if op == "spans":
                    want = spans(p, s)
                elif op == "split":
                    want = "".join((x or "").encode().hex() + "," for x in p.split(s))
                else:
                    want = "1" if p.fullmatch(s) else "0"
                lines.append("\t".join([op, pat.encode().hex(), str(nflags), s.encode().hex()]))
                expect.append((want.encode().hex(), pat, op, s))
    out = subprocess.run([a.binary, "re"], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    got = out.stdout.splitlines()
    bad = 0
    for (want, pat, op, s), g in zip(expect, got):
        if g != want:
            bad += 1
            if bad <= 10:
                print("MISMATCH", op, repr(pat), repr(s), g[:80], want[:80])
    print(f"{len(expect)} regex checks, {bad} mismatches")
    sys.exit(1 if bad or len(got) != len(expect) else 0)

main()
