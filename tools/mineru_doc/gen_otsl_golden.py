"""Build tests/data/otsl_profiles_golden.tsv for tests/test_otsl_table.cpp.

Each row: hex(input) TAB hex(mineru_basic_v1 output) TAB hex(marker_strict
output) TAB hex(marker warnings joined by \\x1e) TAB hex(upstream docling output).
The basic/marker columns come from the implementations *before* extraction
into otsl_table.h (built by the session from git history: mineru_layout.h at
7331d89b and marker_document.cpp at 7331d89b) via the oracle binary on stdin;
the upstream column comes from the pinned mineru-vl-utils convert_otsl_to_html.

usage: python -I gen_otsl_golden.py ORACLE_BINARY OUT.tsv
"""
import random, subprocess, sys
from mineru_vl_utils.post_process.otsl2html import convert_otsl_to_html

rng = random.Random(20261031)
TOK = ["<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>"]
TXT = ["A", " b ", "A & <b>", "'q'", '"', "中文", "　", "x\ty", "1.5", ""]
cases = ["", "<fcel>A<nl>", "<fcel>A<lcel><nl><ucel><xcel><nl>", "<fcel>A<ecel><nl><fcel>B<nl>", "<lcel><fcel>A<nl>",
         "lead<fcel>A<nl>", "<fcel>A<foo><nl>", "<fcel>A<nl", "<ecel>ignore me<nl>", "<table><tr><td>x</td></tr></table>",
         "<fcel>A<nl><fcel>B<fcel>C<nl>", "<fcel>A&\"<fcel>B<nl><fcel>C<ecel><nl>", "<lcel><nl>", "not a table",
         "<fcel>A<lcel><nl><ucel><ecel><nl>", "<fcel>a<ucel><nl><fcel>b<ucel><nl>", "<fcel>" + "<" * 50 + "<nl>"]
for _ in range(600):
    s = "".join(rng.choice(TOK) + (rng.choice(TXT) if rng.random() < .5 else "") for _ in range(rng.randint(1, 20)))
    if rng.random() < .1:
        s = rng.choice(TXT) + s
    if rng.random() < .05:
        s += "<bad>"
    cases.append(s)
inp = "\n".join(c.encode().hex() for c in cases) + "\n"
p = subprocess.run([sys.argv[1]], input=inp, capture_output=True, text=True)
sys.stderr.write(p.stderr)
if p.returncode:
    sys.exit("old and moved implementations differ")
rows = p.stdout.splitlines()
with open(sys.argv[2], "w") as f:
    for c, row in zip(cases, rows):
        up = convert_otsl_to_html(c).encode().hex()
        f.write(row + "\t" + up + "\n")
print(len(rows), "golden rows")
