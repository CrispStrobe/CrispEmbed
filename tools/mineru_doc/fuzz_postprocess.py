"""Differential fuzz: native MinerU post-processing vs the pinned Python helper.

Every case calls the unchanged mineru-vl-utils function and the native port
(tests/test_mineru_doc.cpp mode `fn`) on the same generated input and compares
the exact output strings. Generators produce structured, adversarial inputs
(OTSL token soups, LaTeX with \\left/\\right/array, Markdown and chart tables
with repetition, Mermaid flowcharts, tagged image-analysis output, layout
token streams with unicode digits/whitespace).

usage: python -I fuzz_postprocess.py BINARY [--cases N] [--seed S] [--report OUT.json]
"""
import argparse
import json
import random
import subprocess
import sys

from mineru_vl_utils.mineru_client import DEFAULT_PROMPTS, DEFAULT_SAMPLING_PARAMS, MinerUClientHelper
from mineru_vl_utils import post_process as pp
from mineru_vl_utils.post_process import (
    chart_invalid_repetition as chart, equation_big, equation_double_subscript as dsub,
    equation_left_right as lr, equation_unbalanced_braces as braces, image_analysis_postprocess as ia,
    mermaid_invalid_edge_lines as medge, mermaid_invalid_subgraph_lines as msub,
    mermaid_node_id_legalize as mlegal, mermaid_repetition as mrep, otsl2html, table_image_processor as tip,
    text_display2inline as d2i, text_inline_spacing as spacing, text_move_underscores_outside as unders,
)
from loguru import logger

logger.remove()
HELPER = MinerUClientHelper("transformers", DEFAULT_PROMPTS, DEFAULT_SAMPLING_PARAMS, (1036, 1036), 28, 50,
                            False, True, False, False, False, False)

def text_chain(s):
    s = d2i.try_convert_display_to_inline(s)
    s = spacing.try_fix_macro_spacing_in_markdown(s)
    return unders.try_move_underscores_outside(s)

def image(s):
    r = ia.process_image_or_chart(s)
    return "\x1f".join([r["class"], r["sub_class"], r["caption"], r["content"]])

def layout(s):
    return json.dumps(HELPER.parse_layout_output(s), ensure_ascii=False, separators=(",", ":"))

FUNCS = {
    "otsl": otsl2html.convert_otsl_to_html,
    "equation": lambda s: pp._process_equation(s, debug=False),
    "brackets": pp._add_equation_brackets,
    "left_right": lr.try_match_equation_left_right,
    "big": equation_big.try_fix_equation_big,
    "braces": braces.try_fix_unbalanced_braces,
    "double_subscript": dsub.try_fix_equation_double_subscript,
    "text": text_chain,
    "eq_wrap": lambda s: tip.replace_table_formula_delimiters(s, enabled=True),
    "md_table": lambda s: ia.convert_markdown_table_to_html(s) or "<None>",
    "malformed": lambda s: "1" if ia.has_malformed_markdown_table(s) else "0",
    "mermaid_strict": ia.extract_and_validate_mermaid_strict,
    "mermaid_rep": lambda s: "1" if mrep.detect_mermaid_repetition_issues(s) else "0",
    "mermaid_legal": mlegal.try_fix_mermaid_node_id_legalization,
    "mermaid_edges": medge.try_fix_mermaid_invalid_edge_lines,
    "mermaid_subgraph": msub.try_fix_mermaid_invalid_subgraph_lines,
    "chart_rep": chart.try_fix_chart_invalid_repetition,
    "chart_valid": lambda s: "1" if chart.validate_content_repetition(s) else "0",
    "image": image,
    "pure_table": pp._convert_pure_table_content_to_html,
    "lower": str.lower,
    "layout": layout,
}

WS = [" ", "  ", "\t", "\n", "　", "\xa0", " ", ""]

def pick(rng, items, n):
    return "".join(rng.choice(items) for _ in range(rng.randint(0, n)))

def gen_otsl(rng):
    toks = ["<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>"]
    cells = ["a", "B &", "<b>", " x ", "中文", "'q'", '"', "", "1.5", "　"]
    out = "".join(rng.choice(toks) + (rng.choice(cells) if rng.random() < .6 else "") for _ in range(rng.randint(0, 25)))
    if rng.random() < .1:
        out = "<table><tr><td>x</td></tr></table>"
    return out

LATEX = ["\\left(", "\\right)", "\\left[", "\\right]", "\\left\\{", "\\right\\}", "\\left.", "\\right.", "\\left|",
         "\\right|", "\\rightarrow", "\\left\\langle", "\\right\\rangle", "\\begin{array}{cc}", "\\begin{array}{l}",
         "\\end{array}", "&", "\\\\", "x", "y^{2}", "_{i}", "_ {j}", "_{a}_{b}", "{", "}", "\\{", "\\}", "\\\\{",
         "\\big{)}", "\\Big{\\}}", "\\bigg{|}", "\\Biggm{[}", "\\biggl{\\langle}", "\\bigtimes", "<", "\\eqqcolon",
         "\\coloneqq", "\\tag{1}", " ", "\\[", "\\]", "\\frac{a}{b}", "\\left\\lVert", "\\right\\rVert", "中"]

def gen_latex(rng):
    return pick(rng, LATEX, 18)

TEXT = ["\\[", "\\]", "\\(", "\\)", "1-2", "3,4", "–", "a", "\\timesX", "\\to y", "\\top", "\\int x", "\\inX",
        "___", "____", "_", " ", "\n", "\\cong A", "\\subsetB", "word", "١٢", "\\times b"]

def gen_text(rng):
    if rng.random() < .5:
        return "".join(rng.choice(["\\(" + pick(rng, TEXT, 5) + "\\)", "\\[" + pick(rng, TEXT, 4) + "\\]",
                                   pick(rng, TEXT, 3)]) for _ in range(rng.randint(1, 5)))
    return pick(rng, TEXT, 16)

def gen_md(rng):
    if rng.random() < .5:
        cols = rng.randint(2, 4)
        cell = lambda: rng.choice(["a", "b\\|c", "1", "中", "&<>", "x y", "`c`"])
        rows = ["| " + " | ".join(cell() for _ in range(cols)) + " |",
                "|" + "|".join(rng.choice(["---", ":---:", "----:"]) for _ in range(cols)) + "|"]
        rows += ["| " + " | ".join(cell() for _ in range(cols)) + " |" for _ in range(rng.randint(0, 5))]
        return "\n".join(rows)
    rows = []
    cols = rng.randint(1, 4)
    for r in range(rng.randint(0, 6)):
        if r == 1 and rng.random() < .8:
            rows.append("|" + "|".join(rng.choice(["---", ":---:", "--", " --- "]) for _ in range(cols)) + "|")
            continue
        cells = [rng.choice(["a", "b\\|c", "x\\", "`c|d`", "1", "", "中", "&<>"]) for _ in range(cols + (rng.random() < .1))]
        line = "|".join(cells)
        if rng.random() < .7:
            line = "| " + line + " |"
        rows.append(line)
    if rng.random() < .2:
        rows.insert(rng.randint(0, len(rows)), "")
    return "\n".join(rows)

def gen_chart(rng):
    base = gen_md(rng)
    k = rng.random()
    if k < .2:
        base += "\n" + "\n".join(["| same row repeated here for testing | 1 |"] * rng.randint(2, 5))
    elif k < .3:
        base += "\n\n\n\n\n\n"
    elif k < .4:
        base += " " + "a" * rng.randint(15, 25)
    elif k < .5:
        base = (base + " ") * rng.randint(2, 4)
    elif k < .6:
        base += "\n| " + "; ".join(["foo bar"] * rng.randint(5, 10)) + " | x |"
    elif k < .7:
        base += " tok" * rng.randint(5, 10)
    elif k < .75:
        base = base.replace("|", "｜")
    return base

MERMAID = ["graph TD", "flowchart LR", "grap TD", "  A[Start] --> B[End]", "  A --> B", "  node one --> node two",
           "  A -> B", "  A - -> B", "  A -- > B", "  A <--> B", "  A < - - > B", "  subgraph 1) Trigger",
           "  subgraph sg1[\"Title\"]", "  subgraph Main", "  end", "  C[\"quoted \\\"x\\\"\"] --> D",
           "  G -->node(DRAM)| H", "  A -->|label| B", "  A --> |lab (x)| Top Layer", "  X[\"[a]\"]",
           "  E[line\nline]", "%% comment", "  A --> B --> C", "  A; B --> C;", "  中文 --> 节点",
           "  AnAoApAqArAsAtAuAvAwAx --> B", "  A[\"rep\\nrep\\nrep\\nrep\\nrep\\nrep\\nrep\\nrep\\nrep\"]",
           "  Z1 --> Z2", "```mermaid", "```", "mermaid"]

def gen_mermaid(rng):
    lines = [rng.choice(MERMAID) for _ in range(rng.randint(0, 12))]
    if rng.random() < .2:
        lines += ["  Q --> R[\"same\"]"] * rng.randint(7, 10)
    s = "\n".join(lines)
    if rng.random() < .3:
        s += "\n"
    return s

def gen_image(rng):
    cls = rng.choice(["chart", "flowchart", "chemical", "pure_table", "pure_formula", "natural_image", "Chart ",
                      "diagram", ""])
    content = rng.choice([gen_chart(rng), gen_mermaid(rng), gen_otsl(rng), gen_md(rng), "x = y", ""])
    parts = [f"<|class_start|>{cls}<|class_end|>"]
    if rng.random() < .8:
        parts.append(f"<|sub_class_start|>{rng.choice(['bar', 'Line', ''])}<|sub_class_end|>")
    if rng.random() < .8:
        parts.append(f"<|caption_start|> {rng.choice(['A cap', '中文图注', ''])} <|caption_end|>")
    r = rng.random()
    if r < .7:
        parts.append(f"<|content_start|>{content}<|content_end|>")
    elif r < .85:
        parts.append(f"<|content_start|>{content}")
    rng.shuffle(parts) if rng.random() < .1 else None
    return "\n".join(parts)

TYPES = ["text", "title", "table", "equation", "image", "chart", "Unknown", "inline_formula", "list", "TEXT",
         "image_caption", "header", "bogus", "equation_block", "image_block", "list_item", "中"]

def gen_layout(rng):
    out = []
    for _ in range(rng.randint(0, 8)):
        coords = [str(rng.choice([rng.randint(0, 1000), rng.randint(0, 1200), 0, 1000])) for _ in range(4)]
        if rng.random() < .05:
            coords[0] = "٣٠٠"
        sep = rng.choice([" ", "  ", "\t", "　", "\n"])
        rot = rng.choice(["", "<|rotate_up|>", "<|rotate_right|>", "<|rotate_down|>", "<|rotate_left|>"])
        tail = rng.choice(["", "txt_contd_tgt", "junk", "\n"])
        out.append(f"<|box_start|>{sep.join(coords)}<|box_end|><|ref_start|>{rng.choice(TYPES)}<|ref_end|>{rot}{tail}")
    if out and rng.random() < .3:
        out.append(out[rng.randrange(len(out))])
    return "".join(out)

GENS = {
    "otsl": gen_otsl, "equation": gen_latex, "brackets": gen_latex, "left_right": gen_latex, "big": gen_latex,
    "braces": gen_latex, "double_subscript": gen_latex, "text": gen_text, "eq_wrap": gen_text, "md_table": gen_md,
    "malformed": gen_md, "mermaid_strict": gen_mermaid, "mermaid_rep": gen_mermaid, "mermaid_legal": gen_mermaid,
    "mermaid_edges": gen_mermaid, "mermaid_subgraph": gen_mermaid, "chart_rep": gen_chart, "chart_valid": gen_chart,
    "image": gen_image, "pure_table": lambda r: rng_choice_table(r), "lower": lambda r: pick(r, ["A", "İ", "Σ", "ẞ", "K", "x"], 8),
    "layout": gen_layout,
}

def rng_choice_table(rng):
    return rng.choice([gen_md(rng), gen_otsl(rng), "<TABLE>x</table>", " ", "plain"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--report")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lines, expect = [], []
    for name, fn in FUNCS.items():
        for _ in range(a.cases):
            s = GENS[name](rng)
            try:
                want = fn(s).encode("utf-8", "surrogateescape").hex()
            except Exception as e:  # noqa: BLE001 - record upstream exception class
                want = "ERR " + type(e).__name__
            lines.append(f"{name}\t{s.encode('utf-8', 'surrogateescape').hex()}")
            expect.append((name, s, want))
    out = subprocess.run([a.binary, "fn"], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    got = out.stdout.splitlines()
    per = {}
    first = {}
    for (name, s, want), g in zip(expect, got):
        ok = g == want or (want.startswith("ERR") and g.startswith("ERR"))
        st = per.setdefault(name, [0, 0])
        st[0] += 1
        st[1] += 0 if ok else 1
        if not ok and name not in first:
            first[name] = {"input": s, "python": want, "native": g}
    total_bad = sum(v[1] for v in per.values())
    for name, (n, bad) in per.items():
        print(f"{name:18s} {n:6d} cases {bad:5d} mismatches")
    for name, f in first.items():
        print("FIRST MISMATCH", name, repr(f["input"])[:300])
        print("   python", bytes.fromhex(f["python"]).decode("utf-8", "replace")[:300] if not f["python"].startswith("ERR") else f["python"])
        print("   native", bytes.fromhex(f["native"]).decode("utf-8", "replace")[:300] if not f["native"].startswith("ERR") else f["native"])
    if a.report:
        with open(a.report, "w", encoding="utf-8") as fh:
            json.dump({"seed": a.seed, "cases_per_function": a.cases, "python": sys.version.split()[0],
                       "functions": {k: {"cases": v[0], "mismatches": v[1]} for k, v in per.items()},
                       "first_mismatch": first, "total_mismatches": total_bad,
                       "lines_returned": len(got), "lines_sent": len(expect)}, fh, ensure_ascii=False, indent=1)
    sys.exit(1 if total_bad or len(got) != len(expect) else 0)

main()
