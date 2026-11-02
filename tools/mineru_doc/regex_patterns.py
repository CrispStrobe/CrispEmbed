"""Every regular expression used by the pinned mineru-vl-utils document helper.

Used by the differential regex fuzz (tools/mineru_doc/fuzz_regex.py) that
checks the native pyre engine against CPython `re` on the exact patterns.
"""
import re

from mineru_vl_utils.post_process import equation_left_right as lr

LEFT = "|".join(lr.LEFT_TOKEN_LIST)
RIGHT = "|".join(lr.RIGHT_TOKEN_LIST)
NODE_CLASS = "[A-Za-z0-9_\u4e00-\u9fa5\\-]"

PATTERNS = [
    (r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(?:(<\|rotate_(?:up|right|down|left)\|>))?(.*?)(?=<\|box_start\|>|$)", re.DOTALL),
    (r"^:?-{3,}:?$", 0),
    (r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*|[^\s]", 0),
    (r"([A-Za-z0-9])\1{19,}", 0),
    (r"\s*(?:::|;|,|\|)\s*", 0),
    (r"\s+", 0),
    (r"\n{5,}", 0),
    (r"\\tag\s*\{[^}]*\}", 0),
    (r"\\tag\s*\{([^}]*)\}", 0),
    (r"_\s*\{([^{}]|\{[^{}]*\})*\}\s*_\s*\{([^{}]|\{[^{}]*\})*\}", 0),
    (r"<", 0),
    (LEFT, 0),
    (RIGHT, 0),
    (r"(\&|\\\\|\\begin{array}\s*{[a-zA-Z\s]*}|\\end{array})", 0),
    ("(" + LEFT + "|" + RIGHT + ")", 0),
    (r"\\eqqcolon", 0),
    (r"-->\s*\|[^|\n]+\|\s*\S", 0),
    (r"-->\s*[^|\n]*\|\s*[^|\n]+$", 0),
    (r"[\u4e00-\u9fff\u3400-\u4dbf]", 0),
    (r"^(\s*)subgraph\s+(.+?)\s*$", 0),
    (r"^[A-Za-z_][A-Za-z0-9_]*$", 0),
    (r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?=[\[\(])", 0),
    (r"[^A-Za-z0-9_]+", 0),
    (r"_+", 0),
    (r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", 0),
    (r"```mermaid\s*(.*?)\s*```", re.DOTALL),
    (r"```\s*(.*?)\s*```", re.DOTALL),
    (r"^(grap|grapg|graphh)\b", re.IGNORECASE),
    (r"^(flowchar|flowchartt)\b", re.IGNORECASE),
    (r"<\s*-\s*-\s*>", 0),
    (r"-\s+->", 0),
    (r"--\s+>", 0),
    (r"-\s+-\s+>", 0),
    (r"(?<![-=])\s+->\s+", 0),
    (r"[ \t]+", 0),
    (r"([a-zA-Z0-9_\u4e00-\u9fa5\-]+(?:[ \t]+[a-zA-Z0-9_\u4e00-\u9fa5\-]+)*)[ \t]*\[(.*?)\]", re.DOTALL),
    (r"(<nl>|<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)", 0),
    (r"\\\((.+?)\\\)", re.DOTALL),
    (r"\\\[(.+?)\\\]", re.DOTALL),
    (r"(<eq>.*?</eq>)", re.DOTALL),
    (r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[\[\(\{]", 0),
    (r"[A-Za-z0-9]+", 0),
    (r"[^A-Za-z0-9_]", 0),
    (r" {2,}", 0),
    (r"^\|(?P<label>.*?)\|\s*(?P<dst>.+?)\s*$", 0),
    (r"(_{3,})", 0),
    (r"_{3,}", 0),
    (r'(?P<node>[A-Za-z0-9_\u4e00-\u9fa5\-]+)\s*\["(?P<label>(?:[^"\\]|\\.)*)"\]', re.DOTALL),
    (r"^\s*(?P<src>[A-Za-z0-9_\u4e00-\u9fa5\-]+)\s*-->\s*(?:\|[^|\n]*\|\s*)?(?P<dst>[A-Za-z0-9_\u4e00-\u9fa5\-]+)(?:\s*(?:\[[^\]]*\]|\([^\)]*\)|\{[^\}]*\}))?\s*$", 0),
    (r"<br\s*/?>", re.IGNORECASE),
    (r"^\s*(?P<src>[A-Za-z0-9_\u4e00-\u9fa5\-]+)\s*-->\s*(?P<dst>[A-Za-z0-9_\u4e00-\u9fa5\-]+)\s*\[\"(?P<label>(?:[^\"\\]|\\\\.)*)\"\]\s*$", 0),
    (r"[A-Z][a-z]|[A-Z]{2}", 0),
    (r"\\+n", 0),
    (r'^[\[\]\(\)"\'\s>]+', 0),
    (r'[\[\]\(\)"\'\s,.;:]+$', 0),
    (r"[A-Za-z]{8,}", 0),
    (r"^\|[^|\n]*\|\s*", 0),
    (r"^([A-Za-z0-9_\u4e00-\u9fa5\-]+)", 0),
    (r"[–\d\-,\s]+", 0),
    (r"\\\[(.*?)\\\]", re.DOTALL),
    (r"(\\\(.*?\\\))", re.DOTALL),
    (re.escape(r"\times") + r"([a-zA-Z])(?![a-zA-Z])", 0),
    (r"\[\s*" + re.escape("AC2X") + r"\s*\]", 0),
]
