#!/usr/bin/env python3
"""Retokenize MathWriting labels into PosFormer's 110-token vocabulary.

Strategy: parse raw LaTeX character-by-character, mapping to PosFormer
tokens. Unknown commands get mapped to close equivalents or dropped.
Unknown letters get mapped to lowercase. Environments get stripped.
The goal is maximum coverage, not perfect LaTeX fidelity.
"""

import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

POSFORMER_DICT = Path("/mnt/volume1/PosFormer-fresh/Pos_Former/datamodule/dictionary.txt")

VOCAB: Set[str] = set()
COMMANDS: Set[str] = set()
SINGLES: Set[str] = set()

# Map unknown commands to PosFormer equivalents
COMMAND_MAP = {
    # Greek (map to closest available or strip)
    '\\epsilon': '\\phi',      # visually similar enough
    '\\varepsilon': '\\phi',
    '\\omega': '\\theta',
    '\\Omega': '\\Pi',
    '\\xi': '\\phi',
    '\\zeta': '\\phi',
    '\\eta': '\\theta',
    '\\kappa': '\\lambda',
    '\\nu': '\\mu',
    '\\rho': '\\sigma',
    '\\tau': '\\sigma',
    '\\chi': '\\phi',
    '\\psi': '\\phi',
    '\\varphi': '\\phi',
    '\\vartheta': '\\theta',
    '\\varsigma': '\\sigma',
    '\\varpi': '\\pi',
    '\\varrho': '\\sigma',
    '\\Xi': '\\Pi',
    '\\Phi': '\\Pi',
    '\\Psi': '\\Pi',
    '\\Gamma': '\\Pi',
    '\\Lambda': '\\Pi',
    '\\Sigma': '\\Pi',
    '\\Theta': '\\Pi',
    '\\Upsilon': '\\Pi',

    # Operators → closest match
    '\\partial': 'd',
    '\\nabla': '\\Delta',
    '\\approx': '=',
    '\\equiv': '=',
    '\\cong': '=',
    '\\simeq': '=',
    '\\sim': '=',
    '\\propto': '=',
    '\\le': '\\leq',
    '\\ge': '\\geq',
    '\\ll': '<',
    '\\gg': '>',
    '\\ne': '\\neq',
    '\\not': None,  # drop
    '\\neg': None,

    # Arrows
    '\\leftarrow': '\\rightarrow',
    '\\Rightarrow': '\\rightarrow',
    '\\Leftarrow': '\\rightarrow',
    '\\leftrightarrow': '\\rightarrow',
    '\\Leftrightarrow': '\\rightarrow',
    '\\longrightarrow': '\\rightarrow',
    '\\mapsto': '\\rightarrow',
    '\\to': '\\rightarrow',
    '\\hookrightarrow': '\\rightarrow',
    '\\longleftarrow': '\\rightarrow',
    '\\iff': '\\rightarrow',
    '\\implies': '\\rightarrow',

    # Decorations → strip (return content only)
    '\\hat': None,       # handled specially
    '\\tilde': None,
    '\\bar': None,
    '\\vec': None,
    '\\dot': None,
    '\\ddot': None,
    '\\overline': None,
    '\\underline': None,
    '\\widehat': None,
    '\\widetilde': None,
    '\\overrightarrow': None,

    # Formatting → strip
    '\\mathbb': None,
    '\\mathcal': None,
    '\\mathrm': None,
    '\\mathbf': None,
    '\\mathit': None,
    '\\text': None,
    '\\textbf': None,
    '\\operatorname': None,
    '\\boldsymbol': None,

    # Spacing → drop
    '\\quad': None,
    '\\qquad': None,
    '\\hspace': None,
    '\\,': None,
    '\\;': None,
    '\\!': None,

    # Delimiters → map to ( ) [ ]
    '\\langle': '(',
    '\\rangle': ')',
    '\\lfloor': '(',
    '\\rfloor': ')',
    '\\lceil': '(',
    '\\rceil': ')',
    '\\left': None,
    '\\right': None,
    '\\big': None,
    '\\Big': None,
    '\\bigg': None,
    '\\Bigg': None,

    # Products/sums
    '\\prod': '\\sum',
    '\\bigcup': '\\sum',
    '\\bigcap': '\\sum',
    '\\bigoplus': '\\sum',
    '\\bigotimes': '\\sum',
    '\\coprod': '\\sum',
    '\\iint': '\\int',
    '\\iiint': '\\int',
    '\\oint': '\\int',

    # Misc
    '\\infin': '\\infty',
    '\\infinity': '\\infty',
    '\\star': '\\times',
    '\\circ': '.',
    '\\bullet': '.',
    '\\otimes': '\\times',
    '\\oplus': '+',
    '\\ominus': '-',
    '\\cup': '+',
    '\\cap': '.',
    '\\subset': '<',
    '\\supset': '>',
    '\\subseteq': '\\leq',
    '\\supseteq': '\\geq',
    '\\emptyset': '0',
    '\\notin': '\\neq',
    '\\models': '=',
    '\\vdash': '|',
    '\\Vdash': '|',
    '\\perp': '\\bot',
    '\\bot': '\\perp' if '\\perp' in VOCAB else '.',
    '\\top': 'T',
    '\\dagger': '+',
    '\\ddagger': '+',
    '\\setminus': '-',
    '\\backslash': '-',
    '\\lnot': '\\neg' if '\\neg' in VOCAB else None,
    '\\wedge': '.',
    '\\vee': '.',
    '\\land': '.',
    '\\lor': '.',
}

# Letters not in PosFormer vocab → map to lowercase or closest
LETTER_MAP = {
    'D': 'd', 'J': 'j', 'K': 'k', 'O': 'o', 'Q': 'q',
    'U': 'u', 'W': 'w', 'Z': 'z',
}


def init_vocab():
    global VOCAB, COMMANDS, SINGLES
    with open(POSFORMER_DICT) as f:
        for line in f:
            t = line.strip()
            if t:
                VOCAB.add(t)
                if t.startswith('\\'):
                    COMMANDS.add(t)
                else:
                    SINGLES.add(t)


def tokenize_latex(latex: str) -> Optional[List[str]]:
    """Parse raw LaTeX into PosFormer tokens with fuzzy mapping."""
    if not VOCAB:
        init_vocab()

    tokens = []
    i = 0
    s = latex.strip()

    while i < len(s):
        # Skip whitespace
        if s[i].isspace():
            i += 1
            continue

        # LaTeX command
        if s[i] == '\\':
            # Check for \begin{env}...\end{env}
            begin_m = re.match(r'\\begin\{(\w+)\}', s[i:])
            if begin_m:
                env = begin_m.group(1)
                i += len(begin_m.group(0))
                # Find matching \end{env}
                end_pat = f'\\end{{{env}}}'
                end_idx = s.find(end_pat, i)
                if end_idx >= 0:
                    inner = s[i:end_idx]
                    i = end_idx + len(end_pat)
                    # Flatten matrix content: & → , and \\ → ,
                    inner = inner.replace('&', ' , ')
                    inner = inner.replace('\\\\', ' , ')
                    # Recursively tokenize the flattened content
                    inner_tokens = tokenize_latex(inner)
                    if inner_tokens:
                        tokens.extend(inner_tokens)
                else:
                    i += len(begin_m.group(0))  # no matching end, skip
                continue

            end_m = re.match(r'\\end\{(\w+)\}', s[i:])
            if end_m:
                i += len(end_m.group(0))
                continue

            # Match command name
            m = re.match(r'\\[a-zA-Z]+', s[i:])
            if m:
                cmd = m.group(0)
                i += len(cmd)

                if cmd in COMMANDS:
                    tokens.append(cmd)
                elif cmd in COMMAND_MAP:
                    mapped = COMMAND_MAP[cmd]
                    if mapped is not None:
                        if mapped in COMMANDS or mapped in SINGLES:
                            tokens.append(mapped)
                        else:
                            tokens.append(mapped)
                    # For decorations (\hat, \tilde etc.): we dropped the
                    # command but keep the braced content — { } parse normally.
                    # For \dot{x} → x' (add prime)
                    if cmd in ('\\dot', '\\ddot'):
                        # Parse the {content} then append prime
                        # The { will be parsed normally, after it closes
                        # we'll need to inject prime. For simplicity,
                        # just let { } parse and don't add prime.
                        pass
                else:
                    # Unknown command — skip command, keep braced content
                    pass

            elif i + 1 < len(s):
                # \{ \} \\ \, etc
                pair = s[i:i+2]
                if pair in VOCAB:
                    tokens.append(pair)
                    i += 2
                elif pair == '\\\\':
                    i += 2  # LaTeX newline — skip
                elif pair == '\\,':
                    i += 2  # thin space — skip
                elif pair == '\\;':
                    i += 2
                elif pair == '\\!':
                    i += 2
                elif pair == '\\:':
                    i += 2
                else:
                    i += 2  # skip unknown \x
            else:
                i += 1  # trailing backslash

        # \begin{...} / \end{...} — skip environments
        # (already handled above since \begin is a command)

        # Known single character
        elif s[i] in SINGLES:
            tokens.append(s[i])
            i += 1

        # Unknown letter → map
        elif s[i].isalpha() and s[i] in LETTER_MAP:
            tokens.append(LETTER_MAP[s[i]])
            i += 1

        # Unknown character → skip
        elif s[i] in (':', ';', '&', '#', '~', '@', '`', '"', "'",
                       '?', '*', '\\', '\n', '\r', '\t'):
            i += 1  # skip

        else:
            i += 1  # skip any other unknown char

    return tokens if tokens else None


def retokenize_label(raw_label: str) -> Optional[str]:
    """Convert raw LaTeX to PosFormer space-separated tokens.

    Returns the retokenized string, or None if nothing survived.
    """
    tokens = tokenize_latex(raw_label)
    if tokens is None or len(tokens) < 2:
        return None
    return ' '.join(tokens)


if __name__ == "__main__":
    import xml.etree.ElementTree as ET
    from collections import Counter

    init_vocab()
    print(f"PosFormer vocab: {len(VOCAB)} tokens")
    print(f"  Commands: {len(COMMANDS)}, Singles: {len(SINGLES)}")
    print(f"  Command mappings: {len(COMMAND_MAP)}")
    print()

    # Test on MathWriting excerpt
    excerpt = Path("/mnt/volume1/mathwriting/mathwriting-2024-excerpt")
    total = 0
    compatible = 0
    short = 0
    examples = []

    for split in ['train', 'valid', 'test']:
        split_dir = excerpt / split
        if not split_dir.exists():
            continue
        for inkml in sorted(split_dir.glob('*.inkml')):
            tree = ET.parse(inkml)
            label = None
            for ann in tree.getroot().iter():
                if ann.tag.endswith('annotation'):
                    if ann.get('type') == 'normalizedLabel':
                        label = ann.text
                        break
                    elif ann.get('type') == 'label' and label is None:
                        label = ann.text
            if not label:
                continue

            total += 1
            result = retokenize_label(label)
            if result:
                compatible += 1
                if len(examples) < 15:
                    examples.append((label[:60], result[:60]))
            else:
                short += 1

    print(f"Results: {compatible}/{total} compatible ({100*compatible/max(total,1):.1f}%)")
    print(f"Rejected (too short/empty): {short}")
    print()
    print("Examples:")
    for raw, tok in examples:
        print(f"  {raw}")
        print(f"  → {tok}")
        print()
