#!/usr/bin/env python3
"""Build the expanded PosFormer v2 dictionary (169 tokens)."""

existing = open('/mnt/volume1/PosFormer-fresh/Pos_Former/datamodule/dictionary.txt').read().strip().split('\n')
print(f'Existing: {len(existing)} tokens')

additions = [
    # Matrix support
    '&', '\\\\', '\\begin', '\\end', 'matrix',
    # Greek (all common)
    '\\partial', '\\delta', '\\nu', '\\omega', '\\xi', '\\epsilon',
    '\\Omega', '\\psi', '\\kappa', '\\rho', '\\tau', '\\eta', '\\chi',
    '\\zeta', '\\varphi', '\\Sigma', '\\Lambda', '\\Phi', '\\Psi',
    '\\Theta', '\\Xi', '\\Upsilon',
    # Operators/relations
    '\\approx', '\\equiv', '\\sim', '\\le', '\\ge', '\\ne',
    '\\nabla', '\\prod', '\\propto', '\\gg',
    # Decorations
    '\\hat', '\\overline', '\\tilde', '\\underline', '\\vec', '\\dot',
    # Delimiters
    '\\langle', '\\rangle', '\\lfloor', '\\rfloor', '\\lceil', '\\rceil',
    # Missing uppercase letters
    'D', 'J', 'K', 'O', 'Q', 'U', 'W', 'Z',
    # Other common
    ':', ';', '*',
    '\\iint', '\\oint',
    '\\oplus', '\\otimes', '\\cup', '\\cap',
    '\\emptyset', '\\notin', '\\subseteq',
    '\\Rightarrow', '\\Leftrightarrow',
    '\\dagger', '\\perp',
]

# Dedupe and remove already-existing
additions = sorted(set(a for a in additions if a not in existing))
new_dict = sorted(existing + additions)

print(f'Additions: {len(additions)} tokens')
print(f'New total: {len(new_dict)} tokens')

with open('/mnt/volume1/dictionary_v2.txt', 'w') as f:
    f.write('\n'.join(new_dict) + '\n')

print(f'Saved to /mnt/volume1/dictionary_v2.txt')
print()
print('Added tokens:')
for t in additions:
    print(f'  + {t}')
