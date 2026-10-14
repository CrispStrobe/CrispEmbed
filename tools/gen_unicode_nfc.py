#!/usr/bin/env python3
"""Generate canonical NFC data; Unicode 16.0, Unicode license in core/third_party/unicode."""
import hashlib
import urllib.request


def generate():
    data = {}
    for name in ['UnicodeData.txt', 'DerivedNormalizationProps.txt']:
        url = 'https://www.unicode.org/Public/16.0.0/ucd/' + name
        with urllib.request.urlopen(url, timeout=30) as response:
            data[name] = response.read()
        expected = {'UnicodeData.txt': 'ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f',
                    'DerivedNormalizationProps.txt': '4d4c03892dea9146d674b686e495df2d55a28d071ac474041d73518f887abddc'}
        assert hashlib.sha256(data[name]).hexdigest() == expected[name]
    excluded = set()
    for line in data['DerivedNormalizationProps.txt'].decode().splitlines():
        line = line.split('#')[0].strip()
        if not line:
            continue
        fields = [x.strip() for x in line.split(';')]
        if fields[1] == 'Full_Composition_Exclusion':
            bounds = [int(x, 16) for x in fields[0].split('..')]
            excluded.update(range(bounds[0], bounds[-1] + 1))
    decompositions, classes, compositions = [], [], []
    for line in data['UnicodeData.txt'].decode().splitlines():
        fields = line.split(';')
        cp, cls = int(fields[0], 16), int(fields[3])
        if cls:
            classes.append((cp, cls))
        if fields[5] and not fields[5].startswith('<'):
            sequence = [int(x, 16) for x in fields[5].split()]
            assert len(sequence) in (1, 2)
            decompositions.append((cp, sequence[0], sequence[1] if len(sequence) == 2 else 0))
            if len(sequence) == 2 and cp not in excluded:
                compositions.append(((sequence[0] << 21) | sequence[1], cp))
    print('#pragma once\n#include <cstdint>\nnamespace core_nfc {')
    print('// Generated Unicode 16.0 canonical data; license: core/third_party/unicode/LICENSE.')
    for name, contents in data.items():
        print('// ' + name + ' SHA256 ' + hashlib.sha256(contents).hexdigest())
    print('struct decomposition { uint32_t cp, first, second; };\ninline constexpr decomposition decompositions[] = {')
    for row in decompositions:
        print('    {' + ','.join(hex(x) for x in row) + '},')
    print('};\nstruct combining { uint32_t cp; unsigned value; };\ninline constexpr combining classes[] = {')
    for row in classes:
        print('    {' + ','.join(hex(x) for x in row) + '},')
    print('};\nstruct composition { uint64_t pair; uint32_t cp; };\ninline constexpr composition compositions[] = {')
    for key, cp in sorted(compositions):
        print('    {' + hex(key) + 'ULL,' + hex(cp) + '},')
    print('};\n} // namespace core_nfc')


if __name__ == '__main__':
    generate()
