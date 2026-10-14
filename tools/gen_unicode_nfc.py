#!/usr/bin/env python3
"""Generate canonical NFC data; Unicode 9.0, Unicode license in core/third_party/unicode."""
import hashlib
import urllib.request


def generate():
    data = {}
    for name in ['UnicodeData.txt', 'DerivedNormalizationProps.txt']:
        url = 'https://www.unicode.org/Public/9.0.0/ucd/' + name
        with urllib.request.urlopen(url, timeout=30) as response:
            data[name] = response.read()
        expected = {'UnicodeData.txt': '68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f',
                    'DerivedNormalizationProps.txt': '02d8e12cea7c61a9f3cc5fbf8eaceadf8da23fe83d60c64cf647088830b810cc'}
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
    print('// Generated Unicode 9.0 canonical data; license: core/third_party/unicode/LICENSE.')
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
