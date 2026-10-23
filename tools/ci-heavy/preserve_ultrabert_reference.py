#!/usr/bin/env python3
"""Verify and inventory a remote copy of the frozen original reference artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ultrabert_contract import sha256
from ultrabert_native import read_frozen_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    import gguf
    import numpy as np
    baseline = json.loads((ROOT / 'tests/results/ultrabert/reference-audit.json').read_text())
    original = read_frozen_manifest(baseline, args.directory)
    cases = original['reference']['cases']
    if len(cases) != 7 or {x['dtype'] for x in original['conversions']} != {'f32', 'f16'}:
        raise ValueError('Incomplete original case/conversion identities')
    stages = 0
    for case in cases:
        name = case['archive']
        if Path(name).name != name:
            raise ValueError('Invalid archive name')
        path = args.directory / 'reference' / name
        if path.stat().st_size != case['archive_bytes'] or sha256(path) != case['archive_sha256']:
            raise ValueError('Original archive bytes differ: ' + name)
        reader = gguf.GGUFReader(str(path))
        tensors = {t.name: t for t in reader.tensors}
        if set(tensors) != set(case['stages']):
            raise ValueError('Original stage set differs')
        for name, record in case['stages'].items():
            t = tensors[name]
            if (t.tensor_type != gguf.GGMLQuantizationType.F32
                    or list(reversed(t.shape.tolist())) != record['shape']
                    or not np.isfinite(t.data).all()
                    or hashlib.sha256(t.data.tobytes()).hexdigest() != record['payload_sha256']):
                raise ValueError('Original tensor differs: ' + name)
            stages += 1
        del tensors, reader
    if stages != 2730:
        raise ValueError('Incomplete original stage count')
    files = {}
    for path in sorted(args.directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Artifact symlink refused')
        if path.is_file():
            files[path.relative_to(args.directory).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    result = dict(original_source=original['source_commit'], original_run=38997562520,
                  raw_manifest_sha256=baseline['audit_provenance']['raw_report_sha256'],
                  cases=7, stages=stages, conversion_records_exact=True,
                  scope='Byte-exact reference copy; no model inference or regenerated conversions', files=files)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print('Verified seven archives/2730 stage payloads and original conversion records')


if __name__ == '__main__':
    main()
