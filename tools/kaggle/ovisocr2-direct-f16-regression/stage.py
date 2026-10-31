"""Stage a source-closed Ovis direct-F16 regression kernel."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PATHS = ('src/qwen35_linear.cpp', 'tools/kaggle/ovisocr2-direct-f16-regression/run.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.commit):
        parser.error('commit must be a full lowercase SHA')
    if args.directory.exists():
        parser.error('stage directory already exists')
    bundle = {}
    for name in PATHS:
        content = (ROOT / name).read_bytes()
        bundle[name] = {'base64': base64.b64encode(content).decode(), 'sha256': sha(content)}
    raw = json.dumps(bundle, sort_keys=True, separators=(',', ':')).encode()
    kernel = (HERE / 'kernel.py').read_text()
    for key, value in (('@COMMIT@', args.commit), ('@SOURCES_B64@', base64.b64encode(raw).decode()),
                       ('@SOURCES_SHA256@', sha(raw))):
        if kernel.count(key) != 1:
            raise ValueError('invalid placeholder ' + key)
        kernel = kernel.replace(key, value)
    compile(kernel, 'kernel.py', 'exec')
    metadata = (HERE / 'kernel-metadata.json').read_bytes()
    json.loads(metadata)
    args.directory.mkdir(parents=True)
    (args.directory / 'kernel.py').write_text(kernel)
    (args.directory / 'kernel-metadata.json').write_bytes(metadata)
    print(json.dumps({'commit': args.commit, 'bundle_sha256': sha(raw),
                      'kernel_sha256': sha(kernel.encode()),
                      'sources': {name: entry['sha256'] for name, entry in bundle.items()}}, indent=2))


if __name__ == '__main__':
    main()
