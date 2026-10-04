"""Report exact-source native/WASM dry-run artifact hashes without publishing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def manifest(root):
    root = Path(root)
    source = Path('src/posformer_ocr.cpp')
    if b'_probe' in source.read_bytes() or b'POSFORMER_REFERENCE' in source.read_bytes():
        raise ValueError('Release source contains diagnostic instrumentation')
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        row = {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
               'sha256': digest(path)}
        members = None
        if path.name.endswith('.tar.gz'):
            with tarfile.open(path, 'r:gz') as archive:
                entries = archive.getmembers()
                members = [item.name for item in entries if not item.isdir()]
                row['links'] = [{'path': item.name, 'target': item.linkname,
                                 'kind': 'symlink' if item.issym() else 'hardlink'}
                                for item in entries if item.issym() or item.islnk()]
        elif path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                members = [item.filename for item in archive.infolist() if not item.is_dir()]
        if members is not None:
            link_targets = [item['target'] for item in row.get('links', [])]
            if any(name.lower().endswith('.gguf') for name in members + link_targets):
                raise ValueError('Model weights unexpectedly bundled in runtime artifact')
            row['members'] = members
        files.append(row)
    if not files:
        raise ValueError('Empty artifact inventory')
    return {'format': 'crispembed.runtime-artifact-provenance',
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'ggml_commit': subprocess.check_output(['git', 'ls-tree', 'HEAD', 'ggml'], text=True).split()[2],
            'posformer_source_sha256': digest(source),
            'version': Path('VERSION').read_text().strip(), 'production_instrumentation_absent': True,
            'model_weights_packaged': False, 'files': files,
            'limits': ['Hashes describe these actual workflow artifacts; an untagged dry-run does not publish a release.',
                       'Source/version/checksum review and exact tagged-source validation precede app promotion.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(manifest(args.artifact_root), indent=2) + '\n')
