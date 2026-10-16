"""Stream-hash reviewed registry files at immutable revisions on hosted runners."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_model_hashes as pins


def main():
    repo = Path(__file__).resolve().parents[2]
    discovery = repo / 'tests/results/registry-pin-audit/discovery.json'
    raw = discovery.read_bytes(); audit = json.loads(raw)
    if (audit['unreachable'] or audit['unpinned'] or audit['removed']
            or hashlib.sha256(pins.REGISTRY_CPP.read_bytes()).hexdigest() != audit['registry_sha256']
            or hashlib.sha256(pins.HEADER_OUT.read_bytes()).hexdigest() != audit['header_sha256']):
        raise ValueError('Incomplete discovery or registry/header changed before byte verification')
    rows = audit['added'] + audit['changed']
    if not rows or len({row['url'] for row in rows}) != len(rows):
        raise ValueError('Missing/repeated reviewed registry candidates')
    report = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                  discovery_sha256=hashlib.sha256(raw).hexdigest(), files=[], result='not completed',
                  scope='Integrity-only stream hashing of existing registry artifacts; no model loading, inference, license promotion or replacement-byte approval.')
    target = Path(os.environ['HEAVY_OUT']) / 'registry-pin-verified.json'
    try:
        for row in rows:
            changed = row['old_sha256'] is not None
            revision = row.get('trusted_revision') if changed else row['current_revision']
            digest = row['old_sha256'] if changed else row['current_sha256']
            if not re.fullmatch(r'[0-9a-f]{40}', revision or '') or not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError('Missing immutable revision or expected digest')
            parts = pins.URL_RE.fullmatch(row['url'])
            if not parts or parts['path'] != row['file'] or parts['owner'] + '/' + parts['repo'] != row['model']:
                raise ValueError('Changed reviewed URL schema')
            url = row['url'].replace('/resolve/' + parts['rev'] + '/', '/resolve/' + revision + '/')
            entry = pins.fetch_tree(parts['owner'], parts['repo'], revision).get(row['file'])
            if not entry or entry[0] != digest or not 0 < entry[1] <= 8 * 1024**3:
                raise ValueError('Immutable LFS hash/size does not match reviewed candidate')
            expected_size = entry[1]
            hasher = hashlib.sha256(); size = 0
            request = urllib.request.Request(url + '?download=true', headers={'User-Agent': 'crispembed-registry-byte-audit'})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 200:
                    raise ValueError('Incomplete weight response')
                for chunk in iter(lambda: response.read(4 * 1024**2), b''):
                    size += len(chunk)
                    if size > expected_size: raise ValueError('Download exceeds immutable LFS size')
                    hasher.update(chunk)
            actual = hasher.hexdigest()
            if size != expected_size or actual != digest:
                raise ValueError('Downloaded bytes differ from reviewed immutable LFS identity')
            result = dict(previous_url=row['url'], immutable_url=url, sha256=actual, bytes=size,
                          retained_previous_digest=changed, revision=revision)
            report['files'].append(result)
            target.write_text(json.dumps(report, indent=2) + '\n')
            print(row['model'], row['file'], size, actual, 'verified', flush=True)
        report['result'] = 'all reviewed immutable files independently stream-hash verified; no replacement bytes adopted'
    finally:
        target.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()
