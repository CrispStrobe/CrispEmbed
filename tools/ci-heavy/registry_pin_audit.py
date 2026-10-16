"""Read-only registry drift inventory; never approves replacement model bytes."""
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


def api(path):
    request = urllib.request.Request('https://huggingface.co/api/models/' + path,
                                     headers={'User-Agent': 'crispembed-registry-audit'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    repo = Path(__file__).resolve().parents[2]
    out = Path(os.environ['HEAVY_OUT'])
    urls = pins.registry_urls(pins.REGISTRY_CPP)
    current, missing, unreachable = pins.resolve(urls)
    old = dict(re.findall(r'\{ "(https://[^"\n]+)",\s*"([0-9a-f]{64})" \}', pins.HEADER_OUT.read_text()))
    report = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                  registry_sha256=hashlib.sha256(pins.REGISTRY_CPP.read_bytes()).hexdigest(),
                  header_sha256=hashlib.sha256(pins.HEADER_OUT.read_bytes()).hexdigest(),
                  registered=len(urls), resolved=len(current), unreachable=[list(k) for k in sorted(unreachable)],
                  unpinned=missing, added=[], changed=[], removed=[url for url in old if url not in urls],
                  policy='Read-only discovery. Changed bytes are not approved. Historical matching LFS metadata is not independent downloaded-byte verification.')
    try:
        if missing or unreachable: raise ValueError('Incomplete registry inventory')
        for url, digest in current.items():
            if old.get(url) == digest: continue
            parts = urls[url]; model = parts['owner'] + '/' + parts['repo']
            info = api(model + '/revision/' + parts['rev'])
            row = dict(url=url, old_sha256=old.get(url), current_sha256=digest,
                       current_revision=info['sha'], file=parts['path'], model=model)
            if url not in old:
                report['added'].append(row)
            else:
                report['changed'].append(row)
                commits = api(model + '/commits/' + parts['rev'])
                row['searched_commits'] = 0
                for commit in commits[:80]:
                    revision = commit['id']; row['searched_commits'] += 1
                    tree = pins.fetch_tree(parts['owner'], parts['repo'], revision)
                    entry = tree.get(parts['path'])
                    if entry and entry[0] == old[url]:
                        row.update(trusted_revision=revision, trusted_bytes=entry[1],
                                   trusted_url=url.replace('/resolve/' + parts['rev'] + '/', '/resolve/' + revision + '/'))
                        break
            (out / 'registry-pin-audit.json').write_text(json.dumps(report, indent=2) + '\n')
        report['result'] = 'complete metadata inventory; retain old trusted bytes and verify immutable download hashes before changing registry'
    finally:
        (out / 'registry-pin-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
