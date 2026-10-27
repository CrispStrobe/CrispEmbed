#!/usr/bin/env python3
"""Hosted CPU archive audit only: never install, import or execute any wheel."""
import argparse
import csv
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler, Request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'docs/josev2b-wheel-metadata-protocol-v1.json'
SOURCES = ('tools/ci-heavy/josev2b_wheel_metadata.py',
           'docs/josev2b-wheel-metadata-protocol-v1.json',
           'tests/test_josev2b_wheel_metadata.py', '.github/workflows/josev2b-wheel-metadata.yml')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def metadata_fields(data):
    message = BytesParser().parsebytes(data)
    if message.defects or len(message.get_all('Name', [])) != 1 or len(message.get_all('Version', [])) != 1:
        raise ValueError('Malformed/ambiguous core metadata identity')
    return {'name': canonical(message['Name']), 'version': message['Version'],
            'requires_python': message.get('Requires-Python') or '',
            'requires_dist': sorted(message.get_all('Requires-Dist') or []),
            'provides_extra': sorted(message.get_all('Provides-Extra') or [])}


def archive_metadata(path, row, sidecar, limits):
    """Inspect all candidate members, but authenticate exactly one top-level owner."""
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate wheel ZIP member')
        candidates = [i for i in infos if PurePosixPath(i.filename).name == 'METADATA'
                      and PurePosixPath(i.filename).parent.name.endswith('.dist-info')]
        if (sum(i.file_size for i in candidates) > limits['max_metadata_bytes_per_wheel'] or
                any(i.file_size > limits['max_metadata_bytes'] for i in candidates)):
            raise ValueError('Wheel core metadata extent exceeds bound')
        inventory = []
        owner, owner_path = None, None
        for info in candidates:
            parts = PurePosixPath(info.filename).parts
            if info.filename.startswith('/') or '..' in parts or '\\' in info.filename:
                raise ValueError('Unsafe wheel metadata path')
            data = z.read(info)
            fields = metadata_fields(data)
            inventory.append({'path': info.filename, 'bytes': len(data), 'sha256': sha(data),
                              'name': fields['name'], 'version': fields['version'], 'top_level': len(parts) == 2})
            if len(parts) == 2:
                if owner is not None:
                    raise ValueError('Multiple top-level owner METADATA files')
                directory_name, directory_version = parts[0][:-len('.dist-info')].rsplit('-', 1)
                if (canonical(directory_name) != row['name'] or directory_version != row['version'] or
                        fields['name'] != row['name'] or fields['version'] != row['version']):
                    raise ValueError('Top-level owner METADATA identity differs')
                owner = data
                owner_path = info.filename
        if owner is None:
            raise ValueError('Top-level owner METADATA missing')
        record_path = str(PurePosixPath(owner_path).with_name('RECORD'))
        if z.getinfo(record_path).file_size > limits['max_metadata_bytes_per_wheel']:
            raise ValueError('Wheel RECORD extent exceeds bound')
        record_bytes = z.read(record_path)
        record = list(csv.reader(io.StringIO(record_bytes.decode('utf-8'))))
        record_names = [r[0] for r in record if len(r) == 3]
        if len(record_names) != len(record) or len(set(record_names)) != len(record_names):
            raise ValueError('Malformed/duplicate wheel RECORD path')
        recorded_candidates = [name for name in record_names if PurePosixPath(name).name == 'METADATA'
                               and PurePosixPath(name).parent.name.endswith('.dist-info')]
        if set(recorded_candidates) != {r['path'] for r in inventory}:
            raise ValueError('Archive METADATA coverage differs from wheel RECORD')
    actual = metadata_fields(owner)
    published = metadata_fields(sidecar) if sidecar is not None else None
    expected_fields = {'name': row['name'], 'version': row['version'],
                       'requires_python': row['requires_python'], 'requires_dist': sorted(row['requires_dist'])}
    field_equal = {key: published is not None and actual[key] == published[key] == value
                   for key, value in expected_fields.items()}
    field_equal['provides_extra'] = published is not None and actual['provides_extra'] == published['provides_extra']
    result = {'candidate_inventory': inventory, 'wheel_RECORD_sha256': sha(record_bytes),
              'wheel_RECORD_bytes': len(record_bytes), 'wheel_RECORD_entries': len(record_names),
              'legacy_selector_candidate_count': len(recorded_candidates),
              'legacy_selector_unique': len(recorded_candidates) == 1,
              'owner_metadata_bytes': len(owner), 'owner_metadata_sha256': sha(owner),
              'locked_metadata_sha256': row['core_metadata_sha256'],
              'sidecar_metadata_bytes': len(sidecar) if sidecar is not None else None,
              'sidecar_metadata_sha256': sha(sidecar) if sidecar is not None else None,
              'sidecar_matches_lock': sidecar is not None and sha(sidecar) == row['core_metadata_sha256'] and len(sidecar) == row['core_metadata_bytes'],
              'owner_matches_lock': sha(owner) == row['core_metadata_sha256'] and len(owner) == row['core_metadata_bytes'],
              'owner_equals_sidecar': owner == sidecar, 'dependency_fields_exact': field_equal,
              'owner_fields': actual, 'sidecar_fields': published,
              'dependency_semantics': 'identical declaration strings' if all(field_equal.values()) else
                  'different declarations retained; no semantic equivalence or acceptance inferred'}
    result['passed'] = (result['sidecar_matches_lock'] and result['owner_matches_lock'] and
                        result['owner_equals_sidecar'] and all(field_equal.values()))
    return result, owner


def request_url(url, row, sidecar):
    expected = row['url']+('.metadata' if sidecar else '')
    u = urlsplit(url)
    if (url != expected or u.scheme != 'https' or u.hostname != 'files.pythonhosted.org' or
            u.username or u.password or u.port not in (None, 443) or u.query or u.fragment or
            not u.path.startswith('/packages/') or u.path.rsplit('/', 1)[-1] != row['filename']+('.metadata' if sidecar else '')):
        raise ValueError('Unpinned payload/sidecar URL rejected before transport')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Unexpected locked wheel/metadata redirect')


def transfer(row, target, *, sidecar, deadline, limits):
    url = row['url']+('.metadata' if sidecar else '')
    request_url(url, row, sidecar)
    bound = limits['max_metadata_bytes'] if sidecar else row['bytes']
    start = time.monotonic()
    total, h = 0, hashlib.sha256()
    with build_opener(NoRedirect).open(Request(url, headers={'User-Agent': 'CrispEmbed-wheel-metadata-audit-v1'}), timeout=60) as src, target.open('xb') as out:
        length = src.headers.get('Content-Length')
        if length and (int(length) > bound or not sidecar and int(length) != row['bytes']):
            raise ValueError('Published transfer size differs from exact bound')
        while True:
            if time.monotonic() > min(deadline, start+limits['max_wheel_seconds']):
                raise ValueError('Bounded wheel/metadata transfer deadline exceeded')
            chunk = src.read(1024*1024)
            if not chunk:
                break
            total += len(chunk)
            if total > bound:
                raise ValueError('Wheel/metadata response exceeds exact bound')
            h.update(chunk)
            out.write(chunk)
    if not sidecar and (total != row['bytes'] or h.hexdigest() != row['sha256']):
        raise ValueError('Locked wheel payload hash/bytes differ')
    return {'bytes': total, 'sha256': h.hexdigest(), 'seconds': time.monotonic()-start}


def source():
    cfg = json.loads(PROTOCOL.read_text())
    for name, expected in cfg['old_source_files'].items():
        old = subprocess.check_output(['git', 'show', cfg['source']+':'+name], cwd=ROOT)
        if sha(old) != expected:
            raise ValueError('Historical worker source/lock differs: '+name)
        if name != 'tools/josev2b_environment.py' and sha((ROOT/name).read_bytes()) != expected:
            raise ValueError('Frozen environment lock/contract/requirements changed: '+name)
    paths = [*SOURCES, *cfg['old_source_files']]
    if subprocess.check_output(['git', 'status', '--porcelain', '--', *paths], cwd=ROOT):
        raise ValueError('Dirty/untracked measured wheel-audit source')
    lock = json.loads((ROOT/'docs/josev2b-environment-lock.json').read_text())
    if len(lock['packages']) != cfg['package_count'] or sum(p['bytes'] for p in lock['packages']) != cfg['total_wheel_bytes']:
        raise ValueError('Original57-wheel coverage or resource bound changed')
    return cfg, lock, {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'files': {p: sha((ROOT/p).read_bytes()) for p in paths}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        p.error('All wheel payload transfers belong on hosted CPU CI')
    if a.output.exists():
        p.error('Fresh output required; preserve previous audit')
    a.output.mkdir(parents=True)
    report = {'schema': 'crispembed-josev-wheel-metadata-audit-v1', 'completed': False, 'passed': False,
              'framework_imported': False, 'model_tokenizer_accessed': False, 'wheel_installed_or_executed': False,
              'packages': [], 'errors': []}
    def save():
        (a.output/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    try:
        cfg, lock, report['source'] = source()
        report['protocol_sha256'] = sha(PROTOCOL.read_bytes())
        deadline = time.monotonic()+cfg['max_total_seconds']
        for row in lock['packages']:
            result = {'name': row['name'], 'version': row['version'], 'filename': row['filename'], 'passed': False}
            report['packages'].append(result)
            try:
                if time.monotonic() > deadline:
                    raise ValueError('Total audit deadline exceeded')
                # Exactly one wheel is retained in task scratch at a time. The
                # temporary directory is removed even after a package rejects.
                with tempfile.TemporaryDirectory(prefix='crisp-josev-wheel-') as scratch:
                    scratch = Path(scratch)
                    if shutil.disk_usage(scratch).free < row['bytes']+512*1024**2:
                        raise ValueError('Insufficient bounded single-wheel scratch')
                    sidecar = a.output/(row['name']+'.sidecar.metadata')
                    sidecar_bytes = None
                    try:
                        result['sidecar_transfer'] = transfer(row, sidecar, sidecar=True, deadline=deadline, limits=cfg)
                        sidecar_bytes = sidecar.read_bytes()
                    except Exception as exc:
                        # Still inspect the authenticated wheel when a live
                        # sidecar is unavailable; retain both distinct outcomes.
                        result['sidecar_error'] = type(exc).__name__+': '+str(exc)
                    wheel = scratch/row['filename']
                    result['wheel_transfer'] = transfer(row, wheel, sidecar=False, deadline=deadline, limits=cfg)
                    inspected, owner = archive_metadata(wheel, row, sidecar_bytes, cfg)
                    result.update(inspected)
                    (a.output/(row['name']+'.owner.metadata')).write_bytes(owner)
            except Exception as exc:
                result['error'] = type(exc).__name__+': '+str(exc)
            if not result['passed']:
                report['errors'].append({'name': row['name'], 'error': result.get('error', 'Exact owner/sidecar/dependency identity rejected')})
            save()
            print(json.dumps({'name': row['name'], 'passed': result['passed'],
                'legacy_candidates': result.get('legacy_selector_candidate_count'), 'error': result.get('error')}), flush=True)
        report['completed'] = len(report['packages']) == cfg['package_count']
        report['passed'] = report['completed'] and not report['errors']
    except Exception as exc:
        report['errors'].append({'phase': 'source/preflight', 'error': type(exc).__name__+': '+str(exc)})
    finally:
        report['files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
                           for p in a.output.iterdir() if p.is_file() and p.name != 'audit.json'}
        save()
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
