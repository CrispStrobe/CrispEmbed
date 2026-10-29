"""Exact v3 SDK GET delivery on hosted CPU; URLs and credentials stay in memory."""
import argparse
import concurrent.futures
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).parent))
import contract as c
import retained_v3_gha as audit

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'docs/unlimited-retained-gha-fetch-v1.json'
SCRIPT_BYTES = 7899
SCRIPT_SHA = 'dd11f6e7b530489fa3bcf09b858051b47230cd3f77871b7c7beafe74ec396b6e'
PREFIX = 'unlimited-q4k/panel/'
SLUG = 'crispembed-unlimited-ocr2-q4k'


class NoArrays:
    """Block optional SDK dependency imports until the full byte barrier passes."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in ('numpy', 'torch', 'tensorflow', 'transformers'):
            raise ImportError('Array/framework import blocked before retained byte authentication')
        return None


def source_guard(source):
    c.require(re.fullmatch('[0-9a-f]{40}', source or '') and c.git('rev-parse', 'HEAD') == source,
              'Exact reviewed checkout required')
    cfg = json.loads(PROFILE.read_text())
    c.require(cfg['original_forwards'] == cfg['native_forwards'] == cfg['quantizer_runs'] == 0,
              'Read-only scope required')
    for name, pin in cfg['protected_sha256'].items():
        c.require(c.digest(ROOT / name) == pin and
                  hashlib.sha256(subprocess.check_output(['git', 'show', cfg['base'] + ':' + name], cwd=ROOT)).hexdigest() == pin and
                  hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)).hexdigest() == pin,
                  'Protected audit source differs: ' + name)
    for name, pin in cfg['owned_sha256'].items():
        c.require(c.digest(ROOT / name) == pin and
                  hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)).hexdigest() == pin,
                  'Owned transfer source differs: ' + name)
    c.require(not c.git('status', '--porcelain', '--untracked-files=no'), 'Dirty tracked source')
    allowed = set(cfg['owned_sha256']) | {str(PROFILE.relative_to(ROOT))}
    c.require(set(c.git('diff', '--name-only', cfg['base'], 'HEAD').splitlines()) <= allowed,
              'Undeclared retained-transfer source delta')
    return cfg, c.guard()


def safe_name(name):
    rel = PurePosixPath(name)
    c.require(isinstance(name, str) and name and not rel.is_absolute() and '..' not in rel.parts and
              '\\' not in name and str(rel) == name, 'Safe relative producer member required')
    return rel


def download(url, path, pin, *, opener=urllib.request.urlopen):
    """No HEAD: the service supports GET/Range but returns404 for HEAD.

    Large files use exact64MiB ranges, bounded4MiB buffers and whole-file SHA.
    A bad response never becomes an admitted file. No signed URL is serialized.
    """
    path = Path(path)
    c.require(type(pin['bytes']) is int and 0 < pin['bytes'] <= 8 * 2**30 and
              re.fullmatch('[0-9a-f]{64}', pin['sha256']), 'Explicit retained byte/SHA pin required')
    partial = path.with_name(path.name + '.partial')
    c.require(not path.exists() and not partial.exists() and not path.is_symlink() and not partial.is_symlink(),
              'Fresh task-owned transfer destination required')
    path.parent.mkdir(parents=True, exist_ok=True)
    size, h, total = pin['bytes'], hashlib.sha256(), 0
    ranged = size >= 2**30
    try:
        with partial.open('xb') as out:
            for start in range(0, size, 64 * 2**20 if ranged else size):
                end = min(size - 1, start + 64 * 2**20 - 1) if ranged else size - 1
                headers = {'Accept-Encoding': 'identity'}
                if ranged:
                    headers['Range'] = f'bytes={start}-{end}'
                request = urllib.request.Request(url, headers=headers)
                with opener(request, timeout=180) as response:
                    expected = end - start + 1
                    c.require(response.status == (206 if ranged else 200), 'GET status differs')
                    if ranged:
                        c.require(response.headers.get('Content-Range') == f'bytes {start}-{end}/{size}',
                                  'Exact GET Content-Range required')
                    c.require(int(response.headers.get('Content-Length', '-1')) == expected and
                              response.headers.get('Content-Encoding', 'identity') == 'identity',
                              'Exact unencoded response extent required')
                    left = expected
                    while left:
                        block = response.read(min(left, 4 * 2**20))
                        c.require(block and len(block) <= left, 'Truncated GET response')
                        out.write(block)
                        h.update(block)
                        total += len(block)
                        left -= len(block)
                    c.require(not response.read(1), 'Excess GET response')
            out.flush()
            os.fsync(out.fileno())
        c.require(total == size and h.hexdigest() == pin['sha256'], 'Whole retained file SHA/extent differs')
        partial.replace(path)
        return dict(bytes=total, sha256=h.hexdigest(), ranged_get=ranged)
    except Exception:
        raise ValueError('Retained GET rejected; unadmitted partial preserved') from None


def output_urls(api, owner):
    # Kaggle's package exports `api` as an authenticated instance, shadowing
    # its subpackage. Resolve the exact module rather than parent attributes.
    sdk = importlib.import_module('kaggle.api.kaggle_api_extended')
    def request(cls):
        value = cls()
        value.user_name = owner
        value.kernel_slug = SLUG
        value.version_label = 'v3'
        c.require(json.loads(str(value))['versionLabel'] == 'v3', 'Serialized exact versionLabel required')
        return value
    urls, pages, logs = {}, 0, set()
    with api.build_kaggle_client() as client:
        kernel = client.kernels.kernels_api_client
        original = json.loads(str(kernel.get_kernel(request(sdk.ApiGetKernelRequest))))
        script = original['blob']['source'].encode()
        c.require(len(script) == SCRIPT_BYTES and hashlib.sha256(script).hexdigest() == SCRIPT_SHA and
                  audit.PRODUCER_SOURCE.encode() in script, 'Immutable submitted v3 script differs')
        state = str(kernel.get_kernel_session_status(request(sdk.ApiGetKernelSessionStatusRequest)).status).split('.')[-1]
        c.require(state == 'ERROR', 'Exact stopped-v3 terminal state required')
        token = None
        while True:
            q = request(sdk.ApiListKernelSessionOutputRequest)
            q.page_size = 100
            if token:
                q.page_token = token
            page = kernel.list_kernel_session_output(q)
            pages += 1
            c.require(pages <= 20, 'Bounded complete SDK pagination required')
            for item in page.files or []:
                name = item.file_name
                safe_name(name)
                c.require(name not in urls and item.url.startswith('https://'), 'Unique HTTPS output members required')
                urls[name] = item.url
            next_token = page.next_page_token
            if not next_token:
                break
            c.require(next_token not in logs, 'Repeated output page token')
            logs.add(next_token)
            token = next_token
    return urls, dict(version_label='v3', state=state, pages=pages, listed_members=len(urls),
                      producer_source=audit.PRODUCER_SOURCE, submitted_script_bytes=SCRIPT_BYTES,
                      submitted_script_sha256=SCRIPT_SHA)


def main(args):
    source_guard(args.source)
    c.require(os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_SHA') == args.source,
              'Genuine source-bound hosted transfer required')
    available = int(next(r.split()[1] for r in Path('/proc/meminfo').read_text().splitlines()
                         if r.startswith('MemAvailable:'))) * 1024
    c.require(available >= 10 * 2**30 and shutil.disk_usage(args.scratch.parent).free >= 16 * 2**30,
              '10GiB RAM/16GiB free hosted scratch required before SDK/payload access')
    args.scratch.mkdir(parents=True, exist_ok=False)
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='crispembed-unlimited-retained-gha-fetch-v1', source=args.source, completed=False,
                  original_forwards=0, native_forwards=0, quantizer_runs=0, publication_qualified=False,
                  protocol_sha256=c.digest(PROFILE), phase='service-admission', files={})
    def save():
        path = args.out / 'transfer.json'
        temporary = path.with_name(path.name + '.partial')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    save()
    c.require(not any(k in sys.modules for k in ('numpy', 'torch', 'tensorflow', 'transformers')),
              'Array/framework imported before transfer admission')
    block_arrays = NoArrays()
    sys.meta_path.insert(0, block_arrays)
    try:
        c.require(os.environ.get('KAGGLE_ACCOUNT') and os.environ.get('KAGGLE_API_TOKEN'),
                  'Existing private read-service configuration required')
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        owner = os.environ.get('KAGGLE_ACCOUNT', '')
        c.require(owner, 'Configured private service account required')
        urls, result['service'] = output_urls(api, owner)
        result['phase'] = 'authenticate-panel'
        save()
        root = args.scratch / 'panel'
        path = root / 'panel.json'
        key = PREFIX + 'panel.json'
        c.require(key in urls, 'Exact producer panel output missing')
        result['panel'] = download(urls[key], path, dict(bytes=audit.PANEL_BYTES, sha256=audit.PANEL_SHA))
        producer = json.loads(path.read_text())
        c.require(producer['source'] == audit.PRODUCER_SOURCE and producer['completed'] is False and
                  producer['f16_all_four_strict_control_passed'] is True and not producer['candidates'] and
                  producer['error'] == 'Protected source type/shape/payload differs: v.pos_embed',
                  'Exact immutable stopped producer required')
        files = producer['files']
        c.require(len(files) == 540 and sum(p['bytes'] for p in files.values()) == 7901314457,
                  'Exact full partial-output inventory required')
        for name in files:
            safe_name(name)
            c.require(PREFIX + name in urls, 'Producer member absent from exact-v3 listing')
        def fetch(item):
            name, pin = item
            rel = safe_name(name)
            try:
                return name, download(urls[PREFIX + name], root.joinpath(*rel.parts), pin)
            except Exception:
                raise ValueError('Retained member GET rejected: ' + name) from None
        # Two IO workers only; no CPU/GPU inference, array import or model build.
        result['phase'] = 'authenticate-540-payloads'
        save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for name, proof in pool.map(fetch, files.items()):
                result['files'][name] = proof
                save()
        c.require('numpy' not in sys.modules, 'Array package imported before full payload authentication')
        result['all_540_payloads_authenticated_before_array_import'] = True
        result['phase'] = 'read-only-structural-audit'
        save()
        sys.meta_path.remove(block_arrays)
        proof = audit.audit(args.scratch, args.out, args.source)
        (args.out / 'audit.json').write_text(json.dumps(proof, indent=2, allow_nan=False) + '\n')
        result.update(completed=True, phase='read-only-audit-complete', structural_audit_completed=True,
                      limits='Read-only structural result. No Q4 forward or numerical/decoded qualification.')
        save()
    except Exception as error:
        result['error'] = 'Retained delivery/audit rejected: ' + type(error).__name__
        if str(error).startswith('Retained member GET rejected: '):
            result['failed_member'] = str(error).removeprefix('Retained member GET rejected: ')
        save()
        raise ValueError(result['error']) from None
    finally:
        if block_arrays in sys.meta_path:
            sys.meta_path.remove(block_arrays)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--source-only', action='store_true')
    p.add_argument('--scratch', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.source_only:
        cfg, inventory = source_guard(a.source)
        print(json.dumps(dict(protected=len(cfg['protected_sha256']), source_files=len(inventory))))
    else:
        c.require(a.scratch and a.out, 'Explicit hosted scratch/output required')
        main(a)
