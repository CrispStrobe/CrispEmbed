"""Standard-library authentication and continuity gates for original observers.

Raw ZIP/payload access is hosted-only. Importing this module loads no framework.
"""
from collections.abc import Mapping
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile

from contract import CONTRACT, PANEL, ROOT, digest

PROTOCOL = ROOT/'docs/moedern-colbert-observer-correction-v1.json'


def protocol():
    return json.loads(PROTOCOL.read_text())


def capture_leaves(stage, value, is_tensor):
    """BatchEncoding/BatchFeature are Mapping, not built-in dict.

    Keep the actual tensor and its dimensions; the caller snapshots immediately
    before another pipeline module mutates the feature mapping.
    """
    if is_tensor(value):
        yield stage, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from capture_leaves(f'{stage}.{key}', item, is_tensor)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from capture_leaves(f'{stage}.{index}', item, is_tensor)


def indexed(rows):
    result = {(row['case'], row['stage']): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate original stage descriptor')
    return result


def pipeline_coverage(report, panel):
    rows = indexed(report['stage_arrays'])
    for case in report['cases']:
        if case['id'] not in panel['full_capture']:
            continue
        for index in (1, 2, 3):
            name = f'pipeline.{index}.output#0.token_embeddings'
            row = rows.get((case['id'], name), {})
            if row.get('shape') != [1, len(case['token_ids']), 128] or row.get('dtype') != 'f32':
                raise ValueError('Required original pipeline coverage missing: '+case['id']+'/'+name)
            if any(cid == case['id'] and stage.startswith(f'pipeline.{index}.output#1') for cid, stage in rows):
                raise ValueError('Repeated original pipeline invocation')
    if {case['id'] for case in report['cases'] if case['id'] in panel['full_capture']} != set(panel['full_capture']):
        raise ValueError('Full-capture case coverage incomplete')


def continuity(current, prior):
    """Every retained old stage is an exact subset, including ordinary outputs.

    No batch-axis rewrite, normalization, float tolerance or recaptured gold.
    """
    for key in ('schema', 'variant', 'revision', 'source_contract_sha256', 'panel_sha256',
                'versions', 'pipeline', 'tokenizer_descriptor', 'checkpoint_files', 'inventory',
                'loaded_tensor_sha256', 'scores'):
        if current.get(key) != prior.get(key):
            raise ValueError('Prior original continuity differs: '+key)
    def cases(value):
        return [{key: item for key, item in row.items() if key != 'seconds_including_observer'}
                for row in value['cases']]
    if cases(current) != cases(prior):
        raise ValueError('Prior original request/ID/mask/output-shape continuity differs')
    new, old = indexed(current['stage_arrays']), indexed(prior['stage_arrays'])
    for key, row in old.items():
        if new.get(key) != row:
            raise ValueError('Prior original array bytes/shape differ: '+'/'.join(key))
    return {'prior_case_count': len(prior['cases']), 'exact_prior_arrays': len(old),
            'exact_scores': len(prior['scores']), 'all_prior_outputs_bit_exact': True}


def validate_prior(directory):
    """Rehash locally retained authenticated ZIPs and every old array on CI."""
    cfg = protocol()
    directory = Path(directory)
    if digest(CONTRACT) != cfg['original_contract_sha256'] or digest(PANEL) != cfg['panel_sha256']:
        raise ValueError('Frozen original contract/panel changed')
    for kind in ('review', 'raw'):
        identity = cfg['prior_'+kind+'_artifact']
        path = directory/(kind+'.zip')
        if path.stat().st_size != identity['bytes'] or digest(path) != identity['sha256']:
            raise ValueError('Prior '+kind+' ZIP identity differs')
    root = directory/'raw/moedern-original'
    report_path = root/'review/original.json'
    if (digest(report_path) != cfg['prior_receipt_sha256'] or
            report_path.read_bytes() != (directory/'review/moedern-original/review/original.json').read_bytes() or
            digest(root/'model-F32.conversion.json') != cfg['prior_conversion_receipt_sha256'] or
            digest(root/'model-F32.gguf') != cfg['prior_gguf_sha256']):
        raise ValueError('Prior report/conversion/GGUF identity differs')
    report = json.loads(report_path.read_text())
    if (report.get('completed') is not True or report.get('github_sha') != cfg['prior_source'] or
            report.get('variant') != cfg['variant'] or len(report['cases']) != cfg['prior_cases'] or
            len(report['stage_arrays']) != cfg['prior_arrays']):
        raise ValueError('Prior scoped original identity/coverage differs')
    for name, expected in report['driver_sources'].items():
        value = subprocess.check_output(['git', 'show', cfg['prior_source']+':'+name], cwd=ROOT)
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError('Prior original source differs')
        # Converter/schema math stays byte-identical; only observers change.
        if name != 'tools/moedern-colbert/reference.py' and digest(ROOT/name) != expected:
            raise ValueError('Original schema/converter source changed')
    indexed(report['stage_arrays'])
    for row in report['stage_arrays']:
        relative = PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in row['path']:
            raise ValueError('Unsafe prior stage path')
        path = root/relative
        width = {'f32': 4, 'i64': 8, 'u8': 1}[row['dtype']]
        if (path.stat().st_size != row['bytes'] or row['bytes'] != math.prod(row['shape'])*width
                or digest(path) != row['sha256']):
            raise ValueError('Prior actual array hash/extent differs')
    return root, report


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', 'repos/CrispStrobe/CrispEmbed/'+path], timeout=120))


def prepare(directory):
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise ValueError('Prior raw artifact access is hosted-only')
    cfg = protocol()
    run = api('actions/runs/'+str(cfg['prior_run']))
    if (run['head_sha'] != cfg['prior_source'] or run['run_attempt'] != 1 or
            run['status'] != 'completed' or run['conclusion'] != 'success'):
        raise ValueError('Prior actual service run differs')
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Fresh prior-artifact directory required')
    directory.mkdir(parents=True)
    service = {'run': {key: run[key] for key in
                      ('id', 'head_sha', 'run_attempt', 'status', 'conclusion', 'html_url', 'path')},
               'artifacts': {}}
    for kind in ('review', 'raw'):
        pin = cfg['prior_'+kind+'_artifact']
        actual = api('actions/artifacts/'+str(pin['id']))
        if (actual['workflow_run']['id'] != cfg['prior_run'] or
                actual['workflow_run']['head_sha'] != cfg['prior_source'] or actual['expired'] or
                actual['name'] != pin['name'] or actual['size_in_bytes'] != pin['bytes'] or
                actual['digest'] != 'sha256:'+pin['sha256']):
            raise ValueError('Prior artifact service lineage differs')
        archive = directory/(kind+'.zip')
        with archive.open('xb') as out:
            subprocess.run(['gh', 'api', 'repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(pin['id'])+'/zip'],
                           stdout=out, check=True, timeout=600)
        if archive.stat().st_size != pin['bytes'] or digest(archive) != pin['sha256']:
            raise ValueError('Prior downloaded ZIP differs')
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            if (len(names) != len(set(names)) or sum(i.file_size for i in z.infolist()) > 256*1024**2 or
                    any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names) or
                    any(stat.S_ISLNK(i.external_attr >> 16) for i in z.infolist())):
                raise ValueError('Unsafe prior artifact members')
            z.extractall(directory/kind)
        service['artifacts'][kind] = actual
    validate_prior(directory)
    return service


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    record = {'schema': protocol()['schema'], 'completed': False, 'protocol_sha256': digest(PROTOCOL)}
    try:
        record['service'] = prepare(a.prior)
        record['completed'] = True
    except Exception as exc:
        record['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        (a.output/'prior-authentication.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
