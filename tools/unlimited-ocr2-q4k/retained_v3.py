"""Read-only hosted audit of the stopped v3 conversion; no model forwards."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import contract as c
import panel

PRODUCER_SOURCE = '410541f9c2587d345fb0d575a3315ce3c032a02a'
PANEL_BYTES = 514832
PANEL_SHA = '9c0aae284ef9af732c51bf3ccfe5996b71cb5d8b448dc1bf5f6802468cf53ba9'
MODEL_BYTES = 6481651840
MODEL_SHA = '8b9d8f28e225a229107da1c2dd043452d5c30a3f568337eff1741181a5da4f37'


def member(root, name):
    rel = PurePosixPath(name)
    c.require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name and str(rel) == name,
              'Safe immutable producer member required')
    path = root.joinpath(*rel.parts)
    c.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
              'Missing/unsafe immutable producer member')
    return path


def locate(mount):
    hits = [p for p in mount.rglob('panel.json') if p.is_file() and not p.is_symlink()
            and p.stat().st_size == PANEL_BYTES and c.digest(p) == PANEL_SHA]
    c.require(len(hits) == 1, 'One exact v3 panel mount required')
    return hits[0]


def audit(mount, out, source):
    c.require(Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir() and
              os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Batch', 'Interactive'), 'Genuine hosted CPU audit required')
    c.require(c.git('rev-parse', 'HEAD') == source, 'Explicit audit checkout differs')
    hashes = c.guard()
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:'))) * 1024
    c.require(available >= 10 * 2**30 and shutil.disk_usage(out.parent).free >= 2 * 2**30,
              '10GiB RAM/2GiB output headroom required before retained payload reads')
    path = locate(mount)
    root = path.parent
    producer = json.loads(path.read_text())
    c.require(producer['source'] == PRODUCER_SOURCE and producer['completed'] is False and
              producer['f16_all_four_strict_control_passed'] is True and not producer['candidates'] and
              producer['error'] == 'Protected source type/shape/payload differs: v.pos_embed',
              'Exact stopped producer state required')
    files = producer['files']
    c.require(len(files) == 540 and sum(r['bytes'] for r in files.values()) == 7901314457,
              'Complete immutable partial-output inventory required')
    # Authenticate every retained byte before importing array packages or mapping weights.
    for name, identity in files.items():
        p = member(root, name)
        c.require(p.stat().st_size == identity['bytes'] and c.digest(p) == identity['sha256'],
                  'Retained producer payload differs: ' + name)
    refs, _ = c.references()
    c.require([r['id'] for r in producer['baseline']] == c.CASES, 'Four baseline cases required')
    controls = []
    for row in producer['baseline']:
        case = row['id']
        c.require(set(row['modes']) == {'plain', 'ordinary', 'observed', 'repeat'}, 'Four baseline modes required')
        for mode, old in row['modes'].items():
            actual = panel.parsed(root / 'baseline' / case / mode)
            c.require(actual['ids'] == old['ids'] == refs[case]['ids'] and
                      actual['text'] == old['text'] == refs[case]['text'], 'Retained F16 original IDs/text differ')
        stages = {}
        for mode in ('observed', 'repeat'):
            actual = panel.stage_manifest(root / 'baseline' / case / (mode + '-trace'))
            expected = row['modes'][mode]['stages']
            c.require(set(actual) == set(expected) and all(all(actual[k][f] == expected[k][f]
                      for f in ('shape', 'type', 'bytes', 'sha256')) for k in actual),
                      'Retained first-prefill stage declaration differs')
            stages[mode] = actual
        c.require(all(stages['observed'][k]['sha256'] == stages['repeat'][k]['sha256'] for k in stages['observed']),
                  'Retained stage repeat is not byte exact')
        controls.append(dict(id=case, tokens=len(refs[case]['ids']), modes=4, stages=58,
                             original_ids_text_exact=True, observed_repeat_stages_exact=True))
    model = member(root, 'unlimited-last1.gguf')
    c.require(model.stat().st_size == MODEL_BYTES and c.digest(model) == MODEL_SHA, 'Immutable first candidate differs')
    actual = c.inventory(model)
    proof = c.protected(producer['baseline_inventory'], actual, 'last1')
    return dict(schema='crispembed-unlimited-retained-v3-audit-v1', completed=True, source=source,
                producer_source=PRODUCER_SOURCE, producer_panel_sha256=PANEL_SHA,
                producer_negative_preserved=True, original_rerun=False, native_forwards=0,
                quantizer_runs=0, packages_imported_after_full_authentication=True,
                authenticated_files=540, authenticated_bytes=7901314457,
                controls=controls, model=dict(bytes=MODEL_BYTES, sha256=MODEL_SHA),
                actual_inventory=actual, protected=proof, source_hashes=hashes,
                publication_qualified=False, numerical_qualified=False,
                limits='Read-only structural proof. Q4 numerical, decoded and operator comparisons remain unexecuted.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('mount', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source', required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='crispembed-unlimited-retained-v3-audit-v1', completed=False,
                  source=args.source, original_rerun=False, native_forwards=0, publication_qualified=False)
    try:
        result = audit(args.mount, args.out, args.source)
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        (args.out / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
