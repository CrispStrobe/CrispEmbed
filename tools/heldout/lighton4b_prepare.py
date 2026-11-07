#!/usr/bin/env python3
"""Model-free held-out admission. Reads only frozen inputs/references, never model outputs.

Produces a blocked execution plan; deliberately has no process/network executor.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/lighton4b-heldout-v1.json'
CASE_IDS = ['historical-receipt', 'two-column-page', 'inventory-table']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def admit(cfg):
    require(cfg['policy'] == 'lighton4b-heldout-v1', 'protocol identity')
    require([c['id'] for c in cfg['cases']] == CASE_IDS, 'frozen case order')
    q = cfg['quota']
    expected = dict(cases=3, real_documents=2, owned_controlled_tables=1,
                    variants_per_case=1, replacement_cases=0, max_new_tokens=1536,
                    context=2048, maximum_prompt_tokens=512,
                    maximum_merged_image_tokens=256, threads=4, build_threads=2,
                    max_workers=1, case_worker_seconds=6000, arm_seconds=1200,
                    total_case_worker_seconds=18000, minimum_available_ram_gib=10,
                    minimum_scratch_gib=25, minimum_output_free_gib=8)
    require(q == expected, 'closed quota')
    for case in cfg['cases']:
        require(case['max_new_tokens'] == 1536 and case['crop'] is None and
                case['prompt'] == '', 'no variant/prompt adaptation')
        require(case['prepared']['nominal_patches'] % 4 == 0 and
                case['prepared']['nominal_patches'] // 4 <= 256, 'image budget')


def budget(prompt_tokens, image_tokens):
    require(isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool)
            and 1 <= prompt_tokens <= 512, 'prompt admission')
    require(isinstance(image_tokens, int) and not isinstance(image_tokens, bool)
            and 1 <= image_tokens <= 256 and image_tokens <= prompt_tokens,
            'image admission')
    require(prompt_tokens + 1536 <= 2048, 'physical cache capacity')


def prepare(cfg, archive, root=ROOT):
    admit(cfg)
    # The small prepared-input archive is authenticated before ZIP parsing.
    raw = Path(archive).read_bytes()
    require(len(raw) == cfg['prepared_archive']['bytes'] and
            sha(raw) == cfg['prepared_archive']['sha256'], 'immutable prepared archive')
    from PIL import Image
    cases = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members = z.infolist()
        require(len(members) <= 128 and sum(m.file_size for m in members) <= 16 << 20,
                'bounded input archive')
        require(all(not Path(m.filename).is_absolute() and '..' not in Path(m.filename).parts
                    for m in members), 'safe archive paths')
        for c in cfg['cases']:
            for key in ('source', 'gold'):
                require(sha((root / c[key]).read_bytes()) == c[key + '_sha256'],
                        'frozen ' + key + ' ' + c['id'])
            row = c['prepared']
            matches = [m for m in members if Path(m.filename).name == row['file']]
            require(len(matches) == 1, 'unique prepared image ' + c['id'])
            image_bytes = z.read(matches[0])
            require(sha(image_bytes) == row['png_sha256'], 'prepared PNG')
            with Image.open(io.BytesIO(image_bytes)) as im:
                rgb = im.convert('RGB')
                require(list(rgb.size) == row['size'] and sha(rgb.tobytes()) == row['rgb_sha256'],
                        'prepared RGB and dimensions')
            cases.append(dict(id=c['id'], png_sha256=row['png_sha256'],
                              rgb_sha256=row['rgb_sha256'], size=row['size'],
                              gold_sha256=c['gold_sha256'],
                              nominal_merged_image_tokens=row['nominal_patches'] // 4,
                              execution_order=cfg['arms']))
    return dict(schema='lighton4b-heldout-admission/v1', status='inputs_verified_execution_blocked',
                quality_qualified=False, reference_status=cfg['reference']['status'],
                model_inference_performed=False, candidate_outputs_read=False,
                cases=cases, blockers=cfg['execution_blockers'],
                required_runtime_contract=dict(cap=1536, threads=4, context=2048,
                    modes=['composed-primary', 'source-feature-secondary'],
                    prompt_tokens_max=512, image_tokens_max=256,
                    completed_outputs_required=True, generic_driver_available=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-archive', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    protocol_bytes = PROTOCOL.read_bytes()
    result = prepare(json.loads(protocol_bytes), a.prepared_archive)
    result['protocol_sha256'] = sha(protocol_bytes)
    a.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
