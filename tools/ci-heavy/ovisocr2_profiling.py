"""Hosted paired Ovis profiling. Frozen weights/inputs, no inference math changes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import urllib.request

import ovisocr2_published_consumer as consumer
from ovisocr2_quality_contract import source_schema
from qwen35_profiling import parse_benchmark, check_bytes, paired_summary, OUTPUT_FIELDS

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('tests/test_qwen35_profile.cpp', 'tools/ci-heavy/ovisocr2_profiling.py',
           'tools/ci-heavy/qwen35_profiling.py', 'tools/ci-heavy/ovisocr2_published_consumer.py')


def require_answer(row, expected):
    if (row['ids'] != expected['token_ids'] or row['text'] != expected['raw_text']
            or row['stop'] != expected['stop_kind'] or row['stop'] != 1
            or len(row['ids']) != 9 or row['ids'][-1] != 248046):
        raise ValueError('Frozen actual IDs/text/EOS differ')
    # Across hosts retain actual score differences; same-host off/on must be exact.
    maximum = max(abs(a-b) for a, b in zip(row['confidences'], expected['confidences']))
    mean = abs(row['mean_confidence']-expected['mean_confidence'])
    return {'selected_probability_max_abs': maximum, 'mean_abs': mean,
            'scope': 'cross-host diagnostic; same-host neutrality requires exact scores',
            'reference_source': 'first qualified F16 dev-hello'}


def require_neutral(rows):
    expected_order = [(pair, bool(on)) for pair in range(3)
                      for on in ((0, 1) if pair % 2 == 0 else (1, 0))]
    if [(row['pair'], row['profile_active']) for row in rows] != expected_order:
        raise ValueError('Missing/reordered alternating profiling pairs')
    first = rows[0]
    if any(any(row[key] != first[key] for key in OUTPUT_FIELDS) for row in rows):
        raise ValueError('Profiler changes actual decoded output or selected scores')
    return paired_summary(rows)


def attribution(row):
    events = {(r['phase'], r['event']): r for r in row['events']}
    result = {}
    for phase in ('vision', 'prefill', 'decode'):
        parent = 'cached_decode' if phase == 'decode' else phase
        ns = events[phase, parent]['nanoseconds']
        result[phase] = {'phase_ms': ns/1e6,
            'projection_ms': events[phase, 'project_total']['nanoseconds']/1e6,
            'materialize_ms': events[phase, 'project_materialize']['nanoseconds']/1e6,
            'compute_ms': events[phase, 'project_compute']['nanoseconds']/1e6,
            'f32_bytes_materialized': sum(m['f32_bytes'] for m in row['matrices'] if m['phase']==phase)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    publication, fixture, reference = consumer.preflight('f16')
    if args.preflight:
        print(json.dumps({'native_source_closure': publication['native_source_closure'],
                          'model': publication['remote'], 'fixture': fixture}))
        return
    out = Path(os.environ['HEAVY_OUT']); out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'ovis-profile'; scratch.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'crispembed-ovisocr2-profiling-v1', 'status': 'started',
        'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256': {p: consumer.sha(ROOT/p) for p in SOURCES},
        'native_source_closure': publication['native_source_closure'],
        'model': publication['remote'], 'fixture': fixture, 'cpu': consumer.cpu_record(),
        'commands': {}, 'rows': [], 'limits': [
            'One frozen 448x448 owned image; nine IDs/EOS, not a document performance benchmark.',
            'Fresh process/context each arm; initialization separate from recognition; GNU time RSS includes both.',
            'Profile counters are nested. Never sum source/projection/phase parents as independent totals.',
            'No math/default/weight change. No cross-host speed claim. Original references are reused, not regenerated.']}
    def save():
        (out/'ovisocr2-profiling.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def command(label, argv, env=None):
        row=consumer.run(label,argv,out,env=env); report['commands'][label]=row;save()
        if row['exit_code'] != 0:raise ValueError(label+' failed')
        return row
    save()
    try:
        image, report['small_provenance'] = consumer.fetch_small_provenance(publication,fixture,scratch)
        model = scratch/'ovisocr2-f16.gguf'
        if model.exists():raise ValueError('Fresh task model destination required')
        with urllib.request.urlopen(publication['registry']['url'], timeout=90) as response, model.open('wb') as target:
            size=0
            while block:=response.read(1024*1024):
                size+=len(block)
                if size>publication['remote']['bytes']:raise ValueError('Model exceeds expected size')
                target.write(block)
        report['download_verified']=consumer.check_cached(model,publication['remote']);save()
        build=scratch/'build'
        command('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',
            '-DCRISPEMBED_NATIVE=OFF','-DGGML_NATIVE=OFF','-DGGML_BLAS=OFF','-DGGML_METAL=OFF'])
        command('cpu-baseline',[sys.executable,ROOT/'scripts/check-cpu-baseline.py',build])
        command('build',['cmake','--build',build,'--target','test-qwen35-profile','-j2'])
        command('compiler',['c++','--version'])
        report['binary_sha256']=consumer.sha(build/'test-qwen35-profile')
        env=consumer.env_for(scratch/'unused-cache',publication['qualification']['selected_vision_preset'],'f16')
        command('counter-controls',[build/'test-qwen35-profile'],env)
        report['environment']={k:v for k,v in env.items() if k.startswith('CRISPEMBED_QWEN35_') or k.startswith('CRISPEMBED_OVISOCR2_') or k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}
        expected=json.loads((ROOT/'tests/results/qwen35-ocr/ovisocr2-f16-quality.json').read_text())['producer']['native_cases']['f16']['dev-hello']
        schema={name:{'shape':shape,'dtype':'float16' if len(shape)==2 and 'norm' not in name else 'float32'} for name,shape in source_schema().items()}
        for pair in range(3):
            for on in ((0,1) if pair%2==0 else (1,0)):
                label=f'pair-{pair}-profile-{on}';resources=out/(label+'-resources.json')
                metadata=command(label,['/usr/bin/time','-f','{"max_rss_kib":%M,"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S}',
                    '-o',resources,build/'test-qwen35-profile',model,image,'12','2','ovisocr2'],
                    dict(env,CRISPEMBED_QWEN35_PROFILE=str(on)))
                row=parse_benchmark((out/metadata['log']).read_text());check_bytes(row,schema)
                if row['profile_active']!=bool(on):raise ValueError('Profile gate differs')
                row.update(pair=pair,resources=json.loads(resources.read_text()),log=metadata,
                           frozen_answer=require_answer(row,expected))
                if row['resources']['max_rss_kib']<=0:raise ValueError('Invalid isolated process RSS')
                if on:row['attribution']=attribution(row)
                report['rows'].append(row);save()
        report['summary']=require_neutral(report['rows'])
        report['status']='completed paired profiler neutrality and attribution'
        save()
    except Exception as error:
        report.update(status='failed',error=str(error));save();raise


if __name__=='__main__':
    main()
