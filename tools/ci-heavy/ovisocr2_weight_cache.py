"""Hosted, bounded decoder-weight cache audit; reuse original arrays and weights."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess

import ovisocr2_large_inputs as large
import ovisocr2_model_audit as audit
import ovisocr2_native_replay as replay
import ovisocr2_vision_audit as ab
from ovisocr2_published_consumer import cpu_record

ROOT = Path(__file__).resolve().parents[2]
BASE = 'a8912d73f01eaaf28cbf4b057a9a0e0837f24700'
BUDGET = 256 * 1024**2
# Exact reviewed prototype files, populated before dispatch; unrelated source
# must remain identical to BASE. Changing these pins requires reviewing the diff.
NATIVE_PINS = {
    "CMakeLists.txt": "3493586d63b6f1c1fe337c4560777e0f314daa510147ee51153f7c65b8e208ca",
    "src/qwen35_attention.cpp": "58287bdf61483bbd4ab8681f3c71f74d6a264c5c372609de665125909f22e619",
    "src/qwen35_attention.h": "3f0b62c4238b199a4eb61d17f4223f1b98bf82b3986373fedb0ae96709770905",
    "src/qwen35_decoder.cpp": "db4e23fbd60d9ff7b1593ac54cf471c775433883aab179c0d98ce8bfd77f3aaa",
    "src/qwen35_decoder.h": "9f37cbf630341ab3749e9d1975ca154ad650e61c7f123deece8527133ebd0f64",
    "src/qwen35_linear.cpp": "de90487fc6d9ce016428935129e4b839c8c53e09fd4c0a39d33702cffaed4a55",
    "src/qwen35_linear.h": "8f1f3a7f4d8e4c1641aa7975ff71dbfea87dbe525612503e566da298f4241570",
    "src/qwen35_math.h": "0696f1781ce2cb49fc639dcc405262ffc1beb63bd1549aa571391f6b30e06d96",
    "src/qwen35_ocr.cpp": "01e67496cefb5b1063c1c601f9a77fb3945e8400f396bf800f0fd84faaa8484f",
    "src/qwen35_weight_cache.h": "ca17d1ca90db283fad37f79930944b1660dba8756f9889f7a6a853fe5cc45152",
    "tests/qwen35_parity.h": "b003edf078884107df753a95c957e260d3e3a63c0356b5eb57ec2092a4b746bf",
    "tests/test_ovisocr2_cache_bench.cpp": "272fce1185099ab9cc02ca272206c8ec25c905ad712b0672fec1f80a73b35785",
    "tests/test_qwen35_decoder.cpp": "c8755b7ad7bb08cbe83077fe8e43123e8d129b9e10eede0fdac483c46fbca095",
    "tests/test_qwen35_weight_cache.cpp": "8f2451c89c1e8e42d9e45b0f6316a64aeec4e9edb0a82d54776339ba9ac551a6",
    "tools/ci-heavy/ovisocr2_large_inputs.py": "69fbbfd0f2246dd0654ad8c8526c2880ab727aee8046def14dae0b65b5dc6318",
    "tools/ci-heavy/ovisocr2_native_replay.py": "cc515a3fab916692c1139de5db5a7bb46762ae677a72f93af7a691ccd0bd4dae",
    "tools/ci-heavy/ovisocr2_quality_run.py": "110056eb9b727d1fac7ff54d0cba509030a41982eef5adf83040ad23f2fedc5f"
}
TARGETS = ('test-qwen35-decoder', 'test-qwen35-vision', 'test-qwen35-delta',
           'test-ovisocr2-context', 'test-ovisocr2-cache-bench', 'test-qwen35-weight-cache')
ANSWER_KEYS = ('token_ids', 'confidences', 'raw_text_utf8_hex', 'stop_kind', 'mean_confidence')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()


def source_contract():
    try:
        git('cat-file', '-e', BASE + '^{commit}')
    except subprocess.CalledProcessError:
        subprocess.run(['git', 'fetch', '--no-tags', '--depth=1', 'origin', BASE], cwd=ROOT,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    changed = set(git('diff', '--name-only', BASE, '--', 'src', 'CMakeLists.txt',
                      'tests/test_qwen35_decoder.cpp').splitlines())
    required = {name for name in NATIVE_PINS if name.startswith('src/') or name in
                ('CMakeLists.txt', 'tests/test_qwen35_decoder.cpp')}
    if not NATIVE_PINS or changed != required:
        raise ValueError('Cache runtime changes differ from the explicit source scope')
    actual = {name: audit.sha(ROOT / name) for name in NATIVE_PINS}
    if actual != NATIVE_PINS:
        raise ValueError('Cache native/checker prototype hashes differ')
    for args in [('diff', '--name-only', 'HEAD', '--', 'src', 'CMakeLists.txt'),
                 ('ls-files', '--others', '--exclude-standard', '--', 'src')]:
        if git(*args):
            raise ValueError('Dirty or untracked runtime source')
    old_ggml = git('ls-tree', BASE, 'ggml')
    if old_ggml != git('ls-tree', 'HEAD', 'ggml') or not old_ggml.startswith('160000 commit '):
        raise ValueError('Cache experiment changed the ggml gitlink')
    ggml = old_ggml.split()[2]
    if git('-C', str(ROOT/'ggml'), 'rev-parse', 'HEAD') != ggml:
        raise ValueError('Actual ggml checkout differs')
    if git('-C', str(ROOT/'ggml'), 'status', '--porcelain'):
        raise ValueError('Dirty ggml checkout')
    replay.source_helpers()  # Original observer/comparator/delta trace pins.
    return {'base': BASE, 'source_commit': git('rev-parse', 'HEAD'),
            'src_tree': git('rev-parse', 'HEAD:src'), 'native_sha256': actual,
            'ggml_gitlink': ggml, 'intentional_changes': sorted(changed),
            'scope': 'Intentional default-off cache runtime; original math/comparators are retained'}


def environment(on):
    # Drop ambient experimental gates and all threshold/dump overrides. Transfer
    # only the explicit, already reviewed frozen-control arithmetic preset.
    source = large.native_environment('frozen-control')
    names = set(audit.NATIVE_ENV) | {'CRISPEMBED_QWEN35_VISION_PRECISE_NORM',
        'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN', 'CRISPEMBED_QWEN35_VISION_CASCADE_NORM',
        'CRISPEMBED_OVISOCR2_LARGE_IMAGES'}
    result = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
    result.update({name: source[name] for name in names})
    result.update(CRISPEMBED_QWEN35_WEIGHT_CACHE=str(int(on)),
                  CRISPEMBED_QWEN35_WEIGHT_CACHE_BYTES=str(BUDGET),
                  CRISPEMBED_QWEN35_PARALLEL_F16='0', CRISPEMBED_QWEN35_PROFILE='0',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return result


def benchmark_rows(text, two=False):
    rows = [json.loads(line.removeprefix('CACHE_BENCH_RESULT ')) for line in text.splitlines()
            if line.startswith('CACHE_BENCH_RESULT ')]
    expected = ['cold', 'warm', 'independent', 'survivor'] if two else ['cold', 'warm']
    if [row.get('label') for row in rows] != expected:
        raise ValueError('Cold/warm/independent/survivor result coverage differs')
    for row in rows:
        if (row.get('contexts') != (2 if two else 1) or row.get('token_ids') != replay.FROZEN_IDS
                or row.get('raw_text_utf8_hex') != replay.FROZEN_TEXT.encode().hex()
                or row.get('stop_kind') != 1 or len(row.get('confidences', [])) != 9):
            raise ValueError('Benchmark IDs/text/EOS/confidence coverage differs')
        for key in ('init_seconds', 'recognize_seconds', 'mean_confidence'):
            if type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError('Invalid benchmark measurement')
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in row['confidences']):
            raise ValueError('Invalid selected probability')
        if abs(sum(row['confidences'])/9-row['mean_confidence']) > 1e-6:
            raise ValueError('Selected mean probability differs')
    return rows


def require_exact(rows):
    if not rows or any(any(row.get(k) != rows[0].get(k) for k in ANSWER_KEYS) for row in rows):
        raise ValueError('Cache changes actual IDs/text/EOS/selected confidences')


def verify_quality(path):
    current = json.loads(Path(path).read_text())
    frozen = json.loads((ROOT/'tests/results/qwen35-ocr/ovisocr2-f16-quality.json').read_text())['producer']
    actual = current.get('native_cases', {}).get('f16', {})
    old = frozen['native_cases']['f16']
    keys = ('input_rgb_sha256', 'input_png_sha256', 'raw_text', 'processed_text', 'token_ids',
            'stop_id', 'finish_reason', 'stop_kind', 'cap', 'confidences', 'mean_confidence')
    if len(actual) != 11 or set(actual) != set(old):
        raise ValueError('Full eleven-case first-F16 comparison absent')
    for name in old:
        if any(actual[name].get(k) != old[name].get(k) for k in keys):
            raise ValueError('Cache F16 quality/selected probabilities differ: '+name)
    for key in ('f16_paired_synthetic', 'f16_paired_documents'):
        if current.get(key) != frozen[key] or current[key].get('screen_passed') is not True:
            raise ValueError('Frozen paired quality scores differ')
    if current.get('scores', {}).get('f16') != frozen['scores']['f16']:
        raise ValueError('First F16 scores differ')
    return {'eleven_cases_exact': True, 'token_count': sum(len(row['token_ids']) for row in actual.values()),
            'confidences_exact': True, 'quality_screens_exact': True}


def run():
    closure = source_contract()
    out = Path(os.environ['HEAVY_OUT']); out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ['HEAVY_SCRATCH'])/'cache'; scratch.mkdir(parents=True, exist_ok=True)
    build = Path(os.environ['AUDIT_BIN_DIR'])
    report = {'schema': 'crispembed-ovisocr2-weight-cache-v1', 'status': 'started',
              'source': closure, 'cpu': cpu_record(), 'budget_bytes_per_context': BUDGET,
              'commands': {}, 'arms': {}, 'benchmarks': [], 'limits': [
                  'Default-off experiment, decoder matrices only; no model/default/release promotion.',
                  'Stable first-fit/no eviction. The full 1017118720-byte head cannot fit this budget.',
                  'Budget covers retained CPU buffers; model weights, conversion scratch and graph arenas are separate.',
                  'Three alternating same-host pairs on one owned image; not a document latency benchmark.',
                  'Fresh-process RSS includes loading and recognition. Two-context tests report both owners.',
                  'Original943 stages/qualified GGUFs are reused without original inference or conversion.']}
    def save():
        large.save_report(out, report)
        (out/'ovisocr2-large-inputs.json').replace(out/'ovisocr2-weight-cache.json')
    def command(label, args, env):
        passed = large.timed_command(label, args, out, report, env, timeout=7200)
        save()
        return passed
    save()
    try:
        manifest=json.loads((build/'binaries.json').read_text())
        if manifest['source_commit'] != closure['source_commit'] or set(manifest['binary_sha256']) != set(TARGETS):
            raise ValueError('Cache build manifest differs')
        for name, digest in manifest['binary_sha256'].items():
            if audit.sha(build/name) != digest:raise ValueError('Cache binary hash differs')
            (build/name).chmod(0o700)
        report['build']=manifest
        if not command('component-controls', [build/'test-qwen35-weight-cache'], environment(False)):
            raise ValueError('Native cache controls failed before model transfer')
        if shutil.disk_usage(scratch).free < 18 * 1024**3:
            raise ValueError('At least18GiB hosted scratch required before model transfer')
        source = large.source_and_exports(out, scratch, report)
        reference = replay.source_reference(ab, scratch, out, report, source)
        save()
        failures=[]
        for on in (0, 1):
            report['arms'][str(on)]={}
            for precision, (filename, _, _) in ab.MODELS.items():
                for name, args in audit.native_arms(precision, build, scratch/filename,
                        scratch/'owned-readable.png', scratch/'official-delta.gguf').items():
                    label=f'cache-{on}-{name}'
                    folder=scratch/'native-diff'/label;folder.mkdir(parents=True,exist_ok=True)
                    passed=command(label,args,dict(environment(on),CRISPEMBED_DIFF_DUMP_DIR=str(folder)))
                    log=(out/(label+'.log')).read_text()
                    try:
                        raw=large.raw_manifest(folder,reference,log,name)
                        for row in raw.get('files',[]):
                            row['file']=row['file'].replace('native-diff/'+name+'/', 'native-diff/'+label+'/', 1)
                        proof=replay.verify_arm_log(name,log,reference['stages'],ab)
                        if '-decoder-' in name:
                            match=re.search(r'^WEIGHT_CACHE bytes=(\d+) budget=(\d+) entries=(\d+) hits=(\d+)$',log,re.M)
                            if not match:raise ValueError('Cache decoder counters absent')
                            used,budget,entries,hits=map(int,match.groups())
                            if budget!=BUDGET or used>budget or (on and (not used or not entries or not hits)) or (not on and (used or entries or hits)):
                                raise ValueError('Actual cache admission/hits/budget differ')
                            proof['cache']={'bytes':used,'budget':budget,'entries':entries,'hits':hits}
                        result={'passed':passed,'raw':raw,'checks':proof}
                        if not passed or not raw['coverage_complete']:failures.append(label)
                    except (ValueError,KeyError) as error:
                        result={'passed':False,'error':str(error)};failures.append(label)
                    report['arms'][str(on)][name]=result;save()
        for name, old in report['arms']['0'].items():
            new=report['arms']['1'][name]
            def raw_identity(row):
                return sorted((r['stage'],r.get('variant','primary'),r['bytes'],r['sha256']) for r in row.get('raw',{}).get('files',[]))
            if not old['passed'] or not new['passed'] or raw_identity(old)!=raw_identity(new):
                failures.append('cache-on-off-raw-'+name)
        report['frozen_failures']=failures;save()
        all_answers=[]
        for pair in range(3):
            for on in ((0,1) if pair%2==0 else (1,0)):
                label=f'pair-{pair}-cache-{on}'
                if not command(label,[build/'test-ovisocr2-cache-bench',scratch/'ovisocr2-f16.gguf',scratch/'owned-readable.png'],environment(on)):
                    raise ValueError('Paired cache benchmark failed')
                rows=benchmark_rows((out/(label+'.log')).read_text());all_answers.extend(rows)
                require_exact(all_answers)
                report['benchmarks'].append({'pair':pair,'cache':bool(on),'rows':rows,
                                             'resources':report['commands'][label]['resources']});save()
        for on in (0,1):
            label=f'two-contexts-cache-{on}'
            if not command(label,[build/'test-ovisocr2-cache-bench',scratch/'ovisocr2-f16.gguf',scratch/'owned-readable.png','--two-contexts'],environment(on)):
                raise ValueError('Two-context cache benchmark failed')
            rows=benchmark_rows((out/(label+'.log')).read_text(),True);all_answers.extend(rows);require_exact(all_answers)
            report.setdefault('two_contexts',{})[str(on)]={'rows':rows,'resources':report['commands'][label]['resources']};save()
        report['paired_summary']={}
        for phase in ('cold','warm'):
            values={on:[next(r['recognize_seconds'] for r in b['rows'] if r['label']==phase)
                        for b in report['benchmarks'] if b['cache']==bool(on)] for on in (0,1)}
            ratios=[a/b for a,b in zip(values[1],values[0])]
            report['paired_summary'][phase]={'off_seconds':values[0],'on_seconds':values[1],
                'paired_on_over_off':ratios,'median_paired_on_over_off':statistics.median(ratios)}
        report['status']='completed cache paired/frozen audit' if not failures else 'rejected cache frozen audit'
        report['frozen_fourteen_per_gate_positive']=not failures
        report['same_host_all_ids_text_confidences_exact']=True
        save()
        if failures:raise ValueError('Cache frozen audit failed; all arms preserved')
    except Exception as error:
        report['error']=re.sub(r'https?://\S+','<url>',str(error))
        if report['status']=='started':report['status']='failed before complete observations'
        save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--verify-quality',type=Path)
    args=parser.parse_args()
    if args.preflight:print(json.dumps(source_contract(),sort_keys=True))
    elif args.verify_quality:
        result=verify_quality(args.verify_quality)
        args.verify_quality.with_name('ovisocr2-weight-cache-quality-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
    else:run()
