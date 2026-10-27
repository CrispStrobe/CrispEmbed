"""Hosted-only build-once SciFact native-v3, against frozen original-v2 arrays."""
import argparse
import hashlib
import importlib.util
import inspect
import importlib.metadata
import platform
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from ultrabert_scifact_native_v3 import CONTRACT, protocol, sha, safe_zip, check_token_proof, check_rows, check_actual_ids, check_shard
from ultrabert_scifact import shard_rows, rank_scores, quality
import scifact_native_loader_smoke as loader
from scifact_tokenizer_diagnostic import check_jobs

spec = importlib.util.spec_from_file_location('scifact_v2_lowlevel', ROOT / 'tools/ci-heavy/ultrabert_scifact.py')
if sha(ROOT / 'tools/ci-heavy/ultrabert_scifact.py') != protocol()['helper_sha256']['tools/ci-heavy/ultrabert_scifact.py']:
    raise ValueError('Historical lowlevel helper bytes differ before import')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
write, load, inventory = old.write, old.load, old.inventory
REPO = 'CrispStrobe/CrispEmbed'
OWN = (CONTRACT, 'tools/ultrabert_scifact_native_v3.py', 'tools/ci-heavy/ultrabert_scifact_native_v3.py',
       'tests/test_ultrabert_scifact_native_v3.py', '.github/workflows/ultrabert-scifact-native-v3.yml',
       '.github/workflows/ovisocr2-metadata.yml', 'docs/ultrabert-scifact-native-v3.md',
       'tests/results/ultrabert/initial-scifact-native-v3-loader-failure.json',
       'tools/scifact_native_loader_smoke.py')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def source():
    p = protocol(); paths = p['runtime_paths']
    objects = {x: git('rev-parse', p['runtime_base'] + ':' + x) for x in paths}
    if (objects != {x: git('rev-parse', 'HEAD:' + x) for x in paths}
            or git('diff', 'HEAD', '--name-only', '--', *paths, *OWN, *p['helper_sha256'])
            or git('ls-files', '--others', '--exclude-standard', '--', *paths)
            or git('-C', str(ROOT / 'ggml'), 'rev-parse', 'HEAD') != p['ggml']
            or git('-C', str(ROOT / 'ggml'), 'status', '--porcelain')):
        raise ValueError('Exact new runtime/binding/build source closure differs')
    if any(sha(ROOT / x) != h for x, h in {**p['helper_sha256'], **p['token_proof']['copied_runtime_sha256']}.items()):
        raise ValueError('Historical helpers/reference gates or copied tokenizer source differ')
    # Own harness must be committed and clean as well; never import untracked wrappers.
    for x in OWN:
        if git('rev-parse', 'HEAD:' + x) != git('hash-object', x):
            raise ValueError('Uncommitted caller source: ' + x)
    return dict(head=git('rev-parse', 'HEAD'), ggml=p['ggml'], runtime_base=p['runtime_base'],
                runtime_git_objects=objects, files={x:sha(ROOT / x) for x in (*OWN, *p['helper_sha256'])},
                contract_sha256=sha(ROOT / CONTRACT))


def api(path):
    return old.api(path)


def run_identity(run, head, *, complete=True):
    meta = api('actions/runs/' + str(run))
    if (meta['head_sha'] != head or meta['path'] not in ('.github/workflows/ovisocr2-metadata.yml',
            '.github/workflows/ultrabert-scifact-native-v3.yml') or meta['event'] != 'workflow_dispatch'
            or meta.get('run_attempt') != 1 or complete and (meta['status'] != 'completed' or meta['conclusion'] != 'success')):
        raise ValueError('Producer workflow/source/attempt/state differs')
    return meta


def fetch_artifact(run, head, pin, target, *, complete=True, job=None):
    run_identity(run, head, complete=complete)
    if job:
        jobs = api('actions/runs/' + str(run) + '/jobs?per_page=100')
        matches = [j for j in jobs['jobs'] if j['name'].split(' / ')[-1] == job]
        if jobs['total_count'] > 100 or len(matches) != 1 or matches[0]['conclusion'] != 'success' or matches[0]['status'] != 'completed':
            raise ValueError('Same-run producer job did not complete successfully: ' + job)
    artifacts = api('actions/runs/' + str(run) + '/artifacts?per_page=100')
    found = [a for a in artifacts['artifacts'] if a['name'] == pin['name'] and not a['expired']]
    if artifacts['total_count'] > 100 or len(found) != 1:
        raise ValueError('Artifact absent/ambiguous')
    a = found[0]
    if any(a[k] != pin[q] for k,q in [('id','id'),('size_in_bytes','bytes')] if q in pin) or ('sha256' in pin and a.get('digest') != 'sha256:' + pin['sha256']):
        raise ValueError('Pinned service artifact identity differs')
    archive = target.parent / (target.name + '.zip'); target.mkdir(parents=True, exist_ok=True)
    with archive.open('wb') as stream:
        subprocess.run(['gh','api','repos/'+REPO+'/actions/artifacts/'+str(a['id'])+'/zip'], stdout=stream, check=True, timeout=1200)
    if archive.stat().st_size != a['size_in_bytes'] or 'sha256:' + sha(archive) != a.get('digest'):
        raise ValueError('Downloaded service ZIP size/SHA differs')
    with zipfile.ZipFile(archive) as z:
        safe_zip(z); z.extractall(target)
    return dict(run=run, source=head, artifact_id=a['id'], name=a['name'], bytes=a['size_in_bytes'],
                sha256=sha(archive), expires_at=a['expires_at'])


def receipt(target, expected_sha=None):
    if expected_sha and sha(target / 'receipt.json') != expected_sha:
        raise ValueError('Immutable receipt bytes differ')
    r = load(target / 'receipt.json'); files = inventory(target); files.pop('receipt.json')
    if files != r['files']:
        raise ValueError('Payload exact member coverage/hash differs')
    return r


def materialized_references(scratch):
    p = protocol(); token = p['token_proof']; original = p['original']; prep = p['prepared']
    path = scratch / 'token-proof/ab.json'
    if sha(path) != token['report_sha256']:
        raise ValueError('Reviewed token-only report bytes differ')
    check_token_proof(load(path))
    frozen_path = scratch / 'freeze/original-freeze.json'
    if sha(frozen_path) != original['freeze_report_sha256']:
        raise ValueError('Frozen original8 receipt changed')
    frozen = load(frozen_path)
    if (frozen['schema'] != 'crispembed-scifact-original-freeze-v2' or frozen['original_rows'] != 5483
            or frozen['no_native_outputs_yet'] is not True or frozen['source']['head'] != original['source']
            or frozen['service'] != original['shards']
            or frozen['status'] != 'all8 original shards frozen before native'
            or frozen['protocol_sha256'] != p['frozen_protocol_sha256']):
        raise ValueError('Frozen original completion/lineage differs')
    prepared = scratch / 'prepared'; r = receipt(prepared, prep['receipt_sha256'])
    f = load(prepared / 'data-freeze.json')
    if (r['source']['head'] != prep['source'] or r['schema'] != 'crispembed-scifact-preparation-v2'
            or r['protocol_sha256'] != p['frozen_protocol_sha256'] or r['original_hook_neutral'] is not True
            or r['projected_original_within_bound'] is not True
            or sha(prepared / 'data-freeze.json') != prep['data_freeze_sha256']
            or f['dataset_sha256'] != sha(prepared / 'dataset.json') or f['lengths_sha256'] != sha(prepared / 'lengths.json')
            or load(prepared / 'protocol.json') != old.protocol()
            or frozen['data_freeze_sha256'] != prep['data_freeze_sha256']):
        raise ValueError('Frozen preparation/input/gold/protocol differs')
    data = load(prepared / 'dataset.json'); tokens = check_rows(data, load(prepared / 'lengths.json'))
    return prepared, data, tokens, frozen



def references(scratch):
    p = protocol(); token = p['token_proof']; original = p['original']; prep = p['prepared']
    proof = fetch_artifact(token['run'], token['source'], token['artifact'], scratch / 'token-proof')
    meta = run_identity(original['run'], original['source'], complete=False)
    states = check_jobs(meta, api('actions/runs/' + str(original['run']) + '/jobs?per_page=100'))
    frozen_proof = fetch_artifact(original['run'], original['source'], original['freeze_artifact'], scratch / 'freeze', complete=False, job='freeze-original')
    prep_proof = fetch_artifact(prep['run'], prep['source'], prep['artifact'], scratch / 'prepared')
    return (*materialized_references(scratch), dict(token_proof=proof, original_freeze=frozen_proof, preparation=prep_proof, old_jobs=states))


def command(argv, log, *, env=None, timeout=3600):
    with log.open('w') as stream:
        subprocess.run([str(a) for a in argv], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def dependency_check(library, log, env):
    command(['ldd',library],log,env=env)
    text=log.read_text()
    if 'not found' in text:raise ValueError('Bundled dependency missing')
    for line in text.splitlines():
        if 'libggml' in line or 'libcrispembed' in line:
            parts=line.split()
            if len(parts)<3 or Path(parts[2]).resolve().parent!=library.parent.resolve():
                raise ValueError('Bundled ggml dependency resolves outside exact payload')


def smoke_environment(directory, policy):
    env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')and k not in ('LD_PRELOAD','LD_AUDIT')}
    env['LD_LIBRARY_PATH']=loader.loader_path(directory,policy)
    return env


def library_smoke(payload,out):
    directory=payload/'lib';manifest=payload/'loader-libraries.json';result_path=payload/'loader-smoke.json'
    write(manifest,{p.name:dict(bytes=p.stat().st_size,sha256=sha(p))for p in sorted(directory.iterdir())})
    attempts=[];policy=dict(mode='bundle-only')
    argv=[sys.executable,str(ROOT/'tools/scifact_native_loader_smoke.py'),'--directory',str(directory),
          '--manifest',str(manifest),'--manifest-sha',sha(manifest),'--output',str(result_path)]
    for i in range(2):
        log=out/('loader-smoke-'+str(i)+'.log')
        with log.open('w')as stream:
            r=subprocess.run([*argv,'--mode',policy['mode']],cwd=ROOT,env=smoke_environment(directory,policy),
                             stdout=stream,stderr=subprocess.STDOUT,timeout=60)
        attempts.append(dict(mode=policy['mode'],exit=r.returncode,log_sha256=sha(log)))
        write(out/'loader-smoke-attempts.json',attempts)
        if r.returncode==0:
            result=load(result_path)
            if (result.get('completed')is not True or result.get('schema')!='crispembed-scifact-native-loader-smoke-v1'
                    or result.get('mode')!=policy['mode'] or result.get('python_library')!=policy.get('python_library')
                    or result.get('library_manifest_sha256')!=sha(manifest)
                    or result.get('library_sha256')!=sha(directory/'libcrispembed.so')
                    or result.get('symbols')!=list(loader.SYMBOLS)or result.get('all_ggml_mappings_in_bundle')is not True
                    or any(result.get(k)is not False for k in ('model_files_accessed','framework_imported','forward_called'))):
                raise ValueError('Actual fresh-loader smoke receipt differs')
            write(out/'loader-smoke.json',result)
            return result
        # A missing GGML/CrispEmbed library never permits a different search path.
        # Only the dynamic loader's interpreter-startup libpython rejection may
        # add the exact currently mapped sys.base_prefix/lib runtime directory.
        if i or r.returncode!=127 or result_path.exists():
            raise ValueError('Actual fresh Python/C API loader smoke failed')
        runtime=loader.python_library();message=log.read_text()
        if 'error while loading shared libraries: '+runtime['library']+':' not in message:
            raise ValueError('Fresh loader failure is not Python interpreter startup')
        policy=dict(mode='bundle-and-python-lib',python_library=loader.fingerprint(runtime))
    raise ValueError('Actual loader smoke did not complete')


def build(out, scratch, closure):
    references(scratch)  # All reviewed identity barriers before any weight download.
    if shutil.disk_usage(scratch).free < 10 * 1024**3:
        raise ValueError('At least10GiB hosted scratch required')
    payload = out / 'payload'; (payload / 'lib').mkdir(parents=True)
    directory = old.original_snapshot(True)
    converter_spec = importlib.util.spec_from_file_location('native_v3_converter', ROOT / 'models/convert-ultrabert-to-gguf.py')
    converter = importlib.util.module_from_spec(converter_spec); converter_spec.loader.exec_module(converter)
    conversion = converter.convert(directory, payload / 'native-f32.gguf', 'f32')
    if sha(payload / 'native-f32.gguf') != protocol()['native_model_sha256']:
        raise ValueError('Qualified F32 conversion bytes differ')
    write(out / 'conversion.json', conversion)
    command([sys.executable,'-m','pip','freeze'], out / 'environment.log')
    build_dir = scratch / 'build'
    flags = ['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DGGML_NATIVE=OFF',
             '-DCRISPEMBED_NATIVE=OFF','-DGGML_BLAS=OFF','-DGGML_METAL=OFF','-DGGML_CUDA=OFF',
             '-DGGML_VULKAN=OFF','-DGGML_OPENCL=OFF','-DGGML_LLAMAFILE=OFF','-DCRISPEMBED_BUILD_SHARED=ON']
    flags += ['-D'+name+'='+str(scratch / 'absent-vendor') for name in ('CRISP_AUDIO_DIR','CRISP_PUNC_DIR','CRISP_LID_DIR','CRISP_TRUECASE_DIR')]
    command(['cmake','-S',ROOT,'-B',build_dir,*flags], out / 'configure.log')
    command(['cmake','--build',build_dir,'--target','crispembed-shared','test-ultrabert-truncation','-j2'], out / 'build.log', timeout=5400)
    command([sys.executable,'scripts/check-cpu-baseline.py',build_dir], out / 'baseline.log')
    for path in sorted(build_dir.rglob('*.so*')):
        if path.is_file():
            dest = payload / 'lib' / path.name
            if dest.exists() and sha(dest) != sha(path):
                raise ValueError('Conflicting bundled library basenames')
            shutil.copyfile(path, dest)
    library = payload / 'lib/libcrispembed.so'
    if not library.is_file(): raise ValueError('Required actual shared library absent')
    env = {k:v for k,v in os.environ.items() if not k.startswith('CRISPEMBED_')}
    env.update(protocol()['gates']); env['LD_LIBRARY_PATH'] = str(payload / 'lib')
    command([build_dir / 'test-ultrabert-truncation', payload / 'native-f32.gguf'], out / 'truncation.log', env=env, timeout=120)
    dependency_check(library, out / 'ldd.log', env)
    smoke=library_smoke(payload,out)
    shutil.copyfile(build_dir / 'compile_commands.json', out / 'compile_commands.json')
    r = dict(schema='crispembed-scifact-native-build-v3', source=closure, model_sha256=sha(payload / 'native-f32.gguf'),
             library_sha256=sha(library), loader_smoke=smoke, files=inventory(payload), logs={p.name:sha(p) for p in out.glob('*.log')})
    write(payload / 'receipt.json', r); write(out / 'build.json', r)
    return 0


def current_payload(name, job, target, closure):
    proof = fetch_artifact(int(os.environ['GITHUB_RUN_ID']), closure['head'], dict(name=name), target, complete=False, job=job)
    r = receipt(target)
    if r['source'] != closure: raise ValueError('Same-run source closure differs')
    return r, proof


def checked_bundle(bundle, closure):
    b = receipt(bundle)
    if (b.get('schema') != 'crispembed-scifact-native-build-v3' or b['source'] != closure
            or b['model_sha256'] != protocol()['native_model_sha256']
            or sha(bundle / 'native-f32.gguf') != protocol()['native_model_sha256']
            or b['library_sha256'] != sha(bundle / 'lib/libcrispembed.so')):
        raise ValueError('Authenticated build model/library/source identity differs')
    smoke=load(bundle/'loader-smoke.json')
    if (b.get('loader_smoke')!=smoke or smoke.get('schema')!='crispembed-scifact-native-loader-smoke-v1'
            or smoke.get('mode')not in loader.MODES or smoke.get('completed')is not True or smoke.get('all_ggml_mappings_in_bundle')is not True
            or smoke.get('library_sha256')!=b['library_sha256'] or smoke.get('symbols')!=list(loader.SYMBOLS)
            or any(smoke.get(k)is not False for k in ('model_files_accessed','framework_imported','forward_called'))):
        raise ValueError('Build fresh-Python loader smoke proof differs')
    loader.checked_manifest(bundle/'lib',bundle/'loader-libraries.json',smoke['library_manifest_sha256'])
    loader.loader_path(bundle/'lib',smoke)
    return b


def check_cached_artifact(proof, target, *, complete=True, job=None):
    """Reauthenticate service identity and retained ZIP, without downloading again."""
    run_identity(proof['run'], proof['source'], complete=complete)
    a = api('actions/artifacts/' + str(proof['artifact_id']))
    if (a['workflow_run']['id'] != proof['run'] or a['workflow_run']['head_sha'] != proof['source']
            or a['name'] != proof['name'] or a['size_in_bytes'] != proof['bytes'] or a['expired']
            or a.get('digest') != 'sha256:' + proof['sha256']):
        raise ValueError('Cached artifact service proof differs')
    if job:
        jobs = api('actions/runs/' + str(proof['run']) + '/jobs?per_page=100')
        found = [r for r in jobs['jobs'] if r['name'].split(' / ')[-1] == job]
        if jobs['total_count'] > 100 or len(found) != 1 or found[0]['status'] != 'completed' or found[0]['conclusion'] != 'success':
            raise ValueError('Cached producer job proof differs')
    archive = target.parent / (target.name + '.zip')
    if archive.is_symlink() or archive.stat().st_size != proof['bytes'] or sha(archive) != proof['sha256']:
        raise ValueError('Cached authenticated archive bytes differ')
    with zipfile.ZipFile(archive) as z:
        safe_zip(z)
        if (target / 'receipt.json').is_file():
            if hashlib.sha256(z.read('receipt.json')).hexdigest() != sha(target / 'receipt.json'):
                raise ValueError('Cached receipt is not the authenticated archive member')


def check_pinned_proof(proof, pin, run, head):
    expected = dict(run=run, source=head, artifact_id=pin['id'], name=pin['name'], bytes=pin['bytes'], sha256=pin['sha256'])
    if any(proof.get(k) != v for k, v in expected.items()):
        raise ValueError('Staged immutable reference proof differs')


def validate_stage(path, expected_sha, index, scratch, closure):
    if path.resolve() != (scratch / 'native-stage.json').resolve() or path.is_symlink() or path.stat().st_size > 4*1024**2 or sha(path) != expected_sha:
        raise ValueError('Native child stage path/bytes/hash differs')
    stage = load(path)
    if (stage.get('schema') != 'crispembed-scifact-native-child-stage-v1' or stage.get('shard') != index
            or stage.get('source') != closure or set(stage.get('directories', {})) != {'token-proof','freeze','prepared','bundle'}):
        raise ValueError('Native child stage source/shard/scope differs')
    for name, files in stage['directories'].items():
        if inventory(scratch / name) != files:
            raise ValueError('Staged materialized bytes differ: ' + name)
    p = protocol(); lineage = stage['lineage']
    for label, group, proof_key in [('token-proof',p['token_proof'],'token_proof'),('freeze',p['original'],'original_freeze'),('prepared',p['prepared'],'preparation')]:
        pin = group['freeze_artifact'] if label == 'freeze' else group['artifact']
        check_pinned_proof(lineage[proof_key], pin, group['run'], group['source'])
        check_cached_artifact(lineage[proof_key], scratch / label, complete=label != 'freeze', job='freeze-original' if label == 'freeze' else None)
    old_meta = run_identity(p['original']['run'], p['original']['source'], complete=False)
    if check_jobs(old_meta, api('actions/runs/' + str(p['original']['run']) + '/jobs?per_page=100')) != lineage['old_jobs']:
        raise ValueError('Retained cancelled original parent/job proof differs')
    bp = stage['build']
    if bp['run'] != int(os.environ['GITHUB_RUN_ID']) or bp['source'] != closure['head'] or bp['name'] != 'ultrabert-scifact-native-v3-build':
        raise ValueError('Staged same-run build proof differs')
    check_cached_artifact(bp, scratch / 'bundle', complete=False, job='build')
    prepared, data, tokens, frozen = materialized_references(scratch)
    b = checked_bundle(scratch / 'bundle', closure)
    return prepared, data, tokens, frozen, lineage, b, bp


def launch_plan(index, scratch, stage_sha, smoke_policy=None):
    library_directory = (scratch / 'bundle/lib').resolve()
    if not library_directory.is_dir() or ':' in str(library_directory):
        raise ValueError('Fresh child requires an already materialized exact library directory')
    env = smoke_environment(library_directory,smoke_policy or dict(mode='bundle-only'))
    env.update(protocol()['gates'])
    argv = [sys.executable, str(ROOT / 'tools/ci-heavy/ultrabert_scifact_native_v3.py'), '--phase', 'native-child',
            '--shard', str(index), '--stage', str(scratch / 'native-stage.json'), '--stage-sha', stage_sha]
    return argv, env


def native(index, out, scratch, closure):
    # Materialize and authenticate first. No NumPy/binding/ctypes import here.
    prepared, data, tokens, frozen, lineage = references(scratch)
    bundle = scratch / 'bundle'
    b, bp = current_payload('ultrabert-scifact-native-v3-build', 'build', bundle, closure)
    checked_bundle(bundle, closure)
    stage = scratch / 'native-stage.json'
    if stage.exists():raise ValueError('Fresh native child stage required')
    write(stage, dict(schema='crispembed-scifact-native-child-stage-v1', shard=index, source=closure,
                     lineage=lineage, build=bp, directories={name:inventory(scratch / name) for name in ('token-proof','freeze','prepared','bundle')}))
    stage_sha = sha(stage)
    argv, env = launch_plan(index, scratch, stage_sha, b['loader_smoke'])
    write(out / 'native-child-launch.json', dict(fresh_python_process=True, stage_sha256=stage_sha,
          exact_existing_library_directory=True, loader_mode=b['loader_smoke']['mode'], inherited_preload=False, downloads_in_child=False))
    command(argv, out / 'native-child.log', env=env, timeout=protocol()['execution']['shard_deadline_seconds']+300)
    return 0


def native_child(index, out, scratch, closure, stage, stage_sha):
    prepared, data, tokens, frozen, lineage, b, bp = validate_stage(stage, stage_sha, index, scratch, closure)
    bundle = scratch / 'bundle'
    env = protocol()['gates']
    if (os.environ.get('LD_LIBRARY_PATH') != loader.loader_path(bundle / 'lib',b['loader_smoke'])
            or any(os.environ.get(k) != v for k,v in env.items()) or os.environ.get('LD_PRELOAD') or os.environ.get('LD_AUDIT')):
        raise ValueError('Fresh native child loader/gate environment differs')
    import numpy as np
    dependency_check(bundle / 'lib/libcrispembed.so', out / 'ldd.log', os.environ.copy())
    # Bundle existed when this fresh child started; never mutate loader paths here.
    sys.path.insert(0, str(ROOT / 'python'))
    from crispembed import CrispEmbed
    model = CrispEmbed(str(bundle / 'native-f32.gguf'), n_threads=2, lib_path=str(bundle / 'lib/libcrispembed.so'), offline=True)
    if (Path(model._lib._name).resolve() != (bundle / 'lib/libcrispembed.so').resolve()
            or Path(inspect.getfile(CrispEmbed)).resolve() != (ROOT / 'python/crispembed/_binding.py').resolve()):
        raise ValueError('Binding/library fallback differs')
    loaded = [line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/libggml' in line or '/libcrispembed' in line]
    if not loaded or any(Path(path).resolve().parent != (bundle / 'lib').resolve() for path in loaded):
        raise ValueError('Actual loaded ggml/CrispEmbed dependency fallback differs')
    write(out / 'loaded-libraries.json', {Path(path).name:sha(path) for path in set(loaded)})
    rows = shard_rows(data,index); raw=[]; normalized=[]; ids=[]; timing=[]
    deadline = time.monotonic() + protocol()['execution']['shard_deadline_seconds']
    payload = out / 'payload'; payload.mkdir()
    for row in rows:
        if time.monotonic()>deadline: raise TimeoutError('Native shard bounded deadline')
        model.set_prefix(old.QUERY if row['id'].startswith('q:') else '')
        start=time.perf_counter(); actual, hidden=model.encode_tokens(row['text'],normalize=False)
        observed=actual.tolist()
        write(out / 'latest-id-preflight.json', dict(id=row['id'],actual=observed,expected=tokens[row['id']]['ids']))
        ids.append(check_actual_ids(row['id'], observed, tokens[row['id']]))
        if hidden.shape != (len(observed),768) or hidden.dtype != np.dtype('float32') or not np.isfinite(hidden).all():
            raise ValueError('Actual raw CAPI hidden dimensions/dtype/finite differs')
        raw.append(hidden[0].copy()); timing.append(dict(id=row['id'],raw_seconds=time.perf_counter()-start,tokens=len(observed)))
        if len(raw) % 16 == 0 or len(raw) == len(rows):
            np.save(payload / 'unaccepted-raw.npy', np.asarray(raw,dtype='<f4'),allow_pickle=False)
            write(payload / 'unaccepted-id-preflight.json',ids)
        write(out / 'id-preflight.json', dict(accepted_vectors=False, rows_checked=len(ids), total=len(rows), mask_provenance='DERIVED'))
    # All assigned IDs/lengths and derived all1 masks pass before accepting vectors.
    write(out / 'id-preflight.json', dict(accepted_vectors=True, rows_checked=len(ids),total=len(rows),mask_provenance='DERIVED'))
    for n,row in enumerate(rows):
        if time.monotonic()>deadline: raise TimeoutError('Native shard bounded deadline')
        model.set_prefix(old.QUERY if row['id'].startswith('q:') else '')
        start=time.perf_counter(); vector=model.encode(row['text'])
        if vector.shape!=(768,) or vector.dtype!=np.dtype('float32') or not np.isfinite(vector).all(): raise ValueError('Normalized CAPI vector invalid')
        norm=float(np.linalg.norm(raw[n].astype('float64')))
        if norm<=1e-12 or np.max(abs(raw[n].astype('float64')/norm-vector))>1e-6: raise ValueError('Raw CLS/normalized scalar inconsistency')
        normalized.append(vector); timing[n]['normalized_seconds']=time.perf_counter()-start
        if len(normalized) % 16 == 0 or len(normalized) == len(rows):
            old.save_vectors(payload, raw[:len(normalized)], normalized, ids[:len(normalized)], timing[:len(normalized)])
    (payload / 'unaccepted-raw.npy').unlink(missing_ok=True)
    (payload / 'unaccepted-id-preflight.json').unlink(missing_ok=True)
    shutil.copytree(prepared / 'notices',payload / 'notices')
    r=dict(schema='crispembed-scifact-native-shard-v3',status='completed native shard',source=closure,shard=index,
           rows=[r['id'] for r in rows], data_freeze_sha256=protocol()['prepared']['data_freeze_sha256'],
           frozen_protocol_sha256=protocol()['frozen_protocol_sha256'], original_freeze_sha256=protocol()['original']['freeze_report_sha256'],
           all_actual_ids_preflight_exact=True, mask_contract=protocol()['mask_contract'],gates=env,
           model_sha256=b['model_sha256'],library_sha256=b['library_sha256'],build=bp,lineage=lineage,
           loader_child=dict(stage_sha256=stage_sha, fresh_python_process=True, authenticated_materialized_inputs=True,
                             loader_mode=b['loader_smoke']['mode'], loader_smoke_sha256=sha(bundle/'loader-smoke.json')),files=inventory(payload))
    write(payload / 'receipt.json',r);write(out / ('native-'+str(index)+'.json'),r)
    return 0


def aggregate(out,scratch,closure):
    import numpy as np
    prepared,data,tokens,frozen,lineage=references(scratch)
    records={}; services=[]; libraries=set();builds=set()
    ordered=[r['id'] for r in data['queries']+data['documents']]
    for mode in ('original','native'):
        values={}
        for i in range(8):
            target=scratch/(mode+'-'+str(i)); rows=[r['id'] for r in shard_rows(data,i)]
            if mode=='original':
                pin=protocol()['original']['shards'][i]
                proof=fetch_artifact(pin['run'],pin['head'],dict(name=pin['name'],id=pin['artifact_id'],bytes=pin['bytes'],sha256=pin['digest'].removeprefix('sha256:')),target,complete=False,job='original-shard-'+str(i))
                r=receipt(target,pin['receipt_sha256'])
            else:
                r,proof=current_payload('ultrabert-scifact-native-v3-'+str(i),'native-shard-'+str(i),target,closure)
                libraries.add(r['library_sha256']);builds.add(r['build']['sha256'])
            check_shard(r,mode,i,rows,frozen['shards'][i] if mode=='original' else None)
            a=np.load(target/'raw.npy',allow_pickle=False);b=np.load(target/'normalized.npy',allow_pickle=False); ids=load(target/'ids.json')
            if a.dtype!=np.dtype('<f4') or b.dtype!=a.dtype or a.shape!=(len(rows),768) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or [x['id'] for x in ids]!=rows:
                raise ValueError('Actual F32 vector/ID shapes/finite coverage differs')
            for n,name in enumerate(rows):
                check_actual_ids(name,ids[n]['ids'],tokens[name])
                if mode=='native' and ids[n]!=check_actual_ids(name,ids[n]['ids'],tokens[name]):raise ValueError('Derived mask receipt differs')
                norm=float(np.linalg.norm(a[n].astype('float64')))
                if name in values or norm<=1e-12 or np.max(abs(a[n].astype('float64')/norm-b[n]))>1e-6:raise ValueError('Duplicate row or raw/normalized consistency differs')
                values[name]=dict(raw=a[n],normalized=b[n])
            services.append(dict(mode=mode,shard=i,service=proof))
        if set(values)!=set(ordered):raise ValueError('Full5483 stable-ID union differs')
        records[mode]=values
    if len(libraries)!=1 or len(builds)!=1:raise ValueError('Native shards did not share exact build/library')
    comparisons=[]
    for name in ordered:
        for mode in ('raw','normalized'):
            a=records['original'][name][mode].astype('float64');b=records['native'][name][mode].astype('float64');e=abs(a-b)
            an=float(np.linalg.norm(a));bn=float(np.linalg.norm(b));cos=float(np.dot(a,b)/(an*bn)) if an*bn else 0.
            comparisons.append(dict(id=name,mode=mode,max_abs=float(e.max()),mean_abs=float(e.mean()),rms=float(np.sqrt(np.mean(e**2))),original_norm=an,native_norm=bn,original_min=float(a.min()),original_max=float(a.max()),native_min=float(b.min()),native_max=float(b.max()),cosine=cos,passed=float(e.max())<=.002 and cos>=.99999))
    payload=out/'payload';payload.mkdir();rankings={};documents=[r['id'] for r in data['documents']]
    for mode in ('original','native'):
        matrix=np.asarray([records[mode][name]['normalized'] for name in ordered],dtype='float64');matrix/=np.linalg.norm(matrix,axis=1,keepdims=True)
        scores=matrix[:300]@matrix[300:].T;np.save(payload/(mode+'-scores.npy'),scores,allow_pickle=False)
        rankings[mode]=[dict(query_id=r['id'],**rank_scores(documents,scores[i],r['relevant_document_ids'])) for i,r in enumerate(data['queries'])]
        write(payload/(mode+'-rankings.json'),rankings[mode])
    result=quality(rankings['original'],rankings['native']); passed=result['passed'] and all(r['passed'] for r in comparisons)
    write(payload/'vector-comparisons.json',comparisons);shutil.copytree(prepared/'notices',payload/'notices')
    r=dict(schema='crispembed-scifact-quality-v3',source=closure,status='passed bounded frozen SciFact gates' if passed else 'rejected frozen SciFact gates',
           protocol_sha256=sha(ROOT/CONTRACT),frozen_protocol_sha256=protocol()['frozen_protocol_sha256'],data_freeze_sha256=protocol()['prepared']['data_freeze_sha256'],
           lineage=lineage,service=services,queries=300,corpus=5183,vector_comparisons=len(comparisons),all5483_actual_ids_exact=True,
           mask_contract=protocol()['mask_contract'],quality=result,vector_failures=[x for x in comparisons if not x['passed']],
           first_vector_failure=next((x for x in comparisons if not x['passed']),None),limits=protocol()['limits'],files=inventory(payload))
    write(payload/'receipt.json',r);write(out/'scifact-quality-v3.json',r)
    return int(not passed)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('build','native','native-child','aggregate','preflight'),required=True);parser.add_argument('--shard',type=int,choices=range(8));parser.add_argument('--stage',type=Path);parser.add_argument('--stage-sha');args=parser.parse_args()
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);out.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    report=dict(phase=args.phase,status='started',source=None)
    try:
        closure=source();report['source']=closure;write(out/'source.json',closure)
        packages={'numpy':'2.2.6'}
        if args.phase=='build':packages.update({'gguf':'0.19.0','huggingface-hub':'1.33.0','safetensors':'0.8.0'})
        if args.phase!='preflight' and any(importlib.metadata.version(k)!=v for k,v in packages.items()):
            raise ValueError('Pinned native dependency versions differ')
        write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),packages=packages,threads=2))
        if args.phase=='preflight':return 0
        code={'build':lambda:build(out,scratch,closure),'native':lambda:native(args.shard,out,scratch,closure),'native-child':lambda:native_child(args.shard,out,scratch,closure,args.stage,args.stage_sha),'aggregate':lambda:aggregate(out,scratch,closure)}[args.phase]()
        report.update(status='completed' if code==0 else 'retained numerical/quality rejection',exit_code=code);return code
    except Exception as error:
        message=str(error)
        for value,label in ((str(ROOT),'<source>'),(str(scratch),'<scratch>'),(str(out),'<out>'),(os.environ.get('HF_HOME',''),'<hf-cache>')):
            if value:message=message.replace(value,label)
        report.update(status='structural rejection',error=message);return 1
    finally:write(out/('phase-native-child.json' if args.phase=='native-child' else 'phase.json'),report)


if __name__=='__main__':sys.exit(main())
