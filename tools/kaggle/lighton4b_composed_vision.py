#!/usr/bin/env python3
"""Observed-case paired projector diagnostic; no held-out or shipping qualification."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline, command


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


f = load('form_tools', ROOT / 'tools/kaggle/lighton4b_physical_f16_full_form.py')
require, sha = f.require, f.sha
CASES = ['simple_form', 'simple_table', 'arabic_printed_line']
FORM = '0a0c2084baa4e46aa80f09f7ae94fd4b1c56ccae2bbf2f9ef6210e81e8cf1d01'
PHYSICAL = '712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2'
CALIBRATION = '7639d6559da7cc58443ccfbe1dac518af0d70f3e7a83718d037a97bb147eb2b7'
SOURCE = '4a63d73bee7289deab41fb20f67765d3c6800182'


def admit(cfg):
    require(cfg['schema'] == 'lighton4b-composed-vision/v1' and cfg['cases'] == CASES, 'fixed observed cases')
    require(cfg['arms'] == ['source-features', 'composed'] and cfg['cap'] == 512 and
            cfg['threads'] == 4 and cfg['context'] == 2048 and
            cfg['arm_seconds'] == 2400 and cfg['worker_seconds'] == 6000, 'closed bounded pair')
    require(cfg['receipts'] == {'form': FORM, 'physical': PHYSICAL, 'calibration': CALIBRATION}, 'frozen receipts')
    require(cfg['production_source'] == SOURCE and cfg['quality_qualified'] is False, 'source and limits')


def native_args(native, candidate, rgb, height, width, teacher, dest, arm):
    require(arm in ('source-features', 'composed'), 'closed native arm')
    args = [native, candidate, rgb, height, width, teacher, dest, 512, 4, 'greedy-full']
    if arm == 'source-features':
        args += ['--upstream-vision', rgb.parent / 'observer-on/source.vision.0.f32']
    return args + ['--observed-composed-cap512']


def vector_metric(a, b):
    import numpy as np
    x, y = np.fromfile(a, dtype='<f4'), np.fromfile(b, dtype='<f4')
    require(x.size > 0 and x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), 'equivalent finite stage')
    d = x.astype('f8') - y.astype('f8')
    return dict(bit_exact=x.tobytes() == y.tobytes(), elements=int(x.size),
                rms=float(np.sqrt(np.mean(d*d))), max_abs=float(abs(d).max()),
                reference_max_abs=float(abs(y).max()), candidate_sha256=sha(a), reference_sha256=sha(b))


def verify(path, row):
    require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'authenticated artifact ' + path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--controls-only', action='store_true', help='Stop before any model acquisition/forward')
    options = parser.parse_args()
    import numpy as np
    from PIL import Image
    require(Path('/kaggle/input').is_dir() and not list(Path('/dev').glob('nvidia[0-9]*')), 'Kaggle CPU only')
    require(np.__version__ == '2.3.5' and Image.__version__ == '11.3.0', 'serializer pins')
    cfgpath = ROOT / 'docs/lighton4b-composed-vision-v1.json'
    cfg = json.loads(cfgpath.read_text()); admit(cfg)
    out = Path(os.environ['HEAVY_OUT']); scratch = Path(os.environ['HEAVY_SCRATCH']) / 'composed-vision'
    scratch.mkdir(); started = time.monotonic(); deadline = Deadline(6000); deadline.__enter__()
    report = dict(schema=cfg['schema'], case=options.case, status='failed', protocol_sha256=sha(cfgpath),
                  quality_qualified=False, heldout=False, original_forward_executed=False,
                  f32_model_instantiated=False, arms={}, commands={},
                  projector_weight_equivalence='unproven; deployed projectors, not pure arithmetic comparison',
                  unsupported_upstream_stages=['preprocessing patches', 'patch/rotary', 'vision block intermediates'],
                  reference_limits='BF16 capture is a reference output, not human ground truth; upstream table256 is capped')
    def save():
        (out / 'composed-vision.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    def run(label, args, env=None, timeout=600, on_tick=None):
        report['phase'] = label; report['commands'][label] = {}; save(); print('phase', label, flush=True)
        try:
            command(args, out / (label+'.log'), min(timeout, 6000-(time.monotonic()-started)),
                    report['commands'][label], env=env, cwd=ROOT, on_tick=on_tick)
        finally:
            save()
    def locate(name, digest):
        found = [p for p in Path('/kaggle/input').rglob(name) if p.is_file() and p.stat().st_size < 4 << 20 and sha(p) == digest]
        require(len(found) == 1, 'unique mounted receipt ' + name)
        return found[0]
    try:
        run('source', ['git', 'rev-parse', 'HEAD']); report['source_sha'] = (out/'source.log').read_text().strip()
        require(report['source_sha'] == os.environ['LIGHTON4B_EXPECTED_SOURCE'], 'exact controller source')
        base = load('physical_base', ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py')
        sys.path.insert(0, str(ROOT/'tools')); import lighton4b_upstream_physical_pack as pack
        report['preflight'] = dict(memory=base.memory(), scratch_free=shutil.disk_usage(scratch).free,
                                   output_free=shutil.disk_usage(out).free, load=os.getloadavg())
        require(report['preflight']['memory']['MemAvailable'] >= 10 << 30 and
                report['preflight']['scratch_free'] >= 25 << 30 and report['preflight']['output_free'] >= 8 << 30, 'effective capacity')
        fp = locate('physical-f16-full-form.json', FORM); form = json.loads(fp.read_text())
        pp = locate('physical-decode.json', PHYSICAL); physical = json.loads(pp.read_text())
        cp = locate('physical-f16-calibration.json', CALIBRATION); calibration = json.loads(cp.read_text())
        require(form['status'] == 'measured_full_form' and form['source_sha'] == SOURCE and
                physical['status'] == 'measured' and calibration['status'] == 'measured_two_case_calibration', 'completed dependencies')
        protocol = json.loads((ROOT/'docs/lighton4b-upstream-physical-decode-v1.json').read_text())
        case = next(c for c in protocol['cases'] if c['id'] == options.case)
        folder = pp.parent/options.case; art = physical['cases'][options.case]['artifacts']
        require(physical['cases'][options.case]['source_observer_neutral'] is True, 'actual source two-prefix observer neutrality')
        required = ['input.rgb', 'original.json', 'retained-upstream-response.json', 'upstream_features/prompt.ids',
                    'upstream_features/modalities.ids', 'upstream_features/prefill.input.f32', 'upstream_features/prefill.positions.f32',
                    'observer-on/source.vision.0.f32', 'observer-on/source.logits.0.f32']
        for rel in required: verify(folder/rel, art[rel])
        require(sha(folder/'original.json') == case['reference_sha256'] and
                sha(folder/'retained-upstream-response.json') == case['upstream_response_sha256'], 'reference identity')
        require(sha(ROOT/case['image']) == case['image_sha256'], 'frozen original image')
        image = Image.open(ROOT/case['image']).convert('RGB')
        require(image.tobytes() == (folder/'input.rgb').read_bytes(), 'actual captured RGB')
        ids = list(map(int, (folder/'upstream_features/prompt.ids').read_text().split()))
        mods = list(map(int, (folder/'upstream_features/modalities.ids').read_text().split()))
        buf = io.BytesIO(); np.save(buf, np.array([ids], dtype='<i8'), allow_pickle=False)
        require(hashlib.sha256(buf.getvalue()).hexdigest() == case['expected_input_ids_npy_sha256'] and
                len(ids) == len(mods) == case['expected_prompt_tokens'] and mods.count(1) == case['image_tokens'] and
                len(ids)+512 <= 2048, 'prompt/modality/context closure')
        original = json.loads((folder/'original.json').read_text()); retained = json.loads((folder/'retained-upstream-response.json').read_text())
        require(len(original['generated_ids']) == case['reference_generated_ids'] and original['generated_ids'][-1] in (248044,248046), 'complete original reference IDs')
        source_row = np.fromfile(folder/'observer-on/source.logits.0.f32', dtype='<f4')
        require(source_row.size == 248320 and np.isfinite(source_row).all(), 'actual source score')
        teacher = scratch/'one.ids'; teacher.write_text(str(original['generated_ids'][0])+'\n')
        golden = fp.parent/'f16-full' if options.case == 'simple_form' else cp.parent/options.case
        golden_art = form['artifacts'] if options.case == 'simple_form' else calibration['artifacts']
        prefix = 'f16-full/' if options.case == 'simple_form' else options.case+'/'
        old = form['arms']['f16-full']['generated_ids'] if options.case == 'simple_form' else calibration['cases'][options.case]['generated_ids']
        # Authenticate every old row and stage before rebuilding or acquiring any model.
        for rel in list(f.STAGES) + [f'greedy.{i}.logits.f32' for i in range(len(old))]: verify(golden/rel, golden_art[prefix+rel])
        report['inputs_authenticated'] = True; report['receipts'] = cfg['receipts']; report['actual_input_hashes']={rel:art[rel] for rel in required}; save()
        # Only the explicitly reviewed test CLI differs; production/CMake remain tested bytes.
        run('fetch-production', ['git','fetch','--depth','1','origin',SOURCE])
        run('production-tree', ['git','diff','--exit-code',SOURCE,'HEAD','--','src','ggml'])
        tested = json.loads((ROOT/'docs/lighton4b-physical-f16-full-form-v1.json').read_text())
        require(sha(ROOT/'CMakeLists.txt') == tested['reviewed_cmake_sha256'], 'tested CMake identity')
        run('submodule', ['git','submodule','update','--init','ggml']); build=scratch/'build'
        flags = {'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON',
                 'GGML_LLAMAFILE':'ON','GGML_AVX512':'OFF','GGML_BLAS':'OFF','GGML_BACKEND_DL':'OFF',
                 'GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
        run('configure', ['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}' for k,v in flags.items()]])
        cache=(build/'CMakeCache.txt').read_text().splitlines()
        for k,v in flags.items(): require(f'{k}:BOOL={v}' in cache, 'actual build flag '+k)
        shutil.copy(build/'CMakeCache.txt',out/'CMakeCache.txt')
        run('build', ['cmake','--build',build,'--target','lighton4b-native-document','test-lighton4b-physical-f16-attention','--parallel','2'])
        run('supported-host', [build/'test-lighton4b-physical-f16-attention'])
        require('PHYSICAL_F16_ATTENTION PASS ' in (out/'supported-host.log').read_text(), 'supported CPU/private ABI')
        native=build/'lighton4b-native-document'; report['binary_sha256']=sha(native)
        # Actual binary CLI admission occurs before acquiring any model payload.
        import subprocess
        dummy = scratch/'missing.gguf'; dummy_rgb = scratch/'missing.rgb'
        checks = [
            ('old256', [native,dummy,dummy_rgb,10,10,teacher,out,256,4,'greedy-full'],4),
            ('new-composed', native_args(native,dummy,dummy_rgb,10,10,teacher,out,'composed'),4),
            ('new-fed', native_args(native,dummy,dummy_rgb,10,10,teacher,out,'source-features'),4),
            ('table-legacy', [native,dummy,dummy_rgb,10,10,teacher,out,512,4,'greedy-full','--upstream-vision',dummy_rgb,'--extended-table-cap512'],4),
            ('unflagged512', [native,dummy,dummy_rgb,10,10,teacher,out,512,4,'greedy-full'],2),
            ('reject513', [native,dummy,dummy_rgb,10,10,teacher,out,513,4,'greedy-full','--observed-composed-cap512'],2),
            ('reject-threads', [native,dummy,dummy_rgb,10,10,teacher,out,512,2,'greedy-full','--observed-composed-cap512'],2),
            ('reject-mode', [native,dummy,dummy_rgb,10,10,teacher,out,512,4,'full','--observed-composed-cap512'],2),
            ('reject-unknown', [native,dummy,dummy_rgb,10,10,teacher,out,512,4,'greedy-full','--unknown'],2)]
        for label,args,expected in checks:
            try: run('cli-'+label,args,timeout=30)
            except subprocess.CalledProcessError as e: require(e.returncode==expected,'actual CLI return '+label)
            else: require(expected==0,'expected CLI rejection/missing-metadata '+label)
        report['actual_cli_controls']=len(checks)
        if options.controls_only:
            report['status']='model_free_controls_pass';report['model_acquisition_performed']=False
            return

        env=dict(os.environ); env.update({k:'0' for k in env if k.startswith('CRISPEMBED_')}); env.update(form['arms']['f16-full']['gates'])
        require(env['CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION']=='1' and env['CRISPEMBED_LIGHTON4B_HEAD_BF16']=='0', 'frozen tested gate tuple')
        libs=sorted(p for p in build.rglob('libggml*.so*') if p.is_file()); require(libs,'actual runtime libraries')
        env['LD_LIBRARY_PATH']=':'.join(sorted({str(p.parent) for p in libs})); report['library_hashes']={p.name:sha(p) for p in libs}; report['gates']=form['arms']['f16-full']['gates']
        row=protocol['models']['Q4_K_M']; up=scratch/row['file']; report['phase']='acquire-upstream';save()
        base.download('https://huggingface.co/'+protocol['model_repo']+'/resolve/'+protocol['model_revision']+'/'+row['file'],up,row['bytes'],row['sha256'])
        row=protocol['base_model']; protected=scratch/row['file']; report['phase']='acquire-protected';save()
        base.download('https://huggingface.co/'+row['repository']+'/resolve/'+row['revision']+'/'+row['file'],protected,row['bytes'],row['sha256'])
        candidate=scratch/'physical-native.gguf'; report['phase']='lossless-pack';save()
        audit=pack.convert(protected,up,candidate,out/'physical-pack.json')
        require(audit['candidate_sha256']==form['candidate_sha256'], 'same physical model');report['candidate_sha256']=audit['candidate_sha256'];protected.unlink()
        for arm in cfg['arms']:
            require(base.memory()['MemAvailable']>=10<<30 and shutil.disk_usage(out).free>=4<<30, 'per-arm capacity')
            dest=out/arm; dest.mkdir();last=[0.]
            def tick():
                if time.monotonic()-last[0]<30:return
                last[0]=time.monotonic();report['progress']={'arm':arm,'raw_greedy_rows':len(list(dest.glob('greedy.*.logits.f32')))};save();print('heartbeat',json.dumps(report['progress']),flush=True)
            run(arm,native_args(native,candidate,folder/'input.rgb',image.height,image.width,teacher,dest,arm),env,2400,tick)
            summary=json.loads((dest/'native-summary.json').read_text()); generated=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode('utf8')
            require(generated and len(generated)<=512 and summary['eos']==(generated[-1]in(248044,248046)) and
                    (summary['eos']or len(generated)==512) and summary['teacher_positions']==1 and
                    summary['source_fed_vision']==(arm=='source-features'), 'full independent decode/cap/mode closure')
            require((dest/'prompt.ids').read_bytes()==(folder/'upstream_features/prompt.ids').read_bytes() and
                    (dest/'modalities.ids').read_bytes()==(folder/'upstream_features/modalities.ids').read_bytes() and
                    sha(dest/'prefill.positions.f32')==art['upstream_features/prefill.positions.f32']['sha256'], 'actual prompt/modality/positions')
            for i,token in enumerate(generated):
                row=np.fromfile(dest/f'greedy.{i}.logits.f32',dtype='<f4');require(row.size==248320 and np.isfinite(row).all() and int(row.argmax())==token,'all raw greedy argmax rows')
            require(sha(dest/'teacher.0.logits.f32')==sha(dest/'greedy.0.logits.f32')==sha(dest/'prefill.logits.f32'),'first-score identity')
            if arm=='source-features':
                require(generated[:len(old)]==old and (len(generated)==len(old) if old[-1]in(248044,248046) else len(generated)>=len(old)), 'same old prefix/EOS')
                for rel in list(f.STAGES)+[f'greedy.{i}.logits.f32' for i in range(len(old))]: require(sha(dest/rel)==sha(golden/rel),'all old stage/prefix bytes '+rel)
                require(sha(dest/'vision.merged.f32')==art['observer-on/source.vision.0.f32']['sha256'] and sha(dest/'prefill.input.f32')==art['upstream_features/prefill.input.f32']['sha256'],'source feature splice closure')
            report['arms'][arm]=dict(summary=summary,generated_ids=generated,decoded=text,eos=summary['eos'],cap_hit=not summary['eos'],
                vs_upstream=f.metric(base,generated,text,retained['tokens'],retained['content']),
                vs_original_reference=f.metric(base,generated,text,original['generated_ids'],original['decoded']),
                merged_vs_actual_upstream=vector_metric(dest/'vision.merged.f32',folder/'observer-on/source.vision.0.f32'),
                first_score_vs_actual_upstream=vector_metric(dest/'prefill.logits.f32',folder/'observer-on/source.logits.0.f32'))
            save()
        observed=[]
        for rel in ['processor.patches.f32','text.embedded.f32','vision.merged.f32','prefill.input.f32','prefill.positions.f32']+[f'prefill.layer.{i}.f32' for i in range(32)]+['prefill.norm.f32','prefill.logits.f32']:
            observed.append(dict(stage=rel,**vector_metric(out/'composed'/rel,out/'source-features'/rel)))
        require(sha(out/'composed/processor.grid.ids')==sha(out/'source-features/processor.grid.ids'),'same processor grid')
        report['paired_native_stages']=observed;report['earliest_different_comparable_stage']=next((r['stage'] for r in observed if not r['bit_exact']),None)
        report['stage_scope']='Paired native same-decoder stages; upstream internal visual/text blocks unavailable. Actual upstream comparison only merged/first score.'
        require(sha(candidate)==form['candidate_sha256'],'weights unchanged');report['status']='measured_observed_projector_pair';candidate.unlink()
    except BaseException as error:
        report['error_type']=type(error).__name__;report['error']=str(error);raise
    finally:
        deadline.close();report['elapsed_seconds']=time.monotonic()-started
        report['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for arm in ('source-features','composed') for p in(out/arm).rglob('*') if p.is_file()};save()


if __name__=='__main__':main()
