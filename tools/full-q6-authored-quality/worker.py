#!/usr/bin/env python3
"""Paired fresh authored-document quality, using retained quantized models/runtime."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'tools'))
import ocr_quality
spec = importlib.util.spec_from_file_location('observed_utils', ROOT/'tools/full-q6-observed/worker.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
sha, require = shared.sha, shared.require


def find_file(name, expected=None):
    candidates = list(Path('/kaggle/input').rglob(name))
    if expected is not None:
        candidates = [p for p in candidates if sha(p) == expected]
    require(len(candidates) == 1, 'one retained ' + name + ', found ' + str(len(candidates)))
    return candidates[0]


def paired_stages(q4, q6, np):
    names = ['vision.patch', 'vision.position', 'vision.embedded']
    names += ['vision.layer.'+str(i) for i in range(27)]
    names += ['vision.merged', 'prefill.input']
    names += ['prefill.layer.'+str(i) for i in range(32)]
    names += ['prefill.norm', 'prefill.logits']
    stages = []
    for name in names:
        left, right = q4/(name+'.f32'), q6/(name+'.f32')
        if not left.exists() and not right.exists():
            continue
        require(left.exists() and right.exists(), 'both paired stages ' + name)
        values = shared.numerical(np.fromfile(right, dtype='<f4'), np.fromfile(left, dtype='<f4'), np)
        stages.append(dict(stage=name, **values))
    return dict(candidate='q6', reference='q4', stages=stages,
                earliest_saved_difference=next((s['stage'] for s in stages if not s['bit_exact']), None),
                limit='Cross-precision differences, not same-weight operator parity; only saved boundaries inspected.')


def main():
    import numpy as np
    from PIL import Image
    require(Path('/kaggle/input').is_dir(), 'Kaggle execution required')
    cfg = json.loads((HERE/'protocol.json').read_text())
    fixtures = json.loads((HERE/'fixtures.json').read_text())
    require([c['id'] for c in fixtures['cases']] == cfg['cases'], 'fixed authored case order')
    out = Path(os.environ['HEAVY_OUT'])
    work = Path(os.environ['HEAVY_SCRATCH'])/'authored-quality'
    work.mkdir()
    started = time.monotonic()
    report = dict(schema=cfg['schema'], status='failed', commands={}, cases={}, qualified=False,
                  original_forward_executed=False, source_forward_executed=False,
                  previous_case_forward_executed=False, runtime_rebuilt=False, model_reimported=False,
                  fixture_sha256=sha(HERE/'fixtures.json'), protocol_sha256=sha(HERE/'protocol.json'),
                  controller_source=subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip(),
                  runtime_source=cfg['runtime_source'], quality_policy=ocr_quality.POLICY)
    def save():
        (out/'authored-quality.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    def run(label,args,env):
        report['phase']=label
        save()
        print('PHASE '+label,flush=True)
        receipt=report['commands'].setdefault(label,{})
        try:
            shared.command(args,out/(label+'.log'), min(cfg['arm_seconds'],cfg['worker_seconds']-(time.monotonic()-started)),
                           receipt,env=env,cwd=ROOT)
        finally:
            save()
    with shared.Deadline(cfg['worker_seconds']):
        try:
            report['capacity']=dict(available_bytes=shared.memory_available(), output_free=shutil.disk_usage(out).free,
                                    scratch_free=shutil.disk_usage(work).free, load=os.getloadavg())
            require(report['capacity']['available_bytes']>=10<<30 and report['capacity']['output_free']>=12<<30,
                    'sufficient memory/output disk before model execution')
            retained=find_file('full-q6-observed.json')
            previous=json.loads(retained.read_text())
            require(previous['source_sha']==cfg['runtime_source'], 'retained runtime source identity')
            bundle=out/'runtime'
            for relative, record in cfg['runtime_bundle'].items():
                source=shared.verified(retained.parent/'runtime'/relative,record)
                dest=bundle/relative
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,dest)
            binary=bundle/'bin/lighton4b-native-document'
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            report['runtime_bundle']=cfg['runtime_bundle']
            models={arm:find_file(cfg[arm+'_model']['file'],cfg[arm+'_model']['sha256']) for arm in cfg['arms']}
            report['models']={arm:dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for arm,p in models.items()}
            # The reference tool requires one teacher ID; greedy-full does not step with it.
            # IDs/text come solely from actual independent argmax rows, never from gold text.
            teacher=work/'dummy-teacher.ids'
            teacher.write_text('1\n')
            clean=dict(os.environ)
            clean.update({key:'0' for key in clean if key.startswith('CRISPEMBED_')})
            clean.update(LD_LIBRARY_PATH=str(bundle/'lib'),OPENBLAS_NUM_THREADS='1')
            for case in fixtures['cases']:
                require(sha(HERE/case['image'])==case['png_sha256'],'frozen PNG '+case['id'])
                image=Image.open(HERE/case['image']).convert('RGB')
                require(image.size==(case['width'],case['height']),'frozen geometry')
                rgb=work/(case['id']+'.rgb')
                rgb.write_bytes(image.tobytes())
                require(sha(rgb)==case['rgb_sha256'],'frozen actual RGB')
                row=report['cases'].setdefault(case['id'],dict(arms={},gold=case['gold'],input=dict(
                    png_sha256=case['png_sha256'],rgb_sha256=case['rgb_sha256'],width=image.width,height=image.height)))
                for arm in cfg['arms']:
                    require(shared.memory_available()>=8<<30,'per-arm RAM')
                    dest=out/case['id']/arm
                    dest.mkdir(parents=True)
                    gates=dict(cfg['q6_gates'])
                    gates[shared.GATE]='1' if arm=='q6' else '0'
                    gates[shared.REPACK]='0' if arm=='q6' else '1'
                    env=dict(clean,**gates)
                    run(case['id']+'-'+arm,[binary,models[arm],rgb,image.height,image.width,teacher,dest,
                        cfg['cap'],cfg['threads'],'greedy-full','--observed-composed-cap512'],env)
                    summary=json.loads((dest/'native-summary.json').read_text())
                    ids=list(map(int,(dest/'generated.ids').read_text().split()))
                    text=(dest/'generated.txt').read_bytes().decode('utf-8')
                    shared.stop_state(summary,ids,cfg['cap'])
                    hashes=shared.rows(dest,ids,np)
                    require(summary['physical_full_q6']==(arm=='q6') and summary['physical_q4_repack']==(arm=='q4'),
                            'actual precision/repack profile')
                    require(summary['converted_f32_weight_entries']==summary['converted_f32_weight_bytes']==0,
                            'no F32 text-weight cache')
                    if arm=='q6':
                        require(summary['physical_repack_entries']==summary['physical_repack_bytes']==0,'no Q4 repack in Q6')
                    score=ocr_quality.score(case['gold'],text,complete=summary['eos'])
                    row['arms'][arm]=dict(summary=summary,gates=gates,generated_ids=ids,decoded=text,
                        quality=score, raw_character_edits=shared.edits(text,case['gold']['text']),
                        raw_word_edits=shared.edits(text.split(),case['gold']['text'].split()),
                        content_character_edits=shared.edits(score['parsed_text'],ocr_quality.words(case['gold']['text'])) if score['content'] else None,
                        content_word_edits=shared.edits(score['parsed_text'].split(),case['gold']['text'].split()) if score['content'] else None,
                        vocabulary_rows=len(hashes),row_sha256=hashes,
                        elapsed_seconds=report['commands'][case['id']+'-'+arm].get('elapsed_seconds'))
                    save()
                row['paired_prefill']=paired_stages(out/case['id']/'q4',out/case['id']/'q6',np)
                q4,q6=row['arms']['q4']['quality'],row['arms']['q6']['quality']
                row['comparison']=dict(content_character_edit_delta_q6_minus_q4=(
                    shared.edits(q6['parsed_text'],ocr_quality.words(case['gold']['text']))-shared.edits(q4['parsed_text'],ocr_quality.words(case['gold']['text']))) if q4['content'] and q6['content'] else None)
                save()
            report['status']='measured_authored_q6_q4_quality'
        except BaseException as error:
            report.update(error_type=type(error).__name__,error=str(error))
            raise
        finally:
            report['elapsed_seconds']=time.monotonic()-started
            report['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))
                for p in out.rglob('*') if p.is_file() and p.name!='authored-quality.json' and p.suffix!='.log'}
            save()

if __name__=='__main__':
    main()
