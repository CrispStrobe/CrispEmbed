"""Only immutable last1;16 F16 +12 Q4 calls, no original/quantizer invocation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import lane as c
import panel as old
from process import run as child


def equal_stage(a,b):return all(a[k]==b[k]for k in ('shape','type','bytes','sha256'))

def barrier(current,retained,refs):
    """Strict bytes include signed zero; never replace a failed byte check with cosine."""
    c.require(set(current)==set(retained)=={'plain','ordinary','observed','repeat'},'four F16 modes required')
    for mode,row in current.items():
        target=retained[mode]
        c.require(row['ids']==target['ids']==refs['ids']and row['text']==target['text']==refs['text']and row['command']['stdout_sha256']==target['command']['stdout_sha256'],'F16 original/retained IDs or raw text bytes differ')
        c.require(row['output_count']==target['output_count']and row['token_cap_reached']is False and
            row['eos_observed']==target['eos_observed']is False and row['stop_contract']==target['stop_contract'],'F16 exact no-cap/source-derived stop contract differs')
    a,b=current['observed'],current['repeat']
    c.require(a['vision']==b['vision']==retained['observed']['vision']==retained['repeat']['vision'],'F16 retained/current/repeat vision bytes differ')
    c.require(set(a['stages'])==set(b['stages'])==set(retained['observed']['stages'])==set(retained['repeat']['stages']),'F16 exact58 coverage differs')
    c.require(len(a['stages'])==58,'exact58 F16 observations required')
    for name in a['stages']:
        c.require(all(equal_stage(a['stages'][name],v)for v in (b['stages'][name],retained['observed']['stages'][name],retained['repeat']['stages'][name])),'F16 retained/current/repeat raw stage differs: '+name)
    return dict(passed=True,modes=4,stages=58,ids_text_exact=True,vision_byte_exact=True,observed_repeat_retained_byte_exact=True,
        eos_id_observed=False,stop_contract=a['stop_contract'])


def run(a):
    cfg=c.guard(a.source,True);a.out.mkdir(parents=True,exist_ok=False);c.require(0<a.budget_seconds<=cfg['panel_seconds'],'bounded predeclared remaining panel time required');deadline=time.monotonic()+a.budget_seconds
    report=dict(schema='crispembed-unlimited-reused-q4-panel-v1',source=a.source,completed=False,phase='input-authentication',
        recipe='last1',allocated_budget_seconds=a.budget_seconds,original_forwards=0,quantizer_runs=0,original_rerun=False,original_intermediate_gold=False,comparison_observation_order=c.stage_order(),
        publication_qualified=False,production_qualified=False,baseline=[],candidates=[],commands=[],
        planned_counts=cfg['counts'],actual_calls=dict(f16_attempts=0,f16_commands_completed=0,q4_attempts=0,q4_commands_completed=0,
            projection_attempts=0,projection_commands_completed=0))
    save=lambda:c.atomic(a.out/'panel.json',report);save()
    def command(name,argv,env=None,timeout=3600,allowed=(0,)):
        remaining=deadline-time.monotonic();c.require(remaining>0,'declared panel deadline reached before next call')
        directory=a.out/name;directory.mkdir(parents=True,exist_ok=False)
        status=child(argv,directory/'stderr',stdout_path=directory/'stdout',cwd=c.ROOT,env=env,timeout=min(timeout,remaining))
        status.update(log=name,stdout_sha256=c.digest(directory/'stdout'),stderr_sha256=c.digest(directory/'stderr'))
        c.atomic(directory/'command.json',status);report['commands'].append(status);save()
        c.require(not status['timed_out']and status['exit']in allowed,'bounded hosted command rejected: '+name)
        return status
    def compare(pairs,name):
        import numpy as np
        directory=a.out/name;directory.mkdir(parents=True,exist_ok=False);independent={}
        for pair in pairs:
            x,y=pair['original'],pair['native']
            independent[pair['name']]=old.metrics(np.memmap(x['path'],dtype='<f4',mode='r',shape=tuple(x['shape'])),np.memmap(y['path'],dtype='<f4',mode='r',shape=tuple(y['shape'])))
        adapters=old.write_pair(directory/'control.gguf',directory/'candidate.gguf',pairs)
        status=command(name+'/ref',[a.diff,directory/'control.gguf',directory/'candidate.gguf',directory/'ref.json'],timeout=1200,allowed=(0,1))
        actual=old.collect(status['exit'],directory/'ref.json',independent)
        result=dict(passed=all(r['passed']for r in independent.values()),cpp=actual,stages=independent,adapters=adapters,
            first_divergence=next((k for k,v in independent.items()if not v['passed']),None),worst_absolute=max(v['max_absolute']for v in independent.values()),
            reference_kind='native F16 control, not original intermediate gold')
        c.atomic(directory/'comparison.json',result);c.require(actual['all_metrics_agree'],'actual Ref and independent metrics disagree; proof retained');return result
    try:
        proof=json.loads((a.auth/'inputs.json').read_text());structural=json.loads((a.auth/'structural.json').read_text())
        c.require(proof['source']==a.source and proof['completed']and proof['all_544_authenticated_before_array_image_import']and
            proof['panel_files_authenticated']==540 and proof['additional_images_authenticated']==4 and c.digest(a.auth/'structural.json')==proof['structural_report_sha256']and structural['completed'],'complete544-input structural proof required')
        retained=a.inputs/'panel';producer_path=retained/'panel.json';pin=json.loads((c.ROOT/'docs/unlimited-retained-gha-fetch-v1.json').read_text())
        c.require(c.digest(producer_path)==cfg['producer_panel_sha256'],'retained producer panel changed')
        producer=json.loads(producer_path.read_text())
        # All retained entries are rehashed before any new forward; candidate is never rewritten.
        import retained_v3_gha as audit
        for name,p in producer['files'].items():
            q=audit.member(retained,name);c.require(q.stat().st_size==p['bytes']and c.digest(q)==p['sha256'],'retained input changed before forward')
        refs,images=c.frozen.references()
        for case in c.frozen.CASES:c.require(c.digest(a.inputs/'pages'/(case+'.png'))==images[case],'frozen PNG changed')
        candidate=retained/'unlimited-last1.gguf';baseline=a.control
        c.require(candidate.stat().st_size==cfg['candidate']['bytes']and c.digest(candidate)==cfg['candidate']['sha256'],'immutable candidate changed')
        c.require(baseline.stat().st_size==cfg['control']['bytes']and c.digest(baseline)==cfg['control']['sha256'],'exact published F16 required')
        source_inventory=c.frozen.inventory(baseline);candidate_inventory=c.frozen.inventory(candidate)
        c.require(source_inventory==producer['baseline_inventory']and candidate_inventory==structural['actual_inventory'],'all739 retained source/candidate tensors differ')
        report['protected']=c.frozen.protected(source_inventory,candidate_inventory,'last1');report['models']=dict(control=cfg['control'],candidate=cfg['candidate'])
        report['binaries']={n:dict(bytes=getattr(a,n).stat().st_size,sha256=c.digest(getattr(a,n)))for n in ('cli','observed','diff','projection')}
        env={k:v for k,v in os.environ.items()if not k.startswith(('CRISPEMBED_','UOCR'))}
        env.update(CRISPEMBED_UNLIMITED_OCR2='1',CRISPEMBED_UNLIMITED_OCR2_FAST='1',UOCR_DBG='1',UOCR2_FAST_CHUNK='2048',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
        def forward(model,case,mode,kind):
            count=report['actual_calls'];limit=16 if kind=='f16'else 12
            c.require(count[kind+'_attempts']<limit,'declared full-forward count exceeded')
            if kind=='q4':c.require(report.get('f16_all_four_strict_retained_barrier_passed')is True,'all16 F16 controls must close before ANY Q4 call')
            c.resources(a.out);count[kind+'_attempts']+=1;save();name=kind+'/'+case+'/'+mode;active=dict(env)
            if mode in ('observed','repeat'):
                trace=a.out/kind/case/(mode+'-trace');trace.mkdir(parents=True,exist_ok=False)
                active.update(UOCR_Q4_AUDIT_DIR=str(trace),UOCR2_DUMP_VISION=str(trace/'assembled-vision.f32'))
            binary=a.cli if mode=='plain'else a.observed
            status=command(name,[binary,'-m',model,'--ocr',a.inputs/'pages'/(case+'.png'),'--ocr-max-tokens','1024','-t','4'],active)
            count[kind+'_commands_completed']+=1;save()
            row=old.parsed(a.out/name);row.update(command=status,quality=old.task_quality(row,refs[case]))
            if mode in ('observed','repeat'):
                row['stages']=old.stage_manifest(trace);vision=trace/'assembled-vision.f32'
                c.require(vision.is_file()and vision.stat().st_size>0 and vision.stat().st_size%(1280*4)==0,'assembled vision extent differs')
                row['vision']=dict(bytes=vision.stat().st_size,sha256=c.digest(vision))
            return row
        controls={};report['phase']='all16-matched-F16-before-Q4';save()
        for case in c.frozen.CASES:
            entry=dict(id=case,modes={},completed=False);report['baseline'].append(entry);save()
            for mode in ('plain','ordinary','observed','repeat'):entry['modes'][mode]=forward(baseline,case,mode,'f16');save()
            current=entry['modes'];old_case=next(r for r in producer['baseline']if r['id']==case)
            target=old.stage_manifest(retained/'baseline'/case/'observed-trace')
            pairs=[dict(name=k,original=old.float_spec(target[k]),native=old.float_spec(current['observed']['stages'][k]))for k in c.stage_order()if target[k]['type']=='f32']
            c.require(len(pairs)==47,'47 F32 and11 integer original native stages required')
            entry['retained_stage_actual_ref']=compare(pairs,'f16/'+case+'/retained-comparison');save()
            c.require(entry['retained_stage_actual_ref']['passed'],'retained/current F16 actual Ref control rejected')
            entry['barrier']=barrier(current,old_case['modes'],refs[case]);entry['completed']=True;controls[case]=current['observed']['stages'];save()
        c.require(report['actual_calls']['f16_commands_completed']==16 and all(r['completed']for r in report['baseline']),'all16 original-matched F16 controls required')
        report.update(f16_all_four_strict_retained_barrier_passed=True,phase='immutable-last1-only');save()
        item=dict(recipe='last1',cases=[],candidate=cfg['candidate'],model_reused=True,model_written=False);report['candidates'].append(item);save()
        for case in c.frozen.CASES:
            row=dict(id=case,completed=False);item['cases'].append(row);save()
            try:
                modes={}
                for mode in ('plain','observed','repeat'):modes[mode]=forward(candidate,case,mode,'q4');row[mode]=modes[mode];save()
                plain,observed,repeated=(modes[k]for k in ('plain','observed','repeat'))
                c.require(plain['ids']==observed['ids']==repeated['ids']and plain['text']==observed['text']==repeated['text']and plain['command']['stdout_sha256']==observed['command']['stdout_sha256']==repeated['command']['stdout_sha256'],'Q4 ordinary/observer/repeat decoded bytes differ')
                c.require(all(equal_stage(observed['stages'][k],repeated['stages'][k])for k in observed['stages']),'Q4 repeat raw stage bits differ')
                base=next(r for r in report['baseline']if r['id']==case)
                c.require(observed['vision']==repeated['vision']==base['modes']['observed']['vision'],'protected F16/Q4 vision bytes differ')
                row.update(ordinary_observer_exact=True,raw_repeat_bit_exact=True,protected_vision_bit_exact=True,raw_stdout_equal_f16=plain['command']['stdout_sha256']==base['modes']['plain']['command']['stdout_sha256'])
                expected,actual=controls[case],observed['stages']
                row['natural_router_ids_equal_f16']={k:equal_stage(expected[k],actual[k])for k in c.stage_order()if expected[k]['type']=='i32'}
                row['first_integer_router_difference']=next((k for k,v in row['natural_router_ids_equal_f16'].items()if not v),None)
                pairs=[dict(name=k,original=old.float_spec(expected[k]),native=old.float_spec(actual[k]))for k in c.stage_order()if expected[k]['type']=='f32']
                row['natural_f16_to_q4']=compare(pairs,'q4/'+case+'/natural-comparison');row['same_input_projection']=[];save()
                packet,n=old.projection_inputs(expected,11,a.out/'q4'/case/'input-11')
                for part in ('gate','up'):
                    weight='l.blk.11.ffn_'+part+'_exps.weight';prefix=a.out/'q4'/case/('projection-11-'+part)
                    for label,model in [('f16',baseline),('q4',candidate)]:
                        calls=report['actual_calls'];c.require(calls['projection_attempts']<16,'local projection execution budget exceeded')
                        calls['projection_attempts']+=1;save()
                        command('q4/'+case+'/operator-11-'+part+'-'+label,[a.projection,model,weight,packet,str(prefix)+'-'+label,'4'],env,180)
                        calls['projection_commands_completed']+=1;save()
                    def spec(label):
                        path=Path(str(prefix)+'-'+label+'.f32');return dict(path=str(path),shape=[n,6,896],sha256=c.digest(path))
                    measured=compare([dict(name='f16_capture_vs_fixed_f16',original=old.float_spec(expected['layer_11_chunk_0_'+part]),native=spec('f16.native')),
                        dict(name='fixed_f16_vs_q4',original=spec('f16.native'),native=spec('q4.native')),
                        dict(name='q4_stored_vs_dequant_f32',original=spec('q4.dequant'),native=spec('q4.native'))], 'q4/'+case+'/projection-comparison-11-'+part)
                    row['same_input_projection'].append(dict(weight=weight,control_weight=source_inventory[weight],candidate_weight=candidate_inventory[weight],input_packet_sha256=c.digest(packet),fixed_ids_source='current F16 first16 captured rows; test-only local operator',**measured));save()
                row.update(completed=True,numeric_passed=row['natural_f16_to_q4']['passed']and all(p['passed']for p in row['same_input_projection']),task_quality_passed=plain['quality']['task_completed'])
            except Exception as error:row['error']=type(error).__name__+': '+str(error)
            save()
        report.update(completed=all(r['completed']for r in item['cases']),all_numeric_passed=all(r.get('numeric_passed',False)for r in item['cases']),
            all_task_quality_passed=all(r.get('task_quality_passed',False)for r in item['cases']),all_generated_ids_raw_text_exact=all(r.get('plain',{}).get('quality',{}).get('ids_exact',False)and r.get('plain',{}).get('quality',{}).get('raw_text_exact',False)and r.get('raw_stdout_equal_f16',False)for r in item['cases']))
        if report['completed']:
            c.require(report['actual_calls']==dict(f16_attempts=16,f16_commands_completed=16,q4_attempts=12,q4_commands_completed=12,projection_attempts=16,projection_commands_completed=16),'completed panel call counts differ')
        c.require(candidate.stat().st_size==cfg['candidate']['bytes']and c.digest(candidate)==cfg['candidate']['sha256'],'immutable reused candidate changed after replay')
        c.require(baseline.stat().st_size==cfg['control']['bytes']and c.digest(baseline)==cfg['control']['sha256'],'published F16 changed after replay')
        report.update(candidate_preserved=True,phase='panel-complete',limits='Four owned development documents only. Same-host native F16 intermediate controls, not original intermediate gold. Native trace omits EOS; source-derived stop/no-cap recorded honestly. No Q8,last4/all11,quantizer,original forward,default or publication change.');save()
    except Exception as error:report['error']=type(error).__name__+': '+str(error);save();raise
    finally:
        report['files']={str(p.relative_to(a.out)):dict(bytes=p.stat().st_size,sha256=c.digest(p))for p in sorted(a.out.rglob('*'))if p.is_file()and p!=a.out/'panel.json'};save()
    return 0 if report['completed']and report['all_numeric_passed']and report['all_task_quality_passed']else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--budget-seconds',type=int,required=True)
    for n in ('auth','inputs','control','out','cli','observed','diff','projection'):p.add_argument('--'+n,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
