"""NEW20 strict stages plus separately declared public task scoring.

Comparator body adapted from tools/josev2b_composed_panel.py at710e64a2.
Old four-case module/guard is never modified or invoked as a new-panel guard.
Only low-level manifest/coverage/metrics and actual Ref adapters are reused.
"""
import json
import subprocess
from josev2b_composed_panel import require, sha, write, manifest, coverage, metrics
from josev2b_composed_diff import collect, write_pair
from josev2b_task_quality_v1 import native_prerequisites, compare_public

def compare(reference_dir, freeze, report, outputs, diff, out, *, precision):
    # The hosted owner must authenticate reference ZIP/member/source hashes first.
    # Repeat these small immutable identities; never regenerate original arrays.
    require(sha(reference_dir / 'reference.json') == native_prerequisites()['original_reference_sha256'],
            'original reference changed')
    require(len(report['arrays']) == 1480 and len(freeze['records']) == 20, 'original scope differs')
    require(set(outputs) == {'plain', 'observed', 'repeat'}, 'three actual native variants required')
    native = {mode:[json.loads(line) for line in (directory/'native.jsonl').read_text().splitlines()]
              for mode,directory in outputs.items()}
    require(all(len(rows)==20 for rows in native.values()), 'twenty actual results per variant required')
    out.mkdir(parents=True, exist_ok=False)
    result = dict(completed=False, full_model_qualified=False, model_quality=False,
                  compared_stages=0, cases=[], failures=[], limits='Twenty authored requests/ten translated pairs only; strict F32 flags separate from NEW20 Q4 task gates; no broad benchmark/release/publication.')
    for i, (record, original_case) in enumerate(zip(freeze['records'], report['cases'])):
        case = dict(id=record['id'], completed=False); result['cases'].append(case)
        try:
            require(record['id'] == original_case['id'], 'frozen case order differs')
            rows = {mode:entries[i] for mode,entries in native.items()}
            require(all(r['id']==record['id'] and r['completed'] is True and not r['error'] for r in rows.values()),
                    'native inference failure or case order differs')
            require(len({r['response'] for r in rows.values()}) == 1, 'ordinary/observer/repeat response bytes differ')
            observed_dir = outputs['observed']/str(i)
            encoded = json.loads((observed_dir/'encoded.json').read_text())
            expected = dict(record['encoded'], attention_mask=record['attention_mask'])
            for q in expected['questions']:
                q['lexical_ids'] = [expected['input_ids'][a:b] for a,b in q['option_spans']]
            require(encoded == expected, 'native IDs/masks/spans/lexical IDs/order differ')
            manifests = {mode:manifest(directory/str(i)) for mode,directory in outputs.items()}
            mapped, unmapped = coverage(report, record['id'], manifests['observed'])
            require(len(mapped) == 68, 'mapped stage count differs')
            fields=original_case['fields']
            final_keys={(f'result/{kind}/{f}',0) for f in range(len(fields)) for kind in ('logits','probabilities')}
            extra_keys={(f'composed/{kind}/{f}',0) for f in range(len(fields)) for kind in ('lexical','logits','probabilities')}
            expected_keys={(r['stage'],r['occurrence']) for r in mapped}|final_keys|extra_keys
            require(set(manifests['observed'])==expected_keys and set(manifests['repeat'])==expected_keys and
                    set(manifests['plain'])==final_keys, 'actual variant/extra trace coverage differs')
            for key,declaration in manifests['observed'].items():
                repeated=manifests['repeat'][key]
                require(declaration['shape']==repeated['shape'] and
                        (observed_dir/declaration['file']).read_bytes()==
                        (outputs['repeat']/str(i)/repeated['file']).read_bytes(), 'observed/repeat stage bytes differ')
            for f,field in enumerate(fields):
                for kind,label,occurrence in (('lexical','head/option_lexical_projection/input/0',f),
                                              ('logits','logits/'+field['id'],0),
                                              ('probabilities','probabilities/'+field['id'],0)):
                    first=manifests['observed'][(f'composed/{kind}/{f}',0)]
                    second=manifests['observed'][(label,occurrence)]
                    require(first['shape']==second['shape'] and
                            (observed_dir/first['file']).read_bytes()==(observed_dir/second['file']).read_bytes(),
                            'composed input/output trace differs from actual head input/output')
            case['unmapped'] = unmapped
            case_dir = out/str(i); case_dir.mkdir()
            pairs=[]; independent={}; details=[]
            for index,row in enumerate(mapped):
                key=(row['stage'],row['occurrence']); declaration=manifests['observed'][key]
                original=reference_dir/row['file']; path=observed_dir/declaration['file']
                require(sha(original)==row['sha256'], 'original array changed')
                import numpy as np
                a=np.load(original, mmap_mode='r', allow_pickle=False)
                b=np.memmap(path, mode='r', dtype='<f4', shape=tuple(row['shape']))
                measured=metrics(a,b); name=f'stage.{index:03d}'
                raw=case_dir/(name+'.original.f32'); a.tofile(raw)
                pairs.append(dict(name=name,original=dict(path=raw,shape=row['shape'],sha256=sha(raw)),
                                  native=dict(path=path,shape=row['shape'],sha256=sha(path))))
                independent[name]=measured
                details.append(dict(stage=row['stage'],occurrence=row['occurrence'],**measured))
            # Compare the same final raw F32 fields from all real library variants.
            raw_equal=[]; ordinal=[]
            for f,field in enumerate(original_case['fields']):
                for kind,budget in (('logits',.002),('probabilities',.0001)):
                    key=('result/'+kind+'/'+str(f),0)
                    payloads=[(outputs[m]/str(i)/manifests[m][key]['file']).read_bytes() for m in outputs]
                    require(all(p==payloads[0] for p in payloads), 'ordinary/observer/repeat raw final bytes differ')
                    captured=manifests['observed'][(kind+'/'+field['id'],0)]
                    require((observed_dir/captured['file']).read_bytes()==payloads[0], 'observed final/raw result differ')
                    b=np.frombuffer(payloads[0],dtype='<f4'); a=np.asarray(field[kind],dtype='<f4')
                    require(a.shape==b.shape and np.isfinite(b).all(), 'final field shape/finite differs')
                    error=float(abs(a.astype('float64')-b.astype('float64')).max())
                    raw_equal.append(dict(field=field['id'],kind=kind,max_absolute=error,passed=error<=budget))
                    q=record['encoded']['questions'][f]
                    if kind=='probabilities' and q['question_type']==2:
                        original_score=sum(n*float(prob) for n,prob in enumerate(a))
                        native_score=sum(n*float(prob) for n,prob in enumerate(b))
                        delta=abs(native_score-original_score)
                        ordinal.append(dict(field=field['id'],original=original_score,native=native_score,
                                            absolute_error=delta,budget=.001,passed=delta<=.001))
            adapters=write_pair(case_dir/'original.gguf',case_dir/'native.gguf',pairs)
            with (case_dir/'diff.log').open('w') as log:
                process=subprocess.run([str(diff),str(case_dir/'original.gguf'),str(case_dir/'native.gguf'),str(case_dir/'diff.json')],
                                       stdout=log,stderr=subprocess.STDOUT,timeout=120)
            actual=collect(process.returncode,case_dir/'diff.json',independent)
            response=json.loads(rows['observed']['response'])
            exact=response==original_case['official_response']
            case.update(completed=True,mapped=len(mapped),metrics=details,actual_ref=actual,adapters=adapters,
                        raw_finals=raw_equal,ordinal=ordinal,response=response,original_response=original_case['official_response'],
                        first_byte_difference=next((r for r in details if not r['bit_exact']),None),
                        first_failing_stage=next((r for r in details if not r['passed']),None),
                        complete_response_exact=exact,ordinary_observer_repeat_raw_equal=True)
            case['passed']=actual['all_metrics_agree'] and all(m['passed'] for m in independent.values()) and all(r['passed'] for r in raw_equal) and all(r['passed'] for r in ordinal)
            result['compared_stages']+=len(mapped)
        except Exception as error:
            case['error']=str(error);case['passed']=False
        if not case['passed']:result['failures'].append(record['id'])
        write(out/'panel.json',result)
    result['completed']=True
    result['strict_f32_numerical_flags_passed']=not result['failures'] and result['compared_stages']==1360
    original={c['id']:c['official_response'] for c in report['cases']}
    candidate={c['id']:c['response'] for c in result['cases'] if c.get('completed')}
    result['precision']=precision
    result['task_quality']=compare_public(original,candidate,precision=precision) if len(candidate)==20 else None
    result['qualified_for_owned_panel']=bool(result['task_quality'] and result['task_quality']['independent_task_quality_passed'] and all(c.get('completed') and c['actual_ref']['all_metrics_agree'] for c in result['cases']) and (precision=='q4' or result['strict_f32_numerical_flags_passed']))
    result['publication_authorized']=False
    write(out/'panel.json',result)
    return result
