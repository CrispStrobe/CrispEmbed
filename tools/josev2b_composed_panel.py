"""Original-first comparison for native raw-request composition; no inference gold."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

from josev2b_composed_diff import collect, write_pair

UNMAPPED = {'backbone/rotary/position_ids', 'backbone/rotary/0', 'backbone/rotary/1',
            'backbone/layer/0/input/position_ids', 'backbone/layer/3/input/position_ids',
            'backbone/layer/3/input/attention_mask'}

def require(value, why):
    if not value:
        raise ValueError(why)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def manifest(directory):
    result = {}
    for line in (directory / 'stages.jsonl').read_text().splitlines():
        row = json.loads(line)
        key = (row['stage'], row['occurrence'])
        require(key not in result, 'duplicate native stage occurrence')
        require(Path(row['file']).name == row['file'] and row['file'].endswith('.f32'), 'unsafe raw member')
        require((directory / row['file']).stat().st_size == 4 * math.prod(row['shape']), 'raw stage size differs')
        result[key] = row
    return result

def coverage(report, case, native):
    originals = []
    occurrences = {}
    for row in report['arrays']:
        if row['case'] != case:
            continue
        label = row['stage']
        occurrence = occurrences.get(label, 0)
        originals.append(dict(row, occurrence=occurrence))
        occurrences[label] = occurrence + 1
    unmapped = [r for r in originals if r['stage'] in UNMAPPED]
    require(len(unmapped) == 6, 'unmapped source trace coverage differs')
    mapped = [r for r in originals if r['stage'] not in UNMAPPED]
    keys = [(r['stage'], r['occurrence']) for r in mapped]
    require(len(keys) == len(set(keys)), 'duplicate original stage occurrence')
    actual = {key for key in native if not key[0].startswith(('composed/', 'result/'))}
    require(actual == set(keys), 'missing/extra native captured stage')
    for r in mapped:
        require(r['dtype'] == 'float32' and native[(r['stage'], r['occurrence'])]['shape'] == r['shape'],
                'original/native logical shape or precision differs')
    return mapped, unmapped

def metrics(a, b):
    import numpy as np
    require(a.dtype == np.dtype('<f4') and b.dtype == a.dtype and a.shape == b.shape and a.size and
            np.isfinite(a).all() and np.isfinite(b).all(), 'finite exact original/native F32 shape required')
    x = a.astype(np.float64); y = b.astype(np.float64); delta = y - x
    ar = x.reshape(-1, x.shape[-1]); br = y.reshape(-1, y.shape[-1])
    aa = (ar * ar).sum(axis=1); bb = (br * br).sum(axis=1); dot = (ar * br).sum(axis=1)
    cosine = np.zeros_like(dot); valid = (aa > 1e-18) & (bb > 1e-18)
    cosine[(aa <= 1e-18) & (bb <= 1e-18)] = 1
    cosine[valid] = dot[valid] / (np.sqrt(aa[valid]) * np.sqrt(bb[valid]))
    result = dict(shape=list(a.shape), elements=a.size, max_absolute=float(abs(delta).max()),
                  mean_absolute=float(abs(delta).mean()), mean_error=float(delta.mean()),
                  rms=float(np.sqrt((delta * delta).mean())), min_row_cosine=float(cosine.min()),
                  original_l2=float(np.sqrt((x*x).sum())), native_l2=float(np.sqrt((y*y).sum())),
                  bit_exact=bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))))
    for label, value in (('original', x), ('native', y)):
        result.update({label+'_min':float(value.min()), label+'_max':float(value.max()),
                       label+'_abs_min':float(abs(value).min()), label+'_abs_max':float(abs(value).max())})
    result['passed'] = result['max_absolute'] <= .002 and result['min_row_cosine'] >= .99999
    return result

def compare(reference_dir, freeze, report, outputs, diff, out):
    # The hosted owner must authenticate reference ZIP/member/source hashes first.
    # Repeat these small immutable identities; never regenerate original arrays.
    require(sha(reference_dir / 'reference.json') == '51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090',
            'original reference changed')
    require(len(report['arrays']) == 251 and len(freeze['records']) == 4, 'original scope differs')
    require(set(outputs) == {'plain', 'observed', 'repeat'}, 'three actual native variants required')
    native = {mode:[json.loads(line) for line in (directory/'native.jsonl').read_text().splitlines()]
              for mode,directory in outputs.items()}
    require(all(len(rows)==4 for rows in native.values()), 'four actual results per variant required')
    out.mkdir(parents=True, exist_ok=False)
    result = dict(completed=False, full_model_qualified=False, model_quality=False,
                  compared_stages=0, cases=[], failures=[], limits='Four batch1/all-attended text requests only; no public/release/task quality.')
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
            require(len(mapped) in (68,59,50), 'mapped stage count differs')
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
            raw_equal=[]
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
            adapters=write_pair(case_dir/'original.gguf',case_dir/'native.gguf',pairs)
            with (case_dir/'diff.log').open('w') as log:
                process=subprocess.run([str(diff),str(case_dir/'original.gguf'),str(case_dir/'native.gguf'),str(case_dir/'diff.json')],
                                       stdout=log,stderr=subprocess.STDOUT,timeout=120)
            actual=collect(process.returncode,case_dir/'diff.json',independent)
            response=json.loads(rows['observed']['response'])
            exact=response==original_case['official_response']
            case.update(completed=True,mapped=len(mapped),metrics=details,actual_ref=actual,adapters=adapters,
                        raw_finals=raw_equal,response=response,original_response=original_case['official_response'],
                        complete_response_exact=exact,ordinary_observer_repeat_raw_equal=True)
            case['passed']=exact and actual['all_metrics_agree'] and all(m['passed'] for m in independent.values()) and all(r['passed'] for r in raw_equal)
            result['compared_stages']+=len(mapped)
        except Exception as error:
            case['error']=str(error);case['passed']=False
        if not case['passed']:result['failures'].append(record['id'])
        write(out/'panel.json',result)
    result['completed']=True
    result['numeric_and_complete_typed_parity']=not result['failures'] and result['compared_stages']==227
    write(out/'panel.json',result)
    return result
