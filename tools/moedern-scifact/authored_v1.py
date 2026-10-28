"""Frozen ten-text/20-vector/three-score control using the actual root C ABI."""
import json
import math
from pathlib import Path
import sys
from contract_native_v1 import ROOT,require,sha
from providers_v1 import module

# The immutable helper uses sibling imports, not a Python package. Declare its
# exact protected directory before loading it; this must also work under -I.
sys.path.insert(0,str(ROOT/'tools/moedern-native-v2'))
reference=module('frozen_vector_judge',ROOT/'tools/moedern-native-v2/replay.py')
adapter=module('frozen_authored_adapters',ROOT/'tools/moedern-native-v2/diff_adapter.py')


def ref(binary,folder,measurements,specs,command,env):
    adapters=adapter.write_pair(folder/'original.gguf',folder/'native.gguf',specs)
    run=command([binary,folder/'original.gguf',folder/'native.gguf',folder/'ref.json'],folder/'ref.log',env,allowed=(0,1))
    result=adapter.collect(run['exit'],folder/'ref.json',measurements)
    require(result['all_metrics_agree'],'authored actual Ref/NumPy disagreement')
    return dict(actual_ref=result,adapters=adapters,process=run)


def evaluate(model,original,arrays,binaries,out,command,env):
    import numpy as np
    out.mkdir(parents=True,exist_ok=False);text={c['id']:c['text']for c in original['cases']};hx=lambda s:s.encode().hex()or'-'
    cases,pairs=out/'cases.tsv',out/'pairs.tsv'
    cases.write_text(''.join(c['id']+'\t'+c['role']+'\t'+hx(c['text'])+'\n'for c in original['cases']))
    pairs.write_text(''.join(s['query']+'|'+s['document']+'\t'+hx(text[s['query']])+'\t'+hx(text[s['document']])+'\n'for s in original['scores']))
    process=command([binaries/'replay-consumer',model,cases,pairs],out/'consumer.log',env,stdout_path=out/'consumer.jsonl')
    # JSON -0 is a valid integer token, but its float32 sign is meaningful.
    def number(value):return -0. if value=='-0'else int(value)
    def constant(_value):raise ValueError('nonfinite actual consumer value')
    rows=[json.loads(line,parse_int=number,parse_constant=constant)for line in(out/'consumer.jsonl').read_text().splitlines()if line.strip()]
    cs=[r for r in rows if r['kind']=='case'];ss=[r for r in rows if r['kind']=='score']
    require(len(rows)==13 and len(ss)==3 and[c['id']for c in cs]==[c['id']for c in original['cases']],'ten ordered cases/three scores required')
    measurements={};specs=[];byid={r['id']:r for r in cs}
    for target,native in zip(original['cases'],cs):
        supplied=list(map(int,target['supplied_attention_mask']))
        expected=dict(ids=target['token_ids'],supplied_mask=supplied,effective_mask=list(map(int,target['effective_attention_mask'])),
                      scoring_mask=list(map(int,target['scoring_mask'])),expansion_positions=[1-x for x in supplied]if target['role']=='query'else[])
        require(native['role']==target['role']and all(native[k]==v for k,v in expected.items()),'authored IDs/masks/expansion differ: '+target['id'])
        for kind,stage in(('all','ordinary.features.token_embeddings'),('selected','ordinary.selected')):
            left=arrays[target['id'],stage];a=np.fromfile(left['path'],dtype='<f4').reshape(left['shape'])
            b=np.asarray(native[kind+'_vectors'],dtype='<f4');require(b.size==a.size,'authored vector extent differs');b=b.reshape(a.shape)
            name=target['id'].replace('-','_')+'.'+kind;path=out/(name+'.f32');path.write_bytes(b.tobytes())
            measurements[name]=reference.compare(a,b)
            specs.append(dict(name=name,original=left,native=dict(path=path,shape=left['shape'],sha256=sha(path))))
    actual=ref(binaries/'replay-authored-diff',out,measurements,specs,command,env);scores=[]
    for target,native in zip(original['scores'],ss):
        require(native['id']==target['query']+'|'+target['document'],'authored score order differs')
        q=np.asarray(byid[target['query']]['selected_vectors'],dtype='<f4').reshape(-1,128)
        d=np.asarray(byid[target['document']]['selected_vectors'],dtype='<f4').reshape(-1,128)
        protocol=float((d@q.T).max(axis=0).astype(np.float64).sum());value=float(native['score'])
        require(math.isfinite(value)and abs(value-protocol)<=.002,'authored C MaxSim differs from F32dot/F64sum')
        scores.append(dict(id=native['id'],native=value,protocol=protocol,original=target['original_maxsim'],
                           max_absolute=abs(value-target['original_maxsim']),passed=abs(value-target['original_maxsim'])<=.002))
    return dict(process=process,typed_cases=10,vectors=measurements,scores=scores,**actual,
                strict_numeric_passed=all(v['passed']for v in measurements.values())and all(s['passed']for s in scores)),rows


def family_control(model,original,arrays,rows,binaries,out,command,env):
    import numpy as np
    out.mkdir(parents=True,exist_ok=False);tokens=out/'native-derived-ids.tsv'
    tokens.write_text(''.join(r['id']+'\t'+r['role']+'\t'+','.join(map(str,r['ids']))+'\t'+','.join(map(str,r['supplied_mask']))+'\n'for r in rows if r['kind']=='case'))
    process=command([binaries/'replay-family-control',model,tokens,out,2],out/'family.log',env)
    measurements={};specs=[]
    for target in original['cases']:
        cid=target['id'];left=arrays[cid,'ordinary.features.token_embeddings'];path=out/(cid+'.f32')
        a=np.fromfile(left['path'],dtype='<f4').reshape(left['shape']);b=np.fromfile(path,dtype='<f4')
        require(a.size==b.size,'family control vector extent differs');name=cid.replace('-','_')
        measurements[name]=reference.compare(a,b.reshape(a.shape));specs.append(dict(name=name,original=left,native=dict(path=path,shape=left['shape'],sha256=sha(path))))
    actual=ref(binaries/'replay-authored-diff',out,measurements,specs,command,env)
    return dict(process=process,native_frontend_ids_only=True,strict_numeric_passed=all(v['passed']for v in measurements.values()),vectors=measurements,**actual)
