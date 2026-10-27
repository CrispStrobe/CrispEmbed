"""Frozen four-case token/stage contracts. No framework or payload import."""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE='7e70d35324162b6f4379d7550a5f0f18c3d0fc98'
GGML='a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
SOURCE='b2edf8c83c9aca1f8a9210198d47798ac13828b8'
RECEIPT='51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090'
FREEZE='9c85fae56186517b5721be547bc74e8f92bd8e7cdb45d38afb4340e029161fe7'
RUN=40429513945
PRODUCER='5da7c64d1a95df0f9a829c62938fc8d7d64f77b7'
ARTIFACTS={'review':dict(id=12702676604,bytes=33345,sha256='f452ef7c9583258a77341c56d96fd0c3f257926f3e053862d469e3501399e1e8',name=f'josev-head-review-{RUN}'),
 'raw':dict(id=12702553413,bytes=317814519,sha256='eff25c8ec207646602cb885abc6d17f7fc646cd2b0d7e5f7ab9b27650e175708',name=f'josev-head-original-and-component-{RUN}')}
CASES=(('mixed-de',319),('null-and-options',197),('json-unicode',165),('state-only-clip',512))
NEW=('tools/josev2b_backbone_probe.py','tools/josev2b_backbone_diff.py','tools/ci-heavy/josev2b_backbone.py',
 'tests/josev-backbone/probe.cpp','tests/josev-backbone/diff.cpp','tests/josev-backbone/CMakeLists.txt',
 'tests/test_josev2b_backbone_probe.py','.github/workflows/josev2b-backbone.yml',
 '.github/workflows/ovisocr2-metadata.yml','docs/josev2b-backbone-probe.md')
PROTECTED=('src','ggml','cmake','CMakeLists.txt','tools/convert_josev2b_backbone_f32.py',
 'tools/josev2b_head_schema.py','tools/convert_josev2b_head_f32.py','tools/ci-heavy/josev2b_head_artifact.py',
 'tests/josev-backbone/metadata.cpp','tests/test_josev2b_backbone_contract.py',
 'tests/test_josev2b_backbone_conversion.py','tests/results/josev2b/original-reference.json',
 'tests/results/josev2b/original-input-freeze.json','docs/josev2b-source-contract.json')


def sha(data):return hashlib.sha256(data).hexdigest()


def source_guard():
    changed=subprocess.check_output(['git','diff','--name-only',BASE,'HEAD','--',*PROTECTED],cwd=ROOT,text=True)
    if changed:raise ValueError('Declared backbone/converter/observer source changed: '+changed)
    if subprocess.check_output(['git','status','--porcelain','--',*PROTECTED,*NEW],cwd=ROOT):
        raise ValueError('Dirty/untracked measured backbone source')
    if subprocess.check_output(['git','rev-parse','HEAD:ggml'],cwd=ROOT,text=True).strip()!=GGML:
        raise ValueError('Declared ggml dependency differs')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=sorted(set(NEW)|set(subprocess.check_output(['git','ls-files','--',*PROTECTED],cwd=ROOT,text=True).splitlines())-{'ggml'})
    return dict(head=head,base=BASE,ggml=GGML,git_objects={p:subprocess.check_output(['git','rev-parse','HEAD:'+p],cwd=ROOT,text=True).strip() for p in (*PROTECTED,*NEW)},
        files={p:sha((ROOT/p).read_bytes()) for p in files})


def stages(tokens):
    if type(tokens)is not int or not 1<=tokens<=512:raise ValueError('Token bound differs')
    return [dict(stage=s,occurrence=0,shape=[1,tokens,2048]) for s in
            ['backbone/embedding',*[f'backbone/layer/{i}' for i in range(24)],'backbone/final_norm']]


def original_events(report):
    if [(c['id'],c['tokens']) for c in report['cases']]!=list(CASES):raise ValueError('All four original cases required')
    result={}
    for case,tokens in CASES:
        expected=stages(tokens);names={e['stage'] for e in expected}
        selected=[r for r in report['arrays'] if r['case']==case and r['stage'] in names]
        if len(selected)!=26 or [r['stage'] for r in selected]!=[r['stage'] for r in expected]:raise ValueError('Original104stage coverage/order differs')
        for row,event in zip(selected,expected):
            if row['shape']!=event['shape'] or row['dtype']!='float32' or row['negative_infinity']!=0:
                raise ValueError('Original backbone stage dtype/shape/finite policy differs')
        result[case]=selected
    return result


def packet(freeze):
    if [(r['id'],len(r['encoded']['input_ids'])) for r in freeze['records']]!=list(CASES):raise ValueError('Frozen token panel differs')
    result=b'JOSBKV1\0'+RECEIPT.encode()+SOURCE.encode()+FREEZE.encode()+struct.pack('<I',4)
    for record in freeze['records']:
        ids=record['encoded']['input_ids'];mask=record['attention_mask'];name=record['id'].encode()
        if (mask!=[1]*len(ids) or any(type(i)is not int or not 0<=i<248320 for i in ids)):
            raise ValueError('Frozen valid IDs/all-attended mask required')
        result+=struct.pack('<II',len(name),len(ids))+name+struct.pack('<'+'I'*len(ids),*ids)+bytes(mask)
    return result


def validate_native(case,record,directory):
    expected_id,tokens=CASES[case]
    if (record.get('id')!=expected_id or record.get('tokens')!=tokens or record.get('completed')is not True or
            record.get('ordinary_repeat_bit_exact')is not True or record.get('logits_empty')is not True):
        raise ValueError('Native case/neutrality/no-head coverage differs')
    rows=record.get('stages',[])
    if len(rows)!=26:raise ValueError('Native26stage coverage differs')
    for i,(row,event) in enumerate(zip(rows,stages(tokens))):
        if any(row.get(k)!=v for k,v in event.items()) or row.get('file')!=f'{case:02d}/{i:02d}.f32':
            raise ValueError('Native stage/occurrence/shape/path differs')
        path=directory/row['file']
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=4*math.prod(event['shape']):
            raise ValueError('Native F32 stage extent differs')
    return rows


def comparison_records(events,native,metrics):
    if len(events)!=26 or len(native)!=26 or set(metrics)!={f'stage.{i:02d}' for i in range(26)}:
        raise ValueError('All26original/native/comparator stages required')
    result=[]
    for i,(original,row) in enumerate(zip(events,native)):
        measured=metrics[f'stage.{i:02d}']
        if row['stage']!=original['stage'] or measured['shape']!=original['shape'] or row['shape']!=original['shape'] or row['occurrence']!=0:
            raise ValueError('Comparator/original/native shape or occurrence differs')
        result.append(dict(stage=original['stage'],occurrence=0,original_file=original['file'],native_file=row['file'],**measured))
    return result
