"""Small input/event contract helpers for the separate hosted head adapter."""
import hashlib
import json
import math
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'docs/josev2b-head-native-contract.json'


def stage_plan(manifest):
    """Logical PyTorch shapes; repeated labels remain ordered occurrences."""
    tokens=manifest['tokens'];fields=manifest['fields'];total=sum(len(f['options']) for f in fields)
    rows=[];counts={}
    def event(label,shape):
        index=counts.get(label,0);counts[label]=index+1
        rows.append(dict(stage=label,occurrence=index,shape=shape))
    event('head/hidden_norm/input/0',[1,tokens,2048]);event('head/hidden_norm',[1,tokens,2048])
    event('head/memory_projection',[tokens,512])
    for f in fields:
        n=len(f['options']);event('head/option_context_projection',[n,512])
        event('head/option_lexical_projection/input/0',[n,2048]);event('head/option_lexical_projection',[n,512]);event('head/option_question_projection',[512])
    for i in range(2):event(f'head/evidence/{i}',[1,total,512])
    event('head/question_projection/input/0',[len(fields),2048]);event('head/question_projection',[len(fields),512])
    event('head/option_summary_norm',[len(fields),512]);event('head/global_projection',[512]);event('head/type_embedding',[len(fields),512])
    for i in range(4):event(f'head/field/{i}',[1,len(fields),512])
    event('head/field_norm',[len(fields),512])
    for f in fields:
        n=len(f['options']);event('head/option_norm',[n,512]);event('head/residual_scorer/input/0',[n,2048]);event('head/residual_scorer',[n,1])
    for f in fields:
        event('logits/'+f['id'],[len(f['options'])]);event('probabilities/'+f['id'],[len(f['options'])])
    return rows


def validate_manifest(value):
    if set(value)!={'schema','tokens','hidden_f32','fields'} or value['schema']!='crispembed-josev-head-input-v1':raise ValueError('Exact head input schema required')
    tokens=value['tokens'];fields=value['fields']
    if type(tokens)is not int or not 1<=tokens<=512 or not isinstance(fields,list) or not 1<=len(fields)<=16:raise ValueError('Head input bound differs')
    names=[value['hidden_f32']];ids=set();count=0
    for f in fields:
        if set(f)!={'id','type','question_start','question_end','options','lexical_f32'}:raise ValueError('Exact field schema required')
        if (not isinstance(f['id'],str) or not f['id'] or len(f['id'].encode())>256 or '\0' in f['id'] or f['id'] in ids or
                type(f['type'])is not int or not 0<=f['type']<=2 or not isinstance(f['options'],list) or not f['options']):raise ValueError('Field identity/type/options differs')
        ids.add(f['id']);count+=len(f['options']);names.append(f['lexical_f32'])
        ranges=[dict(start=f['question_start'],end=f['question_end']),*f['options']]
        for r in ranges:
            if set(r)!={'start','end'} or any(type(r[k])is not int for k in r) or not 0<=r['start']<r['end']<=tokens:raise ValueError('Empty/out-of-range span')
    if count>128:raise ValueError('Total option bound differs')
    for n in names:
        if not isinstance(n,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',n) or n in ('.','..'):raise ValueError('Flat relative raw filename required')
    return value


def validate_events(manifest,events):
    validate_manifest(manifest);expected=stage_plan(manifest)
    if len(events)!=len(expected):raise ValueError('Head stage coverage differs')
    for index,(actual,reference) in enumerate(zip(events,expected)):
        if (set(actual)!={'stage','occurrence','shape','file','dtype','elements'} or
                any(actual[k]!=v for k,v in reference.items()) or actual['dtype']!='float32' or
                actual['elements']!=math.prod(reference['shape']) or actual['file']!=f'{index:05d}.f32'):
            raise ValueError('Head stage order/occurrence/shape differs')
    return expected
