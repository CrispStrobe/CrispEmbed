"""Eight bounded lossless stream adapters, preserving every128-wide token row.

This format is separate from historical per-case adapters. Each original stream
is already a concatenated F32 matrix; spans end on original case boundaries.
No normalization, token dropping, averaging, transposition or row fusion occurs.
"""
import hashlib
import json
import math
from pathlib import Path
import struct

MAX_ROWS=686*512
MAX_BYTES=512*2**20
FIELDS=('max_absolute','mean_absolute','mean_error','rms','min_row_cosine','original_l2','native_l2',
        'original_min','original_max','original_abs_min','original_abs_max',
        'native_min','native_max','native_abs_min','native_abs_max')


def require(ok,why):
    if not ok:raise ValueError(why)


def bounds(row):
    require(type(row['bytes'])is int and row['bytes']>0 and isinstance(row['shape'],list)and
            len(row['shape'])==2 and all(type(x)is int for x in row['shape'])and
            row['shape']==[row['bytes']//512,128]and 1<=row['shape'][0]<=MAX_ROWS and
            row['bytes']%512==0 and type(row['offset'])is int and row['offset']>=0 and row['offset']%512==0,
            'exact bounded128-wide F32 stream span required')
    path=Path(row['path']);require(path.is_file()and not path.is_symlink()and row['offset']+row['bytes']<=path.stat().st_size,'stream extent differs')
    return path


def write(path,streams):
    require(set(streams)=={'normalized_all','normalized_selected'},'exact two normalized streams required')
    names=sorted(streams);require(sum(r['bytes']for r in streams.values())<=MAX_BYTES,'adapter exceeds512MiB')
    rows=[];offset=0
    def text(value):
        value=value.encode('ascii');return struct.pack('<Q',len(value))+value
    header=b'GGUF'+struct.pack('<IQQ',3,2,0)
    for name in names:
        row=streams[name];bounds(row);n=row['shape'][0]
        header+=text(name)+struct.pack('<IQQIQ',2,128,n,0,offset)
        rows.append((name,row,offset));offset=(offset+row['bytes']+31)//32*32
    start=(len(header)+31)//32*32;identities={}
    path=Path(path)
    with path.open('xb')as target:
        target.write(header);target.write(b'\0'*(start-len(header)))
        for name,row,offset in rows:
            target.write(b'\0'*(start+offset-target.tell()));h=hashlib.sha256();left=row['bytes']
            with Path(row['path']).open('rb')as source:
                source.seek(row['offset'])
                while left:
                    payload=source.read(min(4*2**20,left));require(payload and len(payload)<=left,'truncated stream span')
                    target.write(payload);h.update(payload);left-=len(payload)
            identities[name]=dict(shape=row['shape'],bytes=row['bytes'],span_sha256=h.hexdigest(),stream_offset=row['offset'])
    h=hashlib.sha256()
    with path.open('rb')as source:
        while payload:=source.read(4*2**20):h.update(payload)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest(),tensors=identities,
                reconstructible_from_authenticated_streams=True,rows_transformed=False)


def collect(code,path,independent):
    require(code in(0,1),'actual Ref structural/process rejection')
    result=json.loads(Path(path).read_text());rows=result.get('stages',[])
    require(result['schema']=='crispembed-moedern-scifact-ref-v1'and result['max_absolute_budget']==.002 and
            result['minimum_row_cosine']==.99999 and result['status']==('passed'if code==0 else'numeric_rejection')and
            {r['name']for r in rows}==set(independent)=={'normalized_all','normalized_selected'}and len(rows)==2,
            'actual Ref fixed gates/coverage/status differ')
    numeric=False;agree=True
    for row in rows:
        ref=independent[row['name']]
        require(row['ggml_shape']==list(reversed(ref['shape']))and row['row_axis']==0 and row['elements']==ref['elements']and row['found']is True,'actual Ref shape/row/element coverage differs')
        require(all(math.isfinite(row[k])and math.isfinite(ref[k])for k in FIELDS),'nonfinite metric')
        gate=row['max_absolute']<=.002 and row['min_row_cosine']>=.99999
        require(row['passed']is gate,'actual Ref gate decision differs')
        consistent={k:abs(row[k]-ref[k])<=2e-7*max(1.,abs(row[k]),abs(ref[k]))for k in FIELDS}
        ref.update(actual_ref=row,metric_agreement=consistent,numpy_passed=ref['passed'],passed=ref['passed']and gate and all(consistent.values()))
        numeric|=not gate;agree&=all(consistent.values())
    require(numeric==(code==1),'actual Ref exit differs from numeric flags')
    return dict(exit=code,status=result['status'],all_metrics_agree=agree,stages=2)
