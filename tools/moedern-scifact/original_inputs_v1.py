"""Authenticate complete retained originals before any native model access."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import moedern_scifact as S
from contract_native_v1 import ROOT,admit,git,require,sha


def region(path,offset,count):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        stream.seek(offset)
        while count:
            block=stream.read(min(4*2**20,count));require(block and len(block)<=count,'truncated original vector span')
            h.update(block);count-=len(block)
    return h.hexdigest()


def expected_tokens(row):
    supplied=list(map(int,row['supplied_mask']))
    return dict(ids=row['ids'],supplied_mask=supplied,effective_mask=list(map(int,row['effective_mask'])),
                scoring_mask=list(map(int,row['scoring_mask'])),
                expansion_positions=[1-x for x in supplied]if row['role']=='query'else[])


def locate(mount,pin):
    hits=[p for p in Path(mount).rglob('original.json')if p.is_file()and not p.is_symlink()and
          p.stat().st_size==pin['report_bytes']and sha(p)==pin['report_sha256']]
    require(len(hits)==1,'one exact immutable original report mount required')
    return hits[0]


def authenticate(mount,variant,cfg):
    pin=admit(cfg,variant);path=locate(mount,pin);root=path.parent.parent
    report=json.loads(path.read_text());raw=root/'raw'
    require(report['schema']=='crispembed-moedern-scifact-original-v1'and report['completed']is True and
            report['native_executed']is False and report['variant']==variant and report['source']==pin['source']and
            report['rows']==5483,'complete original5483 source/scope differs')
    require(report['preflight']['all_features_bit_exact']is True and report['preflight']['public_slice_bit_exact']is True and
            len(report['preflight']['keys'])==13,'original observer/public-slice controls differ')
    require(sha(ROOT/'docs/moedern-scifact-parity-v1.json')==report['protocol_sha256'],'original protocol differs')
    for name,value in report['driver_sha256'].items():
        require(hashlib.sha256(git('show',pin['source']+':'+name)).hexdigest()==value,'original driver source differs: '+name)
    names={'data.json','rows.json','all.f32','selected.f32','scores.npy','rankings.json'}
    require(set(report['arrays'])==names,'complete original raw inventory required')
    for name,identity in report['arrays'].items():
        file=raw/name
        require(file.is_file()and not file.is_symlink()and file.resolve().is_relative_to(root.resolve())and
                file.stat().st_size==identity['bytes']and sha(file)==identity['sha256'],'original raw member differs: '+name)
    data=json.loads((raw/'data.json').read_text());journal=json.loads((raw/'rows.json').read_text())
    require(len(data['queries'])==300 and len(data['documents'])==5183 and len(journal)==5483,'full canonical coverage required')
    require(S.digest(data)==report['data']['all_rows_sha256'],'canonical text/data identity differs')
    gold={q['id']:{d:1 for d in q['relevant']}for q in data['queries']}
    require(S.digest(gold)==report['data']['gold_sha256'],'canonical qrels identity differs')
    inputs=[dict(**r,key='q:'+r['id'],role='query')for r in data['queries']]+[dict(**r,key='d:'+r['id'],role='document')for r in data['documents']]
    require(len({r['key']for r in inputs})==5483,'duplicate input identity')
    offsets={'all':0,'selected':0}
    for source,row in zip(inputs,journal):
        require(row['key']==source['key']and row['id']==source['id']and row['role']==source['role'],'original row order differs')
        encoded=source['text'].encode('utf-8');require(b'\0'not in encoded and len(encoded)<=65536,'raw text exceeds published API subset')
        ids=row['ids'];n=len(ids)
        require(3<=n<=512 and ids[:2]==[2,30000 if row['role']=='query'else 30001]and
                all(type(i)is int and 0<=i<30002 for i in ids),'original IDs/role prefix differ')
        for name in ('supplied_mask','effective_mask','scoring_mask'):
            require(len(row[name])==n and all(type(x)in(bool,int)and x in(0,1)for x in row[name]),'original mask differs')
        require(row['supplied_ids']==ids and row['effective_mask']==([1]*n if row['role']=='query'else row['supplied_mask']),
                'original observed supplied/effective IDs or masks differ')
        require(row['role']!='query'or(n==32 and all(row['scoring_mask'])),'query fixed32 scoring differs')
        count=sum(row['scoring_mask']);require(count>0,'empty original scoring selection')
        for kind,length in (('all',n),('selected',count)):
            require(row[kind+'_offset']==offsets[kind]and row[kind+'_bytes']==length*512,'original stream row extent differs')
            require(region(raw/(kind+'.f32'),offsets[kind],length*512)==row[kind+'_sha256'],'original per-row vector SHA differs')
            offsets[kind]+=length*512
    require(all(offsets[kind]==report['arrays'][kind+'.f32']['bytes']for kind in offsets),'original stream totals differ')
    # NumPy is used only on hosted frozen inputs, after every byte is authenticated.
    import numpy as np
    scores=np.load(raw/'scores.npy',mmap_mode='r',allow_pickle=False)
    require(scores.shape==(300,5183)and scores.dtype==np.dtype('<f8')and np.isfinite(scores).all(),'original score matrix differs')
    ranks=json.loads((raw/'rankings.json').read_text());docs=[d['id']for d in data['documents']]
    independently=[dict(query_id=q['id'],**S.rank_scores(docs,scores[i].tolist(),q['relevant']))for i,q in enumerate(data['queries'])]
    require(ranks==independently and S.mean_metrics(ranks)==report['metrics'],'all300 original rankings/metrics differ')
    return dict(root=root,raw=raw,report=report,data=data,rows=inputs,journal=journal,ranks=ranks,
                scores=scores,pin=pin,expansion_truth='Derived fixed32 pad positions from observed supplied mask; not a new captured original tensor.')
