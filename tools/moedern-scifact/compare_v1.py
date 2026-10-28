"""Hosted bounded all-vector/actual-Ref and complete ranking comparison.

The earliest original vector stage retained by the full original capture is
normalized_all. These results cannot identify an internal attention/FFN cause.
"""
import json
from pathlib import Path
import time
from contract_native_v1 import require,sha
import diff_adapter_v1 as adapter
import moedern_scifact as S


def metrics(a,b):
    import numpy as np
    require(a.shape==b.shape and a.ndim==2 and a.shape[1]==128 and len(a)>0,'exact nonempty128-wide matrices required')
    size=a.size;abs_sum=err_sum=square=aa=bb=0.;maximum=0.;cosine=1.
    amin=bmin=aminabs=bminabs=float('inf');amax=bmax=-float('inf');amaxabs=bmaxabs=0.
    for i in range(0,len(a),4096):
        x=np.asarray(a[i:i+4096],dtype=np.float64);y=np.asarray(b[i:i+4096],dtype=np.float64)
        require(np.isfinite(x).all()and np.isfinite(y).all(),'nonfinite normalized vectors')
        delta=y-x;absolute=np.abs(delta);xx=np.sum(x*x,axis=1);yy=np.sum(y*y,axis=1)
        xy=np.sum(x*y,axis=1);den=np.sqrt(xx*yy)
        c=np.divide(xy,den,out=np.where((xx==0)&(yy==0),1.,0.),where=den>0)
        cosine=min(cosine,float(c.min()));maximum=max(maximum,float(absolute.max()))
        abs_sum+=float(absolute.sum());err_sum+=float(delta.sum());square+=float((delta*delta).sum())
        aa+=float(xx.sum());bb+=float(yy.sum())
        amin=min(amin,float(x.min()));amax=max(amax,float(x.max()));bmin=min(bmin,float(y.min()));bmax=max(bmax,float(y.max()))
        ax=np.abs(x);by=np.abs(y)
        aminabs=min(aminabs,float(ax.min()));amaxabs=max(amaxabs,float(ax.max()));bminabs=min(bminabs,float(by.min()));bmaxabs=max(bmaxabs,float(by.max()))
    return dict(shape=list(a.shape),elements=size,max_absolute=maximum,mean_absolute=abs_sum/size,mean_error=err_sum/size,
                rms=(square/size)**.5,min_row_cosine=cosine,original_l2=aa**.5,native_l2=bb**.5,
                original_min=amin,original_max=amax,original_abs_min=aminabs,original_abs_max=amaxabs,
                native_min=bmin,native_max=bmax,native_abs_min=bminabs,native_abs_max=bmaxabs,
                passed=maximum<=.002 and cosine>=.99999)


def partition(total,count=8):
    require(type(total)is int and total>=count and count==8,'eight nonempty corpus shards required')
    return [(i*total//count,(i+1)*total//count)for i in range(count)]


def vectors(original,capture,binary,out,command,env):
    import numpy as np
    out.mkdir(parents=True,exist_ok=False)
    status=json.loads((capture/'capture.json').read_text())
    require(status['completed']and status['rows']==5483 and status['failure_count']==0,'complete typed C-ABI capture required before numeric comparison')
    journal=[json.loads(line)for line in(capture/'rows.jsonl').read_text().splitlines()]
    require(len(journal)==5483,'complete native vector journal required')
    for name,pin in status['files'].items():require((capture/name).stat().st_size==pin['bytes']and sha(capture/name)==pin['sha256'],'native stream integrity differs')
    matrices={};targets=original['journal']
    for kind in('all','selected'):
        path=original['raw']/(kind+'.f32');mine=capture/(kind+'.f32')
        require(path.stat().st_size==mine.stat().st_size and path.stat().st_size%512==0,'exact full stream extent differs')
        matrices[kind]=(np.memmap(path,dtype='<f4',mode='r').reshape(-1,128),np.memmap(mine,dtype='<f4',mode='r').reshape(-1,128))
    failures=0;first=None;worst=None
    with(out/'per-row-metrics.jsonl').open('x')as log:
        for target,row in zip(targets,journal):
            require(row['completed']and row['typed_exact']and row['key']==target['key'],'native row identity/typed coverage differs')
            for kind in('all','selected'):
                require(all(row[kind+s]==target[kind+s]for s in('_offset','_bytes')),'native row geometry differs')
                start=target[kind+'_offset']//512;n=target[kind+'_bytes']//512;a,b=matrices[kind]
                value=metrics(a[start:start+n],b[start:start+n]);record=dict(key=row['key'],stage='normalized_'+kind,**value)
                if not value['passed']:
                    failures+=1
                    if first is None:first=record
                if worst is None or value['max_absolute']>worst['max_absolute']:worst=record
                log.write(json.dumps(record,allow_nan=False)+'\n')
    shards=[]
    for index,(begin,end)in enumerate(partition(5483)):
        folder=out/('shard-'+str(index));folder.mkdir();independent={};specs={'original':{},'native':{}}
        for kind in('all','selected'):
            offset=targets[begin][kind+'_offset'];stop=targets[end-1][kind+'_offset']+targets[end-1][kind+'_bytes'];count=stop-offset
            name='normalized_'+kind;a,b=matrices[kind];independent[name]=metrics(a[offset//512:stop//512],b[offset//512:stop//512])
            for label,path in(('original',original['raw']/(kind+'.f32')),('native',capture/(kind+'.f32'))):
                specs[label][name]=dict(path=path,offset=offset,bytes=count,shape=[count//512,128])
        identities={label:adapter.write(folder/(label+'.gguf'),rows)for label,rows in specs.items()}
        run=command([binary,folder/'original.gguf',folder/'native.gguf',folder/'ref.json'],folder/'ref.log',env,allowed=(0,1))
        actual=adapter.collect(run['exit'],folder/'ref.json',independent)
        require(actual['all_metrics_agree'],'actual Ref/independent full-vector metrics disagree')
        shards.append(dict(index=index,case_range=[begin,end],first_key=targets[begin]['key'],last_key=targets[end-1]['key'],
                           adapter_identities=identities,actual_ref=actual,metrics=independent,process=run))
        # Both source streams and exact spans/hashes are retained. Avoid adding
        # another two complete copies of the multi-GB arrays to remote output.
        for label in specs:(folder/(label+'.gguf')).unlink()
    return dict(rows=5483,vector_matrices=10966,elements=sum(a.size for a,b in matrices.values()),
                failed_matrices=failures,first_numeric_rejection=first,worst_absolute=worst,shards=shards,
                strict_numeric_passed=failures==0 and all(m['passed']for s in shards for m in s['metrics'].values()),
                full_metrics_file=dict(file='per-row-metrics.jsonl',sha256=sha(out/'per-row-metrics.jsonl')),
                earliest_available_original_stage='normalized_all; internal encoder states were not captured by this full-corpus original.')


def ranking(original,capture,out,*,quantized):
    import numpy as np
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    rows=original['journal'];stream=np.memmap(capture/'selected.f32',dtype='<f4',mode='r').reshape(-1,128)
    starts=np.array([r['selected_offset']//512 for r in rows[300:]],dtype=np.int64)
    documents=stream[starts[0]:];starts-=starts[0]
    scores=np.empty((300,5183),dtype='<f8');ranks=[];docs=[d['id']for d in original['data']['documents']]
    max_error=0.;sum_error=0.;square_error=0.;finite=True
    for i,q in enumerate(original['data']['queries']):
        r=rows[i];begin=r['selected_offset']//512;count=r['selected_bytes']//512
        sim=documents@stream[begin:begin+count].T  # Explicit F32 dot, unchanged original protocol.
        scores[i]=np.maximum.reduceat(sim,starts,axis=0).astype(np.float64).sum(axis=1)
        require(np.isfinite(scores[i]).all(),'nonfinite full MaxSim scores')
        delta=scores[i]-original['scores'][i];absolute=np.abs(delta)
        max_error=max(max_error,float(absolute.max()));sum_error+=float(absolute.sum());square_error+=float((delta*delta).sum())
        ranks.append(dict(query_id=q['id'],**S.rank_scores(docs,scores[i].tolist(),q['relevant'])))
    np.save(out/'scores.npy',scores,allow_pickle=False)
    (out/'rankings.json').write_text(json.dumps(ranks,allow_nan=False)+'\n')
    metrics_now=S.mean_metrics(ranks);drops={k:original['report']['metrics'][k]-v for k,v in metrics_now.items()}
    agreement=sum(a['top1']==b['top1']for a,b in zip(original['ranks'],ranks))/300
    pairs=json.loads((capture/'capture.json').read_text())['pairs']
    checks=[dict(**r,protocol=float(scores[r['query'],r['document']]),max_absolute=abs(r['score']-float(scores[r['query'],r['document']])))for r in pairs]
    require(len(checks)==6 and all(c['max_absolute']<=.002 for c in checks),'public raw-text score differs from F32dot/F64sum protocol')
    quality=all(d<=.001 for d in drops.values())and agreement>=.99
    return dict(protocol='F32dot/F64sum, all300x5183 scores; ascending document-ID tie order',
                metrics=metrics_now,original_metrics=original['report']['metrics'],metric_drop=drops,top1_agreement=agreement,
                unchanged_quality_gates=dict(max_metric_drop=.001,min_top1_agreement=.99),quality_passed=quality,
                max_score_absolute=max_error,mean_score_absolute=sum_error/scores.size,score_rms=(square_error/scores.size)**.5,
                strict_score_absolute_passed=max_error<=.002,quantized=quantized,public_score_checks=checks,
                seconds=time.monotonic()-started,files={n:dict(bytes=(out/n).stat().st_size,sha256=sha(out/n))for n in('scores.npy','rankings.json')})
