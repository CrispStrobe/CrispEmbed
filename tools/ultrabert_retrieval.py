"""Model-free scorer for the frozen, owned UltraBERT retrieval smoke.

The producer must separately prove original module/source/prompt and actual native
API provenance. This scorer does not run a model or establish that provenance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

FIXTURE_SHA256 = '80143c78618b835d388d3133dbf508b8ff129c12659a42b94b2823983b1b6dc4'
QUERY = 'Instruct: Given a question, retrieve passages that answer the question\nQuery: '


def fixture(path):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=FIXTURE_SHA256:
        raise ValueError('Frozen retrieval inputs/gold changed')
    return json.loads(path.read_text())


def checked_vectors(data, record):
    ids=[q['id'] for q in data['queries']]+[d['id'] for d in data['documents']]
    if (record.get('pooling')!='cls' or record.get('query_prompt')!=QUERY
            or record.get('document_prompt')!='' or record.get('fixture_sha256')!=FIXTURE_SHA256):
        raise ValueError('Wrong pooling/prompt/input contract')
    rows=record.get('vectors',[])
    if len(rows)!=len(ids) or [r.get('id') for r in rows]!=ids:
        raise ValueError('Missing/extra/reordered retrieval vectors')
    result={}
    for row in rows:
        raw=row.get('raw_cls',[]);normal=row.get('normalized_cls',[])
        if len(raw)!=768 or len(normal)!=768 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in raw+normal):
            raise ValueError('Invalid vector shape/type/nonfinite value')
        norm=math.sqrt(math.fsum(x*x for x in raw))
        if norm<=1e-12 or max(abs(x/norm-y) for x,y in zip(raw,normal))>1e-6:
            raise ValueError('Raw and normalized CLS disagree')
        result[row['id']]=row
    return result


def rankings(data, vectors):
    documents=data['documents'];result=[]
    for q in data['queries']:
        scores=[]
        for doc in documents:
            a=vectors[q['id']]['normalized_cls'];b=vectors[doc['id']]['normalized_cls']
            score=math.fsum(x*y for x,y in zip(a,b))/math.sqrt(math.fsum(x*x for x in a)*math.fsum(x*x for x in b))
            scores.append({'id':doc['id'],'score':score,'relevant':doc['id'] in q['relevant_document_ids']})
        scores.sort(key=lambda x:(-x['score'],x['id']))
        relevant=set(q['relevant_document_ids']);top=scores[:10]
        reciprocal=next((1/(i+1) for i,x in enumerate(top) if x['relevant']),0)
        dcg=math.fsum(1/math.log2(i+2) for i,x in enumerate(top) if x['relevant'])
        ideal=math.fsum(1/math.log2(i+2) for i in range(min(10,len(relevant))))
        result.append({'query_id':q['id'],'ranking':scores,'recall_at_1':float(scores[0]['relevant']),
                       'mrr_at_10':reciprocal,'ndcg_at_10':dcg/ideal,'top_margin':scores[0]['score']-scores[1]['score']})
    return {'queries':result,'metrics':{k:math.fsum(q[k] for q in result)/len(result) for k in ('recall_at_1','mrr_at_10','ndcg_at_10')}}


def evaluate(data, original, native):
    before=checked_vectors(data,original);after=checked_vectors(data,native);comparisons=[]
    limits=data['acceptance']
    for key in before:
        for mode in ('raw_cls','normalized_cls'):
            a=before[key][mode];b=after[key][mode]
            error=max(abs(x-y) for x,y in zip(a,b))
            cosine=math.fsum(x*y for x,y in zip(a,b))/math.sqrt(math.fsum(x*x for x in a)*math.fsum(x*x for x in b))
            comparisons.append({'id':key,'mode':mode,'cosine':cosine,'max_abs':error,
                                'pass':cosine>=limits['vector_cosine_min'] and error<=limits['vector_max_abs']})
    baseline=rankings(data,before);candidate=rankings(data,after)
    agreement=all(a['ranking'][0]['id']==b['ranking'][0]['id'] for a,b in zip(baseline['queries'],candidate['queries']))
    vector_pass=all(c['pass'] for c in comparisons)
    quality_pass=(baseline['metrics']['recall_at_1']>=limits['original_recall_at_1_min'] and
                  candidate['metrics']['recall_at_1']>=limits['native_recall_at_1_min'])
    return {'fixture_sha256':FIXTURE_SHA256,'scope':'Owned eight-query smoke only; producer/source provenance needs separate audit',
            'original':baseline,'native':candidate,'vectors':comparisons,'vector_pass':vector_pass,
            'top1_exact':agreement,'quality_pass':quality_pass,'passed':vector_pass and agreement and quality_pass}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixture',type=Path);p.add_argument('original',type=Path);p.add_argument('native',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();data=fixture(args.fixture)
    result=evaluate(data,json.loads(args.original.read_text()),json.loads(args.native.read_text()))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
