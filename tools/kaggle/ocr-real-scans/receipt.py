"""Input admission and paired receipt review; no inference or package dependencies."""
import hashlib
import json
from pathlib import Path

def admit_image(data,pin):
    actual=hashlib.sha256(data).hexdigest()
    if len(data)!=pin['bytes'] or actual!=pin['sha256']:
        raise ValueError('encoded image identity differs from frozen manifest')
    return actual

def image_status(report,manifest):
    expected={p['id']:p['image']['sha256'] for p in manifest['pages']}
    observed={}
    for row in report['images']:
        ident=row['id']
        if ident not in expected or ident in observed:raise ValueError('unknown/duplicate image receipt')
        observed[ident]=bool(row.get('sha256_ok')is True and row.get('sha256')==expected[ident] and row.get('expected_sha256')==expected[ident])
    if set(observed)!=set(expected):raise ValueError('incomplete image receipt coverage')
    return observed

def paired(native,original,manifest):
    # Invalid image rows stay visible; none enter paired-quality denominators.
    nimage,rimage=image_status(native,manifest),image_status(original,manifest)
    def cases(report):
        result={}
        for row in report['runs']:
            if row['page']in result:raise ValueError('duplicate result row')
            result[row['page']]=row
        if set(result)-set(nimage):raise ValueError('unknown output row')
        return result
    ns,rs=cases(native),cases(original);rows=[]
    for page in manifest['pages']:
        key=page['id'];n,r=ns.get(key),rs.get(key);reason=None
        if not nimage[key]or not rimage[key]:reason='frozen image hash rejected'
        elif not n or not r:reason='missing native/original output'
        elif n.get('exit')!=0 or original.get('reference_meta',{}).get('exit')!=0:reason='unsuccessful producer'
        elif not original.get('reference_meta',{}).get('completed'):reason='incomplete original producer'
        elif not isinstance(n.get('text'),str)or not isinstance(r.get('text'),str):reason='missing text'
        elif not isinstance(n.get('ids'),list)or not isinstance(r.get('ids'),list)or not n['ids']or not r['ids']:reason='missing token IDs'
        elif n.get('max_tokens')!=page['max_tokens']or r.get('max_tokens')!=page['max_tokens']:reason='changed output cap'
        record=dict(page=key,input_verified=nimage[key]and rimage[key],eligible=reason is None)
        if reason:record['rejection']=reason
        else:
            a,b=n['ids'],r['ids'];first=next((i for i,(x,y)in enumerate(zip(a,b))if x!=y),None if len(a)==len(b)else min(len(a),len(b)))
            record.update(ids_exact=a==b,raw_text_exact=n['text']==r['text'],first_token_difference=first,native_tokens=len(a),original_tokens=len(b),
                          native_cap_hit=n.get('cap_hit'),original_cap_hit=r.get('cap_hit'),native_seconds=n.get('wall_seconds'),original_seconds=r.get('wall_seconds'))
        rows.append(record)
    valid=[r for r in rows if r['eligible']]
    return dict(schema='crispembed-ocr-real-scans-paired-review-v1',native_source=native['commit'],original_source=original['commit'],
                pages=len(rows),eligible_pairs=len(valid),exact_ids=sum(r['ids_exact']for r in valid),exact_raw_text=sum(r['raw_text_exact']for r in valid),
                complete_input_proof=all(r['input_verified']for r in rows),model_qualified=False,rows=rows)
