"""Hosted retained-ID decoder proof: no model forward or original regeneration.

Decode the original and initial native real-scan IDs with the production helper,
using the actual published base vocabulary plus the verified831-token fallback.
Original raw strings remain frozen gold; mismatched-image rows stay ineligible.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import urllib.request
ROOT=Path(__file__).resolve().parents[2]

def require(ok,why):
    if not ok:raise ValueError(why)

def hosted():
    require(os.environ.get('GITHUB_ACTIONS')=='true'or(Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()and os.environ.get('KAGGLE_KERNEL_RUN_TYPE')in('Batch','Interactive')),'hosted retained-ID decode only')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        while block:=f.read(1024**2):h.update(block)
    return h.hexdigest()

def inputs():
    folder=ROOT/'tests/results/ocr-real-scans';pins=json.loads((folder/'source-identities.json').read_text())
    reports=[]
    for name in ('unlimited-original.json','unlimited-native-initial.json'):
        p=folder/name;require(p.stat().st_size==pins[name]['bytes']and sha(p)==pins[name]['sha256'],'frozen real-scan producer differs')
        r=json.loads(p.read_text());require(r['commit']==pins[name]['source']and len(r['runs'])==20,'complete real-scan source required');reports.append(r)
    original,native=reports
    require(original['reference_meta']['completed']and original['reference_meta']['exit']==0,'original capture incomplete')
    require(native['alias']['verified']and all(r['exit']==0 for r in native['runs']),'initial native commands incomplete')
    receipt=module('scan_input_review',ROOT/'tools/kaggle/ocr-real-scans/receipt.py')
    paired=receipt.paired(native,original,json.loads((ROOT/'docs/ocr-real-scans-v1.json').read_text()))
    return original,native,paired

def fixture(vocab,added,cases):
    strings=module('decode_fixture_strings',ROOT/'tools/unlimited-ocr2/gen_text_fixture.py')
    used=sorted({i for c in cases for i in c['ids']if i<128000});require(all(0<=i<len(vocab)for i in used),'base token ID out of range')
    lines=['#pragma once','#include <cstdint>','namespace real_decode_fixture {','struct piece { int id; const char * value; };','static const piece pieces[] = {']
    lines += ['{%d,%s},'%(i,strings.c_str(vocab[i]))for i in used];lines+=['};','static const piece added[] = {']
    lines += ['{%d,%s},'%(r['id'],strings.c_str(r['content']))for r in added];lines+=['};']
    for i,c in enumerate(cases):lines+=['static const int32_t ids%d[] = {%s};'%(i,','.join(map(str,c['ids'])))]
    lines+=['struct entry { const char * key; const int32_t * ids; int count; };','static const entry cases[] = {']
    lines+=['{"case_%03d",ids%d,%d},'%(i,i,len(c['ids']))for i,c in enumerate(cases)];lines+=['};','}']
    return '\n'.join(lines)+'\n'

def replay(model,out):
    hosted();original,native,paired=inputs();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='crispembed-unlimited-real-decoder-replay-v1',completed=False,model_inference=False,original_rerun=False,paired_input_review=paired,rows=[])
    def save():(out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        published=json.loads((ROOT/'tests/results/unlimited-ocr2/publication.json').read_text());pin=next(r for r in published['remote']['files']if r['variant']=='F16')
        require(Path(model).stat().st_size==pin['bytes']and sha(model)==pin['sha256'],'exact published F16 metadata owner required')
        from gguf import GGUFReader
        reader=GGUFReader(model);vocab=reader.fields['tokenizer.ggml.tokens'].contents()
        require(len(vocab)==128000 and all(isinstance(v,str)for v in vocab),'published128000-piece base vocabulary required')
        cfg=json.loads((ROOT/'docs/unlimited-ocr2-source.json').read_text());identity=cfg['files']['tokenizer.json']
        url=f"https://huggingface.co/{cfg['repository']}/resolve/{cfg['revision']}/tokenizer.json"
        with urllib.request.urlopen(url,timeout=90)as response:data=response.read(identity['bytes']+1)
        require(len(data)==identity['bytes']and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==identity['git_oid'],'original tokenizer immutable bytes differ')
        tokenizer=json.loads(data);source=tokenizer['model']['vocab'];require(len(source)==128000 and all(vocab[i]==piece for piece,i in source.items()),'published base pieces differ from original tokenizer')
        added=[r for r in tokenizer['added_tokens']if r['id']>=128000];require(sorted(r['id']for r in added)==list(range(128000,128831)),'all831 original added pieces required')
        cases=[dict(c,origin=origin)for origin,r in [('original',original),('initial_native',native)]for c in r['runs']]
        require(all(isinstance(c['ids'],list)and c['ids']and all(type(i)is int and 0<=i<=128830 for i in c['ids'])for c in cases),'actual retained token IDs required')
        header=out/'fixture.h';header.write_text(fixture(vocab,added,cases))
        probe=out/'decode-probe';cmd=['c++','-std=c++17','-O2','-I',str(ROOT/'src'),'-I',str(out),str(ROOT/'tests/unlimited-real-decode/probe.cpp'),'-o',str(probe)]
        with (out/'compile.log').open('w')as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        subprocess.run([str(probe),str(out)],check=True,timeout=30)
        clean=module('independent_clean',ROOT/'tools/unlimited-ocr2/gen_text_fixture.py').clean
        outputs={};eligible={r['page']:r['eligible']for r in paired['rows']}
        for i,c in enumerate(cases):
            raw=(out/('case_%03d.raw.txt'%i)).read_bytes();text=raw.decode('utf-8');cleaned=(out/('case_%03d.clean.txt'%i)).read_bytes()
            require(cleaned==clean(text).encode(),'C++/independent Python clean disagree: '+c['page'])
            if c['origin']=='original':require(raw==c['text'].encode(),'original retained IDs do not reproduce frozen raw string: '+c['page'])
            outputs[c['origin'],c['page']]=(raw,cleaned)
            report['rows'].append(dict(page=c['page'],origin=c['origin'],input_eligible=eligible[c['page']],raw_bytes=len(raw),raw_sha256=hashlib.sha256(raw).hexdigest(),clean_sha256=hashlib.sha256(cleaned).hexdigest(),original_raw_exact=raw==c['text'].encode()if c['origin']=='original'else None))
        comparisons=[]
        for pair in paired['rows']:
            key=pair['page'];a,b=outputs['original',key],outputs['initial_native',key]
            comparisons.append(dict(page=key,eligible=pair['eligible'],ids_exact=pair.get('ids_exact'),corrected_raw_exact=a[0]==b[0],corrected_clean_exact=a[1]==b[1]))
        valid=[r for r in comparisons if r['eligible']];matching=[r for r in valid if r['ids_exact']]
        require(len(valid)==19 and len(matching)==11 and all(r['corrected_raw_exact']for r in matching),'all eligible matching-ID raw strings required')
        report.update(completed=True,all20_original_raw_strings_exact=True,all40_independent_clean_exact=True,all831_added_tokens_exact=True,published_base_vocabulary_matches_original=True,model_sha256=pin['sha256'],tokenizer_sha256=hashlib.sha256(data).hexdigest(),source_sha256={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),ROOT/'src/unlimited_ocr2_text.h',ROOT/'src/core/bpe.h',ROOT/'tests/unlimited-real-decode/probe.cpp']},comparisons=comparisons,eligible_pairs=19,eligible_exact_id_pairs=11,eligible_corrected_raw_exact=sum(r['corrected_raw_exact']for r in valid),eligible_corrected_clean_exact=sum(r['corrected_clean_exact']for r in valid),full_model_real_scan_qualified=False,
                      limits=['Decoder-only execution on frozen streams; it does not run or qualify repaired whole-model inference.','One failed encoded-image proof remains excluded, visible and pending.','Eight meaningful/coordinate token differences require separate model attribution; equal clean strings do not waive raw-ID differences.'])
        save();return report
    except Exception as error:report['error']=str(error);save();raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();replay(a.model,a.out)
