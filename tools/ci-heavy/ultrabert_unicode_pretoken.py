#!/usr/bin/env python3
"""Hosted tokenizer-only original-first A/B. Never run embeddings/weights."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import scifact_tokenizer_diagnostic as frozen_contract
import ultrabert_unicode_pretoken as contract
from ultrabert_contract import tokenizer_metadata
SPEC=importlib.util.spec_from_file_location('retained_diagnostic',ROOT/'tools/ci-heavy/scifact_tokenizer_diagnostic.py')
old=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(old)


def source_guard():
    if os.environ.get('GITHUB_ACTIONS')!='true':raise ValueError('Hosted tokenizer-only A/B required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=os.environ.get('GITHUB_SHA'):raise ValueError('Actual service/source identity differs')
    cfg=json.loads(contract.CONTRACT_PATH.read_text())
    if set(cfg['candidate_sources'])!=set(contract.SOURCES):raise ValueError('Exact candidate source closure required')
    for name,sha in cfg['candidate_sources'].items():
        if frozen_contract.digest((ROOT/name).read_bytes())!=sha:raise ValueError('Candidate source SHA differs: '+name)
    baseline=json.loads(contract.BASELINE_PATH.read_text())
    for name,sha in cfg['frozen_sources'].items():
        if frozen_contract.digest((ROOT/name).read_bytes())!=sha:raise ValueError('Historical diagnostic/core source changed: '+name)
    for name,sha in cfg['historical_only'].items():
        data=subprocess.check_output(['git','show',frozen_contract.BASE+':'+name],cwd=ROOT)
        if frozen_contract.digest(data)!=sha:raise ValueError('Historical-only source differs: '+name)
    if subprocess.check_output(['git','status','--porcelain','--',*cfg['candidate_sources'],*cfg['frozen_sources']],cwd=ROOT):raise ValueError('Dirty measured A/B source')
    return cfg,baseline,head


def execute(args,log,env,timeout=300):
    with log.open('w') as stream:
        subprocess.run([str(a) for a in args],stdout=stream,stderr=subprocess.STDOUT,env=env,check=True,timeout=timeout)


def run(out):
    cfg,baseline,head=source_guard();out.mkdir(parents=True,exist_ok=False);review=out/'review';review.mkdir()
    result=dict(schema='crispembed-ultrabert-unicode-pretoken-ab-v1',source=head,completed=False,tokenizer_parity=False,
        embeddings_authorized=False,new_embeddings_executed=False,model_weights_accessed=False,modes=[],errors=[])
    def save():old.write(review/'ab.json',result)
    save()
    try:
        state=old.api(f'actions/runs/{baseline["run"]}')
        if (state['head_sha']!=baseline['source'] or state['status']!='completed' or state['conclusion']!='success' or state['path']!='.github/workflows/ovisocr2-metadata.yml'):
            raise ValueError('Actual diagnostic run identity/completion differs')
        old.pinned(baseline['run'],f'scifact-tokenizer-diagnostic-review-{baseline["run"]}',baseline['artifact'],out/'diagnostic-review.zip')
        data=old.selected(out/'diagnostic-review.zip',['diagnostic.json'])['diagnostic.json']
        if frozen_contract.digest(data)!=baseline['report_sha256']:raise ValueError('Actual complete diagnostic receipt differs')
        report=json.loads(data)
        if ([r['id'] for r in report['row_summary'] if not r['clipped_ids_exact']]!=baseline['clipped_ids'] or
                [r['id'] for r in report['row_summary'] if not r['untruncated_ids_exact']]!=baseline['full_ids']):raise ValueError('Frozen diagnostic mismatch coverage differs')
        old.pinned(baseline['run'],f'scifact-tokenizer-diagnostic-metadata-{baseline["run"]}',contract.RAW,out/'diagnostic-metadata.zip')
        raw=old.selected(out/'diagnostic-metadata.zip',['native-ids.jsonl'])['native-ids.jsonl']
        if frozen_contract.digest(raw)!=baseline['native_ids_sha256']:raise ValueError('Actual historical native token rows differ')
        legacy=[json.loads(line) for line in raw.splitlines()];result['historical_native_sha256']=frozen_contract.digest(raw);save()
        source=out/'candidate-source'
        for name,sha in {**cfg['frozen_sources'],**cfg['candidate_sources']}.items():
            if not name.startswith('src/'):continue
            target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
        header=source/'src/tokenizer.h';original=header.read_bytes();header.write_text(contract.observed_header(original.decode()))
        old.write(review/'header-overlay.json',dict(before=frozen_contract.digest(original),after=frozen_contract.digest(header.read_bytes()),test_only=True))
        binary=out/'unicode-tokenizer'
        execute(['c++','-std=c++17','-O2','-I'+str(source/'src'),ROOT/'tests/test_ultrabert_unicode_pretoken.cpp',source/'src/tokenizer_bpe.cpp','-o',binary],review/'compile.log',os.environ,180)
        env={k:v for k,v in os.environ.items() if not k.startswith('CRISPEMBED_')};env['CRISPEMBED_SCIFACT_TOKENIZER_DIAGNOSTIC']='1'
        execute([binary,'--controls'],review/'controls.log',env,30)
        rows,tokens,negatives,document=old.prepare(out,json.loads(old.SPEC.read_text()))
        if importlib.metadata.version('tokenizers')!='0.23.3':raise ValueError('Pinned original tokenizer backend required')
        if any(m in sys.modules for m in ('torch','transformers','numpy')):raise ValueError('Model/framework import forbidden')
        import tokenizers
        from tokenizers import Tokenizer
        dist=importlib.metadata.distribution('tokenizers')
        if tokenizers.__version__!='0.23.3' or Path(tokenizers.__file__).resolve()!=Path(dist.locate_file('tokenizers/__init__.py')).resolve():raise ValueError('Loaded original backend origin differs')
        original=Tokenizer.from_file(str(out/'tokenizer.json'));original.no_padding();original.no_truncation()
        for row,token in zip(rows,tokens):
            actual=original.encode(row['input'],add_special_tokens=True)
            if (actual.ids[:512]!=token['ids'] or actual.attention_mask[:512]!=token['attention_mask'] or len(actual.ids)!=token['untruncated_tokens'] or
                    frozen_contract.digest(frozen_contract.canonical(actual.ids))!=token['untruncated_ids_sha256']):raise ValueError('Frozen original replay differs')
        cases=json.loads(contract.CASES_PATH.read_text())
        for case in cases:
            pieces=[frozen_contract.piece_bytes(p).decode() for p,_ in original.pre_tokenizer.pre_tokenize_str(case['text'])]
            if pieces!=case['pretokens']:raise ValueError('Authored splitter case differs from actual original backend: '+case['id'])
        result['all_original_ids_masks_fullhashes_exact']=True;result['authored_cases_original_exact']=len(cases);save()
        vocab,_,merges=tokenizer_metadata(document);packet=out/'input.packet';packet.write_bytes(frozen_contract.packet(vocab,merges,rows))
        if frozen_contract.digest(packet.read_bytes())!=baseline['input_packet_sha256']:raise ValueError('Frozen literal input/vocab/merge packet differs')
        for mode in ('unset','empty','0','1'):
            actual_env=dict(env)
            if mode!='unset':actual_env['CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN']='' if mode=='empty' else mode
            path=out/('native-'+mode+'.jsonl')
            with path.open('w') as stream,(review/('native-'+mode+'.stderr')).open('w') as errors:
                subprocess.run([str(binary),str(packet)],env=actual_env,stdout=stream,stderr=errors,check=True,timeout=300)
            actual=[json.loads(line) for line in path.read_text().splitlines()]
            try:measured=contract.compare_rows(actual,tokens,legacy,baseline,mode)
            except contract.ParityError as error:
                result['modes'].append(dict(mode=mode,passed=False,**error.details));save();raise
            if mode=='1':
                details=[]
                for row,token,candidate in zip(rows,tokens,actual):
                    if row['id'] not in negatives:continue
                    reference_parts=[frozen_contract.piece_bytes(p).decode() for p,_ in original.pre_tokenizer.pre_tokenize_str(row['input'])]
                    if candidate['actual_native_pretokens']!=reference_parts:raise ValueError('Corrected two-row pretoken boundaries differ from original')
                    details.append(dict(id=row['id'],actual_native_pretokens=candidate['actual_native_pretokens'],original_pretokens=reference_parts,ids=candidate['ids'],mask=candidate['attention_mask']))
                old.write(review/'corrected-two-row-details.json',details)
            result['modes'].append(dict(mode=mode,passed=True,raw_sha256=frozen_contract.digest(path.read_bytes()),**measured));save()
        result.update(completed=True,tokenizer_parity=True,limits='Default-off tokenizer-only corpus parity; no embedding/retrieval/model-quality qualification. Existing core Unicode14 categories; broader Unicode-version equivalence unqualified.')
    except Exception as error:
        result['errors'].append(dict(type=type(error).__name__,detail=str(error)));save();raise
    finally:
        result['files']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=frozen_contract.digest(p.read_bytes())) for p in out.rglob('*') if p.is_file() and p!=review/'ab.json'};save()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
if __name__=='__main__':main()
