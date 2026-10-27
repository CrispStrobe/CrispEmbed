#!/usr/bin/env python3
"""Hosted tokenizer-only retained-input diagnosis. No inference weights or vectors."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import urllib.request
import zipfile
import unicodedata
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from scifact_tokenizer_diagnostic import *
from ultrabert_contract import tokenizer_metadata
SPEC=ROOT/'docs/scifact-tokenizer-diagnostic.json'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def api(path):
    return json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/'+path],timeout=90))


def pinned(run,name,identity,out):
    listing=api(f'actions/runs/{run}/artifacts?per_page=100')
    matches=[r for r in listing['artifacts'] if r['name']==name and not r['expired']]
    if listing['total_count']>100 or len(matches)!=1:raise ValueError('Retained artifact absent/ambiguous')
    row=matches[0]
    if (row['id'],row['size_in_bytes'],row['digest'])!=(identity['id'],identity['bytes'],'sha256:'+identity['sha256']):
        raise ValueError('Frozen artifact service identity differs')
    with out.open('xb') as stream:
        subprocess.run(['gh','api',f'repos/CrispStrobe/CrispEmbed/actions/artifacts/{row["id"]}/zip'],stdout=stream,check=True,timeout=180)
    if out.stat().st_size!=identity['bytes'] or digest(out.read_bytes())!=identity['sha256']:
        raise ValueError('Frozen artifact ZIP bytes differ')
    return {k:row[k] for k in ('id','name','size_in_bytes','digest')}


def selected(archive,names):
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();paths=[PurePosixPath(i.filename) for i in infos]
        if (len(infos)>200 or len(set(i.filename for i in infos))!=len(infos) or
                any(p.is_absolute() or '..' in p.parts or '\\' in str(p) for p in paths) or
                sum(i.file_size for i in infos)>128*1024**2):raise ValueError('Retained ZIP scope/budget differs')
        return {name:z.read(name) for name in names}


def source_guard():
    if os.environ.get('GITHUB_ACTIONS')!='true':raise ValueError('Hosted metadata/source execution only')
    current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if current!=os.environ.get('GITHUB_SHA'):raise ValueError('Current service/source differs')
    spec=json.loads(SPEC.read_text());validate_manifest(spec)
    for name,value in spec['diagnostic_sources'].items():
        if digest((ROOT/name).read_bytes())!=value:raise ValueError('Diagnostic source differs: '+name)
    return spec


def prepare(out,spec):
    state=check_jobs(api(f'actions/runs/{RUN}'),api(f'actions/runs/{RUN}/jobs?per_page=100'))
    write(out/'review/retained-jobs.json',state)
    prep=api(f'actions/runs/{PREP_RUN}')
    if (prep['head_sha'],prep['status'],prep['conclusion'],prep['path'])!=(BASE,'completed','success','.github/workflows/ovisocr2-metadata.yml'):
        raise ValueError('Reviewed preparation service differs')
    proof={'prepare':pinned(PREP_RUN,f'ultrabert-scifact-prepared-{PREP_RUN}',PREP,out/'prepared.zip'),
           'freeze':pinned(RUN,'ultrabert-scifact-original-freeze-report',FREEZE,out/'freeze-report.zip')}
    proof['original_backend']=pinned(RUN,'ultrabert-scifact-original-report-0',ORIGINAL_REPORT,out/'original-report.zip')
    environment=selected(out/'original-report.zip',['environment.txt'])['environment.txt']
    if digest(environment)!=ORIGINAL_ENV_SHA or environment.decode().splitlines().count('tokenizers==0.23.3')!=1:
        raise ValueError('Executed original tokenizer backend differs')
    (out/'review/original-environment.txt').write_bytes(environment)
    frozen=selected(out/'freeze-report.zip',['original-freeze.json'])['original-freeze.json']
    if digest(frozen)!=FREEZE_REPORT:raise ValueError('Original freeze report SHA differs')
    frozen=json.loads(frozen)
    if (frozen['source']['head']!=BASE or frozen['status']!='all8 original shards frozen before native' or
            frozen['original_rows']!=5483 or frozen['data_freeze_sha256']!=DATA_FREEZE or frozen['no_native_outputs_yet'] is not True):
        raise ValueError('Original freeze is incomplete/different')
    names=['receipt.json','dataset.json','lengths.json','data-freeze.json','protocol.json']
    files=selected(out/'prepared.zip',names);receipt=json.loads(files['receipt.json'])
    if digest(files['receipt.json'])!=PREP_RECEIPT or receipt['source']['head']!=BASE or receipt['data_freeze_sha256']!=DATA_FREEZE:
        raise ValueError('Prepared receipt differs')
    for name in names[1:]:
        if {'bytes':len(files[name]),'sha256':digest(files[name])}!=receipt['files'][name]:raise ValueError('Prepared metadata member differs')
    notices=[n for n in receipt['files'] if n.startswith('notices/')]
    if len(notices)!=5:raise ValueError('Prepared attribution scope differs')
    for name,data in selected(out/'prepared.zip',notices).items():
        if {'bytes':len(data),'sha256':digest(data)}!=receipt['files'][name]:raise ValueError('Retained notice bytes differ')
        target=out/'review'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    if digest(files['data-freeze.json'])!=DATA_FREEZE:raise ValueError('Prepared input freeze differs')
    proof['negative_reports']={};negatives={}
    for shard,identity in NEGATIVES.items():
        archive=out/f'negative-{shard}.zip'
        proof['negative_reports'][str(shard)]=pinned(RUN,f'ultrabert-scifact-native-report-{shard}',identity,archive)
        members=selected(archive,['initial-id-mismatch.json','phase.json']);mismatch=json.loads(members['initial-id-mismatch.json'])
        phase=json.loads(members['phase.json'])
        if mismatch['id']!=identity['row'] or phase['source']['head']!=BASE or phase['error']!='Actual native text IDs differ: '+identity['row']:
            raise ValueError('Executed native negative provenance differs')
        negatives[mismatch['id']]=mismatch
        write(out/'review'/f'initial-negative-{shard}.json',mismatch)
    data=json.loads(files['dataset.json']);tokens=json.loads(files['lengths.json']);freeze=json.loads(files['data-freeze.json'])
    rows=data['queries']+data['documents']
    if len(data['queries'])!=300 or len(data['documents'])!=5183 or len(tokens)!=5483 or [r['id'] for r in rows]!=[r['id'] for r in tokens]:
        raise ValueError('Complete original ordered input scope differs')
    for row,token in zip(rows,tokens):
        row['input']=(QUERY if row['id'].startswith('q:') else '')+row['text']
        checksum=digest(row['input'].encode())
        if (checksum!=token['input_utf8_sha256'] or checksum!=freeze['mapping']['model_input_utf8_sha256'][row['id']] or
                not row['input'].encode() or len(row['input'].encode())>65536 or '<|' in row['input'] or '\0' in row['input']):
            raise ValueError('Frozen literal input bytes/policy differ')
    proof.update(frozen_metadata={n:{'bytes':len(files[n]),'sha256':digest(files[n])} for n in names},
                 original_shard_vectors_downloaded=False,model_weights_accessed=False)
    write(out/'review/prepared.json',proof)
    # The only HF source payload fetched is the pinned tokenizer JSON, no model.
    url=f'https://huggingface.co/dmis-lab/UltraBERT-Embed-150M/resolve/{REVISION}/tokenizer.json'
    with urllib.request.urlopen(url,timeout=60) as response:
        tokenizer=response.read(8*1024**2+1)
    if len(tokenizer)>8*1024**2 or digest(tokenizer)!=TOKENIZER_SHA:raise ValueError('Original tokenizer bytes differ')
    (out/'tokenizer.json').write_bytes(tokenizer)
    return rows,tokens,negatives,json.loads(tokenizer)


def run(out):
    spec=source_guard();out.mkdir(parents=True,exist_ok=False);(out/'review').mkdir()
    write(out/'review/source.json',dict(head=os.environ['GITHUB_SHA'],contract_sha256=digest(SPEC.read_bytes()),**spec))
    rows,frozen,negatives,document=prepare(out,spec)
    if importlib.metadata.version('tokenizers')!='0.23.3':raise ValueError('Original backend version differs')
    if any(name in sys.modules for name in ('torch','transformers','numpy')):raise ValueError('Unexpected model/framework import')
    import tokenizers
    from tokenizers import Tokenizer
    distribution=importlib.metadata.distribution('tokenizers')
    if tokenizers.__version__!='0.23.3' or Path(tokenizers.__file__).resolve()!=Path(distribution.locate_file('tokenizers/__init__.py')).resolve():
        raise ValueError('Loaded original tokenizer module identity differs')
    original=Tokenizer.from_file(str(out/'tokenizer.json'));original.no_padding();original.no_truncation()
    vocab,_,merges=tokenizer_metadata(document)
    original_rows={}
    for row,token in zip(rows,frozen):
        encoded=original.encode(row['input'],add_special_tokens=True)
        if (encoded.ids[:512]!=token['ids'] or encoded.attention_mask[:512]!=token['attention_mask'] or
                len(encoded.ids)!=token['untruncated_tokens'] or digest(canonical(encoded.ids))!=token['untruncated_ids_sha256']):
            raise ValueError('Original tokenizer replay differs from frozen rows: '+row['id'])
        if row['id'] in negatives:original_rows[row['id']]=encoded
    write(out/'review/original-token-replay.json',{'rows':5483,'all_frozen_ids_masks_full_hashes_exact':True,
        'new_original_embeddings':False,'tokenizers':'0.23.3','tokenizer_sha256':TOKENIZER_SHA})
    source=out/'frozen-source'
    for name,value in spec['runtime_sources'].items():
        data=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
        if digest(data)!=value:raise ValueError('Frozen native tokenizer source differs')
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    header=source/'src/tokenizer.h';before=header.read_bytes();header.write_text(expose_pretokens(before.decode()))
    write(out/'review/header-overlay.json',{'source':BASE,'before':digest(before),'after':digest(header.read_bytes()),
        'change':'test-only public const accessor; unchanged GPT2 splitter/encode/merge arithmetic'})
    binary=out/'native-tokenizer'
    with (out/'review/compile.log').open('w') as stream:
        subprocess.run(['c++','-std=c++17','-O2','-I'+str(source/'src'),str(ROOT/'tests/test_scifact_tokenizer_diagnostic.cpp'),
                        str(source/'src/tokenizer_bpe.cpp'),'-o',str(binary)],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=180)
    for flag in (None,'','0'):
        env=dict(os.environ);env.pop('CRISPEMBED_SCIFACT_TOKENIZER_DIAGNOSTIC',None)
        if flag is not None:env['CRISPEMBED_SCIFACT_TOKENIZER_DIAGNOSTIC']=flag
        control=subprocess.run([str(binary),'absent.packet'],env=env,capture_output=True,timeout=10)
        if control.returncode!=2 or control.stdout or control.stderr:
            raise ValueError('Unset/empty/literal0 diagnostic gate did not reject before input access')
    on_control=subprocess.run([str(binary),'absent.packet'],env=dict(os.environ,CRISPEMBED_SCIFACT_TOKENIZER_DIAGNOSTIC='1'),capture_output=True,timeout=10)
    if on_control.returncode!=2 or b'wrong packet magic' not in on_control.stderr:raise ValueError('Positive diagnostic gate did not reach packet validation')
    packet_path=out/'input.packet';packet_path.write_bytes(packet(vocab,merges,rows))
    output=out/'native-ids.jsonl'
    with output.open('w') as stream,(out/'review/native.stderr').open('w') as errors:
        subprocess.run([str(binary),str(packet_path)],env=dict(os.environ,CRISPEMBED_SCIFACT_TOKENIZER_DIAGNOSTIC='1'),
            stdout=stream,stderr=errors,check=True,timeout=300)
    actual=[json.loads(line) for line in output.read_text().splitlines()]
    if len(actual)!=5483 or [r['id'] for r in actual]!=[r['id'] for r in rows] or [r['row'] for r in actual]!=list(range(5483)):
        raise ValueError('Complete native tokenizer rows differ')
    summary=[];details=[]
    for row,token,native in zip(rows,frozen,actual):
        if (native['observer_bit_exact'] is not True or not native['ids'] or native['ids'][0]!=0 or
                native['ids']!=native['untruncated_ids'][:512] or
                any(type(i) is not int or not 0<=i<49152 for i in native['untruncated_ids'])):
            raise ValueError('Native observer/BOS contract differs')
        exact=native['ids']==token['ids'];full=digest(canonical(native['untruncated_ids']))==token['untruncated_ids_sha256']
        first=next((i for i,(a,b) in enumerate(zip(native['ids'],token['ids'])) if a!=b),min(len(native['ids']),len(token['ids'])) if not exact else None)
        summary.append({'id':row['id'],'input_utf8_sha256':token['input_utf8_sha256'],'native_tokens':len(native['ids']),
            'original_tokens':len(token['ids']),'clipped_ids_exact':exact,'untruncated_ids_exact':full,
            'first_differing_token':first})
        if row['id'] in negatives:
            old=negatives[row['id']]
            if native['ids']!=old['actual'] or token['ids']!=old['original']:raise ValueError('Standalone native probe did not reproduce executed negative')
            encoded=original_rows[row['id']]
            pretokens=original.pre_tokenizer.pre_tokenize_str(row['input'])
            original_spans=pretoken_spans([piece_bytes(p) for p,_ in pretokens],row['input'])
            native_spans=pretoken_spans([p.encode() for p in native['actual_native_pretokens']],row['input'])
            first_boundary=next((i for i,(a,b) in enumerate(zip(original_spans,native_spans)) if a!=b),
                min(len(original_spans),len(native_spans)) if original_spans!=native_spans else None)
            details.append({'id':row['id'],'input_utf8_sha256':token['input_utf8_sha256'],'literal_text':row['input'],
                'python_unidata_version':unicodedata.unidata_version,'codepoints':codepoints(row['input']),
                'first_differing_token':first,'native_ids':native['ids'],'original_frozen_ids':token['ids'],
                'original_backend_offsets_character_units':encoded.offsets,
                'original_backend_tokens':encoded.tokens,
                'original_pretokens':[{'bytelevel_piece':piece,'offsets_character_units':list(offset),
                    'decoded_raw_hex':piece_bytes(piece).hex()} for piece,offset in pretokens],
                'actual_native_pretokens':native['actual_native_pretokens'],
                'original_derived_pretoken_byte_spans':original_spans,'native_derived_pretoken_byte_spans':native_spans,
                'first_differing_pretoken':first_boundary,
                'original_id_pieces':describe_ids(token['ids'],vocab,row['input']),
                'native_id_pieces':describe_ids(native['ids'],vocab,row['input']),
                'native_piece_span_provenance':'Derived byte spans from actual native IDs and exact published vocabulary; native splitter strings are captured.'})
    write(out/'review/two-row-details.json',details)
    mismatches=[r for r in summary if not r['clipped_ids_exact']]
    report={'schema':'crispembed-scifact-tokenizer-diagnostic-v1','source':os.environ['GITHUB_SHA'],'runtime_source':BASE,
        'prepared_run':PREP_RUN,'original_run':RUN,'qualification':False,'embeddings_authorized':False,
        'new_embeddings_executed':False,'original_frozen_rows':5483,'native_rows':5483,
        'all_original_token_replay_exact':True,'native_clipped_token_preflight_pass':not mismatches,
        'native_clipped_token_mismatches':len(mismatches),'native_untruncated_mismatches':sum(not r['untruncated_ids_exact'] for r in summary),
        'first_mismatches':mismatches[:20],'executed_negative_two_rows_reproduced':True,
        'all_native_observer_ids_bit_exact':True,'gate_unset_empty_zero_rejected':True,
        'state':'completed-tokenizer-diagnosis','cause_claimed':False,
        'next':'Review actual first differing pretoken/codepoints before any gated runtime correction; require all5483 IDs before new embeddings.',
        'native_binary_sha256':digest(binary.read_bytes()),'input_packet_sha256':digest(packet_path.read_bytes()),
        'native_ids_sha256':digest(output.read_bytes()),'row_summary':summary}
    write(out/'review/diagnostic.json',report)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:run(a.output)
    except Exception as error:
        write(a.output/'review/rejection.json',{'qualification':False,'new_embeddings_executed':False,'error':str(error)})
        raise
if __name__=='__main__':main()
