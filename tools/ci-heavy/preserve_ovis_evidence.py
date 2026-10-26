"""Byte-exact remote preservation; no reference regeneration or inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile
import ovisocr2_vision_audit as service

ROOT=Path(__file__).resolve().parents[2]
RECEIPT=ROOT/'tests/results/qwen35-ocr/ovisocr3-native-model.json'
RUN=40183489780
HEAD='26dcb07f29b0f1bd4e7d1fa2cd762650ab340a51'
RAW=(12595929739,'ovisocr3-native14-diff-40183489780',1237632076,
     '006d73e811f18bad66dabf9c1cf7f8476b0c845957d733e4a2fcd3a599a5cd3b')
SMALL=(12596845085,'ovisocr3-native14-reports-40183489780',334611,
       'f7211bbb70feb7c25e02c6154a4b36cf7d5e44e53bada8c9290fb345f4b19bad')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def safe_members(z):
    names=z.namelist()
    if (len(names)!=len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names)
        or any(n.endswith('/') for n in names) or sum(i.file_size for i in z.infolist())>32*1024**3):
        raise ValueError('Unsafe, duplicate or oversized preserved ZIP')
    return names


def expected_payloads(z):
    receipt=json.loads(RECEIPT.read_text())
    if receipt['run']!=RUN or receipt['source_commit']!=HEAD:raise ValueError('Frozen producer identity differs')
    body=z.read('ovisocr3-native14.json')
    if (hashlib.sha256(body).hexdigest()!=receipt['producer_report_sha256']
        or canonical(json.loads(body))!=receipt['independent_review']['producer_canonical_sha256']):
        raise ValueError('Frozen source producer report differs')
    records={}
    manifests=[n for n in safe_members(z) if n.endswith('-raw-manifest.json')]
    if len(manifests)!=12:raise ValueError('Original twelve raw manifests are incomplete')
    for name in manifests:
        data=json.loads(z.read(name))
        if data.get('coverage_complete') is not True:raise ValueError('Original raw coverage is incomplete')
        for row in data['files']:
            member=row['file']
            if not member.startswith('native-diff/'):raise ValueError('Original native path differs')
            member=member.removeprefix('native-diff/')
            shape=row['source_shape']
            if member in records or row['bytes']!=4*math.prod(shape):raise ValueError('Duplicate/invalid original payload')
            records[member]={'bytes':row['bytes'],'sha256':row['sha256'],'shape':shape,
                             'stage':row['stage'],'variant':row['variant']}
    if len(records)!=1662:raise ValueError('Original1662-vector coverage differs')
    return records


def inventory(path,expected):
    result={}
    with zipfile.ZipFile(path) as z:
        names=safe_members(z)
        if set(names)!=set(expected):raise ValueError('Preserved archive member coverage differs')
        for name in sorted(names):
            record=expected[name]
            if z.getinfo(name).file_size!=record['bytes']:raise ValueError('Preserved size differs: '+name)
            h=hashlib.sha256()
            with z.open(name) as f:
                for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
            if h.hexdigest()!=record['sha256']:raise ValueError('Preserved payload SHA differs: '+name)
            result[name]=record
    return result


def prepare(out,scratch):
    out.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    report={'schema':'crispembed-ovis-evidence-preservation-v1','status':'started',
            'original_run':RUN,'original_source':HEAD,'original_raw_artifact':RAW,
            'preservation_head':os.environ['GITHUB_SHA'],'preservation_run':int(os.environ['GITHUB_RUN_ID']),
            'no_inference_conversion_or_reference_regeneration':True,
            'source_helper_sha256':sha(__file__),'source_receipt_sha256':sha(RECEIPT)}
    def save():(out/'preservation.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        if shutil.disk_usage(scratch).free<8*1024**3:raise ValueError('At least8GiB hosted scratch required')
        report['source_raw_service']=service.service_source(RUN,HEAD,RAW,conclusion='success')
        report['source_small_service']=service.service_source(RUN,HEAD,SMALL,conclusion='success')
        service.download_artifact(SMALL,scratch/'source-reports.zip')
        with zipfile.ZipFile(scratch/'source-reports.zip') as z:expected=expected_payloads(z)
        payload=scratch/'payload';payload.mkdir(exist_ok=True)
        archive=payload/'original-native-vectors.zip';service.download_artifact(RAW,archive)
        records=inventory(archive,expected)
        manifest={'original_run':RUN,'original_head':HEAD,'source_zip_bytes':RAW[2],'source_zip_sha256':RAW[3],
                  'source_small_zip_sha256':SMALL[3],'vector_count':len(records),'members':records}
        (payload/'member-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        report.update(status='prepared immutable evidence copy',vector_count=len(records),
                      total_uncompressed_bytes=sum(r['bytes'] for r in records.values()),
                      member_manifest_sha256=sha(payload/'member-manifest.json'),
                      members_canonical_sha256=canonical(records))
    except Exception as e:report.update(status='failed evidence preparation',error=str(e));save();raise
    save()


def verify(out,scratch,directory):
    report=json.loads((out/'preservation.json').read_text())
    try:
        archive=directory/'original-native-vectors.zip';manifest=directory/'member-manifest.json'
        if (report['status']!='prepared immutable evidence copy' or archive.stat().st_size!=RAW[2]
            or sha(archive)!=RAW[3] or sha(manifest)!=report['member_manifest_sha256']):
            raise ValueError('Readback original ZIP/manifest identity differs')
        records=json.loads(manifest.read_text())['members']
        if canonical(inventory(archive,records))!=report['members_canonical_sha256']:
            raise ValueError('Readback member inventory differs')
        ident=int(os.environ['PRESERVED_ARTIFACT_ID'])
        meta=service.api_json('actions/artifacts/'+str(ident))
        digest=os.environ['PRESERVED_ARTIFACT_DIGEST'].removeprefix('sha256:')
        name='ovisocr3-native14-raw-'+str(RUN)+'-preserved-'+os.environ['GITHUB_RUN_ID']
        if (meta['name']!=name or meta['workflow_run']['id']!=int(os.environ['GITHUB_RUN_ID'])
            or meta['workflow_run']['head_sha']!=os.environ['GITHUB_SHA'] or meta['expired'] is not False
            or meta['digest']!='sha256:'+digest):raise ValueError('Readback artifact service/source differs')
        report.update(status='completed byte-exact evidence preservation',readback_all_1662_vectors_rehashed=True,
                      original_zip_bit_exact=True,preserved_artifact={k:meta[k] for k in
                      ('id','name','size_in_bytes','digest','expires_at')})
    except Exception as e:
        report.update(status='failed evidence readback',error=str(e));(out/'preservation.json').write_text(json.dumps(report,indent=2)+'\n');raise
    (out/'preservation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',type=Path);args=parser.parse_args()
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])
    if args.verify:verify(out,scratch,args.verify)
    else:prepare(out,scratch)
