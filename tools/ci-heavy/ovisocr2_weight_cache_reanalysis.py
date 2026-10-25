"""Hosted analysis-only recovery of preserved e6 cache vectors. No model execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import subprocess
import zipfile

import ovisocr2_weight_cache as cache

ROOT=cache.ROOT
RUN=40215901483
HEAD='e6ee5afd93b2c49d3107e0b518e0944eaa6ada80'
REPORT=(12609596982,'ovisocr2-weight-cache-reports-40215901483',286372,
        '9ce9dc8715a8d88d5c24be152d3cc5893b7f3c8fd6a5784a0b00d0bb83799c9c')
RAW=(12609314203,'ovisocr2-weight-cache-raw-40215901483',2584835432,
     '7ae0f1f9812be851bc490b46a298c9f95b43b24ea70280555c6cdf0a1631d838')
SUFFIXES=('delta','vision','decoder-prepared','decoder-native-vision',
          'decoder-native-positions','decoder-native-frontend','context')
ARMS={f'{precision}-{suffix}' for precision in ('f32','f16') for suffix in SUFFIXES}


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def require_source_producer(producer):
    if (producer.get('schema')!='crispembed-ovisocr2-weight-cache-v1'
            or producer.get('status')!='rejected cache frozen audit'
            or producer.get('source',{}).get('source_commit')!=HEAD
            or producer.get('error')!='Cache frozen audit failed; all arms preserved'):
        raise ValueError('Unexpected original failed cache producer')
    expected=set()
    for on in (0,1):
        if set(producer['arms'][str(on)])!=ARMS:raise ValueError('Original fourteen-arm coverage differs')
        for name,row in producer['arms'][str(on)].items():
            decoder='-decoder-' in name
            if decoder:
                if row != {'passed':False,'error':"'file'"}:raise ValueError('Failure is not the observed manifest field error')
                expected.add(f'cache-{on}-{name}')
            elif row.get('passed') is not True:raise ValueError('Original non-decoder failure cannot be waived')
    expected|={'cache-on-off-raw-'+name for name in ARMS if '-decoder-' in name}
    if set(producer.get('frozen_failures',[]))!=expected:
        raise ValueError('Unexpected initial failure set')
    commands=producer.get('commands',{})
    wanted={'component-controls'}|{f'cache-{on}-{name}' for on in (0,1) for name in ARMS}
    wanted|={f'pair-{pair}-cache-{on}' for pair in range(3) for on in (0,1)}
    wanted|={f'two-contexts-cache-{on}' for on in (0,1)}
    if set(commands)!=wanted or any(row.get('exit_code')!=0 or row.get('passed') is not True
                                    or row.get('first_failure') is not None for row in commands.values()):
        raise ValueError('Original actual program/benchmark failure cannot be waived')
    return commands


def checked_members(archive, *, maximum):
    names=archive.namelist()
    if (len(names)!=len(set(names)) or sum(i.file_size for i in archive.infolist())>maximum
            or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or n.endswith('/') for n in names)):
        raise ValueError('Unsafe, duplicate or oversized preserved archive')
    return names


def preflight(run):
    if run!=RUN:raise ValueError('Only the exact preserved source run is accepted')
    closure=cache.source_contract()
    try:
        cache.git('cat-file','-e',HEAD+'^{commit}')
    except subprocess.CalledProcessError:
        subprocess.run(['git','fetch','--no-tags','--depth=1','origin',HEAD],cwd=ROOT,
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=120)
    # The latest component fixture is stronger; it is not inference code. All
    # runtime/checker/build objects must remain byte-identical to the real run.
    for path,digest in cache.NATIVE_PINS.items():
        if path=='tests/test_qwen35_weight_cache.cpp':continue
        raw=subprocess.check_output(['git','show',HEAD+':'+path],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Inference/source checker differs from preserved run')
    return closure


def run_analysis(run):
    closure=preflight(run)
    out=Path(os.environ['HEAVY_OUT']);out.mkdir(parents=True,exist_ok=True)
    scratch=Path(os.environ['HEAVY_SCRATCH'])/'reanalysis';scratch.mkdir(parents=True,exist_ok=True)
    report={'schema':'crispembed-ovisocr2-weight-cache-reanalysis-v1','status':'started',
            'analysis_source':closure,'source_run':RUN,'source_head':HEAD,'no_inference_or_conversion_rerun':True,
            'limits':['Original producer remains rejected and unchanged; this is a distinct derived analysis.',
                      'Only preserved actual logs/vectors and frozen original943-stage metadata are consumed.',
                      'Timing is three same-host pairs on one image. No default/release/model publication change.']}
    def save():
        (out/'ovisocr2-weight-cache-reanalysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        if shutil.disk_usage(scratch).free<18*1024**3:raise ValueError('At least18GiB hosted scratch required')
        report['service_reports']=cache.ab.service_source(RUN,HEAD,REPORT,conclusion='failure')
        report['service_raw']=cache.ab.service_source(RUN,HEAD,RAW,conclusion='failure')
        archive=scratch/'reports.zip';cache.ab.download_artifact(REPORT,archive)
        with zipfile.ZipFile(archive) as z:
            names=checked_members(z,maximum=64*1024**2)
            if len(names)!=75:raise ValueError('Original small archive member count differs')
            folder=scratch/'reports';z.extractall(folder)
            source_bytes=z.read('ovisocr2-weight-cache.json')
            producer=json.loads(source_bytes)
            report['small_members']={name:{'bytes':z.getinfo(name).file_size,
                'sha256':hashlib.sha256(z.read(name)).hexdigest()} for name in names}
        report['source_report_sha256']=hashlib.sha256(source_bytes).hexdigest()
        report['source_producer_canonical_sha256']=canonical(producer)
        commands=require_source_producer(producer)
        report['original_producer']=producer  # Never overwrite its rejected state.
        report['command_logs']={}
        for name,row in commands.items():
            log=folder/row['log']
            if log.stat().st_size!=row['bytes'] or cache.audit.sha(log)!=row['sha256']:
                raise ValueError('Original command log bytes/SHA differ')
            resources,recovered=cache.large.parse_gnu_time_resources((folder/(name+'-resources.json')).read_text(),0)
            if resources!=row['resources'] or recovered:raise ValueError('Original actual process resources differ')
            report['command_logs'][name]={'bytes':row['bytes'],'sha256':row['sha256'],'resources':resources}
        # Extract just original JSON metadata from the checked immutable reference
        # archive; do not load a framework, weight GGUF or model graph.
        pin=cache.ab.SOURCE_REFERENCE_ARTIFACT
        report['original_reference_service']=cache.ab.service_source(cache.ab.SOURCE_RUN,cache.ab.SOURCE_HEAD,pin,conclusion='failure')
        reference_zip=scratch/'reference.zip';cache.ab.download_artifact(pin,reference_zip)
        with zipfile.ZipFile(reference_zip) as z:
            name,count,digest=cache.ab.SOURCE_REFERENCE_JSON
            cache.ab.extract_member(z,name,scratch/name,count,digest)
        reference_zip.unlink()
        reference=json.loads((scratch/name).read_text())
        if (len(reference.get('stages',{}))!=943 or reference.get('generated_ids')!=[cache.replay.FROZEN_IDS]
                or reference.get('decoded')!=[cache.replay.FROZEN_TEXT] or reference.get('missing_stages')!=[]):
            raise ValueError('Frozen original reference metadata differs')
        report['original_reference_json']={'bytes':count,'sha256':digest,'stage_count':943}
        raw_zip=scratch/'raw.zip';cache.ab.download_artifact(RAW,raw_zip)
        raw_root=scratch/'native-diff';raw_root.mkdir()
        with zipfile.ZipFile(raw_zip) as z:
            names=checked_members(z,maximum=16*1024**3)
            expected_folders={f'cache-{on}-{arm}' for on in (0,1) for arm in ARMS if not arm.endswith('-context')}
            if len(names)!=3324 or {PurePosixPath(n).parts[0] for n in names}!=expected_folders:
                raise ValueError('Preserved actual raw file coverage differs')
            z.extractall(raw_root)
        raw_zip.unlink()
        report['arms']={};all_records=[]
        for on in (0,1):
            report['arms'][str(on)]={}
            for arm in sorted(ARMS):
                label=f'cache-{on}-{arm}';text=(folder/(label+'.log')).read_text()
                raw=cache.normalize_manifest(cache.large.raw_manifest(raw_root/label,reference,text,arm),label)
                proof=cache.replay.verify_arm_log(arm,text,reference['stages'],cache.ab)
                if '-decoder-' in arm:
                    match=re.search(r'^WEIGHT_CACHE bytes=(\d+) budget=(\d+) entries=(\d+) hits=(\d+)$',text,re.M)
                    if not match:raise ValueError('Actual decoder cache counters absent')
                    used,budget,entries,hits=map(int,match.groups())
                    if budget!=cache.BUDGET or used>budget or (on and (not used or not entries or not hits)) or (not on and (used or entries or hits)):
                        raise ValueError('Actual decoder cache budget/admission differs')
                    proof['cache']={'bytes':used,'budget':budget,'entries':entries,'hits':hits}
                report['arms'][str(on)][arm]={'passed':True,'raw':raw,'checks':proof}
                all_records.extend(raw.get('files',[]));save()
        if len(all_records)!=3324 or len({row['file'] for row in all_records})!=3324:
            raise ValueError('Rebuilt raw manifest coverage differs from actual archive')
        for arm in ARMS:
            def identity(on):
                return sorted((r['stage'],r.get('variant','primary'),r['bytes'],r['sha256'])
                              for r in report['arms'][str(on)][arm]['raw'].get('files',[]))
            if identity(0)!=identity(1):raise ValueError('Actual on/off raw bytes differ: '+arm)
        answers=[];bench=[]
        for pair in range(3):
            for on in ((0,1) if pair%2==0 else (1,0)):
                label=f'pair-{pair}-cache-{on}'
                rows=cache.benchmark_rows((folder/(label+'.log')).read_text())
                original=next(b for b in producer['benchmarks'] if b['pair']==pair and b['cache']==bool(on))
                if rows!=original['rows'] or original['resources']!=commands[label]['resources']:
                    raise ValueError('Benchmark log/results differ')
                answers.extend(rows);bench.append(original)
        two={}
        for on in (0,1):
            label=f'two-contexts-cache-{on}'
            rows=cache.benchmark_rows((folder/(label+'.log')).read_text(),True);answers.extend(rows)
            if rows!=producer['two_contexts'][str(on)]['rows']:raise ValueError('Two-context results differ')
            two[str(on)]={'rows':rows,'resources':commands[label]['resources']}
        cache.require_exact(answers)
        report['benchmarks']=bench;report['two_contexts']=two;report['paired_summary']={}
        for phase in ('cold','warm'):
            values={on:[next(row['recognize_seconds'] for row in b['rows'] if row['label']==phase)
                        for b in bench if b['cache']==bool(on)] for on in (0,1)}
            ratios=[a/b for a,b in zip(values[1],values[0])]
            report['paired_summary'][phase]={'off_seconds':values[0],'on_seconds':values[1],
                'paired_on_over_off':ratios,'median_paired_on_over_off':statistics.median(ratios)}
        if report['paired_summary']!=producer['paired_summary']:raise ValueError('Paired summary recomputation differs')
        report.update(status='completed preserved cache raw/log reanalysis',raw_files_rehashed=3324,
            fourteen_arms_per_gate_positive=True,on_off_all_raw_bytes_exact=True,
            all_twenty_benchmark_outputs_exact=True,original_failure_preserved=True)
        save()
    except Exception as error:
        report.update(status='rejected preserved cache reanalysis',error=str(error));save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',type=int,required=True)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:print(json.dumps(preflight(args.source_run),sort_keys=True))
    else:run_analysis(args.source_run)
