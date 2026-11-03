"""Hosted decode-only full20 input audit. No model, resize or normalization."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'tools')]
import contract as c
from ci_process import run as child
import metrics


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def resources(path):
    ram=int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024
    disk=shutil.disk_usage(path).free
    c.require(ram>=4*2**30 and disk>=3*2**30 and os.cpu_count()>=2,'need4GiB RAM/3GiB disk/two CPUs before packages/inputs')
    return dict(available_ram_bytes=ram,free_disk_bytes=disk,threads=2)


def hosted():
    gha=os.environ.get('GITHUB_ACTIONS')=='true'
    kaggle=Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()and os.environ.get('KAGGLE_KERNEL_RUN_TYPE')in('Batch','Interactive')
    c.require(gha or kaggle,'hosted image decode only')
    c.require(platform.system()=='Linux'and platform.machine()=='x86_64','declared Linux x64 production baseline required')
    if kaggle and shutil.which('nvidia-smi'):
        gpu=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,timeout=15)
        c.require(gpu.returncode!=0 or not gpu.stdout.strip(),'decode-only audit must not reserve a GPU')


def command(args,log,*,stdout=None,timeout=300):
    env={k:v for k,v in os.environ.items()if k not in('CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','CMAKE_ARGS')}
    result=child(args,log,cwd=ROOT,env=env,timeout=timeout,stdout_path=stdout)
    write(Path(str(log)+'.process.json'),dict(**result,log=log.name,sha256=c.sha(log)))
    return result


def flags(row):
    args=row.get('arguments')or shlex.split(row['command'])
    c.require(not any(a.startswith('@')for a in args),'response-file flags require explicit review')
    selected=[a for a in args if a.startswith(('-O','-D','-f','-m','-std='))]
    c.require(sorted(selected)==sorted(c.PROFILE),'actual production-prefix compilation differs')
    return args


def macros(text):
    out={}
    for line in text.splitlines():
        parts=line.split(maxsplit=2)
        if len(parts)>=2 and parts[0]=='#define':
            name=parts[1]
            if name.startswith(('STBI_','STB_IMAGE'))or name in('__SSE2__','__AVX2__','__FMA__','__F16C__'):
                c.require(name not in out,'duplicate decoder macro');out[name]=parts[2]if len(parts)>2 else''
    return out


def build(scratch,review):
    generated=scratch/'production_stb.cpp';generated.write_bytes(c.prefix((ROOT/'src/image_preprocess.cpp').read_bytes()))
    b=scratch/'build'
    for name,args in [('configure',['cmake','-S',ROOT/'tests/ocr-rgb-input','-B',b,'-DCMAKE_BUILD_TYPE=Release','-DPRODUCTION_PREFIX='+str(generated)]),
                      ('build',['cmake','--build',b,'--parallel','2'])]:
        run=command(args,review/(name+'.log'));c.require(run['exit']==0 and not run['timed_out'],'tiny decoder '+name+' rejected')
    rows=json.loads((b/'compile_commands.json').read_text());selected=[r for r in rows if Path(r['file']).resolve()==generated.resolve()]
    c.require(len(selected)==1,'one exact production-prefix compile record required');argv=flags(selected[0])
    write(review/'compile-commands.json',rows)
    # Compare the same filtered macro state against the unchanged production
    # translation unit under these exact baseline flags. No root build needed.
    common=[a for a in argv[1:]if a.startswith(('-O','-D','-f','-m','-std=','-I'))]
    observed={}
    for name,path in [('prefix',generated),('production',ROOT/'src/image_preprocess.cpp')]:
        output=review/(name+'-macros.txt')
        run=command([argv[0],*common,'-dM','-E',path],review/(name+'-macros.log'),stdout=output)
        c.require(run['exit']==0 and not run['timed_out'],'production decoder macro inspection rejected')
        observed[name]=macros(output.read_text())
    c.require(observed['prefix']==observed['production']and'STBI_SSE2'in observed['prefix']and
              'STB_IMAGE_IMPLEMENTATION'in observed['prefix']and'STBI_NO_SIMD'not in observed['prefix'],'production decoder macro configuration differs')
    binary=b/'ocr-rgb-input';symbols=review/'symbols.txt'
    run=command(['nm','--defined-only',binary],review/'nm.log',stdout=symbols)
    c.require(run['exit']==0 and any(re.search(r'\bT stbi_load$',line)for line in symbols.read_text().splitlines()),'production nonstatic decoder export missing')
    version=review/'compiler.txt';command([argv[0],'--version'],review/'compiler-version.log',stdout=version)
    return binary,dict(production_prefix_sha256=c.sha(generated),binary_sha256=c.sha(binary),
        compile_commands_sha256=c.sha(b/'compile_commands.json'),compile_record=selected[0],
        baseline_flags=list(c.PROFILE),profile_basis='Pinned production CMake portable x86 baseline; actual flags and original-TU macros checked here, not a claim about unretained historical binaries.',
        macros_exact=True,decoder_macros=observed['prefix'],compiler=version.read_text().splitlines()[0])


def fetch(url,path,maximum):
    c.require(url.startswith('https://')and not path.exists(),'fresh bounded HTTPS input required')
    count=0;digest=hashlib.sha256()
    request=urllib.request.Request(url,headers={'User-Agent':'CrispEmbed-input-audit/1.0 (https://github.com/CrispStrobe/CrispEmbed)'})
    try:
        with urllib.request.urlopen(request,timeout=90)as response,path.open('xb')as target:
            c.require(response.status==200,'input transfer status differs')
            while block:=response.read(2**20):
                count+=len(block);c.require(count<=maximum,'input transfer exceeds audit bound');target.write(block);digest.update(block)
    except Exception:raise ValueError('bounded encoded input transfer rejected; task-owned partial retained')from None
    return dict(bytes=count,sha256=digest.hexdigest())


def encoded(page,images_dir,raw,scratch,parquets):
    target=raw/'encoded'/(page['id']+'.encoded');target.parent.mkdir(exist_ok=True)
    if images_dir is not None:
        hits=[images_dir/(page['id']+ext)for ext in('.jpg','.png','.encoded')]
        hits=[p for p in hits if p.is_file()and not p.is_symlink()]
        c.require(len(hits)==1,'one explicit retained encoded page required')
        c.require(hits[0].stat().st_size<=16*2**20,'retained image exceeds bound');shutil.copyfile(hits[0],target)
        return target,dict(provider='retained encoded input',bytes=target.stat().st_size,sha256=c.sha(target))
    pin=page['image']
    if 'hf_parquet'not in pin:return target,dict(provider='frozen public image URL',**fetch(pin['url'],target,16*2**20))
    hp=pin['hf_parquet'];key=hp['sha256']
    if key not in parquets:
        path=scratch/(key+'.parquet');proof=fetch(hp['url'],path,512*2**20)
        c.require(proof['sha256']==key,'frozen CORD parquet SHA differs before parse');parquets[key]=(path,proof)
    path,proof=parquets[key]
    import pyarrow.parquet as pq
    table=pq.ParquetFile(path);c.require(table.metadata.num_rows==100 and hp['column']=='image.bytes','frozen CORD layout differs')
    row=hp['row'];offset=0;data=None
    for group in range(table.num_row_groups):
        n=table.metadata.row_group(group).num_rows
        if offset<=row<offset+n:
            data=table.read_row_group(group,columns=['image']).slice(row-offset,1).to_pylist()[0]['image']['bytes'];break
        offset+=n
    c.require(isinstance(data,bytes)and len(data)<=16*2**20,'frozen CORD image member missing/oversized');target.write_bytes(data)
    return target,dict(provider='frozen CORD parquet image.bytes',parquet=proof,row=row,bytes=len(data),sha256=c.sha(target))


def main(args):
    hosted();cfg=c.source(header=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    c.require(head==args.source and (os.environ.get('GITHUB_ACTIONS')!='true'or os.environ.get('GITHUB_SHA')==head),'actual submitted source differs')
    args.scratch.mkdir(parents=True,exist_ok=False);args.out.mkdir(parents=True,exist_ok=False)
    review=args.out/'review';raw=args.out/'raw';review.mkdir();raw.mkdir()
    report=dict(schema='crispembed-ocr-rgb-input-audit-v1',source=head,completed=False,model_execution=False,model_qualified=False,
                cause_established=False,manifest_sha256=cfg['manifest']['sha256'],pages=20,rows=[],resources=resources(args.scratch))
    save=lambda:write(review/'rgb-input.json',report);save()
    try:
        import PIL
        from PIL import Image,features
        import numpy as np
        c.require(PIL.__version__=='11.3.0'and np.__version__=='2.2.6'and importlib.metadata.version('pyarrow')=='17.0.0','pinned codec/metric/input packages required')
        c.require(sys.version_info[:2]==(3,12)and sys.prefix!=sys.base_prefix,'isolated CPython3.12 required')
        report['package_owners']={}
        for name in('Pillow','numpy','pyarrow'):
            owner=importlib.metadata.distribution(name);path=Path(owner._path)/'METADATA'
            c.require(path.resolve().is_relative_to(Path(sys.prefix).resolve()),'package owner escaped audit venv')
            report['package_owners'][name]=dict(version=owner.version,metadata_sha256=c.sha(path),inside_venv=True)
        c.require(all(Path(p).resolve().is_relative_to(Path(sys.prefix).resolve())for p in(PIL.__file__,np.__file__)),'loaded codec/metric origin escaped venv')
        report['codec']=dict(pillow=PIL.__version__,jpeg=features.version_codec('jpg'),
            libjpeg_turbo=features.version_feature('libjpeg_turbo')if features.check_feature('libjpeg_turbo')else None,
            zlib=features.version_codec('zlib'),policy="Image.open(path).convert('RGB'); no EXIF transpose, ICC transform, resize or normalization",
            historical_binary_identity='Pillow11.3 version matches reference setup; historical wheel/libjpeg build was not pinned, so binary identity is not asserted.')
        c.require(Path(Image.core.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'loaded Pillow extension escaped venv')
        report['codec']['imaging_extension_sha256']=c.sha(Image.core.__file__)
        control=args.scratch/'metric-control';control.mkdir()
        left=bytes([0,1,255,5,6,7]);right=bytes([1,1,254,5,8,7])
        (control/'a').write_bytes(left);(control/'b').write_bytes(right)
        c.require(metrics.files(control/'a',control/'b',2,1)==metrics.small(left,right,2),'independent RGB metric control differs')
        report['metric_control']=True
        probe,report['build']=build(args.scratch,review)
        manifest=json.loads((ROOT/'docs/ocr-real-scans-v1.json').read_text());parquets={}
        credits=[{k:page[k]for k in('id','source','image','image_license','ground_truth_license','ground_truth_provenance')}for page in manifest['pages']]
        write(raw/'credits.json',credits)
        for page in manifest['pages']:
            ident=page['id'];row=dict(id=ident,manifest_dimensions=[page['image']['width'],page['image']['height']],
                expected_encoded=dict(bytes=page['image']['bytes'],sha256=page['image']['sha256']),input_admitted=False,
                pillow_decode_attempted=False,stb_decode_attempted=False,eligible_for_decode_comparison=False)
            try:
                path,transfer=encoded(page,args.images_dir,raw,args.scratch,parquets);row['encoded']=transfer
                # Both decoders are strictly downstream of this exact-byte gate.
                def decode(_admitted_bytes):
                    row['input_admitted']=True;path.chmod(0o444)
                    pillow=raw/(ident+'-pillow.rgb');stb=raw/(ident+'-stb.rgb');meta=raw/(ident+'-stb.json')
                    row['pillow_decode_attempted']=True
                    try:
                        with Image.open(path)as original:
                            row['pillow_source']=dict(format=original.format,mode=original.mode)
                            rgb=original.convert('RGB');dimensions=list(rgb.size);pillow.write_bytes(rgb.tobytes())
                        row['pillow']=dict(completed=True,dimensions=dimensions,bytes=pillow.stat().st_size,sha256=c.sha(pillow),channels=3)
                    except Exception as error:row['pillow']=dict(completed=False,error=type(error).__name__+': '+str(error))
                    # A Pillow failure must not hide the independent STB result.
                    row['stb_decode_attempted']=True
                    try:
                        run=command([probe,path,stb,meta],review/(ident+'-stb.log'),timeout=60)
                        c.require(meta.is_file(),'STB decoder metadata missing');native=json.loads(meta.read_text());row['stb_process']=run;row['stb']=native
                        c.require(run['exit']==0 and not run['timed_out']and native['completed'],'STB decode rejected')
                        native.update(dimensions=[native['width'],native['height']],sha256=c.sha(stb))
                        c.require(stb.stat().st_size==native['bytes']==native['width']*native['height']*3,'STB raw extents differ')
                    except Exception as error:row['stb']=dict(completed=False,error=type(error).__name__+': '+str(error))
                    c.admit(path.read_bytes(),page['image'])
                    c.require(row['pillow']['completed']and row['stb']['completed'],'one or both independent decoders rejected')
                    dimensions=row['pillow']['dimensions'];native=row['stb']
                    row['manifest_dimensions_match']=dict(pillow=dimensions==row['manifest_dimensions'],stb=native['dimensions']==row['manifest_dimensions'])
                    c.require(dimensions==native['dimensions'],'Pillow/STB decoded geometry differs')
                    row['metrics']=metrics.files(pillow,stb,*dimensions);row['eligible_for_decode_comparison']=True
                    row['raw_rgb']=dict(pillow=str(pillow.relative_to(args.out)),stb=str(stb.relative_to(args.out)))
                c.admitted_decode(path.read_bytes(),page['image'],decode)
            except Exception as error:row.update(error=type(error).__name__+': '+str(error),excluded_before_decode=not row['input_admitted'])
            report['rows'].append(row);save()
        c.require([r['id']for r in report['rows']]==[p['id']for p in manifest['pages']],'all20 input identities required')
        report.update(completed=True,admitted_pages=sum(r['input_admitted']for r in report['rows']),
            compared_pages=sum(r['eligible_for_decode_comparison']for r in report['rows']),
            rgb_exact_pages=sum(r.get('metrics',{}).get('rgb_exact',False)for r in report['rows']),
            credits_sha256=c.sha(raw/'credits.json'),interpretation='Decode differences establish only an input boundary. No model-token mismatch cause or quality qualification is inferred.')
        write(review/'raw-manifest.json',{str(p.relative_to(args.out)):dict(bytes=p.stat().st_size,sha256=c.sha(p))for p in raw.rglob('*')if p.is_file()})
        save();return 0 if report['compared_pages']==20 else 1
    except Exception as error:report['error']=type(error).__name__+': '+str(error);save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--images-dir',type=Path)
    p.add_argument('--scratch',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
