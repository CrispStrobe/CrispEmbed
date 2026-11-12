#!/usr/bin/env python3
"""Fresh hosted consumers of actual source-bound release archives."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
# A clean tooling checkout is part of desktop provenance; importing its generator
# must not create an ignored bytecode file before that check.
_bytecode_setting=sys.dont_write_bytecode
sys.dont_write_bytecode=True
from successor_asset_manifest import sha, archive_members, safe_name, DESKTOP_REQUIRED, NATIVE_OPTIONAL
sys.dont_write_bytecode=_bytecode_setting
REPO='CrispStrobe/CrispEmbed'
MODEL_ALIAS='all-MiniLM-L6-v2'
MODEL_SHA='77087a5dd455f5eeae6d6d098bac82793610438518a91071c72c253940db18a2'
TEXT='A small reproducible embedding consumer.'

def api(path):return json.loads(subprocess.check_output(['gh','api','repos/'+REPO+'/'+path],text=True,encoding='utf-8',timeout=90))
def artifact(run,name,destination,max_bytes):
 listing=api('actions/runs/'+str(run)+'/artifacts?per_page=100')
 found=[a for a in listing['artifacts'] if a['name']==name and not a['expired']]
 if listing['total_count']>100 or len(found)!=1:raise ValueError('Missing/ambiguous artifact: '+name)
 item=found[0]
 if not 0<item['size_in_bytes']<=max_bytes:raise ValueError('Oversize artifact')
 with destination.open('xb') as stream:
  subprocess.run(['gh','api','repos/'+REPO+'/actions/artifacts/'+str(item['id'])+'/zip'],stdout=stream,check=True,timeout=1200)
 if destination.stat().st_size!=item['size_in_bytes'] or item['digest']!='sha256:'+sha(destination):raise ValueError('Artifact ZIP identity differs')
 return {k:item[k] for k in ('id','name','size_in_bytes','digest')}

# Audit-only changes may differ from the immutable build revision. Every
# production/build/binding object must still be identical; no runtime exception.
RUNTIME_PATHS=('src','ggml','cmake','CMakeLists.txt','VERSION','Cargo.lock','examples',
 'python','flutter','crispembed','crispembed-sys','wasm','tools/quantize.cpp',
 'build-wasm.sh','build-embed-wasm.sh','scripts/check-cpu-baseline.py',
 'scripts/check-bundled-deps.py','tools/successor_asset_manifest.py','.gitattributes','.github/workflows/release.yml',
 '.github/workflows/release-wasm.yml','README.md','LICENSE')

def runtime_closure(producer_source):
 def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding='utf-8',timeout=180).strip()
 for source in (producer_source,):
  if not re.fullmatch('[0-9a-f]{40}',source):raise ValueError('Immutable source commit required')
  check=subprocess.run(['git','cat-file','-e',source+'^{commit}'],cwd=ROOT,stderr=subprocess.DEVNULL)
  if check.returncode:subprocess.run(['git','fetch','--depth=1','origin',source],cwd=ROOT,check=True,timeout=180)
 paths=RUNTIME_PATHS
 before=git('ls-tree',producer_source,'--',*paths).splitlines()
 after=git('ls-tree','HEAD','--',*paths).splitlines()
 if before!=after or git('status','--porcelain','--',*paths):
  raise ValueError('Consumer production/build/binding source differs from immutable producer')
 proof={'scope':'Exact production/build/binding Git objects; audit helpers may differ',
        'objects':before,'consumer_source':git('rev-parse','HEAD')}
 return proof

def desktop_source_closure(runtime_source,workflow_source,runtime_root):
 if runtime_root is None:raise ValueError('Separate exact desktop runtime checkout required')
 runtime_root=Path(runtime_root).resolve()
 if runtime_root==ROOT.resolve():raise ValueError('Desktop runtime and workflow checkouts must be separate')
 proof={'scope':'Exact complete runtime and workflow Git trees, including submodules; no path allowlist'}
 for label,root,expected in [('runtime',runtime_root,runtime_source),('workflow',ROOT,workflow_source)]:
  def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,encoding='utf-8',timeout=180).strip()
  if not re.fullmatch('[0-9a-f]{40}',expected) or git('rev-parse','HEAD')!=expected:
   raise ValueError('Desktop '+label+' checkout identity differs')
  # Include ignored files: a fresh source checkout must not carry generated or
  # injected inputs outside the complete committed tree. Outputs belong outside.
  if git('status','--porcelain','--untracked-files=all','--ignored=matching','--ignore-submodules=none'):
   raise ValueError('Desktop '+label+' complete source tree must be clean')
  submodules=git('submodule','status','--recursive').splitlines()
  if any(line and line[0] in '-+U' for line in submodules):
   raise ValueError('Desktop '+label+' submodule identity differs or is uninitialized')
  if git('submodule','foreach','--quiet','--recursive','git status --porcelain --untracked-files=all --ignored=matching --ignore-submodules=none'):
   raise ValueError('Desktop '+label+' submodule source tree must be clean')
  proof[label+'_source_commit']=expected
  proof[label+'_tree']=git('rev-parse','HEAD^{tree}')
  proof[label+'_submodules']=submodules
 proof['consumer_source']=workflow_source
 return proof

def prepare(run,kind,artifact_name,out,desktop_candidate=False,runtime_source_commit=None,runtime_source_root=None):
 if os.environ.get('GITHUB_ACTIONS')!='true':raise ValueError('Hosted only')
 if desktop_candidate and not re.fullmatch('[0-9a-f]{40}',runtime_source_commit or ''):
  raise ValueError('Explicit immutable expected desktop runtime source required')
 consumer_source=os.environ['GITHUB_SHA'];service=api('actions/runs/'+str(run));source=service['head_sha']
 path='.github/workflows/release.yml' if kind=='native' else '.github/workflows/release-wasm.yml'
 if (not re.fullmatch('[0-9a-f]{40}',source) or service['path']!=path
     or service['status']!='completed' or service['conclusion']!='success' or service['event']!='workflow_dispatch'):
  raise ValueError('Successful exact-source dry-run producer required')
 out.mkdir(parents=True,exist_ok=False)
 identity=artifact(run,'successor-'+kind+'-manifest',out/'manifest.zip',4*1024**2)
 with zipfile.ZipFile(out/'manifest.zip') as z:
  if z.namelist()!=['successor-'+kind+'-manifest.json']:raise ValueError('Manifest artifact members differ')
  manifest=json.loads(z.read(z.namelist()[0]))
 workflow_source=source
 if desktop_candidate:
  if kind!='native' or manifest.get('scope')!='desktop' or manifest.get('workflow_source_commit')!=workflow_source:
   raise ValueError('Explicit desktop workflow identity required')
  source=manifest.get('source_commit','')
  if source!=runtime_source_commit:raise ValueError('Desktop runtime source differs from expected immutable source')
  names=[row['name'] for row in manifest['files']]
  if (set(manifest['required_archives'])!=DESKTOP_REQUIRED or len(names)!=len(set(names))
      or not DESKTOP_REQUIRED<=set(names) or set(names)-(DESKTOP_REQUIRED|NATIVE_OPTIONAL)
      or set(manifest['optional_archives_present'])!=set(names)-DESKTOP_REQUIRED):
   raise ValueError('Desktop manifest inventory differs')
  closure=desktop_source_closure(source,workflow_source,runtime_source_root)
 else:
  if manifest.get('scope','full')!='full' or 'workflow_source_commit' in manifest:
   raise ValueError('Desktop manifest requires explicit desktop-candidate validation')
  closure=runtime_closure(source)
 if closure['consumer_source']!=consumer_source:raise ValueError('Consumer checkout/source identity differs')
 generator_blob=subprocess.check_output(['git','show',workflow_source+':tools/successor_asset_manifest.py'],cwd=ROOT)
 generator_source_sha=hashlib.sha256(generator_blob).hexdigest()
 source_root=Path(runtime_source_root) if desktop_candidate else ROOT
 expected_ggml=subprocess.check_output(['git','rev-parse','HEAD:ggml'],cwd=source_root,text=True,encoding='utf-8').strip()
 if (manifest.get('schema')!='crispembed-successor-archive-manifest-v1' or manifest['source_commit']!=source
     or manifest['ggml_gitlink']!=expected_ggml or manifest['generator_sha256']!=generator_source_sha
     or manifest['kind']!=kind or manifest['version']!=(source_root/'VERSION').read_text(encoding='utf-8').strip()):
  raise ValueError('Manifest/source closure differs')
 archive_identity=artifact(run,artifact_name,out/'payload.zip',2*1024**3)
 selected=[]
 with zipfile.ZipFile(out/'payload.zip') as z:
  names=z.namelist()
  if len(names)!=len(set(names)):raise ValueError('Duplicate payload member')
  for info in z.infolist():
   name=safe_name(info.filename)
   if info.is_dir():continue
   rows=[r for r in manifest['files'] if r['name']==name]
   if len(rows)!=1 or info.file_size!=rows[0]['bytes']:raise ValueError('Archive not bound by manifest')
   target=out/name
   with z.open(info) as src,target.open('xb') as dst:
    import shutil;shutil.copyfileobj(src,dst)
   row=rows[0]
   if sha(target)!=row['sha256'] or archive_members(target)!=row['members']:raise ValueError('Archive/member hashes differ')
   destination=out if kind=='wasm' else out/name.removesuffix('.tar.gz').removesuffix('.zip')
   if kind!='wasm':destination.mkdir()
   if name.endswith('.zip'):
    with zipfile.ZipFile(target) as package:package.extractall(destination)
   else:
    with tarfile.open(target,'r:gz') as package:package.extractall(destination,filter='data')
   selected.append({'name':name,'sha256':row['sha256'],'member_count':len(row['members'])})
 receipt={'schema':'crispembed-successor-fresh-consumer-v2','source_commit':source,'consumer_source':consumer_source,'runtime_source_closure':closure,'ggml_gitlink':expected_ggml,
          'generator_git_blob_sha256':generator_source_sha,'consumer_generator_file_sha256':sha(ROOT/'tools/successor_asset_manifest.py'),
          'producer_run':run,'manifest_artifact':identity,'payload_artifact':archive_identity,'archives':selected,
          'qualification':False,'limits':['Archive validation alone is not runtime/model quality qualification.']}
 if desktop_candidate:receipt.update(scope='desktop',workflow_source_commit=workflow_source)
 (out/'consumer.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
 return receipt

def compare(a,b):
 if len(a)!=384 or len(b)!=384 or not all(math.isfinite(v) for v in a+b):raise ValueError('Embedding shape/finite check failed')
 error=max(abs(x-y) for x,y in zip(a,b));aa=sum(x*x for x in a);bb=sum(y*y for y in b)
 if aa<=0 or bb<=0:raise ValueError('Zero embedding')
 cosine=sum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb)
 result={'max_abs':error,'cosine':cosine,'norms':[math.sqrt(aa),math.sqrt(bb)],'passed':error<=.002 and cosine>=.99999}
 if not result['passed']:raise ValueError('Packaged consumer vector parity failed: '+repr(result))
 return result

def header_exports(header):
 # The shipped C header is UTF-8 source, independent of the consumer locale.
 names=re.findall(r'CRISPEMBED_API\s+[^;]*?\b(crispembed_\w+)\s*\(',Path(header).read_text(encoding='utf-8'))
 if not names:raise ValueError('No ABI exports inspected')
 return names

def c_consumer_command(package,source,executable,windows):
 if windows:
  return ['cl','/nologo','/utf-8','/DCRISPEMBED_SHARED','/I'+str(package/'include'),str(source),'/Fe:'+str(executable),'/Fo'+str(executable.with_suffix('.obj')),'/link',str(package/'crispembed.lib')]
 return ['cc',str(source),'-I'+str(package/'include'),'-L'+str(package),'-lcrispembed','-Wl,-rpath,'+str(package.resolve()),'-o',str(executable)]

def desktop(out,runtime_source_root=None):
 receipt=json.loads((out/'consumer.json').read_text(encoding='utf-8'))
 source_root=ROOT
 if receipt.get('scope')=='desktop':
  desktop_source_closure(receipt['source_commit'],receipt['workflow_source_commit'],runtime_source_root)
  source_root=Path(runtime_source_root)
 binary=list(out.glob('crispembed-*/crispembed.exe' if platform.system()=='Windows' else 'crispembed-*/crispembed'))
 if len(binary)!=1:raise ValueError('Expected one desktop package')
 binary=binary[0];package=binary.parent
 library=package/({'Windows':'crispembed.dll','Darwin':'libcrispembed.dylib'}.get(platform.system(),'libcrispembed.so'))
 cache=out/'model-cache'
 if cache.exists():raise ValueError('Fresh cache required')
 os.environ['CRISPEMBED_CACHE_DIR']=str(cache.resolve())
 os.environ['CRISPEMBED_THREADS']='2'
 if os.name!='nt':
  for file in package.glob('crispembed*'):
   if file.is_file():file.chmod(0o700)
 else:os.add_dll_directory(str(package.resolve()))
 # Published model alias through the actual packaged CLI: no manual model fetch.
 def cli(offline):
  cmd=[str(binary.resolve()),'-m',MODEL_ALIAS,'-t','2','--json']
  if offline:cmd.append('--offline')
  cmd.append(TEXT)
  result=subprocess.run(cmd,text=True,encoding='utf-8',capture_output=True,timeout=600)
  (out/('cli-offline.log' if offline else 'cli-fresh.log')).write_text(result.stderr,encoding='utf-8')
  if result.returncode:raise ValueError('Packaged CLI failed: '+str(result.returncode))
  rows=json.loads(result.stdout)
  if len(rows)!=1 or rows[0]['text']!=TEXT:raise ValueError('CLI response invalid')
  return rows[0]['embedding']
 missing_env=dict(os.environ,CRISPEMBED_CACHE_DIR=str((out/'offline-empty-cache').resolve()))
 missing=subprocess.run([str(binary.resolve()),'-m',MODEL_ALIAS,'--offline','--json',TEXT],env=missing_env,text=True,encoding='utf-8',capture_output=True,timeout=60)
 (out/'cli-empty-offline.log').write_text(missing.stderr,encoding='utf-8')
 if missing.returncode==0 or 'is not cached and offline mode is on' not in missing.stderr or list((out/'offline-empty-cache').rglob('*.gguf')):
  raise ValueError('Empty-cache offline request did not reject')
 first=cli(False);models=list(cache.rglob('all-MiniLM-L6-v2-iq4_xs.gguf'))
 if len(models)!=1 or sha(models[0])!=MODEL_SHA:raise ValueError('Fresh published model hash differs')
 # Persist only neutral public filenames in the report.
 header=package/'include/crispembed.h'
 if not header.is_file() or sha(header)!=sha(source_root/'src/crispembed.h'):
  raise ValueError('Packaged public header missing or stale')
 exports=header_exports(header);lib=C.CDLL(str(library.resolve()))
 for name in exports:getattr(lib,name)
 lib.crispembed_init.argtypes=[C.c_char_p,C.c_int];lib.crispembed_init.restype=C.c_void_p
 lib.crispembed_free.argtypes=[C.c_void_p];lib.crispembed_free.restype=None
 lib.crispembed_encode.argtypes=[C.c_void_p,C.c_char_p,C.POINTER(C.c_int)];lib.crispembed_encode.restype=C.POINTER(C.c_float)
 lib.crispembed_set_offline.argtypes=[C.c_int];lib.crispembed_set_offline.restype=None;lib.crispembed_set_offline(1)
 ctx=lib.crispembed_init(os.fsencode(models[0]),2)
 if not ctx:raise ValueError('Packaged C ABI initialization failed')
 try:
  n=C.c_int();ptr=lib.crispembed_encode(ctx,TEXT.encode(),C.byref(n))
  if not ptr or n.value!=384:raise ValueError('C ABI vector absent')
  native=[float(ptr[i]) for i in range(n.value)]
 finally:lib.crispembed_free(ctx)
 sys.path.insert(0,str(source_root/'python'))
 from crispembed import CrispEmbed
 from crispembed._binding import _CrispEmbedHparams
 executable=out/('capi-consumer.exe' if os.name=='nt' else 'capi-consumer')
 compile_command=c_consumer_command(package,source_root/'tests/successor_capi_consumer.c',executable,os.name=='nt')
 subprocess.run(compile_command,check=True,timeout=120)
 actual_env=dict(os.environ,PATH=str(package.resolve())+os.pathsep+os.environ.get('PATH',''))
 compiled=json.loads(subprocess.check_output([str(executable),str(models[0])],env=actual_env,text=True,encoding='utf-8',timeout=600))
 expected_offsets=[getattr(_CrispEmbedHparams,name).offset for name,_ in _CrispEmbedHparams._fields_]
 if compiled['prefix_offsets']!=expected_offsets or compiled['hparams_size']<C.sizeof(_CrispEmbedHparams):
  raise ValueError('Compiled public header/binding prefix ABI layout differs')
 wrapper=CrispEmbed(MODEL_ALIAS,n_threads=2,lib_path=str(library.resolve()),offline=True)
 try:python_vector=wrapper.encode(TEXT).tolist()
 finally:del wrapper
 offline=cli(True)
 receipt.update(abi_exports=sorted(set(exports)),model_sha256=MODEL_SHA,text=TEXT,vector=native,
  comparisons={'cli_to_cabi':compare(first,native),'offline_cli':compare(first,offline),'python_to_cabi':compare(python_vector,native),'compiled_C_to_cabi':compare(compiled['vector'],native)},
  empty_cache_download=True,empty_cache_offline_rejected=True,offline_reuse=True,desktop_bounded_consumer_positive=True,
  library_sha256=sha(library),cli_binary_sha256=sha(binary),compiled_c_consumer_sha256=sha(executable),
  public_header_sha256=sha(header),compiled_hparams_size=compiled['hparams_size'],python_prefix_offsets=expected_offsets,
  limits=['One qualified published MiniLM model; no broad task-quality, GPU or experimental-model qualification.'])
 (out/'consumer.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=int);p.add_argument('--kind',choices=('native','wasm'))
 p.add_argument('--artifact');p.add_argument('--output',type=Path,required=True);p.add_argument('--desktop',action='store_true');p.add_argument('--desktop-candidate',action='store_true',help='Explicit dual workflow/runtime identity validation for desktop-scoped manifests');p.add_argument('--runtime-source-commit',help='Required reviewed immutable runtime source for --desktop-candidate');p.add_argument('--runtime-source-root',type=Path,help='Separate fresh exact runtime checkout; required in both desktop candidate phases');a=p.parse_args()
 if a.desktop:
  desktop(a.output,runtime_source_root=a.runtime_source_root)
  sys.stdout.flush();sys.stderr.flush()
  os._exit(0)  # Test consumer only: avoid ggml process-global GPU teardown.
 else:prepare(a.run,a.kind,a.artifact,a.output,desktop_candidate=a.desktop_candidate,runtime_source_commit=a.runtime_source_commit,runtime_source_root=a.runtime_source_root)
