#!/usr/bin/env python3
"""Hash real candidate archives and every member; never infer runtime quality."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import subprocess
import tarfile
import zipfile

NATIVE_REQUIRED={
 'crispembed-linux-x86_64.tar.gz','crispembed-linux-arm64.tar.gz','crispembed-macos-arm64.tar.gz',
 'crispembed-windows-x86_64.zip','crispembed-android-arm64-v8a.tar.gz',
 'crispembed-android-armeabi-v7a.tar.gz','crispembed-ios-arm64.tar.gz'}
NATIVE_OPTIONAL={'crispembed-linux-x86_64-cuda-bundled.tar.gz','crispembed-linux-x86_64-cuda.tar.gz','crispembed-windows-x86_64-cuda.zip','crispembed-windows-x86_64-vulkan.zip'}
WASM_REQUIRED={'crispembed-ocr-wasm.tar.gz','crispembed-embed-wasm.tar.gz'}

def digest(stream):
 h=hashlib.sha256()
 for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def sha(path):
 with Path(path).open('rb') as stream:return digest(stream)

def safe_name(name):
 if ':' in name or '\\' in name or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
  raise ValueError('Unsafe archive member')
 value=posixpath.normpath(name)
 if value in ('','.'):return None
 return value

def archive_members(path):
 rows=[];seen=set()
 def add(name,size,stream=None,target=None,kind='file'):
  name=safe_name(name)
  if name is None:return
  if name in seen:raise ValueError('Duplicate normalized member')
  seen.add(name)
  if not 0<=size<=1024**3:raise ValueError('Oversize archive member')
  row={'path':name,'kind':kind,'bytes':size}
  if stream is not None:row['sha256']=digest(stream)
  if target is not None:
   if kind=='hardlink':resolved=safe_name(target)
   else:resolved=safe_name(posixpath.join(posixpath.dirname(name),target))
   if not resolved:raise ValueError('Invalid archive link')
   row.update(target=target,resolved_target=resolved)
  rows.append(row)
 if zipfile.is_zipfile(path):
  with zipfile.ZipFile(path) as archive:
   for info in archive.infolist():
    if info.is_dir():continue
    mode=info.external_attr>>16
    if mode&0o170000==0o120000:raise ValueError('ZIP symlinks unsupported')
    with archive.open(info) as stream:add(info.filename,info.file_size,stream)
 else:
  with tarfile.open(path,'r:gz') as archive:
   for info in archive:
    if info.isdir():continue
    if info.isfile():
     with archive.extractfile(info) as stream:add(info.name,info.size,stream)
    elif info.issym() or info.islnk():add(info.name,0,target=info.linkname,kind='symlink' if info.issym() else 'hardlink')
    else:raise ValueError('Unsupported archive member type')
 by_name={r['path']:r for r in rows}
 for row in rows:
  chain=set();current=row
  while 'resolved_target' in current:
   if current['path'] in chain:raise ValueError('Archive link cycle')
   chain.add(current['path'])
   if current['resolved_target'] not in by_name:raise ValueError('Dangling archive link')
   current=by_name[current['resolved_target']]
 if not rows or sum(r['bytes'] for r in rows)>4*1024**3:raise ValueError('Empty/oversize archive')
 return sorted(rows,key=lambda r:r['path'])

def build(root,kind,source_root):
 root=Path(root);source_root=Path(source_root)
 required=NATIVE_REQUIRED if kind=='native' else WASM_REQUIRED
 allowed=required|(NATIVE_OPTIONAL if kind=='native' else set())
 files=sorted(p for p in root.rglob('*') if p.is_file() and (p.name.endswith('.tar.gz') or p.name.endswith('.zip')))
 if len({p.name for p in files})!=len(files):raise ValueError('Duplicate archive basename')
 names={p.name for p in files}
 if not required<=names or names-allowed:raise ValueError('Candidate archive coverage differs: '+repr(sorted(required-names)))
 def git(*args):return subprocess.check_output(['git','-C',str(source_root),*args],text=True).strip()
 return {'schema':'crispembed-successor-archive-manifest-v1','kind':kind,'source_commit':git('rev-parse','HEAD'),
  'ggml_gitlink':git('rev-parse','HEAD:ggml'),'version':(source_root/'VERSION').read_text().strip(),
  'generator_sha256':sha(__file__),'required_archives':sorted(required),'optional_archives_present':sorted(names-required),
  'runtime_or_model_qualification':False,'application_pin_changed':False,
  'files':[{'path':str(p.relative_to(root)),'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'members':archive_members(p)} for p in files]}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--kind',choices=('native','wasm'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.write_text(json.dumps(build(a.root,a.kind,a.source_root),indent=2,allow_nan=False)+'\n')
