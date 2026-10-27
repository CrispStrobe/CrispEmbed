#!/usr/bin/env python3
"""Fresh-Python bundled-library smoke only; no model, references or forward calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import sysconfig

SYMBOLS=('crispembed_init','crispembed_free','crispembed_encode','crispembed_encode_tokens_raw','crispembed_last_token_ids')
MODES=('bundle-only','bundle-and-python-lib')


def require(value,why):
    if not value:raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while block:=stream.read(1024**2):h.update(block)
    return h.hexdigest()


def mapped_libraries():
    return {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
            if '/libggml' in line or '/libcrispembed' in line}


def python_library():
    """Only the interpreter's own base-prefix lib, bound to its actual mapped ELF."""
    directory=(Path(sys.base_prefix).resolve()/'lib').resolve()
    name=sysconfig.get_config_var('INSTSONAME') or sysconfig.get_config_var('LDLIBRARY')
    require(name and Path(name).name==name and name.startswith('libpython'),'No derived Python runtime library')
    require(Path(sysconfig.get_config_var('LIBDIR')).resolve()==directory,'Python LIBDIR is outside base_prefix/lib')
    library=(directory/name).resolve()
    require(directory.is_dir() and library.is_file() and library.parent==directory,'Derived Python runtime library missing/outside directory')
    mapped={Path(line.split()[-1]).resolve()for line in Path('/proc/self/maps').read_text().splitlines() if '/libpython' in line}
    require(library in mapped,'Derived Python runtime is not the actual interpreter library')
    return dict(directory=str(directory),library=library.name,bytes=library.stat().st_size,sha256=sha(library),version=sys.version)


def fingerprint(runtime):
    return {k:v for k,v in runtime.items()if k!='directory'}


def loader_path(directory,policy):
    directory=Path(directory).resolve()
    require(directory.is_dir()and ':'not in str(directory),'Already existing exact library directory required')
    require(policy.get('mode')in MODES,'Unknown reviewed loader environment policy')
    if policy['mode']=='bundle-only':return str(directory)
    runtime=python_library()
    require(fingerprint(runtime)==policy['python_library'],'Current interpreter runtime differs from build-smoke proof')
    require(':'not in runtime['directory'],'Invalid derived Python library directory')
    return str(directory)+':'+runtime['directory']


def checked_manifest(directory,path,expected_sha):
    directory=Path(directory).resolve();path=Path(path)
    require(not path.is_symlink()and path.is_file()and path.stat().st_size<1024**2 and sha(path)==expected_sha,'Exact small library manifest required')
    files=json.loads(path.read_text())
    require(isinstance(files,dict)and files and 'libcrispembed.so'in files,'CrispEmbed library manifest omitted')
    require(set(files)=={p.name for p in directory.iterdir()},'Exact library directory coverage differs')
    for name,row in files.items():
        require(Path(name).name==name and '.so'in name,'Only explicit shared libraries permitted')
        file=directory/name
        require(not file.is_symlink()and file.is_file()and file.resolve().parent==directory and file.stat().st_size==row['bytes']and sha(file)==row['sha256'],'Library manifest bytes differ: '+name)
    return files


def probe(directory,manifest,manifest_sha,mode):
    directory=Path(directory).resolve();files=checked_manifest(directory,manifest,manifest_sha)
    require(mode in MODES,'Unknown loader smoke mode')
    policy=dict(mode=mode)
    if mode=='bundle-and-python-lib':policy['python_library']=fingerprint(python_library())
    require(os.environ.get('LD_LIBRARY_PATH')==loader_path(directory,policy) and not os.environ.get('LD_PRELOAD')and not os.environ.get('LD_AUDIT'),'Fresh smoke loader environment differs')
    # Import only the standard C FFI after every path/hash guard. No GGUF is read.
    import ctypes
    library=ctypes.CDLL(str(directory/'libcrispembed.so'),mode=os.RTLD_NOW|os.RTLD_LOCAL)
    for name in SYMBOLS:
        require(bool(ctypes.cast(getattr(library,name),ctypes.c_void_p).value),'Missing/null C API export: '+name)
    mapped=mapped_libraries()
    require(mapped and directory/'libcrispembed.so'in mapped and any(p.name.startswith('libggml')for p in mapped),'Required CrispEmbed/GGML mappings absent')
    require(all(p.parent==directory and p.name in files and sha(p)==files[p.name]['sha256']for p in mapped),'Actual loaded GGML/CrispEmbed falls outside exact bundle')
    return dict(schema='crispembed-scifact-native-loader-smoke-v1',completed=True,mode=mode,
                python_library=policy.get('python_library'),library_manifest_sha256=manifest_sha,
                library_sha256=files['libcrispembed.so']['sha256'],symbols=list(SYMBOLS),
                mapped_libraries={p.name:files[p.name]for p in sorted(mapped)},all_ggml_mappings_in_bundle=True,
                model_files_accessed=False,framework_imported=False,forward_called=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha',required=True);p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(os.environ.get('GITHUB_ACTIONS')=='true'and sys.platform.startswith('linux'),'Actual library smoke is hosted Linux only')
    require(not a.output.exists(),'Fresh smoke receipt required')
    result=probe(a.directory,a.manifest,a.manifest_sha,a.mode)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
