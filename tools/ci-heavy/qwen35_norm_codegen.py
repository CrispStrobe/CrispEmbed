#!/usr/bin/env python3
"""Hosted static inspection of the actual vision norm compiler output; no weights."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    repo=Path(__file__).resolve().parents[2]
    out=Path(os.environ['HEAVY_OUT'])
    build=Path(os.environ['HEAVY_SCRATCH'])/'qwen35-norm-codegen'
    subprocess.run(['cmake','-S',str(repo),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
        '-DGGML_NATIVE=OFF','-DGGML_BLAS=OFF','-DGGML_METAL=OFF',
        '-DCRISPEMBED_BUILD_SHARED=OFF','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],check=True)
    subprocess.run(['cmake','--build',str(build),'--target','crispembed-qwen35-delta','-j','4'],check=True)
    commands=json.loads((build/'compile_commands.json').read_text())
    command=next(c['command'] for c in commands if c['file'].endswith('/qwen35_vision.cpp') and 'crispembed-qwen35-delta.dir' in c['command'])
    obj=build/'CMakeFiles/crispembed-qwen35-delta.dir/src/qwen35_vision.cpp.o'
    asm=subprocess.check_output(['objdump','-d','-C',obj.name],cwd=obj.parent,text=True)
    (out/'vision-disassembly.txt').write_text(asm)
    normalized=command.replace(str(build),'$BUILD').replace(str(repo),'$REPO')
    compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0]
    report=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        scope='Static compiler output only; no model download or inference',compiler=compiler,command=normalized,
        source_sha256={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in ('src/qwen35_vision.cpp','src/qwen35_norm.cpp')},
        disassembly_sha256=hashlib.sha256(asm.encode()).hexdigest())
    (out/'norm-codegen.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Retained actual vision object disassembly and normalized compiler provenance')


if __name__=='__main__':
    main()
