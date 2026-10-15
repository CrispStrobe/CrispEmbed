#!/usr/bin/env python3
"""Hosted compiler provenance for native and validation attention; no weights."""
import hashlib
import json
import os
from pathlib import Path
import subprocess


def capture(repo, build, out):
    commands = json.loads((build / 'compile_commands.json').read_text())
    records = []
    for name, relative, object_name in (
        ('vision', 'src/qwen35_vision.cpp', 'CMakeFiles/crispembed-qwen35-delta.dir/src/qwen35_vision.cpp.o'),
        ('trace', 'tests/test_qwen35_attention_trace.cpp', 'CMakeFiles/test-qwen35-attention-trace.dir/tests/test_qwen35_attention_trace.cpp.o')):
        command = next(c['command'] for c in commands if c['file'].endswith('/' + relative) and object_name.split('/')[1] in c['command'])
        obj = build / object_name
        asm = subprocess.check_output(['objdump', '-d', '-C', obj.name], cwd=obj.parent, text=True)
        (out / (name + '-attention-disassembly.txt')).write_text(asm)
        records.append(dict(name=name, command=command.replace(str(build), '$BUILD').replace(str(repo), '$REPO'),
            disassembly_sha256=hashlib.sha256(asm.encode()).hexdigest(), bytes=len(asm.encode())))
    report = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        scope='Actual native/validation object disassembly only; compiler inspection is not model parity',
        compiler=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0], objects=records,
        source_sha256={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in
            ('src/qwen35_vision.cpp','src/qwen35_norm.cpp','tests/test_qwen35_attention_trace.cpp','tests/qwen35_attention_audit.h')})
    (out / 'attention-codegen.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    repo = Path(__file__).resolve().parents[2]
    out = Path(os.environ['HEAVY_OUT'])
    build = Path(os.environ['HEAVY_SCRATCH']) / 'qwen35-attention-codegen'
    subprocess.run(['cmake','-S',str(repo),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
        '-DGGML_NATIVE=OFF','-DGGML_BLAS=OFF','-DGGML_METAL=OFF','-DCRISPEMBED_BUILD_SHARED=OFF',
        '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],check=True)
    subprocess.run(['cmake','--build',str(build),'--target','test-qwen35-attention-trace','-j','4'],check=True)
    capture(repo, build, out)


if __name__ == '__main__': main()
