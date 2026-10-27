#!/usr/bin/env python3
"""Genuine GPU head audit; frozen original arrays are never regenerated."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools'))
from josev2b_head_schema import reference, sha256
from convert_josev2b_head_f32 import convert


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def hardware():
    text = subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,compute_cap,memory.total,memory.free,driver_version',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    rows = [[p.strip() for p in line.split(',')] for line in text.strip().splitlines()]
    if not rows or any(len(r) != 5 for r in rows):
        raise ValueError('Actual GPU hardware receipt required')
    for _, cap, total, free, driver in rows:
        if (tuple(map(int, cap.split('.'))) < (7, 5) or int(total) < 14*1024 or
                int(free) < 11*1024 or tuple(map(int,driver.split('.'))) < (580,65,6)):
            raise ValueError('Unsupported GPU or insufficient GPU memory; no CPU fallback')
    return rows


def copy_original(mount, destination, artifact):
    # The account/source mount path is private configuration, never a source pin.
    # Search only small descriptors; the exact immutable receipt selects one root.
    candidates = [p for p in mount.rglob('reference.json')
                  if p.is_file() and not p.is_symlink() and p.stat().st_size < 1024*1024
                  and sha256(p) == artifact.RECEIPT]
    if len(candidates) != 1:
        raise ValueError('Exactly one SHA-pinned original descriptor required')
    source = candidates[0].parent
    report, _, _ = reference(candidates[0], artifact.RECEIPT, artifact.SOURCE)
    freeze = source/'input-freeze.json'
    declarations = [dict(file='reference.json',bytes=candidates[0].stat().st_size,sha256=artifact.RECEIPT),
                    dict(file='input-freeze.json',bytes=freeze.stat().st_size,sha256=report['input_freeze_sha256']),
                    *report['arrays']]
    destination.mkdir()
    for row in declarations:
        name = row['file']
        if name not in ('reference.json','input-freeze.json'):
            artifact.safe_member(name)
        path = source/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Frozen mounted original byte identity differs')
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha256(target) != row['sha256']:
            raise ValueError('Preserved original readback differs')
    return len(declarations)


def prepare_artifact(mount, output, artifact):
    output.mkdir()
    review = output/'review'; review.mkdir()
    original = output/'original'
    count = copy_original(mount, original, artifact)
    report, cfg, freeze, events = artifact.verify_original(original)
    (review/'ordered-events.json').write_text(json.dumps(events,indent=2)+'\n')
    inputs = artifact.export_probe_inputs(original,report,freeze,events,output/'inputs')
    (review/'native-input-adapters.json').write_text(json.dumps(inputs,indent=2)+'\n')
    head = output/'head'; head.mkdir()
    with tempfile.TemporaryDirectory(prefix='josev-head-checkpoint-') as scratch:
        checkpoint = Path(scratch)/'joint_head.safetensors'
        artifact.download(cfg,'joint_head.safetensors',checkpoint)
        for name in ('LICENSE','README.md','joint_schema_model.py','joint_head_config.json',
                     'config.json','unsloth_decision_config.json','tokenizer_config.json','processor_config.json'):
            artifact.download(cfg,name,head/name)
        converted = convert(checkpoint,original/'reference.json',artifact.RECEIPT,artifact.SOURCE,head/'joint-head-f32.gguf')
    shutil.copyfile(head/'joint-head-f32.conversion.json',review/'head-conversion.json')
    proof = dict(schema='crispembed-josev-head-artifact-v1',completed=True,
        reference_sha256=artifact.RECEIPT,original_source=artifact.SOURCE,
        arrays=len(events),array_payload_bytes=report['array_payload_bytes'],
        gguf_sha256=converted['gguf_sha256'],gguf_bytes=converted['gguf_bytes'],
        tensors=converted['tensor_count'],full_model=False,native_parity=False,model_quality=False,
        preserved_files=count,transport='mounted frozen original; exact SHA-selected descriptor',
        files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha256(p))
               for p in output.rglob('*') if p.is_file()})
    (review/'artifact.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mount',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--build',type=Path,required=True)
    args = parser.parse_args()
    if os.environ.get('CRISPEMBED_JOSEV_GPU_HEAD_AUDIT') != '1':
        parser.error('Explicit genuine GPU audit gate required')
    if args.output.exists() or args.build.exists():
        parser.error('Fresh task-owned output/build paths required')
    rows = hardware()  # Before packages, mounted arrays, download or build.
    if shutil.disk_usage(args.output.parent).free < 3*2**30:
        parser.error('At least3GiB output space required')
    args.output.mkdir(parents=True)
    result = dict(schema='crispembed-josev-head-gpu-v1',completed=False,passed=False,
        hardware=rows,phase='verify-source',full_model=False,model_quality=False,retry_permitted=False,
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():
        (args.output/'gpu.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    try:
        artifact = module('josev_head_artifact',ROOT/'tools/ci-heavy/josev2b_head_artifact.py')
        reference(artifact.PUBLIC,artifact.RECEIPT,artifact.SOURCE)
        import importlib.metadata
        for name, version in [('numpy','2.2.6'),('gguf','0.17.1')]:
            if importlib.metadata.version(name) != version:
                raise ValueError('Pinned head tooling differs')
        result['phase'] = 'preserve-convert'; save()
        prepare_artifact(args.mount,args.output/'artifact',artifact)
        result['phase'] = 'build'; save()
        kh = module('josev_build_harness',ROOT/'tools/kaggle/olmocr-convert/kaggle_harness.py')
        kh.install_build_toolchain()
        arch = kh.detect_cuda_arch()
        if arch != min(r[1] for r in rows).replace('.',''):
            raise ValueError('Build architecture differs from actual GPU preflight')
        flags = kh.cuda_build_flags(arch)+kh.cache_and_link_flags()
        with kh.build_heartbeat('josev-head-cuda'):
            with (args.output/'configure.log').open('w') as log:
                subprocess.run(['cmake','-S',str(ROOT/'tests/josev-head'),'-B',str(args.build),
                    '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',*flags],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            with (args.output/'build.log').open('w') as log:
                subprocess.run(['cmake','--build',str(args.build),'--parallel',kh.safe_build_jobs(gpu=True)],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
        result['phase'] = 'cpu-cuda-head-panel'; save()
        panel = module('josev_head_panel',ROOT/'tools/josev2b_head_panel.py')
        component = args.output/'artifact'
        comparison = panel.run_head_panel(component/'original',component/'head/joint-head-f32.gguf',
            component/'inputs',args.build,args.output/'native',['cpu','CUDA0'])
        if (comparison.get('component_parity') is not True or comparison.get('completed') is not True or
                any(comparison.get(key) for key in ('structural_failures','numeric_failures','decoded_failures'))):
            raise ValueError('Actual CPU/CUDA head parity rejected; all recorded failures retained')
        result.update(completed=True,passed=True,phase='completed')
    except Exception as error:
        result.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        result['logs'] = {p.name:dict(bytes=p.stat().st_size,sha256=sha256(p))
                          for p in args.output.glob('*.log')}
        save()


if __name__ == '__main__':
    main()
