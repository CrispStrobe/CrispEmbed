#!/usr/bin/env python3
"""Lossless original-first joint-head artifact only; not a full Josev model.

No framework or payload access on import. Execute conversion only on hosted CI
or inside the authorized original GPU session after reviewing its actual receipt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from josev2b_head_schema import HEAD_CONFIG, REVISION, checked_readback, checked_schema, inventory, reference, sha256, tensor_schema


def convert(checkpoint, original_report, original_sha256, original_source, output):
    # All reference guards precede even optional array/GGUF package imports.
    report, cfg, proof = reference(original_report, original_sha256, original_source)
    checkpoint = Path(checkpoint)
    declared = cfg['files']['joint_head.safetensors']
    if checkpoint.stat().st_size != declared['bytes'] or sha256(checkpoint) != declared['sha256']:
        raise ValueError('Actual original joint-head checkpoint identity differs')
    offset, rows = inventory(checkpoint)
    checked_schema(rows)
    import numpy as np
    import gguf
    output = Path(output)
    receipt = output.with_suffix('.conversion.json')
    if output.exists() or receipt.exists():
        raise ValueError('Fresh output required; preserve prior artifacts')
    output.parent.mkdir(parents=True, exist_ok=True)
    records, converted = [], []
    with checkpoint.open('rb') as stream:
        for name, shape in tensor_schema().items():
            row = rows[name]
            stream.seek(offset+row['offsets'][0])
            raw = stream.read(row['offsets'][1]-row['offsets'][0])
            if len(raw) != row['offsets'][1]-row['offsets'][0]:
                raise ValueError('Truncated original tensor')
            if row['dtype'] == 'BF16':
                # BF16 is promoted exactly by placing its bits in the high half.
                value = (np.frombuffer(raw, dtype='<u2').astype('<u4') << 16).view('<f4')
            else:
                value = np.frombuffer(raw, dtype='<f4').copy()
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite original tensor: '+name)
            value = value.reshape(shape or [1])
            actual_sha = hashlib.sha256(value.tobytes(order='C')).hexdigest()
            if actual_sha != proof[name]['loaded_f32_sha256']:
                raise ValueError('F32 promotion does not equal actual loaded original tensor: '+name)
            expected_dtype = 'BF16' if 'bfloat16' in proof[name]['checkpoint_dtype'] else 'F32'
            if proof[name]['checkpoint_dtype'] == 'BF16':
                expected_dtype = 'BF16'
            if row['dtype'] != expected_dtype:
                raise ValueError('Original source dtype differs from captured loaded-weight proof')
            target = 'head.'+name
            records.append(dict(source=name, target=target, source_shape=shape,
                gguf_shape_numpy=list(value.shape), source_dtype=row['dtype'],
                f32_sha256=actual_sha, transformation='exact BF16 promotion' if row['dtype']=='BF16' else 'identity',
                scalar_adapter='rank-zero to one-element vector' if not shape else None))
            converted.append((target, value))
    arch = 'josev2b_head_reference'
    writer = gguf.GGUFWriter(str(output), arch=arch)
    writer.add_name('Josev-2B original joint head F32 component reference')
    writer.add_string('general.license', 'apache-2.0')
    writer.add_string(arch+'.source_revision', REVISION)
    writer.add_string(arch+'.checkpoint_sha256', declared['sha256'])
    writer.add_string(arch+'.original_receipt_sha256', original_sha256)
    writer.add_string(arch+'.original_worker_source', original_source)
    writer.add_string(arch+'.calibration_profile', 'official-pytorch-no-extra-temperature')
    for key, value in HEAD_CONFIG.items():
        writer.add_uint32(arch+'.'+key, value)
    for name, value in converted:
        writer.add_tensor(name, value, raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    actual = {tensor.name: tensor for tensor in gguf.GGUFReader(str(output)).tensors}
    if set(actual) != {name for name, _ in converted}:
        raise ValueError('Written head tensor inventory differs')
    for name, expected in converted:
        tensor = actual[name]
        if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError('Written head tensor type differs: '+name)
        checked_readback(tensor.shape, tensor.data.astype('<f4', copy=False).tobytes(order='C'),
                         list(expected.shape), hashlib.sha256(expected.tobytes(order='C')).hexdigest())
    result = dict(schema='crispembed-josev-head-f32-conversion-v1', completed=True,
        full_model=False, native_parity=False, representative_quality=False,
        original_receipt_sha256=original_sha256, original_worker_source=original_source,
        checkpoint_sha256=declared['sha256'], gguf_sha256=sha256(output), gguf_bytes=output.stat().st_size,
        tensor_count=len(records), parameter_count=sum(__import__('math').prod(v) for v in tensor_schema().values()),
        tensors=records, limits='Head-only F32 byte preservation, not native head/backbone/inference or calibration parity.')
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--original-report',type=Path,required=True)
    p.add_argument('--original-sha256',required=True)
    p.add_argument('--original-source',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true' and not (Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir()):
        p.error('Checkpoint conversion is hosted-only; no local model/framework work')
    convert(args.checkpoint,args.original_report,args.original_sha256,args.original_source,args.output)


if __name__=='__main__': main()
