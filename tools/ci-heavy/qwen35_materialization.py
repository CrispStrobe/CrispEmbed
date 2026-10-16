"""Test-only exact checkpoint conversion audit; large arrays stay in scratch."""
import hashlib
import json
import math
import os
import re
import subprocess


def parse_matrix_audit(log, schema):
    pattern = (r'^MATERIALIZE_MATRIX tensor=(\S+) elements=(\d+) exact=(\d+) finite=(\d+) workers=(\d+) '
               r'max_abs=(\S+) magnitude=(\S+) norm=(\S+) offset=(\d+)$')
    rows = []
    offset = 0
    for m in re.findall(pattern, log, re.M):
        name, count, exact, finite, workers, error, magnitude, norm, position = m
        count, workers, position = int(count), int(workers), int(position)
        error, magnitude, norm = float(error), float(magnitude), float(norm)
        tensor = schema[name]
        expected_workers = 4 if tensor['dtype'] == 'float16' and count >= 262144 else 1
        if (count != math.prod(tensor['shape']) or position != offset or exact != '1' or finite != '1'
                or workers != expected_workers or error != 0
                or any(not math.isfinite(x) or x < 0 for x in (error, magnitude, norm))):
            raise ValueError('Invalid materialization equality/finite/scheduling/shape/offset report')
        rows.append(dict(tensor=name, elements=count, exact=True, finite=True, workers=workers,
            max_abs=error, magnitude=magnitude, norm=norm, offset=position))
        offset += count * 4
    complete = re.findall(r'^MATERIALIZE_COMPLETE tensors=(\d+) bytes=(\d+)$', log, re.M)
    if (len(complete) != 1 or tuple(map(int, complete[0])) != (len(schema), offset)
            or [r['tensor'] for r in rows] != sorted(schema)):
        raise ValueError('Incomplete/duplicate/out-of-order matrix audit')
    return dict(tensors=rows, bytes=offset)


def audit(repo, build, baseline, manifest, scratch, out):
    import gguf
    import numpy as np
    schema = {r['name']: r for r in manifest['records']}
    target = scratch / 'qwen35-materialization-f32.bin'
    if target.exists():
        raise ValueError('Materialization scratch destination already exists')
    try:
        p = subprocess.run([str(build / 'test-qwen35-materialize'), str(baseline), str(target)],
            capture_output=True, text=True, env=dict(os.environ, CRISPEMBED_QWEN35_PROFILE='0', CRISPEMBED_QWEN35_QUANT='0'))
        log = p.stdout + p.stderr
        (out / 'materialize-checkpoint.log').write_text(log)
        p.check_returncode()
        result = parse_matrix_audit(log, schema)
        if target.stat().st_size != result['bytes']:
            raise ValueError('Truncated materialized checkpoint')
        reader = gguf.GGUFReader(str(baseline))
        tensors = {t.name: t for t in reader.tensors}
        if set(tensors) != set(schema):
            raise ValueError('Changed checkpoint tensor set')
        actual = np.memmap(target, dtype='<f4', mode='r')
        aggregate = hashlib.sha256()
        for row in result['tensors']:
            name = row['tensor']
            source = tensors[name].data.reshape(-1)
            expected_dtype = np.dtype({'float16': '<f2', 'float32': '<f4'}[schema[name]['dtype']])
            if source.dtype != expected_dtype or list(tensors[name].data.shape) != schema[name]['shape']:
                raise ValueError('Changed checkpoint dtype/shape')
            mine = actual[row['offset']//4:row['offset']//4 + row['elements']]
            digest = hashlib.sha256()
            magnitude, square = 0., 0.
            for start in range(0, row['elements'], 1048576):
                reference = source[start:start+1048576].astype('<f4')
                values = mine[start:start+len(reference)]
                if not np.isfinite(reference).all() or not np.array_equal(reference.view('<u4'), values.view('<u4')):
                    raise ValueError('Native materialization differs from independent NumPy conversion: ' + name)
                data = values.tobytes()
                digest.update(data); aggregate.update(data)
                magnitude = max(magnitude, float(np.abs(reference).max(initial=0)))
                square += float(np.square(reference.astype(np.float64)).sum())
            row.update(python_bits_exact=True, python_max_abs=0., python_magnitude=magnitude,
                python_norm=math.sqrt(square), materialized_sha256=digest.hexdigest())
            print('materialized exact', name, row['elements'], 'workers', row['workers'], flush=True)
        result.update(python_bits_exact=True, materialized_sha256=aggregate.hexdigest(),
            reference='NumPy float16/float32 GGUF storage -> contiguous little-endian float32, every element bitwise compared; native serial/parallel memcmp exact',
            workers_scope='Configured scheduling policy, not a claim that OS thread creation never fell back to serial.')
        (out / 'materialization-arrays.json').write_text(json.dumps(result, indent=2) + '\n')
        del actual, reader
        return result
    finally:
        if target.exists(): target.unlink()
