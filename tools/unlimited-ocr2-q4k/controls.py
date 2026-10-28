"""Hosted real Ref smoke controls, without model/framework inputs."""
import argparse
from pathlib import Path
import struct
from adapter import write_gguf
from panel import command, save


def main(diff, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'input.f32'
    source.write_bytes(struct.pack('<4f', 1, 2, 3, 4))
    from contract import digest
    def write(path, shape, raw=source):
        write_gguf(path, [dict(name='stage', path=raw, shape=shape, sha256=digest(raw))])
    a, b = out / 'a.gguf', out / 'b.gguf'
    write(a, [2, 2])
    write(b, [2, 2])
    positive = command([diff, a, b, out / 'positive.json'], out / 'positive')
    bad = out / 'bad.f32'
    bad.write_bytes(struct.pack('<4f', 1, 2, 3.01, 4))
    write(out / 'bad.gguf', [2, 2], bad)
    negative = command([diff, a, out / 'bad.gguf', out / 'negative.json'], out / 'negative', accepted=(1,))
    write(out / 'shape.gguf', [1, 4])
    shape = command([diff, a, out / 'shape.gguf', out / 'shape.json'], out / 'shape', accepted=(2,))
    save(out / 'controls.json', dict(positive=positive, numerical_rejection=negative, shape_rejection=shape))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--diff', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.diff, args.output)
