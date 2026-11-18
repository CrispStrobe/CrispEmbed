#!/usr/bin/env python3
"""Run ORIGINAL spaCy with the weights of a (quantized) spaCy GGUF.

Attribution control for precision experiments: the GGUF's tensors are
dequantized with gguf-py and injected into the original thinc layers through
the same name -> (component, node, parameter) mapping the converter uses, then
the corpus is analyzed by unmodified spaCy. If original spaCy shows the same
annotation changes as the native runtime, those changes are caused by weight
precision, not by native operator code.

    python3 tools/spacy/quant_oracle_replay.py --wheel W --gguf q8_0.gguf \
        --oracle oracle.json --output replay.json [--native-report ab.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spacy_bundle  # noqa: E402
from ggml_ab_oracle import FIELDS, annotate  # noqa: E402


def gguf_tensors(path):
    import gguf
    from gguf import quants
    reader = gguf.GGUFReader(path)
    result = {}
    for tensor in reader.tensors:
        shape = tuple(int(d) for d in reversed(tensor.shape))
        kind = tensor.tensor_type
        if kind == gguf.GGMLQuantizationType.F32:
            values = np.array(tensor.data, dtype=np.float32).reshape(shape)
        else:
            values = quants.dequantize(np.array(tensor.data), kind).reshape(shape).astype(np.float32)
        result[tensor.name] = (values, kind.name)
    return result


def inject(nlp, tensors):
    # Reuse the converter's mapping: walk it once on the F32 model to learn origins.
    _, origins = spacy_bundle.extract_tensors(nlp)
    for name, origin in origins.items():
        values, _ = tensors[name]
        layer = list(nlp.get_pipe(origin["component"]).model.walk())[origin["node_index"]]
        current = layer.get_param(origin["parameter"])
        assert current.shape == values.shape, (name, current.shape, values.shape)
        layer.set_param(origin["parameter"], values.copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, help="tests/spacy_ggml_ab.py report to compare flips with")
    args = parser.parse_args()
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    tensors = gguf_tensors(args.gguf)
    with tempfile.TemporaryDirectory() as directory:
        nlp = spacy_bundle.load_verified_model(args.wheel, Path(directory))
        inject(nlp, tensors)
        flips = []
        for index, case in enumerate(oracle["cases"]):
            replay = annotate(nlp(case["text"]))
            assert [t["text"] for t in replay] == [t["text"] for t in case["tokens"]]
            for i, (r, e) in enumerate(zip(replay, case["tokens"])):
                flips.extend({"case": index, "token": i, "text": e["text"], "field": f, "oracle": e[f],
                              "replay": r[f]} for f in FIELDS if r[f] != e[f])
    key = lambda d: (d["case"], d["token"], d["field"])  # noqa: E731
    result = {"schema": "crispembed.spacy.quant_oracle_replay.v1", "gguf": str(args.gguf),
              "types": sorted({kind for _, kind in tensors.values()}),
              "replay_flips": len(flips), "replay_flipped_cases": len({d["case"] for d in flips}), "flips": flips}
    if args.native_report:
        native = json.loads(args.native_report.read_text(encoding="utf-8"))
        native_keys = {key(d) for d in native["differences"] if "field" in d}
        replay_keys = {key(d) for d in flips}
        result["native_flips"] = len(native_keys)
        result["shared_flips"] = len(native_keys & replay_keys)
        result["native_only"] = sorted(native_keys - replay_keys)
        result["replay_only"] = sorted(replay_keys - native_keys)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "flips"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
