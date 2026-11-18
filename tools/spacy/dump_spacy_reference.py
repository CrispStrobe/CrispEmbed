#!/usr/bin/env python3
"""Dump ORIGINAL spaCy per-stage activations as a crispembed-diff reference.

Hooks the original thinc layers of de_core_news_sm 3.8.0 (no reconstructed
math) and writes one F32 tensor per corpus case and stage, named
`c<case>.<stage>` with the native trace stage names, into a GGUF archive read
by crispembed_diff::Ref (src/crispembed_diff.h). With --gguf, the GGUF's
weights (e.g. a quantized derivative, dequantized) are injected first, so the
reference isolates weight precision from native operator code.

    python3 tools/spacy/dump_spacy_reference.py --wheel W --corpus C --output ref.gguf [--gguf q8_0.gguf]
    crispembed-spacy-diff --ref ref.gguf --corpus C --bundle B [--model M]
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

# (component, walk index, expected layer name, native stage name)
TOK2VEC = [(39, "maxout", "tok2vec.embed.maxout"), (40, "layernorm", "tok2vec.embed.norm")]
for _layer, (_window, _maxout, _residual) in enumerate([(41, 57, 11), (43, 59, 12), (45, 61, 13), (47, 63, 14)]):
    _prefix = f"tok2vec.encoder.layer.{_layer}"
    TOK2VEC += [(_window, "expand_window", _prefix + ".window"), (_maxout, "maxout", _prefix + ".maxout"),
                (_maxout + 1, "layernorm", _prefix + ".norm"), (_residual, None, _prefix + ".residual")]
TOK2VEC.append((0, None, "tok2vec.output"))
HOOKS = ([("tok2vec", index, name, stage) for index, name, stage in TOK2VEC] +
         [(component, 3, "softmax", f"{component}.scores") for component in ("tagger", "morphologizer", "lemmatizer")] +
         [("parser", 6, "linear", "parser.reduce"), ("parser", 2, "precomputable_affine", "parser.lower_cache")])


def as_array(value):
    if isinstance(value, list):  # tok2vec output: one array per doc
        value = value[0]
    array = np.asarray(getattr(value, "get", lambda: value)(), dtype=np.float32)
    return array.reshape(array.shape[0], -1)  # rows = tokens (parser cache: [N+1, 8*64*2])


def install_hooks(nlp, captured):
    for component, index, expected, stage in HOOKS:
        layer = list(nlp.get_pipe(component).model.walk())[index]
        if expected and layer.name != expected:
            raise SystemExit(f"Unexpected thinc graph at {component}/{index}: {layer.name}")
        original = layer._func

        def hooked(model, X, is_train, _original=original, _stage=stage):
            Y, backprop = _original(model, X, is_train)
            captured.setdefault(_stage, as_array(Y))  # first call per document
            return Y, backprop

        layer._func = hooked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, help="Inject these (possibly quantized) weights first")
    args = parser.parse_args()
    import gguf
    texts = json.loads(args.corpus.read_text(encoding="utf-8"))["texts"]
    writer = gguf.GGUFWriter(args.output, "crispembed_diff_reference")
    writer.add_string("reference.source", "original spaCy 3.8.18 / thinc 8.3.14, de_core_news_sm 3.8.0")
    writer.add_string("reference.weights", str(args.gguf.name) if args.gguf else "original F32")
    with tempfile.TemporaryDirectory() as directory:
        nlp = spacy_bundle.load_verified_model(args.wheel, Path(directory))
        if args.gguf:
            from quant_oracle_replay import gguf_tensors, inject
            inject(nlp, gguf_tensors(args.gguf))
        captured: dict[str, np.ndarray] = {}
        install_hooks(nlp, captured)
        for case, text in enumerate(texts):
            captured.clear()
            nlp(text)
            missing = {stage for *_, stage in HOOKS} - set(captured)
            if missing:
                raise SystemExit(f"case {case}: stages not observed: {sorted(missing)}")
            for *_, stage in HOOKS:
                writer.add_tensor(f"c{case:03d}.{stage}", np.ascontiguousarray(captured[stage]))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"{args.output}: {len(texts)} cases x {len(HOOKS)} stages")


if __name__ == "__main__":
    main()
