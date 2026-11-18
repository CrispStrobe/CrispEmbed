#!/usr/bin/env python3
"""Export a spaCy 3.8 CNN pipeline (v2 profile) to model.gguf + resources.json.

Developer tooling only; nothing here runs in an app. Supports the official
`*_core_*_sm` family (tok2vec/MultiHashEmbed + MaxoutWindowEncoder, optional
tagger and morphologizer, transition parser, attribute_ruler, trainable or
rule lemmatizer, NER with its own tok2vec). The German de_core_news_sm v1
bundle (models/convert-spacy-to-gguf.py) is unchanged and remains frozen.

    python models/convert-spacy-pipeline-to-gguf.py --wheel en_core_web_sm-3.8.0-py3-none-any.whl \
        --output bundle-en/

The wheel must match a pinned SHA-256 below. Output: model.gguf (architecture
`spacy_pipeline_v2`), resources.json (format_version 2), LICENSE and source
notices verbatim from the wheel, and manifest.json with every file pin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spacy_bundle as v1  # noqa: E402  (shared tokenizer/norm/Unicode export)

ARCHITECTURE = "spacy_pipeline_v2"
FORMAT_VERSION = 2
WHEELS = {
    "en_core_web_sm-3.8.0-py3-none-any.whl": ("1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85", 12806118),
    "fr_core_news_sm-3.8.0-py3-none-any.whl": ("7d6ad14cd5078e53147bfbf70fb9d433c6a3865b695fda2657140bbc59a27e29", 16271721),
}
SUPPORTED_COMPONENTS = {"tok2vec", "tagger", "morphologizer", "parser", "attribute_ruler", "lemmatizer", "ner"}
TOK2VEC_FEATURES = ["NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"]
NER_FEATURES = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
RULE_LEMMATIZERS = {"en": "EnglishLemmatizer", "fr": "FrenchLemmatizer"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_wheel(wheel: Path, destination: Path):
    pin = WHEELS.get(wheel.name)
    require(pin is not None, f"Unpinned wheel {wheel.name}")
    data = wheel.read_bytes()
    require(len(data) == pin[1] and digest(data) == pin[0], "Wheel size/SHA-256 differs from the pinned source")
    package = wheel.name.split("-")[0]
    version = wheel.name.split("-")[1]
    prefix = f"{package}/{package}-{version}/"
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            parts = PurePosixPath(info.filename)
            require(not parts.is_absolute() and ".." not in parts.parts, "Unsafe archive path")
            if info.filename.startswith(prefix) and not info.is_dir():
                relative = info.filename[len(prefix):]
                require(info.file_size < 64 * 1024 * 1024, "Oversized model resource")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
    import spacy
    nlp = spacy.load(destination)  # source default: disabled components stay disabled
    require(set(nlp.pipe_names) <= SUPPORTED_COMPONENTS, f"Unsupported components {nlp.pipe_names}")
    require(nlp.meta["version"] == version, "Wheel/model version mismatch")
    return nlp, {"name": f"{nlp.lang}_{nlp.meta['name']}", "version": version, "sha256": pin[0],
                 "size_bytes": pin[1], "file": wheel.name}


def walk(model):
    return list(model.walk())


def put(tensors, origins, prefix, layer, parameters, component, index):
    import numpy as np
    require(set(layer.param_names) == set(parameters), f"Unexpected parameters at {component}/{index}")
    for parameter in parameters:
        original = layer.get_param(parameter)
        require(original.dtype == np.dtype("float32") and bool(np.isfinite(original).all()),
                f"Not finite F32: {prefix}.{parameter}")
        tensors[f"{prefix}.{parameter}"] = np.array(original, dtype="<f4", order="C", copy=True)
        origins[f"{prefix}.{parameter}"] = {"component": component, "node_index": index, "node": layer.name,
                                            "parameter": parameter}


def export_tok2vec(nlp, model, prefix, component, features, tensors, origins):
    """MultiHashEmbed + MaxoutWindowEncoder, found by layer name (not index)."""
    nodes = walk(model)
    extract = [n for n in nodes if n.name == "extract_features"]
    require(len(extract) == 1, f"{component}: expected one feature extractor")
    columns = [nlp.vocab.strings[c] if isinstance(c, int) else c for c in extract[0].attrs["columns"]]
    require(columns == features, f"{component}: unexpected feature columns {columns}")
    padded = [n for n in nodes if n.name.startswith("with_array(residual")]
    require(len(padded) == 1 and padded[0].attrs.get("pad") == 4, f"{component}: unexpected encoder padding")
    windows = [n for n in nodes if n.name == "expand_window"]
    require(len(windows) == 4 and all(n.attrs["window_size"] == 1 for n in windows), f"{component}: window")
    hashes = [(i, n) for i, n in enumerate(nodes) if n.name == "hashembed"]
    require(len(hashes) == len(features), f"{component}: hash embedding count")
    rows = []
    for feature, (i, node) in enumerate(hashes):
        require(node.attrs["seed"] == 8 + feature and node.attrs["column"] == feature,
                f"{component}: hash embedding seed/column")
        put(tensors, origins, f"{prefix}embed.{feature}", node, ["E"], component, i)
        rows.append(int(node.get_param("E").shape[0]))
    maxouts = [(i, n) for i, n in enumerate(nodes) if n.name == "maxout"]
    norms = [(i, n) for i, n in enumerate(nodes) if n.name == "layernorm"]
    require(len(maxouts) == 5 and len(norms) == 5 and all(n[0] == m[0] + 1 for m, n in zip(maxouts, norms)),
            f"{component}: maxout/layernorm graph")
    put(tensors, origins, f"{prefix}embed.mix", maxouts[0][1], ["W", "b"], component, maxouts[0][0])
    put(tensors, origins, f"{prefix}embed.norm", norms[0][1], ["G", "b"], component, norms[0][0])
    for layer in range(4):
        put(tensors, origins, f"{prefix}encoder.{layer}.maxout", maxouts[layer + 1][1], ["W", "b"], component,
            maxouts[layer + 1][0])
        put(tensors, origins, f"{prefix}encoder.{layer}.norm", norms[layer + 1][1], ["G", "b"], component,
            norms[layer + 1][0])
    width = int(maxouts[0][1].get_param("W").shape[0])
    require(width == 96, f"{component}: unsupported width {width}")
    return rows


def export_classifier(nlp, name, tensors, origins):
    nodes = walk(nlp.get_pipe(name).model)
    soft = [(i, n) for i, n in enumerate(nodes) if n.name == "softmax"]
    require(len(soft) == 1 and soft[0][1].attrs["softmax_normalize"] is False and
            soft[0][1].attrs["softmax_temperature"] == 1.0, f"{name}: unsupported classifier normalization")
    put(tensors, origins, name, soft[0][1], ["W", "b"], name, soft[0][0])


def export_transition(nlp, name, prefix, tensors, origins, features):
    component = nlp.get_pipe(name)
    model = component.model
    require(model.attrs["has_upper"] is True and not model.attrs["unseen_classes"], f"{name}: class mask/upper")
    require(component.cfg.get("beam_width", 1) == 1, f"{name}: only greedy decoding")
    nodes = walk(model)
    require(nodes[2].name == "precomputable_affine" and nodes[3].name == "linear" and nodes[6].name == "linear",
            f"{name}: unexpected transition model graph")
    put(tensors, origins, f"{prefix}reduce", nodes[6], ["W", "b"], name, 6)
    put(tensors, origins, f"{prefix}lower", nodes[2], ["W", "b", "pad"], name, 2)
    put(tensors, origins, f"{prefix}upper", nodes[3], ["W", "b"], name, 3)
    require(int(nodes[2].get_param("W").shape[0]) == features, f"{name}: state feature count")
    return [component.moves.get_class_name(i) for i in range(component.moves.n_moves)]


UNIVERSAL_POS = ["noun", "verb", "adj", "adv", "propn", "num", "pron", "det", "adp", "aux", "cconj", "sconj",
                 "part", "intj", "punct", "sym", "x", "space", "eol"]


def table_json(name, table):
    """Lookups tables are keyed by spaCy string hashes. POS-keyed tables are
    decoded back to their lowercase POS names (the runtime never re-hashes
    them); `lemma_lookup` stays keyed by the decimal hash of the token text."""
    from spacy.strings import get_string_id
    if name == "lemma_lookup":
        return {"key_kind": "hash_uint64_decimal", "entries": {str(k): v for k, v in table.items()}}
    names = {get_string_id(pos): pos for pos in UNIVERSAL_POS}
    keys = list(table.keys())
    require(all(k in names for k in keys), f"{name}: undecodable POS key")
    return {"key_kind": "pos", "entries": {names[k]: table[k] for k in keys}}


def extract(nlp, source: Path):
    import srsly
    tensors, origins, labels = {}, {}, {}
    pipes = nlp.pipe_names
    require(pipes[0] == "tok2vec" and "parser" in pipes, "Expected a shared tok2vec and a parser")
    rows = export_tok2vec(nlp, nlp.get_pipe("tok2vec").model, "", "tok2vec", TOK2VEC_FEATURES, tensors, origins)
    for name in ("tagger", "morphologizer"):
        if name in pipes:
            export_classifier(nlp, name, tensors, origins)
    if "tagger" in pipes:
        labels["tagger"] = list(nlp.get_pipe("tagger").labels)
    if "morphologizer" in pipes:
        morph = nlp.get_pipe("morphologizer")
        labels["morphologizer"] = {"labels": list(morph.labels), "morph": morph.cfg["labels_morph"],
                                   "pos": morph.cfg["labels_pos"], "overwrite": morph.cfg["overwrite"],
                                   "extend": morph.cfg["extend"]}
    labels["parser"] = {"moves": export_transition(nlp, "parser", "parser.", tensors, origins, 8),
                        "learn_tokens": nlp.get_pipe("parser").cfg.get("learn_tokens", False),
                        "features": ["B0", "B1", "S0", "S1", "S2", "L(B0,1)", "L(S0,1)", "R(S0,1)"]}
    require(labels["parser"]["learn_tokens"] is False, "Subtoken learning parser unsupported")
    if "ner" in pipes:
        ner_rows = export_tok2vec(nlp, nlp.get_pipe("ner").model, "ner.", "ner", NER_FEATURES, tensors, origins)
        require(ner_rows == rows[:4], "NER embedding rows differ from the shared tok2vec rows")
        labels["ner"] = {"moves": export_transition(nlp, "ner", "ner.", tensors, origins, 3)}
    if "attribute_ruler" in pipes:
        from spacy.tokens import MorphAnalysis
        rules = nlp.get_pipe("attribute_ruler").patterns
        for rule in rules:
            # spaCy stores MORPH canonically (sorted fields, "_" -> empty);
            # export exactly what set_token_attrs assigns.
            if "MORPH" in rule["attrs"]:
                rule["attrs"]["MORPH"] = str(MorphAnalysis(nlp.vocab, rule["attrs"]["MORPH"]))
        labels["attribute_ruler"] = rules
    if "lemmatizer" in pipes:
        lemma = nlp.get_pipe("lemmatizer")
        if lemma.__class__.__name__ == "EditTreeLemmatizer":
            raise ValueError("Trainable lemmatizers use the v1 German profile for now")
        require(lemma.mode == "rule" and lemma.__class__.__name__ == RULE_LEMMATIZERS.get(nlp.lang),
                f"Unsupported lemmatizer {lemma.__class__.__name__}/{lemma.mode}")
        require(lemma.overwrite is False, "Unsupported lemmatizer overwrite")
        labels["lemmatizer"] = {"mode": "rule", "language": nlp.lang, "overwrite": False,
                                "tables": {name: table_json(name, lemma.lookups.get_table(name))
                                           for name in sorted(lemma.lookups.tables)}}
    tokenizer_and_norms = shared_lexical(nlp, source)
    return tensors, origins, {"format_version": FORMAT_VERSION, "language": nlp.lang, "component_order": pipes,
                              **tokenizer_and_norms, "unicode": v1.unicode_tables(), "labels": labels,
                              "embedding_rows": rows,
                              "lexical_semantics_source": {"revision": v1.SOURCES["spacy"],
                                  "files": ["spacy/lang/lex_attrs.py", "spacy/strings.pyx", "spacy/tokenizer.pyx"]}}


def shared_lexical(nlp, source: Path):
    """Tokenizer and NORM data, exported exactly as the v1 German bundle does."""
    import functools
    import srsly
    from spacy.attrs import IDS, NORM
    reverse = {value: key for key, value in IDS.items()}
    tokenizer = srsly.msgpack_loads((source / "tokenizer").read_bytes())
    require(set(tokenizer) == {"prefix_search", "suffix_search", "infix_finditer", "token_match",
                               "url_match", "exceptions", "faster_heuristics"}, "Tokenizer schema changed")
    tokenizer["exceptions"] = {key: [{reverse[attr]: value for attr, value in item.items()} for item in entries]
                               for key, entries in tokenizer["exceptions"].items()}
    norm_tables = []
    getter = nlp.vocab.lex_attr_getters[NORM]
    while isinstance(getter, functools.partial):
        require(getter.func.__name__ == "_get_attr_unless_lookup", "Unknown NORM getter")
        for table in getter.args[1]:
            keys = list(table.keys())
            kind = "hash_uint64_decimal" if keys and isinstance(keys[0], int) else "text"
            norm_tables.append({"key_kind": kind, "entries": {str(k): v for k, v in table.items()}})
        getter = getter.args[0]
    require(getter.__name__ == "lower", "Unknown fallback NORM getter")
    return {"tokenizer": tokenizer, "norm_lookup_precedence": norm_tables}


def convert(wheel: Path, output: Path):
    require(not output.exists(), "Output exists; refusing to replace")
    output.parent.mkdir(parents=True, exist_ok=True)
    import gguf
    import numpy as np
    import spacy
    with tempfile.TemporaryDirectory(prefix="spacy-v2-", dir=output.parent) as temporary:
        temp = Path(temporary)
        source, bundle = temp / "source", temp / "bundle"
        source.mkdir()
        bundle.mkdir()
        nlp, model = load_wheel(wheel, source)
        tensors, origins, resources = extract(nlp, source)
        resource_bytes = v1.json_bytes(resources)
        (bundle / "resources.json").write_bytes(resource_bytes)
        for notice in ("LICENSE", "LICENSES_SOURCES"):
            if (source / notice).exists():
                shutil.copyfile(source / notice, bundle / notice)
        writer = gguf.GGUFWriter(str(bundle / "model.gguf"), ARCHITECTURE)
        writer.add_uint32("spacy.format_version", FORMAT_VERSION)
        writer.add_string("spacy.language", nlp.lang)
        writer.add_string("spacy.model", model["name"])
        writer.add_string("spacy.model_version", model["version"])
        writer.add_string("spacy.source.sha256", model["sha256"])
        writer.add_string("spacy.resources.sha256", digest(resource_bytes))
        writer.add_string("spacy.components", ",".join(nlp.pipe_names))
        writer.add_string("spacy.precision", "f32")
        for name in sorted(tensors):
            writer.add_tensor(name, tensors[name])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        reader = gguf.GGUFReader(str(bundle / "model.gguf"))
        require(len(reader.tensors) == len(tensors), "GGUF round-trip tensor count mismatch")
        for tensor in reader.tensors:
            original = tensors[tensor.name]
            require(tensor.tensor_type == gguf.GGMLQuantizationType.F32 and
                    tuple(tensor.shape) == tuple(reversed(original.shape)) and
                    np.array_equal(tensor.data.reshape(original.shape), original),
                    f"GGUF round-trip mismatch: {tensor.name}")
        del reader
        files = sorted(p.name for p in bundle.iterdir())
        manifest = {
            "format_version": FORMAT_VERSION, "architecture": ARCHITECTURE, "language": nlp.lang,
            "model": model, "spacy": spacy.__version__, "component_order": nlp.pipe_names,
            "disabled_components": nlp.disabled,
            "tensors": [{"name": n, "shape": list(tensors[n].shape), "sha256": digest(tensors[n].tobytes()),
                         "source": origins[n]} for n in sorted(tensors)],
            "files": {name: {"size_bytes": (bundle / name).stat().st_size,
                             "sha256": digest((bundle / name).read_bytes())} for name in files},
            "converter_sha256": digest(Path(__file__).read_bytes()),
        }
        (bundle / "manifest.json").write_bytes(v1.json_bytes(manifest))
        os.rename(bundle, output)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = convert(args.wheel, args.output)
    except Exception as error:
        print(f"spaCy v2 conversion failed: {error}", file=sys.stderr)
        return 1
    print(f"Exported {len(manifest['tensors'])} tensors for {manifest['model']['name']}; "
          f"model.gguf {manifest['files']['model.gguf']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
