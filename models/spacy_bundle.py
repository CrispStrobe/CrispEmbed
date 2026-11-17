"""Closed, deterministic de_core_news_sm 3.8.0 export contract.

Developer tooling only. No Python code or MessagePack decoder is part of the
native runtime. Authenticate the wheel before opening its archive or loading a
spaCy data directory; never import executable code from the model wheel.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import tempfile
import unicodedata
import zipfile

FORMAT_VERSION = 1
ARCHITECTURE = "spacy_de_core_news_sm"
WHEEL_NAME = "de_core_news_sm-3.8.0-py3-none-any.whl"
WHEEL_SHA256 = "fec69fec52b1780f2d269d5af7582a5e28028738bd3190532459aeb473bfa3e7"
WHEEL_SIZE = 14639490
MODEL_PREFIX = "de_core_news_sm/de_core_news_sm-3.8.0/"
COMPONENTS = ["tok2vec", "tagger", "morphologizer", "parser", "lemmatizer", "attribute_ruler"]
FEATURES = ["NORM", "PREFIX", "SUFFIX", "SHAPE", "SPACY", "IS_SPACE"]
ROWS = [5000, 1000, 2500, 2500, 50, 50]
ORACLE = {"spacy": "3.8.18", "thinc": "8.3.14", "numpy": "2.4.6", "gguf": "0.19.0"}
SOURCES = {
    "spacy": "43311999028032fd81bef77a75fdb4fa96803c98",
    "thinc": "e617de4ea2530c7091ea2110e011807c33c366f0",
    "ggml": "3a27b6c665b191cb230be367db5c52d81dd3e2ac",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def expected_tensors() -> dict[str, tuple[int, ...]]:
    result = {f"embed.{i}.E": (rows, 96) for i, rows in enumerate(ROWS)}
    result.update({"embed.mix.W": (96, 3, 576), "embed.mix.b": (96, 3),
                   "embed.norm.G": (96,), "embed.norm.b": (96,)})
    for i in range(4):
        result.update({f"encoder.{i}.maxout.W": (96, 3, 288),
                       f"encoder.{i}.maxout.b": (96, 3),
                       f"encoder.{i}.norm.G": (96,), f"encoder.{i}.norm.b": (96,)})
    for name, rows in [("tagger", 55), ("morphologizer", 671), ("lemmatizer", 1311)]:
        result[name + ".W"] = (rows, 96)
        result[name + ".b"] = (rows,)
    result.update({"parser.reduce.W": (64, 96), "parser.reduce.b": (64,),
                   "parser.lower.W": (8, 64, 2, 64), "parser.lower.b": (64, 2),
                   "parser.lower.pad": (1, 8, 64, 2),
                   "parser.upper.W": (137, 64), "parser.upper.b": (137,)})
    return result


def verify_wheel(path: Path) -> bytes:
    require(path.is_file(), "Required source wheel is missing")
    require(path.stat().st_size == WHEEL_SIZE, "Source wheel size mismatch")
    data = path.read_bytes()
    require(digest(data) == WHEEL_SHA256, "Source wheel SHA-256 mismatch")
    return data


def verify_environment():
    require(platform.python_implementation() == "CPython" and platform.python_version() == "3.12.14",
            "Converter requires the frozen CPython 3.12.14 reference environment")
    for name, version in ORACLE.items():
        require(importlib.metadata.version(name) == version, f"Expected {name}=={version}")


def load_verified_model(wheel: Path, destination: Path):
    verify_wheel(wheel)
    import spacy
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate archive members")
        for info in archive.infolist():
            name = info.filename
            require(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts,
                    "Unsafe archive path")
            if name.startswith(MODEL_PREFIX) and not info.is_dir():
                relative = name[len(MODEL_PREFIX):]
                require(relative and info.file_size < 32 * 1024 * 1024, "Oversized model resource")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
    nlp = spacy.load(destination, exclude=["ner", "senter"])
    require(nlp.lang == "de" and nlp.meta["name"] == "core_news_sm" and nlp.meta["version"] == "3.8.0",
            "Unsupported model identity")
    require(nlp.pipe_names == COMPONENTS, "Unsupported component order or missing component")
    return nlp


def validate_tensors(tensors):
    import numpy as np
    expected = expected_tensors()
    require(set(tensors) == set(expected), "Unexpected/missing tensor inventory")
    for name, value in tensors.items():
        require(tuple(value.shape) == expected[name], f"Unexpected tensor shape: {name}")
        require(value.dtype == np.dtype("float32"), f"Non-F32 tensor: {name}")
        require(bool(np.isfinite(value).all()), f"Non-finite trained tensor: {name}")


def extract_tensors(nlp):
    import numpy as np
    tensors, origins = {}, {}

    def put(prefix, component, node, parameters):
        nodes = list(nlp.get_pipe(component).model.walk())
        layer = nodes[node]
        require(set(layer.param_names) == set(parameters), f"Unexpected parameters at {component}/{node}")
        for parameter in parameters:
            name = prefix + "." + parameter
            original = layer.get_param(parameter)
            require(original.dtype == np.dtype("float32") and bool(np.isfinite(original).all()),
                    f"Source parameter is not finite F32: {name}")
            tensors[name] = np.array(original, dtype="<f4", order="C", copy=True)
            origins[name] = {"component": component, "node_index": node, "node_name": layer.name,
                             "parameter": parameter}

    nodes = list(nlp.get_pipe("tok2vec").model.walk())
    require(nodes[3].attrs["columns"] == FEATURES and nodes[2].attrs["pad"] == 4,
            "Unexpected feature order or encoder padding")
    for i, node in enumerate([28, 30, 32, 34, 36, 38]):
        require(nodes[node].name == "hashembed" and nodes[node].attrs["seed"] == 8 + i,
                "Unexpected hash embedding seed/graph")
        put(f"embed.{i}", "tok2vec", node, ["E"])
    put("embed.mix", "tok2vec", 39, ["W", "b"])
    put("embed.norm", "tok2vec", 40, ["G", "b"])
    for i, node in enumerate([57, 59, 61, 63]):
        require(nodes[41 + 2 * i].attrs["window_size"] == 1, "Unexpected encoder window")
        put(f"encoder.{i}.maxout", "tok2vec", node, ["W", "b"])
        put(f"encoder.{i}.norm", "tok2vec", node + 1, ["G", "b"])
    for name in ["tagger", "morphologizer", "lemmatizer"]:
        classifier = list(nlp.get_pipe(name).model.walk())[3]
        require(classifier.name == "softmax" and classifier.attrs["softmax_normalize"] is False and
                classifier.attrs["softmax_temperature"] == 1.0, "Unsupported classifier normalization")
        put(name, name, 3, ["W", "b"])
    parser = nlp.get_pipe("parser").model
    require(parser.attrs["has_upper"] is True and not parser.attrs["unseen_classes"],
            "Unsupported parser class mask/upper head")
    put("parser.reduce", "parser", 6, ["W", "b"])
    put("parser.lower", "parser", 2, ["W", "b", "pad"])
    put("parser.upper", "parser", 3, ["W", "b"])
    validate_tensors(tensors)
    return tensors, origins


def unicode_tables():
    # Freeze the actual Python Unicode tables used by this oracle. Ranges are
    # inclusive scalar code points; surrogate values are deliberately absent.
    predicates = ["isspace", "isalpha", "isupper", "islower", "isdigit"]
    ranges = {name: [] for name in predicates}
    starts = {name: None for name in predicates}
    lower = {}
    for cp in range(0x110000):
        scalar = not 0xD800 <= cp <= 0xDFFF
        ch = chr(cp)
        if scalar and ch.lower() != ch:
            lower[str(cp)] = ch.lower()
        for name in predicates:
            value = scalar and getattr(ch, name)()
            if value and starts[name] is None:
                starts[name] = cp
            elif not value and starts[name] is not None:
                ranges[name].append([starts[name], cp - 1])
                starts[name] = None
    for name in predicates:
        if starts[name] is not None:
            ranges[name].append([starts[name], 0x10FFFF])
    return {"unicode_version": unicodedata.unidata_version, "ranges_inclusive": ranges,
            "scalar_lower": lower,
            "lower_semantics": "CPython str.lower; contextual final sigma must also be implemented",
            "normalization": "none; preserve original UTF-8 source"}


def extract_resources(nlp, source: Path):
    import srsly
    from spacy.attrs import NORM, IDS
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
    morph = nlp.get_pipe("morphologizer")
    lemma = nlp.get_pipe("lemmatizer")
    trees = []
    for i in range(len(lemma.trees)):
        tree = dict(lemma.trees[i])
        for key in ["orig", "subst"]:
            if key in tree:
                value = tree[key]
                tree[key] = {"id": str(value), "text": nlp.vocab.strings[value]}
        trees.append(tree)
    parser = nlp.get_pipe("parser")
    labels = {
        "tagger": list(nlp.get_pipe("tagger").labels),
        "morphologizer": {"labels": list(morph.labels), "morph": morph.cfg["labels_morph"],
                          "pos": morph.cfg["labels_pos"], "overwrite": morph.cfg["overwrite"], "extend": morph.cfg["extend"]},
        "parser": {"moves": [parser.moves.get_class_name(i) for i in range(parser.moves.n_moves)],
                   "labels": list(parser.labels), "features": ["B0", "B1", "S0", "S1", "S2", "L(B0,1)", "L(S0,1)", "R(S0,1)"],
                   "cfg": parser.cfg, "moves_serialized_json": srsly.msgpack_loads((source / "parser/moves").read_bytes())},
        "lemmatizer": {"class_to_tree": list(lemma.labels), "trees": trees,
                       "top_k": lemma.top_k, "backoff": lemma.backoff, "overwrite": lemma.overwrite},
        "attribute_ruler": nlp.get_pipe("attribute_ruler").patterns,
    }
    require(len(labels["tagger"]) == 55 and len(morph.labels) == 671 and parser.moves.n_moves == 137
            and len(lemma.labels) == 1311 and len(trees) == 6092, "Changed label/tree inventory")
    require(lemma.top_k == 1 and lemma.backoff == "orth" and lemma.overwrite is False,
            "Unsupported lemma decoding")
    return {"format_version": FORMAT_VERSION, "language": "de", "component_order": COMPONENTS,
            "tokenizer": tokenizer, "norm_lookup_precedence": norm_tables,
            "unicode": unicode_tables(), "labels": labels,
            "lexical_semantics_source": {"revision": SOURCES["spacy"],
                "files": ["spacy/lang/lex_attrs.py", "spacy/strings.pyx", "spacy/tokenizer.pyx"]}}


def validate_resources(resources):
    """Validate the closed v1 data schema before publishing a converted bundle.

    This checks structure and cross-references, not asset authenticity. The
    authenticated source wheel and manifest resource hash bind actual labels,
    Unicode tables and language data to the trained tensor ordering.
    """
    import re

    def keys(value, expected, where):
        require(type(value) is dict and set(value) == set(expected),
                f"Unexpected resource fields: {where}")

    def text(value, where, *, empty=False, limit=65536):
        require(type(value) is str and (empty or bool(value)) and len(value) <= limit,
                f"Invalid resource text: {where}")
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise ValueError(f"Invalid Unicode scalar: {where}") from error

    def integer(value, maximum, where):
        require(type(value) is int and 0 <= value <= maximum, f"Invalid resource integer: {where}")

    def decimal(value, maximum, where):
        text(value, where, limit=20)
        require(value.isascii() and value.isdigit() and str(int(value)) == value,
                f"Noncanonical decimal identifier: {where}")
        integer(int(value), maximum, where)

    def string_list(value, count, where):
        require(type(value) is list and len(value) == count, f"Unexpected resource count: {where}")
        for item in value:
            text(item, where)
        require(len(set(value)) == count, f"Duplicate resource labels: {where}")

    keys(resources, {"format_version", "language", "component_order", "tokenizer",
                     "norm_lookup_precedence", "unicode", "labels", "lexical_semantics_source"}, "root")
    require(type(resources["format_version"]) is int and resources["format_version"] == FORMAT_VERSION
            and resources["language"] == "de" and resources["component_order"] == COMPONENTS,
            "Unsupported resource version/language/components")
    lexical_source = resources["lexical_semantics_source"]
    keys(lexical_source, {"revision", "files"}, "lexical source")
    require(lexical_source == {"revision": SOURCES["spacy"],
                              "files": ["spacy/lang/lex_attrs.py", "spacy/strings.pyx", "spacy/tokenizer.pyx"]},
            "Unsupported lexical source")

    tokenizer = resources["tokenizer"]
    keys(tokenizer, {"prefix_search", "suffix_search", "infix_finditer", "token_match",
                     "url_match", "exceptions", "faster_heuristics"}, "tokenizer")
    require(tokenizer["token_match"] is None and tokenizer["faster_heuristics"] is True,
            "Unsupported tokenizer matching policy")
    for name in ["prefix_search", "suffix_search", "infix_finditer", "url_match"]:
        text(tokenizer[name], name)
        try:
            re.compile(tokenizer[name])
        except re.error as error:
            raise ValueError(f"Malformed tokenizer pattern: {name}") from error
    exceptions = tokenizer["exceptions"]
    require(type(exceptions) is dict and len(exceptions) == 417, "Changed tokenizer exception count")
    for surface, entries in exceptions.items():
        text(surface, "exception surface", limit=4096)
        require(type(entries) is list and 1 <= len(entries) <= 3, "Invalid tokenizer exception pieces")
        for entry in entries:
            require(type(entry) is dict and set(entry) in ({"ORTH"}, {"ORTH", "NORM"}),
                    "Unsupported tokenizer exception attributes")
            for name, value in entry.items():
                text(value, f"exception {name}", limit=4096)
        require("".join(entry["ORTH"] for entry in entries) == surface,
                "Tokenizer exception does not reconstruct original text")
    norms = resources["norm_lookup_precedence"]
    require(type(norms) is list and len(norms) == 2, "Changed NORM lookup inventory")
    for table, kind, count in zip(norms, ["hash_uint64_decimal", "text"], [2, 45]):
        keys(table, {"key_kind", "entries"}, "NORM lookup")
        require(table["key_kind"] == kind and type(table["entries"]) is dict
                and len(table["entries"]) == count, "Changed NORM lookup order/count")
        for key, value in table["entries"].items():
            if kind == "hash_uint64_decimal":
                decimal(key, (1 << 64) - 1, "NORM key")
            else:
                text(key, "NORM key", limit=4096)
            text(value, "NORM value", limit=4096)

    unicode = resources["unicode"]
    keys(unicode, {"unicode_version", "ranges_inclusive", "scalar_lower", "lower_semantics", "normalization"},
         "Unicode")
    require(unicode["unicode_version"] == "15.0.0"
            and unicode["normalization"] == "none; preserve original UTF-8 source"
            and unicode["lower_semantics"] == "CPython str.lower; contextual final sigma must also be implemented",
            "Unsupported Unicode semantics")
    range_counts = {"isalpha": 659, "isdigit": 83, "islower": 671, "isspace": 10, "isupper": 651}
    keys(unicode["ranges_inclusive"], range_counts, "Unicode predicates")
    for name, count in range_counts.items():
        ranges = unicode["ranges_inclusive"][name]
        require(type(ranges) is list and len(ranges) == count, f"Changed Unicode range count: {name}")
        previous = -1
        for bounds in ranges:
            require(type(bounds) is list and len(bounds) == 2, "Invalid Unicode range")
            start, end = bounds
            integer(start, 0x10FFFF, "Unicode start")
            integer(end, 0x10FFFF, "Unicode end")
            require(previous < start <= end and (end < 0xD800 or start > 0xDFFF),
                    "Overlapping, unsorted or nonscalar Unicode range")
            previous = end
    lower = unicode["scalar_lower"]
    require(type(lower) is dict and len(lower) == 1433, "Changed Unicode lowercase count")
    for codepoint, lowered in lower.items():
        decimal(codepoint, 0x10FFFF, "lowercase key")
        require(not 0xD800 <= int(codepoint) <= 0xDFFF, "Nonscalar lowercase key")
        text(lowered, "lowercase value", limit=3)

    labels = resources["labels"]
    keys(labels, {"tagger", "morphologizer", "parser", "lemmatizer", "attribute_ruler"}, "labels")
    string_list(labels["tagger"], 55, "tagger")
    morph = labels["morphologizer"]
    keys(morph, {"labels", "morph", "pos", "overwrite", "extend"}, "morphologizer")
    string_list(morph["labels"], 671, "morphologizer")
    keys(morph["morph"], morph["labels"], "morphology map")
    keys(morph["pos"], morph["labels"], "POS map")
    require(morph["overwrite"] is True and morph["extend"] is False, "Unsupported morphology assignment policy")
    pos_ids = {"ADJ": 84, "ADP": 85, "ADV": 86, "AUX": 87, "CCONJ": 89, "DET": 90,
               "INTJ": 91, "NOUN": 92, "NUM": 93, "PART": 94, "PRON": 95, "PROPN": 96,
               "PUNCT": 97, "SCONJ": 98, "VERB": 100, "X": 101, "SPACE": 103}
    for label in morph["labels"]:
        fields = label.split("|")
        require(all(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*=[A-Za-z0-9]+(?:,[A-Za-z0-9]+)*", field)
                    for field in fields), "Malformed morphology label")
        names = [field.split("=", 1)[0] for field in fields]
        require(len(set(names)) == len(names) and names.count("POS") == 1,
                "Duplicate or missing morphology features/POS")
        pos = next(field.split("=", 1)[1] for field in fields if field.startswith("POS="))
        require(pos in pos_ids and type(morph["pos"][label]) is int and morph["pos"][label] == pos_ids[pos]
                and morph["morph"][label] == "|".join(field for field in fields if not field.startswith("POS=")),
                "Morphology/POS label mapping differs")

    parser = labels["parser"]
    keys(parser, {"moves", "labels", "features", "cfg", "moves_serialized_json"}, "parser")
    string_list(parser["moves"], 137, "parser moves")
    string_list(parser["labels"], 42, "dependency labels")
    require(parser["features"] == ["B0", "B1", "S0", "S1", "S2", "L(B0,1)", "L(S0,1)", "R(S0,1)"],
            "Unsupported parser feature order")
    cfg = {"beam_density": 0.0, "beam_update_prob": 0.0, "beam_width": 1,
           "incorrect_spans_key": None, "learn_tokens": False, "min_action_freq": 30,
           "moves": None, "multitasks": [], "update_with_oracle_cut_size": 100}
    keys(parser["cfg"], cfg, "parser config")
    require(parser["cfg"] == cfg and all(type(parser["cfg"][key]) is type(value) for key, value in cfg.items()),
            "Unsupported parser configuration")
    serialized = parser["moves_serialized_json"]
    keys(serialized, {"cfg", "moves"}, "serialized parser moves")
    require(serialized["cfg"] == {"neg_key": None}, "Unsupported parser negative-label configuration")
    text(serialized["moves"], "serialized parser moves")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate serialized parser key")
            result[key] = value
        return result

    try:
        actions = json.loads(serialized["moves"], object_pairs_hook=unique_object)
    except json.JSONDecodeError as error:
        raise ValueError("Malformed serialized parser moves") from error
    keys(actions, [str(i) for i in range(5)], "parser action types")
    ordered, dependencies = [], set()
    for i, action in enumerate(["S", "D", "L", "R", "B"]):
        moves = actions[str(i)]
        require(type(moves) is dict and moves, "Empty parser action group")
        if i in (0, 1, 4):
            require(list(moves) == (["ROOT"] if i == 4 else [""]), "Unsupported parser control moves")
        for label, count in moves.items():
            text(label, "parser action label", empty=i < 2)
            integer(count, (1 << 32) - 1, "parser move frequency")
        # TransitionSystem.initialize_actions reconstructs class order by
        # descending (frequency, label), including label order for ties.
        for count, label in sorted(((count, label) for label, count in moves.items()), reverse=True):
            ordered.append(action + ("-" + label if label else ""))
            if i >= 2:
                parts = label.split("||")
                require(1 <= len(parts) <= 2 and all(parts), "Malformed pseudo-projective dependency label")
                dependencies.update(parts)
    require(ordered == parser["moves"], "Parser move ordering differs from serialized actions")
    require(sorted(dependencies) == parser["labels"], "Parser dependency inventory differs")

    lemma = labels["lemmatizer"]
    keys(lemma, {"class_to_tree", "trees", "top_k", "backoff", "overwrite"}, "lemmatizer")
    require(type(lemma["top_k"]) is int and lemma["top_k"] == 1
            and lemma["backoff"] == "orth" and lemma["overwrite"] is False, "Unsupported lemma decoding")
    trees, classes = lemma["trees"], lemma["class_to_tree"]
    require(type(trees) is list and len(trees) == 6092 and type(classes) is list and len(classes) == 1311,
            "Changed edit-tree/class inventory")
    for tree_id in classes:
        integer(tree_id, len(trees) - 1, "lemma class tree")
    require(len(set(classes)) == len(classes), "Duplicate lemma class tree")
    children = []
    for tree in trees:
        require(type(tree) is dict, "Invalid edit-tree node")
        if set(tree) == {"orig", "subst"}:
            for name in ["orig", "subst"]:
                keys(tree[name], {"id", "text"}, "edit-tree string")
                decimal(tree[name]["id"], (1 << 64) - 1, "edit-tree string ID")
                text(tree[name]["text"], "edit-tree string", empty=True, limit=4096)
            children.append([])
        else:
            keys(tree, {"prefix_len", "prefix_tree", "suffix_len", "suffix_tree"}, "edit-tree match")
            refs = []
            for side in ["prefix", "suffix"]:
                integer(tree[side + "_len"], 4096, "edit-tree affix length")
                ref = tree[side + "_tree"]
                integer(ref, (1 << 32) - 1, "edit-tree child")
                require(ref < len(trees) or ref == (1 << 32) - 1, "Edit-tree child is out of range")
                if ref == (1 << 32) - 1:
                    require(tree[side + "_len"] == 0, "Missing edit-tree child has nonzero affix")
                else:
                    refs.append(ref)
            children.append(refs)
    states, depths = [0] * len(trees), [0] * len(trees)
    for first in range(len(trees)):
        stack = [(first, False)]
        while stack:
            node, finished = stack.pop()
            if finished:
                depths[node] = 1 + max((depths[child] for child in children[node]), default=0)
                require(depths[node] <= 64, "Edit-tree depth exceeds admission bound")
                states[node] = 2
            elif states[node] == 0:
                states[node] = 1
                stack.append((node, True))
                stack.extend((child, False) for child in children[node])
            else:
                require(states[node] == 2, "Cyclic edit-tree references")
    require(json_bytes(labels["attribute_ruler"]) == json_bytes([
        {"attrs": {"DEP": "dep"}, "index": 0,
         "patterns": [[{"DEP": {"NOT_IN": [""]}, "IS_SPACE": True}]]},
        {"attrs": {"MORPH": "_", "POS": "SPACE", "TAG": "_SP"}, "index": 0,
         "patterns": [[{"IS_SPACE": True}]]},
    ]), "Unsupported attribute-ruler rules/order")


def convert(wheel: Path, output: Path):
    verify_environment()
    verify_wheel(wheel)
    require(not output.exists(), "Output already exists; refusing to replace a frozen bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    import gguf
    import numpy as np
    with tempfile.TemporaryDirectory(prefix="spacy-convert-", dir=output.parent) as temporary:
        temp = Path(temporary)
        source, bundle = temp / "source", temp / "bundle"
        source.mkdir()
        bundle.mkdir()
        nlp = load_verified_model(wheel, source)
        tensors, origins = extract_tensors(nlp)
        resource_data = extract_resources(nlp, source)
        validate_resources(resource_data)
        resources = json_bytes(resource_data)
        (bundle / "resources.json").write_bytes(resources)
        for notice in ["LICENSE", "LICENSES_SOURCES"]:
            shutil.copyfile(source / notice, bundle / notice)
        writer = gguf.GGUFWriter(str(bundle / "model.gguf"), ARCHITECTURE)
        writer.add_uint32("spacy.format_version", FORMAT_VERSION)
        writer.add_string("spacy.model", "de_core_news_sm")
        writer.add_string("spacy.model_version", "3.8.0")
        writer.add_string("spacy.source.sha256", WHEEL_SHA256)
        writer.add_string("spacy.resources.sha256", digest(resources))
        writer.add_string("spacy.components", ",".join(COMPONENTS))
        writer.add_string("spacy.feature_order", ",".join(FEATURES))
        writer.add_string("spacy.precision", "f32")
        writer.add_uint32("spacy.width", 96)
        writer.add_uint32("spacy.encoder_depth", 4)
        writer.add_uint32("spacy.encoder_padding", 4)
        writer.add_float32("spacy.layer_norm_epsilon", 1e-8)
        for name in sorted(tensors):
            writer.add_tensor(name, tensors[name])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        # Independent container readback checks names, dtype, axes and all bytes.
        reader = gguf.GGUFReader(str(bundle / "model.gguf"))
        require(len(reader.tensors) == len(tensors), "GGUF round-trip tensor count mismatch")
        for tensor in reader.tensors:
            original = tensors[tensor.name]
            require(tensor.tensor_type == gguf.GGMLQuantizationType.F32 and
                    tuple(tensor.shape) == tuple(reversed(original.shape)) and
                    np.array_equal(tensor.data.reshape(original.shape), original),
                    f"GGUF round-trip mismatch: {tensor.name}")
        del reader
        manifest = {
            "format_version": FORMAT_VERSION, "architecture": ARCHITECTURE,
            "model": {"name": "de_core_news_sm", "version": "3.8.0", "language": "de"},
            "source": {"file": WHEEL_NAME, "size_bytes": WHEEL_SIZE, "sha256": WHEEL_SHA256,
                       "url": "https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/" + WHEEL_NAME},
            "reference_versions": ORACLE, "reference_python": "CPython 3.12.14", "source_revisions": SOURCES,
            "component_order": COMPONENTS, "omitted_components": ["ner", "senter"],
            "declared_bundle_capabilities": ["tokens", "tags", "pos", "morphology", "lemmas", "dependencies", "sentences"],
            "runtime_qualification": "not established by conversion",
            "layout": "source C row-major F32; GGUF dimensions are reversed, bytes are not transposed",
            "numeric_contract": {"features": FEATURES, "embedding_rows": ROWS, "embedding_seeds": list(range(8, 14)),
                "hash_buckets_per_feature": 4, "hash_reduction": "sum including repeated buckets",
                "width": 96, "depth": 4, "window": 1, "maxout_pieces": 3,
                "encoder_padding": "four zero rows per side once, evolving across all layers",
                "layer_norm_epsilon": 1e-8, "layer_norm_axis": 1, "scores": "unnormalized affine",
                "atol": 2e-4, "rtol": 2e-5, "decisions": "exact, first maximum on ties"},
            "limits": {"bundle_bytes": 32 * 1024 * 1024, "prepared_tokens": 2048,
                       "note": "engineering admission bounds; not device performance qualification"},
            "tensors": [{"name": name, "shape": list(tensors[name].shape),
                         "gguf_shape": list(reversed(tensors[name].shape)), "dtype": "float32-le",
                         "size_bytes": tensors[name].nbytes, "sha256": digest(tensors[name].tobytes()),
                         "source": origins[name]} for name in sorted(tensors)],
            "source_resources": {p.relative_to(source).as_posix(): {"size_bytes": p.stat().st_size,
                                     "sha256": digest(p.read_bytes())} for p in sorted(source.rglob("*")) if p.is_file()},
            "files": {name: {"size_bytes": (bundle / name).stat().st_size,
                               "sha256": digest((bundle / name).read_bytes())}
                      for name in ["model.gguf", "resources.json", "LICENSE", "LICENSES_SOURCES"]},
            "converter_sha256": digest(Path(__file__).read_bytes()),
        }
        (bundle / "manifest.json").write_bytes(json_bytes(manifest))
        os.rename(bundle, output)
        return manifest
