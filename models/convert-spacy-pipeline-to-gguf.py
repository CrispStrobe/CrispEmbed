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
    "fr_dep_news_trf-3.8.0-py3-none-any.whl": ("df243e539bbc365390744a411aa574fb03eb2ee2f948cf6149ad722c59ab419b", 397740844),
    "en_core_web_trf-3.8.0-py3-none-any.whl": ("272a31e9d8530d1e075351d30a462d7e80e31da23574f1b274e200f3fff35bf5", 457421864),
    "de_dep_news_trf-3.8.0-py3-none-any.whl": ("6da411d41d5b0465beb1725549039e05caf00fec9ae1dc41794c8f7555824905", 410449584),
}
SUPPORTED_COMPONENTS = {"tok2vec", "transformer", "tagger", "morphologizer", "parser", "attribute_ruler",
                        "lemmatizer", "ner"}
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
                require(info.file_size < 1024 * 1024 * 1024, "Oversized model resource")
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
    """TransitionBasedParser.v2: reduce Linear -> PrecomputableAffine (nF x nO x
    nP) -> optional upper Linear. Without the upper layer (spaCy trf parsers)
    the maxout output is the class score vector itself (nO == n_moves)."""
    component = nlp.get_pipe(name)
    model = component.model
    has_upper = model.attrs["has_upper"] is True
    require(not model.attrs["unseen_classes"], f"{name}: class mask")
    require(component.cfg.get("beam_width", 1) == 1, f"{name}: only greedy decoding")
    nodes = walk(model)
    lower = [i for i, n in enumerate(nodes) if n.name == "precomputable_affine"]
    linears = [i for i, n in enumerate(nodes) if n.name == "linear"]
    require(len(lower) == 1 and len(linears) == (2 if has_upper else 1), f"{name}: unexpected transition graph")
    moves = [component.moves.get_class_name(i) for i in range(component.moves.n_moves)]
    reduce_index = linears[-1]
    put(tensors, origins, f"{prefix}reduce", nodes[reduce_index], ["W", "b"], name, reduce_index)
    put(tensors, origins, f"{prefix}lower", nodes[lower[0]], ["W", "b", "pad"], name, lower[0])
    shape = nodes[lower[0]].get_param("W").shape
    require(int(shape[0]) == features, f"{name}: state feature count")
    if has_upper:
        put(tensors, origins, f"{prefix}upper", nodes[linears[0]], ["W", "b"], name, linears[0])
    else:
        require(int(shape[1]) == len(moves), f"{name}: upper-less parser must score every move")
    return moves, has_upper


UNIVERSAL_POS = ["noun", "verb", "adj", "adv", "propn", "num", "pron", "det", "adp", "aux", "cconj", "sconj",
                 "part", "intj", "punct", "sym", "x", "space", "eol"]


def export_transformer(nlp, tensors, origins):
    """spacy-curated-transformers RoBERTa-family post-LN encoders with
    WithStridedSpans: CamemBERT + CamemBERT SentencePiece encoder, or RoBERTa +
    GPT-2 byte-level BPE encoder."""
    import base64
    import numpy as np
    component = nlp.get_pipe("transformer")
    cfg = nlp.config["components"]["transformer"]["model"]
    kinds = {("spacy-curated-transformers.CamembertTransformer.v1",
              "spacy-curated-transformers.CamembertSentencepieceEncoder.v1"): ("camembert", "camembert_sentencepiece"),
             ("spacy-curated-transformers.RobertaTransformer.v1",
              "spacy-curated-transformers.ByteBpeEncoder.v1"): ("roberta", "roberta_byte_bpe"),
             ("spacy-curated-transformers.BertTransformer.v1",
              "spacy-curated-transformers.BertWordpieceEncoder.v1"): ("bert", "bert_wordpiece")}
    family = kinds.get((cfg["@architectures"], cfg["piece_encoder"]["@architectures"]))
    require(family is not None, "Unsupported transformer/piece encoder")
    spans = cfg["with_spans"]
    require(spans["@architectures"] == "spacy-curated-transformers.WithStridedSpans.v1", "Unsupported span extractor")
    require(cfg["hidden_act"] == "gelu" and cfg["type_vocab_size"] == (2 if family[0] == "bert" else 1),
            "Unsupported transformer activation/types")
    for name in nlp.pipe_names[1:]:
        tok2vec = (nlp.config["components"][name].get("model") or {}).get("tok2vec")
        if tok2vec is None and name == "parser":
            raise ValueError("parser without a transformer listener")
        if tok2vec is not None:
            require(tok2vec["@architectures"] == "spacy-curated-transformers.LastTransformerLayerListener.v1" and
                    tok2vec["pooling"]["@layers"] == "reduce_mean.v1" and tok2vec["upstream"] == "transformer",
                    f"{name}: unsupported transformer listener")
    state = component.model.get_ref("transformer").shims[0]._model.state_dict()
    layers = cfg["num_hidden_layers"]
    prefix = "curated_encoder."
    # RoBERTa-family embeddings wrap BertEmbeddings as ".inner".
    inner = "" if family[0] == "bert" else "inner."
    mapping = {f"embeddings.{inner}word_embeddings.weight": "trf.embed.word",
               f"embeddings.{inner}position_embeddings.weight": "trf.embed.position",
               f"embeddings.{inner}token_type_embeddings.weight": "trf.embed.type",
               f"embeddings.{inner}layer_norm.weight": "trf.embed.norm.weight",
               f"embeddings.{inner}layer_norm.bias": "trf.embed.norm.bias"}
    for i in range(layers):
        for source, target in (("mha.input", "qkv"), ("mha.output", "attn_out"),
                               ("attn_output_layernorm", "attn_norm"), ("ffn.intermediate", "ffn_in"),
                               ("ffn.output", "ffn_out"), ("ffn_output_layernorm", "ffn_norm")):
            for kind in ("weight", "bias"):
                mapping[f"layers.{i}.{source}.{kind}"] = f"trf.layer.{i}.{target}.{kind}"
    require(set(state) == {prefix + key for key in mapping}, "Unexpected transformer state dict")
    for key, target in mapping.items():
        value = state[prefix + key].detach().cpu().numpy()
        require(value.dtype == np.dtype("float32") and bool(np.isfinite(value).all()), f"Not finite F32: {key}")
        tensors[target] = np.array(value, dtype="<f4", order="C", copy=True)
        origins[target] = {"component": "transformer", "state_dict": prefix + key}
    result = {"architecture": family[0], "piece_encoder": family[1],
            "layers": layers, "width": cfg["hidden_width"], "heads": cfg["num_attention_heads"],
            "intermediate": cfg["intermediate_width"], "layer_norm_eps": cfg["layer_norm_eps"],
            "padding_idx": cfg["padding_idx"], "max_positions": cfg["max_position_embeddings"],
            "model_max_length": cfg["model_max_length"], "window": spans["window"], "stride": spans["stride"],
            "listener": "last_layer_reduce_mean"}
    if family[0] == "bert":
        # BertEmbeddings: positions 0..T-1, token type 0 (row 0 of the table).
        result["position_offset"] = 0
        encoder = component.model.get_ref("piece_encoder")
        from spacy_curated_transformers.tokenization import wordpiece_encoder as wpe
        require(encoder.attrs["preprocess"] is wpe._bert_preprocess and encoder.attrs["lowercase"] is False and
                encoder.attrs["strip_accents"] is False, "Unsupported WordPiece preprocessing")
        wpp = encoder.attrs["wordpiece_processor"]
        pieces = wpp.to_list()
        require("##" not in pieces, "Empty continuation WordPiece piece (loops in the original)")
        result["wordpiece"] = {"pieces": pieces,
                               "bos": wpp.get_initial(encoder.attrs["bos_piece"]),
                               "eos": wpp.get_initial(encoder.attrs["eos_piece"]),
                               "unk": wpp.get_initial(encoder.attrs["unk_piece"])}
    elif family[0] == "camembert":
        spp = component.model.get_ref("piece_encoder").get_ref("encoder").attrs["sentencepiece_processor"]
        result["sentencepiece_model_base64"] = base64.b64encode(spp.to_protobuf()).decode("ascii")
    else:
        encoder = component.model.get_ref("piece_encoder")
        bbp = encoder.attrs["byte_bpe_processor"]
        vocab = bbp.vocab
        require(sorted(vocab.values()) == list(range(len(vocab))), "Byte-BPE vocabulary ids are not dense")
        pieces = [None] * len(vocab)
        for piece, index in vocab.items():
            pieces[index] = piece
        merges = [list(pair) for pair in bbp.merges]
        require(len({tuple(m) for m in merges}) == len(merges), "Duplicate byte-BPE merges")
        result["byte_bpe"] = {"pieces": pieces, "merges": merges,
                              "split_pattern": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                              "bos": bbp.piece_id(encoder.attrs["bos_piece"]),
                              "eos": bbp.piece_id(encoder.attrs["eos_piece"]),
                              "unk": bbp.piece_id(encoder.attrs["unk_piece"])}
        import curated_tokenizers._bbpe as _bbpe
        require(result["byte_bpe"]["split_pattern"] == _bbpe.SPLIT_PATTERN, "Byte-BPE split pattern changed")
    return result


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
    require(pipes[0] in ("tok2vec", "transformer") and "parser" in pipes, "Expected tok2vec/transformer and a parser")
    if pipes[0] == "transformer":
        labels["transformer"] = export_transformer(nlp, tensors, origins)
        rows = []
    else:
        rows = export_tok2vec(nlp, nlp.get_pipe("tok2vec").model, "", "tok2vec", TOK2VEC_FEATURES, tensors, origins)
    for name in ("tagger", "morphologizer"):
        if name in pipes:
            export_classifier(nlp, name, tensors, origins)
    if "tagger" in pipes:
        labels["tagger"] = list(nlp.get_pipe("tagger").labels)
    if "morphologizer" in pipes:
        from spacy.symbols import NAMES
        from spacy.tokens import MorphAnalysis
        morph = nlp.get_pipe("morphologizer")
        require(morph.cfg["overwrite"] is True and morph.cfg["extend"] is False,
                "Only overwrite/no-extend morphologizers are supported")
        # Per label: POS symbol name and canonical morphology (as assigned).
        labels["morphologizer"] = {
            "labels": list(morph.labels), "overwrite": True, "extend": False,
            "assign": [{"pos": NAMES[morph.cfg["labels_pos"][label]] if morph.cfg["labels_pos"][label] else "",
                        "morph": str(MorphAnalysis(nlp.vocab, morph.cfg["labels_morph"][label]))}
                       for label in morph.labels]}
    parser_moves, parser_upper = export_transition(nlp, "parser", "parser.", tensors, origins, 8)
    labels["parser"] = {"moves": parser_moves,
                        "learn_tokens": nlp.get_pipe("parser").cfg.get("learn_tokens", False),
                        "features": ["B0", "B1", "S0", "S1", "S2", "L(B0,1)", "L(S0,1)", "R(S0,1)"]}
    if not parser_upper:
        # Only written when False, so CNN bundles stay byte-identical to the
        # published v2 bundles (the runtime treats a missing key as True).
        labels["parser"]["has_upper"] = False
    require(labels["parser"]["learn_tokens"] is False, "Subtoken learning parser unsupported")
    if "ner" in pipes:
        if pipes[0] == "transformer":
            # NER listens to the shared transformer (no tok2vec of its own).
            ner_moves, ner_upper = export_transition(nlp, "ner", "ner.", tensors, origins, 3)
            labels["ner"] = {"moves": ner_moves, "listener": "transformer"}
            if not ner_upper:
                labels["ner"]["has_upper"] = False
        else:
            ner_rows = export_tok2vec(nlp, nlp.get_pipe("ner").model, "ner.", "ner", NER_FEATURES, tensors, origins)
            require(ner_rows == rows[:4], "NER embedding rows differ from the shared tok2vec rows")
            ner_moves, ner_upper = export_transition(nlp, "ner", "ner.", tensors, origins, 3)
            require(ner_upper, "NER without an upper layer is unsupported")
            labels["ner"] = {"moves": ner_moves}
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
            require(lemma.top_k == 1 and lemma.backoff == "orth" and lemma.overwrite is False,
                    "Unsupported edit-tree lemmatizer settings")
            export_classifier(nlp, "lemmatizer", tensors, origins)
            trees = []
            for i in range(len(lemma.trees)):
                tree = dict(lemma.trees[i])
                for key in ("orig", "subst"):
                    if key in tree:
                        tree[key] = nlp.vocab.strings[tree[key]]
                trees.append(tree)
            labels["lemmatizer"] = {"mode": "edit_tree", "class_to_tree": list(lemma.labels), "trees": trees,
                                    "top_k": 1, "backoff": "orth", "overwrite": False}
        else:
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
    if tokenizer["token_match"] is not None:
        tokenizer["token_match"] = split_alternation(tokenizer["token_match"])
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


def split_alternation(pattern: str):
    """Split `(?flags)(?:a)|(?:b)|...` into its top-level alternatives.

    spaCy calls token_match with re.match, so the whole pattern matches iff
    one alternative matches at position 0. Separate alternatives stay below
    PCRE2's compiled-pattern size limit (French: one 1.45 MB alternation).
    Rejoining must reproduce the pattern exactly, and every part must compile.
    """
    import re
    flags = re.match(r"\(\?([a-zA-Z]+)\)", pattern)
    require(flags is not None, "token_match without a leading flag group")
    inner, parts, depth, in_class, start, i = pattern[flags.end():], [], 0, False, 0, 0
    while i < len(inner):
        c = inner[i]
        if c == "\\":
            i += 2
            continue
        if in_class:
            in_class = c != "]"
        elif c == "[":
            in_class = True
            if inner[i + 1:i + 2] == "]":
                i += 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "|" and depth == 0:
            parts.append(inner[start:i])
            start = i + 1
        i += 1
    parts.append(inner[start:])
    require(depth == 0 and not in_class and flags.group(0) + "|".join(parts) == pattern,
            "token_match split does not reproduce the pattern")
    for part in parts:
        re.compile(flags.group(0) + part)
    return {"flags": flags.group(1), "alternatives": parts}


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
