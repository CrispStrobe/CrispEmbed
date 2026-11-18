#!/usr/bin/env python3
"""Python/ctypes twin of tools/spacy/spacy_smoke.cpp for prebuilt libraries.

Loads an already built (e.g. published) libcrispembed-spacy through the C ABI
only, enables it with crispembed_spacy_set_enabled (no environment change),
opens a bundle with crispembed_spacy_create_with_resources and writes the same
JSON Lines as spacy_smoke, so tests/spacy_smoke_compare.py can compare the
output with original spaCy. Needs no compiler, so it runs on every desktop
platform against the exact published library bytes.

    python tests/spacy_ctypes_smoke.py --library LIB --bundle DIR --texts texts.txt --output smoke.jsonl
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys

TEXT_FIELDS = ("text", "norm", "prefix", "suffix", "shape", "lemma", "tag", "pos", "morph", "dependency")
SENTENCE_START = 4


class TextView(C.Structure):
    _fields_ = [("data", C.c_void_p), ("length", C.c_size_t)]

    def value(self):
        return C.string_at(self.data, self.length).decode("utf-8") if self.length else ""


class TokenView(C.Structure):
    _fields_ = ([(name, TextView) for name in TEXT_FIELDS]
                + [(name, C.c_size_t) for name in ("byte_start", "byte_end", "scalar_start", "scalar_end",
                                                   "utf16_start", "utf16_end")]
                + [("orth_id", C.c_uint64), ("features", C.c_uint64 * 6), ("head", C.c_int32),
                   ("flags", C.c_uint32)])


class EntityView(C.Structure):
    _fields_ = [("label", TextView)] + [(name, C.c_size_t) for name in (
        "token_start", "token_end", "byte_start", "byte_end", "scalar_start", "scalar_end", "utf16_start",
        "utf16_end")]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def open_library(path):
    lib = C.CDLL(str(path))
    lib.crispembed_spacy_set_enabled.argtypes = [C.c_uint32]
    lib.crispembed_spacy_set_enabled.restype = C.c_uint32
    lib.crispembed_spacy_create_with_resources.restype = C.c_void_p
    lib.crispembed_spacy_create_with_resources.argtypes = [C.c_char_p] * 5 + [C.c_size_t]
    lib.crispembed_spacy_context_capabilities.argtypes = [C.c_void_p]
    lib.crispembed_spacy_context_capabilities.restype = C.c_uint32
    lib.crispembed_spacy_free.argtypes = [C.c_void_p]
    lib.crispembed_spacy_analyze_utf8.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t, C.POINTER(C.c_void_p),
                                                  C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_char_p,
                                                  C.c_size_t]
    lib.crispembed_spacy_document_token_count.argtypes = [C.c_void_p]
    lib.crispembed_spacy_document_token_count.restype = C.c_size_t
    lib.crispembed_spacy_document_token.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(TokenView)]
    lib.crispembed_spacy_document_entity_count.argtypes = [C.c_void_p]
    lib.crispembed_spacy_document_entity_count.restype = C.c_size_t
    lib.crispembed_spacy_document_entity.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(EntityView)]
    lib.crispembed_spacy_document_free.argtypes = [C.c_void_p]
    return lib


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True, help="One UTF-8 text per line")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lib = open_library(args.library)
    lib.crispembed_spacy_set_enabled(1)
    error = C.create_string_buffer(1024)
    model, resources = args.bundle / "model.gguf", args.bundle / "resources.json"
    context = lib.crispembed_spacy_create_with_resources(str(model).encode(), sha256(model).encode(),
                                                         str(resources).encode(), sha256(resources).encode(),
                                                         error, 1024)
    if not context:
        print(f"create failed: {error.value.decode(errors='replace')}", file=sys.stderr)
        return 3
    status = 0
    try:
        with args.output.open("w", encoding="utf-8", newline="\n") as out:
            for line in args.texts.read_text(encoding="utf-8").split("\n"):
                if not line:
                    continue
                data = line.encode("utf-8")
                document = C.c_void_p()
                code = lib.crispembed_spacy_analyze_utf8(context, data, len(data), C.byref(document), None, None,
                                                         None, None, error, 1024)
                if code != 0:
                    print(f"analyze failed ({code}): {error.value.decode(errors='replace')}", file=sys.stderr)
                    status = 4
                    break
                try:
                    tokens = []
                    for i in range(lib.crispembed_spacy_document_token_count(document)):
                        view = TokenView()
                        assert lib.crispembed_spacy_document_token(document, i, C.byref(view)) == 0
                        tokens.append({"text": view.text.value(), "tag": view.tag.value(), "pos": view.pos.value(),
                                       "morph": view.morph.value(), "lemma": view.lemma.value(),
                                       "dep": view.dependency.value(), "head": view.head,
                                       "sent_start": bool(view.flags & SENTENCE_START)})
                    entities = []
                    for i in range(lib.crispembed_spacy_document_entity_count(document)):
                        view = EntityView()
                        assert lib.crispembed_spacy_document_entity(document, i, C.byref(view)) == 0
                        entities.append([view.token_start, view.token_end, view.label.value()])
                finally:
                    lib.crispembed_spacy_document_free(document)
                out.write(json.dumps({"text": line, "tokens": tokens, "entities": entities},
                                     ensure_ascii=False) + "\n")
    finally:
        lib.crispembed_spacy_free(context)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
