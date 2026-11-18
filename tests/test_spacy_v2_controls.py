#!/usr/bin/env python3
"""Admission and API controls for version-2 spaCy pipelines (English/French sm).

    python tests/test_spacy_v2_controls.py --library LIB --bundle bundle-en [--german-bundle bundle]

Requires real converted bundles; no quality claim (see tests/spacy_smoke_compare.py).
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--german-bundle", type=Path)
    args = parser.parse_args()
    lib = C.CDLL(str(args.library))
    lib.crispembed_spacy_set_enabled.argtypes = [C.c_uint32]
    lib.crispembed_spacy_create_with_resources.restype = C.c_void_p
    lib.crispembed_spacy_create_with_resources.argtypes = [C.c_char_p] * 5 + [C.c_size_t]
    lib.crispembed_spacy_context_capabilities.argtypes = [C.c_void_p]
    lib.crispembed_spacy_free.argtypes = [C.c_void_p]
    lib.crispembed_spacy_forward_features.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.c_size_t, C.c_size_t,
                                                      C.POINTER(C.c_void_p), C.c_void_p, C.c_void_p, C.c_void_p,
                                                      C.c_void_p, C.c_char_p, C.c_size_t]
    lib.crispembed_spacy_attach_ner.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_char_p, C.c_size_t]
    error = C.create_string_buffer(512)
    model, resources = args.bundle / "model.gguf", args.bundle / "resources.json"

    def create(model_path, model_sha, resources_path, resources_sha):
        return lib.crispembed_spacy_create_with_resources(str(model_path).encode(), model_sha,
                                                          str(resources_path).encode(), resources_sha, error, 512)

    results = {}
    lib.crispembed_spacy_set_enabled(0)
    results["disabled_without_gate"] = not create(model, sha(model), resources, sha(resources))
    lib.crispembed_spacy_set_enabled(1)
    results["wrong_model_hash_rejected"] = not create(model, b"0" * 64, resources, sha(resources))
    results["wrong_resource_hash_rejected"] = not create(model, sha(model), resources, b"0" * 64)
    if args.german_bundle:
        german = args.german_bundle / "resources.json"
        results["foreign_resources_rejected"] = not create(model, sha(model), german, sha(german))
        german_model = args.german_bundle / "model.gguf"
        results["v1_model_with_v2_resources_rejected"] = not create(german_model, sha(german_model), resources,
                                                                    sha(resources))
    context = create(model, sha(model), resources, sha(resources))
    results["admitted"] = bool(context)
    if context:
        caps = lib.crispembed_spacy_context_capabilities(context)
        results["text_only_capabilities"] = caps & 3 == 0 and caps & 12 == 12
        result = C.c_void_p()
        features = (C.c_uint64 * 6)()
        status = lib.crispembed_spacy_forward_features(context, features, 1, 6, C.byref(result), None, None, None,
                                                       None, error, 512)
        results["numeric_service_rejected"] = status == 2 and not result
        results["attach_ner_rejected"] = lib.crispembed_spacy_attach_ner(context, str(model).encode(), sha(model),
                                                                         error, 512) != 0
        lib.crispembed_spacy_free(context)
    print(json.dumps(results, indent=1))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
