#!/usr/bin/env python3
"""crispembed_spacy_set_enabled(): process-wide enablement without setenv.

Requires a built library and the converted bundle; no inference beyond one
short analysis. Run in a fresh process with CRISPEMBED_SPACY/_NER unset.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_spacy_text_api import NativeAPI, validate_bundle  # noqa: E402

ANALYSIS, NER = 1, 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--ner", type=Path, help="Optional ner.gguf for the NER flag control")
    args = parser.parse_args()
    for name in ("CRISPEMBED_SPACY", "CRISPEMBED_SPACY_NER"):
        os.environ.pop(name, None)
    api = NativeAPI(args.library, args.bundle, validate_bundle(args.bundle))
    lib = api.lib
    lib.crispembed_spacy_set_enabled.argtypes = [C.c_uint32]
    lib.crispembed_spacy_set_enabled.restype = C.c_uint32
    results = {}
    context, error = api.open()
    results["disabled_without_flag_or_env"] = context is None and "disabled" in error.lower()
    results["previous_flags_initially_zero"] = lib.crispembed_spacy_set_enabled(ANALYSIS) == 0
    context, error = api.open()
    assert context, error
    status, document, error = api.call(context, "Angela Merkel war in Berlin.".encode(), analyze=True)
    results["analysis_with_flag"] = status == 0 and len(api.tokens(document)) == 6
    lib.crispembed_spacy_document_free(document)
    if args.ner:
        lib.crispembed_spacy_attach_ner.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_char_p, C.c_size_t]
        sha = hashlib.sha256(args.ner.read_bytes()).hexdigest().encode()
        message = C.create_string_buffer(512)
        results["ner_refused_without_ner_flag"] = lib.crispembed_spacy_attach_ner(
            context, os.fsencode(args.ner), sha, message, 512) == 1
        results["previous_flags_returned"] = lib.crispembed_spacy_set_enabled(ANALYSIS | NER) == ANALYSIS
        results["ner_attached_with_flag"] = lib.crispembed_spacy_attach_ner(
            context, os.fsencode(args.ner), sha, message, 512) == 0
    results["unknown_bits_ignored"] = (lib.crispembed_spacy_set_enabled(0xFFFF0000 | ANALYSIS) & ~3) == 0
    results["cleared_flag_disables_calls"] = lib.crispembed_spacy_set_enabled(0) != 0
    status, document, _ = api.call(context, b"Test.", analyze=True)
    results["calls_disabled_after_clear"] = status == 1 and not document
    os.environ["CRISPEMBED_SPACY"] = "1"
    status, document, _ = api.call(context, b"Test.", analyze=True)
    results["environment_gate_still_works"] = status == 0
    lib.crispembed_spacy_document_free(document)
    lib.crispembed_spacy_free(context)
    print(json.dumps(results, indent=1))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
