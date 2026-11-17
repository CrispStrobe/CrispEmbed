#!/usr/bin/env python3
"""Required real-model prepared-feature parity; missing assets are failures.

This developer comparator does not tokenize, decode parses, or claim linguistic
analysis. Native outputs must come through the compiled C ABI and are compared
to every frozen padded stage and the scorer on every recorded parser state.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL = 2e-4, 2e-5
REFERENCE_SHA256 = "9a3bb6bae6114adcf579840834ddcb64dea4e0aaa8361718dc7a551d663c8f7c"
TRACE = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_char_p, C.POINTER(C.c_float), C.POINTER(C.c_size_t), C.c_size_t)
CANCEL = C.CFUNCTYPE(C.c_int, C.c_void_p)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON number: {value}")))


def validate_inputs(args):
    for name in [args.library, args.bundle / "manifest.json", args.quds_root / "tools/proofing/spacy/verify_reference.py"]:
        require(name.is_file(), f"Required asset is missing: {name.name}")
    reference = args.reference or args.quds_root / "tools/proofing/spacy/fixtures/development"
    cases = args.quds_root / "tools/proofing/spacy/fixtures/cases.json"
    spec = importlib.util.spec_from_file_location("spacy_reference_integrity", args.quds_root / "tools/proofing/spacy/verify_reference.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(args.quds_root / "tools/proofing/spacy"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    require((reference / "manifest.json").is_file() and sha256(reference / "manifest.json") == REFERENCE_SHA256,
            "Reference manifest differs from the remotely frozen W1 development oracle")
    verified = module.verify(reference, cases)
    require(verified["cases"] > 0 and verified["tokens"] > 0 and verified["parser_steps"] > 0,
            "Nonempty actual model/reference denominators are required")
    manifest = read_json(args.bundle / "manifest.json")
    require(manifest["format_version"] == 1 and manifest["architecture"] == "spacy_de_core_news_sm",
            "Wrong model bundle architecture/version")
    require(manifest["source"]["sha256"] == "fec69fec52b1780f2d269d5af7582a5e28028738bd3190532459aeb473bfa3e7",
            "Wrong source model identity")
    require(set(manifest["files"]) == {"model.gguf", "resources.json", "LICENSE", "LICENSES_SOURCES"},
            "Incomplete converted bundle")
    for name, entry in manifest["files"].items():
        path = args.bundle / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == entry["size_bytes"]
                and sha256(path) == entry["sha256"], f"Changed/missing converted asset: {name}")
    # Trust comes from the source-pinned W1 receipt, not a self-asserted manifest.
    pinned = read_json(ROOT / "docs/validation/native-spacy-20261117/bundle-manifest.json")
    require(manifest == pinned, "Bundle differs from the externally retained W1 manifest")
    reference_manifest = read_json(reference / "manifest.json")
    policy = reference_manifest["numerical_policy"]
    require(policy.get("absolute_tolerance") == ATOL and policy.get("relative_tolerance") == RTOL,
            "Reference numerical budget drift")
    require(reference_manifest["split"] == "development", "This driver is the development stage gate; acceptance needs an explicit later protocol")
    return reference, reference_manifest, manifest, verified


class Native:
    def __init__(self, library, model_path, model_hash):
        os.environ["CRISPEMBED_SPACY"] = "1"  # Explicit opt-in for this developer process only.
        self.lib = C.CDLL(str(library.resolve()))
        lib = self.lib
        lib.crispembed_spacy_abi_version.restype = C.c_uint32
        lib.crispembed_spacy_capabilities.restype = C.c_uint32
        require(lib.crispembed_spacy_abi_version() == 1 and lib.crispembed_spacy_capabilities() == 3,
                "Unexpected ABI/capabilities; numeric qualification cannot imply full analysis")
        lib.crispembed_spacy_create.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p, C.c_size_t]
        lib.crispembed_spacy_create.restype = C.c_void_p
        lib.crispembed_spacy_free.argtypes = [C.c_void_p]
        lib.crispembed_spacy_free.restype = None
        lib.crispembed_spacy_result_free.argtypes = [C.c_void_p]
        lib.crispembed_spacy_result_free.restype = None
        lib.crispembed_spacy_forward_features.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.c_size_t,
            C.c_size_t, C.POINTER(C.c_void_p), TRACE, C.c_void_p, CANCEL, C.c_void_p, C.c_char_p, C.c_size_t]
        lib.crispembed_spacy_forward_features.restype = C.c_int
        lib.crispembed_spacy_result_matrix.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.POINTER(C.c_float)),
                                                       C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
        lib.crispembed_spacy_result_matrix.restype = C.c_int
        lib.crispembed_spacy_parser_scores.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_int32), C.c_size_t,
            C.POINTER(C.c_float), C.c_size_t, TRACE, C.c_void_p, CANCEL, C.c_void_p, C.c_char_p, C.c_size_t]
        lib.crispembed_spacy_parser_scores.restype = C.c_int
        self.error = C.create_string_buffer(2048)
        started = time.perf_counter()
        self.context = lib.crispembed_spacy_create(os.fsencode(model_path), model_hash.encode("ascii"), self.error, len(self.error))
        self.open_seconds = time.perf_counter() - started
        require(self.context, "Native model admission failed: " + self.error.value.decode("utf-8", "replace"))

    def close(self):
        if self.context:
            self.lib.crispembed_spacy_free(self.context)
            self.context = None

    def run_case(self, case):
        captured, callback_errors = {}, []
        current_step = [None]

        @TRACE
        def callback(_user, encoded_name, values, shape, rank):
            try:
                require(rank <= 4, "Invalid native trace rank")
                dims = [int(shape[i]) for i in range(rank)]
                require(all(0 <= value <= 1000000 for value in dims), "Invalid native trace dimensions")
                count = int(np.prod(dims, dtype=np.int64))
                require(count <= 16 * 1024 * 1024, "Oversized native trace")
                name = encoded_name.decode("utf-8")
                if current_step[0] is not None:
                    require(name in {"parser.sum", "parser.hidden", "parser.scores"}, "Unexpected parser trace")
                    name = f"parser.step.{current_step[0]}." + name.split(".", 1)[1]
                require(name not in captured, f"Duplicate native stage: {name}")
                require(bool(values) or count == 0, "Null native trace payload")
                array = np.ctypeslib.as_array(values, shape=(count,)).copy() if count else np.empty((0,), dtype=np.float32)
                captured[name] = array.reshape(dims)
                return 0
            except Exception as error:
                callback_errors.append(str(error))
                return 1

        ids = np.asarray([[int(value) for value in row] for row in case["features"]["uint64_decimal"]], dtype=np.uint64)
        tokens = len(ids)
        require(ids.size == tokens * 6, "Incomplete lexical feature rows")
        result = C.c_void_p()
        started = time.perf_counter()
        status = self.lib.crispembed_spacy_forward_features(self.context, ids.ctypes.data_as(C.POINTER(C.c_uint64)),
            tokens, ids.size, C.byref(result), callback, None, CANCEL(), None, self.error, len(self.error))
        require(status == 0 and result.value and not callback_errors,
                f"Native forward failed ({status}): {self.error.value!r}; callback={callback_errors}")
        decisions = {}
        parser_decisions = []
        try:
            for name, columns in [("tok2vec", 96), ("tagger", 55), ("morphologizer", 671), ("lemmatizer", 1311)]:
                values, rows, cols = C.POINTER(C.c_float)(), C.c_size_t(), C.c_size_t()
                status = self.lib.crispembed_spacy_result_matrix(result, name.encode(), C.byref(values), C.byref(rows), C.byref(cols))
                require(status == 0 and rows.value == tokens and cols.value == columns, f"Invalid result view: {name}")
                actual = np.ctypeslib.as_array(values, shape=(tokens * columns,)).copy().reshape(tokens, columns) if tokens else np.empty((0, columns), dtype=np.float32)
                key = "tok2vec.output" if name == "tok2vec" else name + ".scores"
                require(key in captured and np.array_equal(actual, captured[key]), f"Public result differs from traced result: {name}")
                if name != "tok2vec":
                    decisions[name] = actual.argmax(axis=1).tolist()
            for step in case["parser_steps"]:
                current_step[0] = step["step"]
                features = (C.c_int32 * 8)(*step["feature_token_ids"])
                scores = (C.c_float * 137)()
                status = self.lib.crispembed_spacy_parser_scores(self.context, result, features, 8, scores, 137,
                    callback, None, CANCEL(), None, self.error, len(self.error))
                require(status == 0 and not callback_errors,
                        f"Native parser scorer failed ({status}): {self.error.value!r}; callback={callback_errors}")
                actual = np.ctypeslib.as_array(scores).copy()
                require(np.array_equal(actual.reshape(1, 137), captured[f"parser.step.{step['step']}.scores"]),
                        "Public parser scores differ from traced scores")
                legal = np.flatnonzero(np.asarray(step["valid_actions"], dtype=bool))
                parser_decisions.append(int(legal[np.argmax(actual[legal])]))
        finally:
            self.lib.crispembed_spacy_result_free(result)
        return captured, decisions, parser_decisions, time.perf_counter() - started


def compare_stage(name, original, actual):
    require(original.shape == actual.shape, f"Stage shape mismatch: {name}")
    require(np.isfinite(original).all() and np.isfinite(actual).all(), f"Non-finite stage: {name}")
    expected, candidate = original.astype(np.float64), actual.astype(np.float64)
    error = np.abs(candidate - expected)
    violations = error > ATOL + RTOL * np.abs(expected)
    nonzero = expected != 0
    return {"stage": name, "shape": list(original.shape), "elements": original.size,
            "max_reference_magnitude": float(np.abs(expected).max()) if expected.size else 0.0,
            "max_candidate_magnitude": float(np.abs(candidate).max()) if candidate.size else 0.0,
            "max_absolute_error": float(error.max()) if error.size else 0.0,
            "mean_absolute_error": float(error.mean()) if error.size else 0.0,
            "max_relative_error_nonzero_reference": float((error[nonzero] / np.abs(expected[nonzero])).max()) if nonzero.any() else 0.0,
            "violating_elements": int(violations.sum()), "bit_exact": original.tobytes() == actual.astype("<f4").tobytes(),
            "passed": not violations.any()}


def qualify(args):
    reference, manifest, bundle, verified = validate_inputs(args)
    native = Native(args.library, args.bundle / "model.gguf", bundle["files"]["model.gguf"]["sha256"])
    cases = []
    try:
        for item in manifest["cases"]:
            case = read_json(reference / item["id"] / "case.json")
            traces, decisions, parser_decisions, seconds = native.run_case(case)
            require(set(traces) == set(case["traces"]),
                    f"Native stage inventory differs for {item['id']}: missing={set(case['traces'])-set(traces)}, extra={set(traces)-set(case['traces'])}")
            payload = (reference / item["id"] / "traces.f32").read_bytes()
            stages, reference_decisions = [], {}
            for name, entry in sorted(case["traces"].items(), key=lambda pair: pair[1]["offset_bytes"]):
                original = np.frombuffer(payload, dtype="<f4", count=entry["size_bytes"] // 4,
                                         offset=entry["offset_bytes"]).reshape(entry["shape"])
                stages.append(compare_stage(name, original, traces[name]))
                if name in {"tagger.scores", "morphologizer.scores", "lemmatizer.scores"}:
                    reference_decisions[name.split(".")[0]] = original.argmax(axis=1).tolist()
            exact_heads = decisions == reference_decisions
            exact_actions = parser_decisions == [step["chosen_action"] for step in case["parser_steps"]]
            cases.append({"id": item["id"], "tokens": item["tokens"], "parser_states": item["parser_steps"],
                          "classifier_decisions_exact": exact_heads, "legal_action_choices_exact_on_reference_states": exact_actions,
                          "seconds_with_trace_copy": seconds, "stages": stages,
                          "passed": exact_heads and exact_actions and all(stage["passed"] for stage in stages)})
    finally:
        native.close()
    counts = {"cases": len(cases), "tokens": sum(case["tokens"] for case in cases),
              "classifier_decisions": 3 * sum(case["tokens"] for case in cases),
              "parser_states": sum(case["parser_states"] for case in cases),
              "trace_stages": sum(len(case["stages"]) for case in cases),
              "float_elements": sum(stage["elements"] for case in cases for stage in case["stages"])}
    require(counts["tokens"] == verified["tokens"] and counts["trace_stages"] == verified["trace_stages"]
            and counts["parser_states"] == verified["parser_steps"], "Incomplete native qualification denominators")
    sources = ["src/spacy_pipeline.h", "src/spacy_pipeline.cpp", "src/spacy_numeric.h", "src/spacy_numeric.cpp",
               "src/core/gguf_loader.h", "src/core/gguf_loader.cpp", "src/core/sha256.h", "src/core/env_gate.h",
               "tools/spacy/CMakeLists.txt", "tests/spacy_parity.py"]
    return {"schema_version": 1, "scope": "prepared-feature CPU/F32 numerical parity and scorer replay on reference parser states",
            "not_qualified": ["native tokenization", "native parser transitions", "native lemma decoding", "full linguistic analysis",
                              "Dart/editor integration", "grammar quality", "mobile targets", "shipping distribution"],
            "passed": all(case["passed"] for case in cases), "atol": ATOL, "rtol": RTOL,
            "counts": counts, "cases": cases, "model_sha256": bundle["files"]["model.gguf"]["sha256"],
            "library_sha256": sha256(args.library), "reference_manifest_sha256": sha256(reference / "manifest.json"),
            "bundle_manifest_sha256": sha256(args.bundle / "manifest.json"), "native_source_sha256": {name: sha256(ROOT / name) for name in sources},
            "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
            "open_seconds_diagnostic": native.open_seconds,
            "timing_scope": "single local development run with trace callbacks; not a performance benchmark"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--quds-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Refusing to overwrite a qualification receipt")
    try:
        report = qualify(args)
    except Exception as error:
        report = {"schema_version": 1, "passed": False, "scope": "native qualification failed",
                  "error": str(error), "error_type": type(error).__name__}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"cases", "native_source_sha256"}}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
