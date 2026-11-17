"""Model-free admission controls and a separate mandatory real-model replay.

Unit controls (no model/fixtures required):
    python tests/test_spacy_converter.py -v

Actual conversion qualification (missing assets fail, never skip):
    python tests/test_spacy_converter.py --real-model --wheel /path/to/pinned.whl

The synthetic resources below test data admission only. They are deliberately
not a linguistic model or native-inference evidence.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spacy_bundle", ROOT / "models/spacy_bundle.py")
bundle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bundle)


def synthetic_resources():
    """Dimensionally valid, openly synthetic data for parser/schema controls."""
    morph_labels = [f"Case=Synthetic{i}|POS=NOUN" for i in range(671)]
    dependencies = [f"synthetic{i:02}" for i in range(41)]
    arc_labels = dependencies + [f"synthetic{i:02}||synthetic00" for i in range(26)]
    actions = {"0": {"": 1}, "1": {"": 1},
               "2": {label: 67 - i for i, label in enumerate(arc_labels)},
               "3": {label: 67 - i for i, label in enumerate(arc_labels)},
               "4": {"ROOT": 1}}
    moves = [action + ("-" + label if label else "")
             for i, action in enumerate(["S", "D", "L", "R", "B"])
             for label in actions[str(i)]]
    return {
        "format_version": 1, "language": "de", "component_order": list(bundle.COMPONENTS),
        "lexical_semantics_source": {"revision": bundle.SOURCES["spacy"],
                                     "files": ["spacy/lang/lex_attrs.py", "spacy/strings.pyx", "spacy/tokenizer.pyx"]},
        "tokenizer": {"prefix_search": "^x", "suffix_search": "x$", "infix_finditer": "x",
                      "token_match": None, "url_match": "^x$", "faster_heuristics": True,
                      "exceptions": {f"synthetic{i}": [{"ORTH": f"synthetic{i}"}] for i in range(417)}},
        "norm_lookup_precedence": [
            {"key_kind": "hash_uint64_decimal", "entries": {"1": "one", "18446744073709551615": "max"}},
            {"key_kind": "text", "entries": {f"synthetic{i}": "x" for i in range(45)}},
        ],
        "unicode": {"unicode_version": "15.0.0", "normalization": "none; preserve original UTF-8 source",
                    "lower_semantics": "CPython str.lower; contextual final sigma must also be implemented",
                    "ranges_inclusive": {name: [[i * 2, i * 2] for i in range(count)]
                                         for name, count in [("isalpha", 659), ("isdigit", 83),
                                                             ("islower", 671), ("isspace", 10), ("isupper", 651)]},
                    "scalar_lower": {str(0x1000 + i): "x" for i in range(1433)}},
        "labels": {
            "tagger": [f"SYNTHETIC{i}" for i in range(55)],
            "morphologizer": {"labels": morph_labels,
                              "morph": {label: label.removesuffix("|POS=NOUN") for label in morph_labels},
                              "pos": {label: 92 for label in morph_labels}, "overwrite": True, "extend": False},
            "parser": {"moves": moves, "labels": sorted(["ROOT"] + dependencies),
                       "features": ["B0", "B1", "S0", "S1", "S2", "L(B0,1)", "L(S0,1)", "R(S0,1)"],
                       "cfg": {"beam_density": 0.0, "beam_update_prob": 0.0, "beam_width": 1,
                               "incorrect_spans_key": None, "learn_tokens": False, "min_action_freq": 30,
                               "moves": None, "multitasks": [], "update_with_oracle_cut_size": 100},
                       "moves_serialized_json": {"cfg": {"neg_key": None}, "moves": json.dumps(actions)}},
            "lemmatizer": {"class_to_tree": list(range(1311)), "top_k": 1, "backoff": "orth", "overwrite": False,
                           "trees": [{"prefix_len": 0, "prefix_tree": (1 << 32) - 1,
                                      "suffix_len": 0, "suffix_tree": (1 << 32) - 1} for _ in range(6092)]},
            "attribute_ruler": [
                {"attrs": {"DEP": "dep"}, "index": 0,
                 "patterns": [[{"DEP": {"NOT_IN": [""]}, "IS_SPACE": True}]]},
                {"attrs": {"MORPH": "_", "POS": "SPACE", "TAG": "_SP"}, "index": 0,
                 "patterns": [[{"IS_SPACE": True}]]},
            ],
        },
    }


class TensorAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid = {name: np.zeros(shape, dtype="float32") for name, shape in bundle.expected_tensors().items()}

    def test_valid_inventory_has_39_f32_tensors(self):
        self.assertEqual(len(self.valid), 39)
        bundle.validate_tensors(self.valid)

    def test_missing_or_extra_tensor_rejected(self):
        for edit in [lambda data: data.pop("parser.lower.pad"),
                     lambda data: data.update({"unused.weight": np.zeros(1, dtype="float32")})]:
            data = dict(self.valid)
            edit(data)
            with self.subTest(names=len(data)), self.assertRaisesRegex(ValueError, "tensor inventory"):
                bundle.validate_tensors(data)

    def test_wrong_shape_rejected_even_with_same_element_count(self):
        data = dict(self.valid)
        data["parser.lower.W"] = np.zeros((8, 2, 64, 64), dtype="float32")
        with self.assertRaisesRegex(ValueError, "tensor shape: parser.lower.W"):
            bundle.validate_tensors(data)

    def test_non_f32_or_wrong_byte_order_rejected(self):
        for dtype in ["float16", "float64", "int32", ">f4"]:
            data = dict(self.valid)
            data["tagger.b"] = np.zeros(55, dtype=dtype)
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "Non-F32"):
                bundle.validate_tensors(data)

    def test_nan_and_infinity_rejected(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            data = dict(self.valid)
            data["tagger.b"] = np.zeros(55, dtype="float32")
            data["tagger.b"][12] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Non-finite"):
                bundle.validate_tensors(data)


class ResourceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.resources = synthetic_resources()

    def reject(self, expression):
        with self.assertRaisesRegex(ValueError, expression):
            bundle.validate_resources(self.resources)

    def test_structurally_valid_synthetic_resources_and_shared_tree(self):
        # A shared child is a DAG, not a cycle; the validator must accept it.
        trees = self.resources["labels"]["lemmatizer"]["trees"]
        trees[1].update(prefix_tree=0, suffix_tree=0)
        bundle.validate_resources(self.resources)

    def test_unknown_resource_field_rejected(self):
        self.resources["executable_plugin"] = "python-module"
        self.reject("resource fields: root")

    def test_changed_component_order_rejected(self):
        self.resources["component_order"][3:5] = ["lemmatizer", "parser"]
        self.reject("version/language/components")

    def test_unsupported_tokenizer_attribute_rejected(self):
        self.resources["tokenizer"]["exceptions"]["synthetic0"][0]["LEMMA"] = "hidden"
        self.reject("exception attributes")

    def test_exception_must_reconstruct_original_text(self):
        self.resources["tokenizer"]["exceptions"]["synthetic0"][0]["ORTH"] = "changed"
        self.reject("reconstruct original")

    def test_invalid_regex_rejected(self):
        self.resources["tokenizer"]["prefix_search"] = "["
        self.reject("Malformed tokenizer pattern")

    def test_noncanonical_or_overflow_uint64_rejected(self):
        for bad in ["01", "-1", "1.0", "18446744073709551616", "١"]:
            self.resources = synthetic_resources()
            entries = self.resources["norm_lookup_precedence"][0]["entries"]
            entries[bad] = entries.pop("1")
            with self.subTest(identifier=bad):
                self.reject("decimal identifier|resource integer")

    def test_surrogate_text_rejected(self):
        self.resources["tokenizer"]["exceptions"]["synthetic0"][0]["NORM"] = "\ud800"
        self.reject("Invalid Unicode scalar")

    def test_overlapping_or_nonscalar_unicode_ranges_rejected(self):
        for bad in [[0, 0], [0xD800, 0xD800], [0x110000, 0x110000]]:
            self.resources = synthetic_resources()
            self.resources["unicode"]["ranges_inclusive"]["isalpha"][1] = bad
            with self.subTest(bounds=bad):
                self.reject("Unicode range|resource integer")

    def test_missing_morphology_map_entry_rejected(self):
        self.resources["labels"]["morphologizer"]["pos"].pop("Case=Synthetic0|POS=NOUN")
        self.reject("POS map")

    def test_wrong_morphology_or_pos_mapping_rejected(self):
        for field, value in [("morph", ""), ("pos", 100)]:
            self.resources = synthetic_resources()
            self.resources["labels"]["morphologizer"][field]["Case=Synthetic0|POS=NOUN"] = value
            with self.subTest(field=field):
                self.reject("label mapping differs")

    def test_malformed_self_consistent_morphology_label_rejected(self):
        for label in ["=Nom|POS=NOUN", "Case==Nom|POS=NOUN", "Case=|POS=NOUN"]:
            self.resources = synthetic_resources()
            morph = self.resources["labels"]["morphologizer"]
            old = morph["labels"][0]
            morph["labels"][0] = label
            del morph["morph"][old]
            del morph["pos"][old]
            morph["morph"][label] = label.removesuffix("|POS=NOUN")
            morph["pos"][label] = 92
            with self.subTest(label=label):
                self.reject("Malformed morphology label")

    def test_parser_move_order_is_bound_to_serialized_actions(self):
        moves = self.resources["labels"]["parser"]["moves"]
        moves[2], moves[3] = moves[3], moves[2]
        self.reject("move ordering differs")

    def test_parser_move_frequencies_determine_source_class_order(self):
        item = self.resources["labels"]["parser"]["moves_serialized_json"]
        actions = json.loads(item["moves"])
        actions["2"]["synthetic00"] = 0
        item["moves"] = json.dumps(actions)
        self.reject("move ordering differs")

    def test_duplicate_serialized_action_key_rejected(self):
        item = self.resources["labels"]["parser"]["moves_serialized_json"]
        item["moves"] = '{"0":{},' + item["moves"][1:]
        self.reject("Duplicate serialized parser key")

    def test_changed_parser_feature_order_rejected(self):
        self.resources["labels"]["parser"]["features"].reverse()
        self.reject("parser feature order")

    def test_tree_out_of_bounds_class_and_child_rejected(self):
        lemma = self.resources["labels"]["lemmatizer"]
        lemma["class_to_tree"][0] = 6092
        self.reject("lemma class tree")
        lemma["class_to_tree"][0] = 0
        lemma["trees"][0]["prefix_tree"] = 6092
        self.reject("child is out of range")

    def test_tree_cycle_rejected(self):
        trees = self.resources["labels"]["lemmatizer"]["trees"]
        trees[0]["prefix_tree"] = 1
        trees[1]["prefix_tree"] = 0
        self.reject("Cyclic edit-tree")

    def test_tree_depth_bound_rejected(self):
        trees = self.resources["labels"]["lemmatizer"]["trees"]
        for i in range(64):
            trees[i]["prefix_tree"] = i + 1
        self.reject("depth exceeds")

    def test_bool_cannot_be_tree_index(self):
        self.resources["labels"]["lemmatizer"]["class_to_tree"][0] = False
        self.reject("resource integer: lemma class tree")

    def test_attribute_ruler_order_preserved(self):
        self.resources["labels"]["attribute_ruler"].reverse()
        self.reject("attribute-ruler rules/order")

    def test_attribute_ruler_boolean_and_integer_types_are_distinct(self):
        self.resources["labels"]["attribute_ruler"][0]["index"] = False
        self.reject("attribute-ruler rules/order")


class SourceAdmissionTests(unittest.TestCase):
    def test_missing_wheel_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "source wheel is missing"):
                bundle.verify_wheel(Path(directory) / "absent.whl")

    def test_changed_wheel_same_size_fails_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / bundle.WHEEL_NAME
            with path.open("wb") as stream:
                stream.truncate(bundle.WHEEL_SIZE)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                bundle.verify_wheel(path)

    def test_truncated_wheel_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / bundle.WHEEL_NAME
            path.write_bytes(b"PK")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                bundle.verify_wheel(path)

    def test_nonfinite_json_cannot_be_published(self):
        with self.assertRaises(ValueError):
            bundle.json_bytes({"not_a_valid_score": float("nan")})

    def test_required_real_model_command_without_asset_fails(self):
        result = subprocess.run([sys.executable, str(Path(__file__)), "--real-model"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--wheel is required", result.stderr)

    def test_required_real_model_command_with_missing_asset_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(Path(__file__)), "--real-model", "--wheel",
                                     str(Path(directory) / "missing.whl")], capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source wheel is missing", result.stderr)


def qualify_real_model(wheel):
    """Run actual repeated conversion; no fixture/model-absence success path."""
    bundle.verify_wheel(wheel)
    import gguf
    with tempfile.TemporaryDirectory(prefix="spacy-converter-qualification-") as directory:
        first, second = Path(directory) / "first", Path(directory) / "second"
        manifest = bundle.convert(wheel, first)
        bundle.convert(wheel, second)
        expected_files = {"model.gguf", "resources.json", "manifest.json", "LICENSE", "LICENSES_SOURCES"}
        if {path.name for path in first.iterdir()} != expected_files:
            raise ValueError("Incomplete or unexpected converted bundle files")
        if {path.name for path in second.iterdir()} != expected_files:
            raise ValueError("Second export has a different file inventory")
        files = {}
        for name in sorted(expected_files):
            data = (first / name).read_bytes()
            if data != (second / name).read_bytes():
                raise ValueError(f"Independent exports differ: {name}")
            files[name] = {"size_bytes": len(data), "sha256": bundle.digest(data)}
            if name != "manifest.json" and files[name] != manifest["files"][name]:
                raise ValueError(f"Manifest asset hash/size differs: {name}")
        with zipfile.ZipFile(wheel) as archive:
            for name in ["LICENSE", "LICENSES_SOURCES"]:
                if archive.read(bundle.MODEL_PREFIX + name) != (first / name).read_bytes():
                    raise ValueError(f"Source notice changed: {name}")
        resources = json.loads((first / "resources.json").read_text())
        bundle.validate_resources(resources)
        reader = gguf.GGUFReader(str(first / "model.gguf"))
        by_name = {tensor.name: tensor for tensor in reader.tensors}
        if len(by_name) != 39 or len(reader.tensors) != 39 or len(manifest["tensors"]) != 39:
            raise ValueError("Incomplete or duplicate real-model tensor inventory")
        for declared in manifest["tensors"]:
            actual = by_name[declared["name"]]
            if (actual.tensor_type != gguf.GGMLQuantizationType.F32
                    or list(actual.shape) != declared["gguf_shape"]
                    or actual.data.nbytes != declared["size_bytes"]
                    or bundle.digest(actual.data.tobytes()) != declared["sha256"]):
                raise ValueError(f"Tensor readback differs: {declared['name']}")
        del reader
        if manifest["converter_sha256"] != bundle.digest((ROOT / "models/spacy_bundle.py").read_bytes()):
            raise ValueError("Converter source changed during qualification")
        return {"scope": "actual model conversion/reproducibility only; no native inference claim",
                "source_wheel_sha256": bundle.WHEEL_SHA256, "independent_exports": 2,
                "byte_identical_files": len(files), "tensor_readbacks": len(manifest["tensors"]),
                "verbatim_notices": 2, "resource_schema": "passed", "files": files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-model", action="store_true")
    parser.add_argument("--wheel", type=Path)
    args, remainder = parser.parse_known_args()
    if args.real_model:
        if args.wheel is None:
            parser.error("--wheel is required for real-model qualification; missing assets never skip")
        if remainder:
            parser.error("Unknown real-model arguments: " + " ".join(remainder))
        try:
            print(json.dumps(qualify_real_model(args.wheel), indent=2, sort_keys=True))
        except Exception as error:
            print(f"spaCy converter qualification failed: {error}", file=sys.stderr)
            raise SystemExit(1)
    else:
        if args.wheel is not None:
            parser.error("--wheel requires --real-model")
        unittest.main(argv=[sys.argv[0], *remainder])
