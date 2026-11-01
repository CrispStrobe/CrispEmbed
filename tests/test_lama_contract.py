#!/usr/bin/env python3
"""Small stdlib tests for the strict Big-LaMa archive and conversion boundary."""

import copy
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import lama_contract as contract
from latent_inpaint_contract import read_gguf_index
from test_latent_inpaint_contract import integer, row, state_pickle, text


def wrapped_pickle(rows, extra_root=False):
    values = {
        "epoch": 23, "global_step": 575000, "pytorch-lightning_version": "1.2.9", "callbacks": {},
        "lr_schedulers": [], "hparams_name": "config", "hyper_parameters": None, "hparams_type": None,
    }
    data = b"\x80\x02}"
    for name, value in values.items():
        data += text(name) + pickle.dumps(value, protocol=2)[2:-1] + b"s"
    data += text("state_dict") + state_pickle(rows)[2:-1] + b"s"
    if extra_root:
        data += text("unexpected") + integer(1) + b"s"
    return data + b"."


def small_archive(directory, weight_values=(1.25, -2.0), counter=7, extra_inner=()):
    """A few-byte synthetic state; no framework or real model weights."""
    rows = [row("generator.model.1.bn_l.weight", (2,), storage="0"),
            row("generator.model.1.bn_l.num_batches_tracked", (), "int64", "1"),
            row("val_evaluator.net.weight", (1,), storage="2"),
            row("test_evaluator.net.weight", (1,), storage="2")]
    for record in rows:
        record["export_shape"] = contract.export_shape(record["name"], record["shape"], record["dtype"])
        record["export"] = ("f32" if record["export_shape"] is not None else
                            "omit-inference-inert-batch-counter" if record["name"].startswith("generator.") else
                            "omit-evaluation-only-state")
    rows.sort(key=lambda item: item["name"])
    payload = wrapped_pickle(rows)
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/data/0", struct.pack("<2f", *weight_values))
        archive.writestr("archive/data/1", struct.pack("<q", counter))
        archive.writestr("archive/data/2", struct.pack("<f", 4.0))
        archive.writestr("archive/version", b"3\n")
        for name, raw in extra_inner:
            archive.writestr(name, raw)
    inner_bytes = inner.getvalue()
    config = b"generator:\n  kind: ffc_resnet\n  n_blocks: 18\n"
    path = Path(directory) / "official.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("big-lama/config.yaml", config)
        archive.writestr("big-lama/models/best.ckpt", inner_bytes)
        archive.writestr("big-lama/", b"")
        archive.writestr("big-lama/models/", b"")
    with zipfile.ZipFile(path) as archive:
        members = [{"name": item.filename, "flags": item.flag_bits, "method": item.compress_type,
                    "crc32": format(item.CRC, "08x"), "compressed_bytes": item.compress_size,
                    "bytes": item.file_size, "local_offset": item.header_offset} for item in archive.infolist()]
    frozen = copy.deepcopy(contract.load_contract())
    frozen["state"] = rows
    frozen.update(original_tensor_count=4, generator_tensor_count=2, export_tensor_count=1,
                  omitted_generator_counter_count=1, excluded_evaluation_tensor_count=2, export_tensor_bytes=8)
    cp = frozen["checkpoint"]
    cp.update(bytes=path.stat().st_size, sha256=contract.sha256(path), archive_members=members,
              pickle_bytes=len(payload), pickle_sha256=hashlib.sha256(payload).hexdigest(),
              auxiliary_state=[], storage_count=3)
    cp["member"] = {"path": "big-lama/models/best.ckpt", "bytes": len(inner_bytes)}
    cp["config"] = {"path": "big-lama/config.yaml", "bytes": len(config),
                    "sha256": hashlib.sha256(config).hexdigest()}
    return path, frozen, inner_bytes


def converter_module():
    spec = importlib.util.spec_from_file_location("lama_converter_test", ROOT / "models/convert-lama-to-gguf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LaMaContractTest(unittest.TestCase):
    def test_real_frozen_census_and_generated_header_agree(self):
        frozen = contract.load_contract()
        self.assertEqual((frozen["original_tensor_count"], frozen["generator_tensor_count"],
                          frozen["export_tensor_count"], frozen["omitted_generator_counter_count"]),
                         (2185, 989, 837, 152))
        self.assertEqual(frozen["export_tensor_bytes"], 204_228_108)
        self.assertEqual(frozen["excluded_evaluation_tensor_count"], 1196)
        self.assertEqual(frozen["checkpoint"]["bytes"], 381_428_720)
        self.assertEqual(frozen["checkpoint"]["member"]["bytes"], 410_046_389)
        self.assertIsNone(frozen["checkpoint"]["member"]["independently_published_sha256"])
        self.assertEqual(contract.NATIVE_SCHEMA.read_text(), contract.native_schema_text())
        self.assertFalse(frozen["reference"]["native_qualified"])
        from lama_reference_contract import CASES, THRESHOLDS, identity
        self.assertEqual(frozen["reference"]["thresholds"], THRESHOLDS)
        self.assertEqual(frozen["reference"]["default_cases"], [identity(name) for name in CASES])

    def test_generator_transform_is_only_f32_or_named_scalar_counter(self):
        self.assertEqual(contract.export_shape("generator.model.1.weight", [2], "float32"), [2])
        self.assertIsNone(contract.export_shape("generator.model.1.num_batches_tracked", [], "int64"))
        for name, shape, dtype in (("generator.model.1.bias", [], "int64"),
                                    ("generator.model.1.weight", [2], "float16"),
                                    ("generator.model.1.weight", [1, 2, 3, 4, 5], "float32")):
            with self.assertRaises(ValueError):
                contract.export_shape(name, shape, dtype)

    def test_verified_archive_streams_generator_and_accounts_for_evaluation_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, frozen, inner = small_archive(tmp)
            with contract.Checkpoint(archive, frozen, work_dir=tmp) as original:
                self.assertEqual(len(original.state), 2)
                self.assertEqual(len(original.all_state), 4)
                self.assertEqual(original.inner_sha256, hashlib.sha256(inner).hexdigest())
                self.assertEqual(original.inner_bytes, len(inner))
                self.assertEqual(b"".join(original.f32_blocks("generator.model.1.bn_l.weight")),
                                 struct.pack("<2f", 1.25, -2.0))
                self.assertEqual(original.read_counter("generator.model.1.bn_l.num_batches_tracked"), 7)
                omitted = original.validate_omitted()
                self.assertEqual(omitted[0]["sha256"], hashlib.sha256(struct.pack("<q", 7)).hexdigest())
                extracted = original.inner_path
            self.assertFalse(extracted.exists())

    def test_full_archive_hash_is_checked_before_any_zip_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, frozen, _ = small_archive(tmp)
            frozen["checkpoint"]["sha256"] = "0" * 64
            with mock.patch.object(contract.zipfile, "ZipFile", side_effect=AssertionError("parsed before hash")):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    contract.Checkpoint(archive, frozen)

    def test_archive_inventory_and_configuration_identity_are_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, frozen, _ = small_archive(tmp)
            changed = copy.deepcopy(frozen)
            changed["checkpoint"]["archive_members"][0]["flags"] ^= 8
            with self.assertRaisesRegex(ValueError, "member properties"):
                contract.Checkpoint(archive, changed)
            changed = copy.deepcopy(frozen)
            changed["checkpoint"]["config"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "configuration"):
                contract.Checkpoint(archive, changed)

    def test_unknown_pickle_globals_are_never_executed(self):
        payload = b"\x80\x02cposix\nsystem\n" + text("forbidden") + b"\x85R."
        with self.assertRaisesRegex(ValueError, "unsupported pickle global"):
            contract.read_lightning_pickle(payload)

    def test_omegaconf_newobj_build_becomes_only_an_inert_data_record(self):
        payload = (b"\x80\x02}" + text("config") + b"comegaconf.dictconfig\nDictConfig\n)\x81}"
                   + text("__class__") + text("untrusted attribute") + b"sbs.")
        root, storages = contract.read_lightning_pickle(payload)
        self.assertIsInstance(root["config"], contract._InertRecord)
        self.assertEqual(root["config"].record, {"__class__": "untrusted attribute"})
        self.assertEqual(storages, {})
        self.assertNotIn("omegaconf", sys.modules)
        self.assertNotIn("pytorch_lightning", sys.modules)

    def test_trailing_pickle_and_changed_root_census_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "trailing"):
            contract.read_lightning_pickle(pickle.dumps({}, protocol=2) + b"extra")
        frozen = contract.load_contract()
        with self.assertRaisesRegex(ValueError, "root keys"):
            contract.validate_lightning({}, {}, frozen)

    def test_pickle_digest_is_checked_before_inert_unpickling(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, frozen, _ = small_archive(tmp)
            frozen["checkpoint"]["pickle_sha256"] = "0" * 64
            with mock.patch.object(contract, "read_lightning_pickle", side_effect=AssertionError("parsed early")):
                with self.assertRaisesRegex(ValueError, "pickle digest"):
                    contract.Checkpoint(archive, frozen)

    def test_extra_storage_is_rejected_even_when_the_outer_archive_is_authenticated(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, frozen, _ = small_archive(tmp, extra_inner=(("archive/data/99", b"extra"),))
            with self.assertRaisesRegex(ValueError, "storage member inventory"):
                contract.Checkpoint(archive, frozen)

    def test_state_shape_stride_and_alias_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, frozen, _ = small_archive(tmp)
            with contract.Checkpoint(archive, frozen) as original:
                root = {key: None for key in frozen["checkpoint"]["root_keys"]}
                root.update(frozen["checkpoint"]["root_primitive_values"])
                root["state_dict"] = dict(original.all_state)
                selected = "generator.model.1.bn_l.weight"
                tensor = root["state_dict"][selected]
                root["state_dict"][selected] = replace(tensor, shape=(3,))
                storages = {value.storage.key: value.storage for value in original.all_state.values()}
                with self.assertRaisesRegex(ValueError, "shape/type/storage"):
                    contract.validate_lightning(root, storages, frozen)

    def test_nonfinite_generator_and_negative_training_counter_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, frozen, _ = small_archive(tmp, weight_values=(float("nan"), 0.0), counter=-1)
            with contract.Checkpoint(archive, frozen) as original:
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    list(original.f32_blocks("generator.model.1.bn_l.weight"))
                with self.assertRaisesRegex(ValueError, "counter"):
                    original.read_counter("generator.model.1.bn_l.num_batches_tracked")
                with self.assertRaisesRegex(ValueError, "counter"):
                    original.validate_omitted()

    def test_gguf_false_is_a_boolean_and_integer_substitution_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, frozen, _ = small_archive(tmp)
            output = Path(tmp) / "model.gguf"
            metadata = contract.gguf_metadata(frozen)
            with contract.Checkpoint(archive, frozen) as original:
                records = contract.write_streamed_gguf(output, original, contract.generator_rows(frozen), metadata)
            contract.verify_gguf(output, metadata, records)
            actual, _, _ = read_gguf_index(output)
            self.assertIs(actual["lama.lfu"], False)
            bad = dict(metadata, **{"lama.lfu": 0})
            with self.assertRaisesRegex(ValueError, "metadata"):
                contract.verify_gguf(output, bad, records)

    def test_source_identity_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "LICENSE"
            path.write_bytes(b"original licence")
            frozen = {"source": {"files": {"LICENSE": {"bytes": path.stat().st_size,
                                                         "sha256": contract.sha256(path)}}}}
            contract.verify_source(tmp, frozen)
            path.write_bytes(b"modified licence")
            with self.assertRaisesRegex(ValueError, "source identity"):
                contract.verify_source(tmp, frozen)
            output = Path(tmp) / "keep.gguf"
            output.write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "already exists"):
                converter_module().convert("unused", output, "unused")
            self.assertEqual(output.read_bytes(), b"keep")

    def test_tiny_conversion_receipt_binds_both_archive_and_observed_inner_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            archive, frozen, inner = small_archive(tmp)
            source = directory / "source"
            source.mkdir()
            (source / "LICENSE").write_text("test-only licence\n")
            frozen["source"]["files"] = {"LICENSE": {"bytes": (source / "LICENSE").stat().st_size,
                                                       "sha256": contract.sha256(source / "LICENSE")}}
            output = directory / "result.gguf"
            converter = converter_module()
            with mock.patch.object(converter, "load_contract", return_value=frozen):
                receipt = converter.convert(archive, output, source)
            self.assertEqual(receipt["gguf_sha256"], contract.sha256(output))
            self.assertEqual(receipt["checkpoint_sha256"], contract.sha256(archive))
            self.assertEqual(receipt["inner_checkpoint_sha256"], hashlib.sha256(inner).hexdigest())
            self.assertFalse(receipt["inner_checkpoint_sha256_independently_published"])
            self.assertFalse(receipt["native_qualified"])
            self.assertEqual(len(receipt["excluded_evaluation_state"]), 2)
            self.assertEqual(len(receipt["omitted_inference_inert_buffers"]), 1)
            self.assertEqual(receipt["tensors"][0]["sha256"], hashlib.sha256(struct.pack("<2f", 1.25, -2)).hexdigest())
            self.assertEqual(json.loads(Path(str(output) + ".json").read_text()), receipt)
            self.assertTrue((Path(str(output) + ".licenses") / "WEIGHTS-LICENSE-EVIDENCE.json").is_file())
            self.assertIn("--checkpoint-archive", receipt["command"])

    def test_helper_and_converter_import_no_frameworks(self):
        converter_module()
        for module in ("torch", "numpy", "omegaconf", "pytorch_lightning"):
            self.assertNotIn(module, sys.modules)


if __name__ == "__main__":
    unittest.main()
