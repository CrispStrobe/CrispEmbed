#!/usr/bin/env python3
"""Bounded stdlib controls: checkpoint parser, layouts, streaming export and pins."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import latent_inpaint_contract as contract


def text(value):
    raw = value.encode("utf-8")
    return b"X" + struct.pack("<I", len(raw)) + raw


def integer(value):
    return b"J" + struct.pack("<i", value)


def tuple_bytes(values):
    return b"(" + b"".join(integer(n) for n in values) + b"t"


def state_pickle(rows):
    result = b"\x80\x02}"
    storage_types = {"float32": "FloatStorage", "float16": "HalfStorage", "int64": "LongStorage"}
    for row in rows:
        storage = (b"(" + text("storage") + b"ctorch\n" + storage_types[row["dtype"]].encode() + b"\n"
                   + text(row["storage"]) + text("cpu") + integer(row["storage_count"]) + b"tQ")
        tensor = (b"ctorch._utils\n_rebuild_tensor_v2\n(" + storage + integer(row["offset"])
                  + tuple_bytes(row["shape"]) + tuple_bytes(row["stride"])
                  + b"\x89ccollections\nOrderedDict\n)RtR")
        result += text(row["name"]) + tensor + b"s"
    return result + b"."


def row(name="weight", shape=(2, 2), dtype="float32", storage="0"):
    count, stride = 1, []
    for n in reversed(shape):
        stride.insert(0, count)
        count *= n
    out_shape = contract.export_shape(name, shape, dtype)
    return {"name": name, "shape": list(shape), "dtype": dtype, "storage": storage,
            "storage_count": count, "offset": 0, "stride": stride, "export_shape": out_shape,
            "export": "omit-inference-inert-batch-counter" if out_shape is None else "f32"}


def archive(path, rows, payloads, pickle_data=None, byteorder=b"little", extras=()):
    payload = state_pickle(rows) if pickle_data is None else pickle_data
    with zipfile.ZipFile(path, "w") as stream:
        stream.writestr("weights/data.pkl", payload)
        if byteorder is not None:
            stream.writestr("weights/byteorder", byteorder)
        for record, raw in zip(rows, payloads):
            stream.writestr("weights/data/" + record["storage"], raw)
        for name, raw in extras:
            stream.writestr(name, raw)
    return {"checkpoint": {"bytes": path.stat().st_size, "sha256": contract.sha256(path),
                           "pickle_bytes": len(payload), "pickle_sha256": hashlib.sha256(payload).hexdigest()},
            "state": rows}


class LatentInpaintContractTest(unittest.TestCase):
    def test_frozen_census_and_native_schema_agree(self):
        source = contract.load_contract()
        expected = {"moebius": (1203, 1079), "pixelhacker": (753, 753), "sdxl_vae": (248, 248)}
        for model, counts in expected.items():
            record = source["models"][model]
            self.assertEqual((record["original_tensor_count"], record["export_tensor_count"]), counts)
            self.assertFalse(record["initial_native_qualified"])
            self.assertEqual(len(record["checkpoint"]["sha256"]), 64)
            self.assertEqual(len(record["checkpoint"]["revision"]), 40)
        self.assertEqual(contract.NATIVE_SCHEMA.read_text(), contract.native_schema_text())

    def test_hf_declared_sizes_include_one_checkpoint_and_vae(self):
        source = contract.load_contract()["models"]
        self.assertEqual(source["moebius"]["checkpoint"]["bytes"] + source["sdxl_vae"]["checkpoint"]["bytes"], 1_072_692_662)
        self.assertEqual(source["pixelhacker"]["checkpoint"]["bytes"] + source["sdxl_vae"]["checkpoint"]["bytes"], 3_616_739_746)

    def test_only_named_singleton_depth_transform_is_permitted(self):
        name = "diff_model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.pos_conv.weight"
        self.assertEqual(contract.export_shape(name, (40, 1, 1, 15, 15), "float32"), [40, 1, 15, 15])
        for bad_name, shape in (("other", (40, 1, 1, 15, 15)), (name, (40, 1, 2, 15, 15)), (name, (40, 1, 1, 3, 3))):
            with self.assertRaisesRegex(ValueError, "rank-five"):
                contract.export_shape(bad_name, shape, "float32")

    def test_only_scalar_bn_counter_may_be_omitted(self):
        self.assertIsNone(contract.export_shape("bn.num_batches_tracked", (), "int64"))
        for name, shape in (("embedding.weight", ()), ("bn.num_batches_tracked", (1,))):
            with self.assertRaisesRegex(ValueError, "BN tracking"):
                contract.export_shape(name, shape, "int64")

    def test_real_pickle_forms_are_inert_and_dtypes_retained(self):
        rows = [row(), row("vae.weight", (3,), "float16", "1"), row("bn.num_batches_tracked", (), "int64", "2")]
        state = contract.read_state_pickle(state_pickle(rows))
        contract.validate_state(state, rows)
        self.assertEqual([state[r["name"]].storage.dtype for r in rows], ["float32", "float16", "int64"])

    def test_foreign_globals_and_trailing_payload_are_rejected(self):
        for payload, expected in ((b"\x80\x02cos\nsystem\n.", "unsupported pickle global"),
                                  (state_pickle([row()]) + b"extra", "trailing pickle")):
            with self.assertRaisesRegex(ValueError, expected):
                contract.read_state_pickle(payload)

    def test_changed_names_shape_type_offset_stride_are_rejected(self):
        original = row()
        state = contract.read_state_pickle(state_pickle([original]))
        for key, value in (("name", "extra"), ("shape", [4]), ("dtype", "float16"),
                           ("offset", 1), ("stride", [1, 2]), ("storage_count", 5), ("storage", "9")):
            changed = dict(original)
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "differ"):
                contract.validate_state(state, [changed])

    def test_full_hash_precedes_pickle_interpretation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pt"
            path.write_bytes(b"invalid")
            record = {"checkpoint": {"bytes": 7, "sha256": "0" * 64}}
            with mock.patch.object(contract, "read_state_pickle") as parser:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    contract.Checkpoint(path, record)
                parser.assert_not_called()

    def test_f16_widens_exactly_and_f32_bytes_are_preserved(self):
        rows = [row(), row("half.weight", (4,), "float16", "1")]
        single = struct.pack("<4f", -0.0, 1.25, -2.5, 65504)
        half = struct.pack("<4e", -0.0, 1.25, -2.5, 65504)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.pt"
            record = archive(path, rows, [single, half])
            with contract.Checkpoint(path, record) as source:
                self.assertEqual(b"".join(source.f32_blocks("weight")), single)
                self.assertEqual(b"".join(source.f32_blocks("half.weight")), single)

    def test_nonfinite_weight_is_rejected(self):
        for dtype, raw in (("float32", struct.pack("<f", float("nan"))), ("float16", struct.pack("<e", float("inf")))):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "tiny.pt"
                record = archive(path, [row(shape=(1,), dtype=dtype)], [raw])
                with contract.Checkpoint(path, record) as source, self.assertRaisesRegex(ValueError, "nonfinite"):
                    list(source.f32_blocks("weight"))

    def test_training_counter_omission_is_hashed_and_checked(self):
        rows = [row("bn.num_batches_tracked", (), "int64")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.pt"
            record = archive(path, rows, [struct.pack("<q", 73)])
            with contract.Checkpoint(path, record) as source:
                omitted = source.validate_omitted(rows)
                self.assertEqual(omitted[0]["sha256"], hashlib.sha256(struct.pack("<q", 73)).hexdigest())
            record = archive(path, rows, [struct.pack("<q", -1)])
            with contract.Checkpoint(path, record) as source, self.assertRaisesRegex(ValueError, "BN batch counter"):
                source.validate_omitted(rows)

    def test_truncated_big_endian_extra_and_duplicate_storage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.pt"
            rows = [row(shape=(1,))]
            for raw, byteorder, extras, message in (
                    (b"\0", b"little", (), "byte count"),
                    (b"\0" * 4, b"big", (), "little-endian"),
                    (b"\0" * 4, b"little", (("weights/data/99", b"\0" * 4),), "member inventory"),
                    (b"\0" * 4, b"little", (("weights/data/0", b"\0" * 4),), "ZIP inventory")):
                with self.subTest(message=message), warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    record = archive(path, rows, [raw], byteorder=byteorder, extras=extras)
                    with self.assertRaisesRegex(ValueError, message):
                        contract.Checkpoint(path, record)

    def test_streamed_gguf_header_payload_and_readback(self):
        rows = [row("half", (2, 3), "float16"), row("full", (1,), "float32", "1")]
        rows.sort(key=lambda r: r["name"])
        payloads = [struct.pack("<f", 1.25), struct.pack("<6e", *range(6))]
        metadata = {"general.architecture": "authored-fixture", "general.file_type": 0}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, output = Path(tmp) / "tiny.pt", Path(tmp) / "tiny.gguf"
            record = archive(checkpoint, rows, payloads)
            with contract.Checkpoint(checkpoint, record) as source:
                records = contract.write_streamed_gguf(output, source, rows, metadata)
            self.assertTrue(all("original_name" not in entry for entry in records))
            contract.verify_gguf(output, metadata, records)
            _, actual, data_offset = contract.read_gguf_index(output)
            self.assertEqual([r["shape"] for r in actual], [[1], [2, 3]])
            with output.open("r+b") as stream:
                stream.seek(data_offset)
                stream.write(b"\0")
                stream.write(b"\0\0\0")
            with self.assertRaisesRegex(ValueError, "readback hash"):
                contract.verify_gguf(output, metadata, records)

    def test_tensor_alias_byte_boundaries_and_fixed_encoding(self):
        for original in ("x" * 63, "λ" * 31 + "x"):
            self.assertEqual(contract.gguf_tensor_name(original), original)
        for original in ("x" * 64, "λ" * 32, "module." * 20 + "weight"):
            expected = "t." + hashlib.sha256(original.encode("utf-8")).hexdigest()[:56]
            self.assertEqual(contract.gguf_tensor_name(original), expected)
            self.assertEqual(len(expected.encode("utf-8")), 58)
        for original in (None, "", "bad\0name"):
            with self.assertRaisesRegex(ValueError, "original tensor name"):
                contract.gguf_tensor_name(original)

    def test_frozen_census_names_fit_ggml_and_are_collision_free(self):
        source = contract.load_contract()
        for model, alias_count in (("moebius", 540), ("pixelhacker", 384), ("sdxl_vae", 0)):
            rows = source["models"][model]["state"]
            names = contract._tensor_name_map(rows, contract.gguf_metadata(model, source))
            self.assertEqual(len(names), len(set(names.values())))
            exported = [entry["name"] for entry in rows if entry["export_shape"] is not None]
            self.assertEqual(sum(names[name] != name for name in exported), alias_count)
            self.assertTrue(all(len(names[name].encode("utf-8")) < 64 for name in exported))

    def test_streamed_alias_preserves_original_name_and_exact_f32_payload(self):
        original = "diff_model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.g_norm_swish_gate.weight"
        rows = [row(original)]
        payload = struct.pack("<4I", 0x3f800000, 0x80000000, 0x3eaaaaab, 0xc0200000)
        metadata = {"general.architecture": "authored-fixture", "general.file_type": 0,
                    "crispembed.tensor_name_encoding": contract.TENSOR_NAME_ENCODING}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, output = Path(tmp) / "tiny.pt", Path(tmp) / "tiny.gguf"
            record = archive(checkpoint, rows, [payload])
            with contract.Checkpoint(checkpoint, record) as source:
                records = contract.write_streamed_gguf(output, source, rows, metadata)
            contract.verify_gguf(output, metadata, records)
            self.assertEqual(records[0]["original_name"], original)
            self.assertEqual(records[0]["name"], contract.gguf_tensor_name(original))
            self.assertEqual(records[0]["sha256"], hashlib.sha256(payload).hexdigest())
            actual_meta, actual, start = contract.read_gguf_index(output)
            self.assertEqual(actual_meta, metadata)
            self.assertEqual(actual[0]["name"], records[0]["name"])
            with output.open("rb") as stream:
                stream.seek(start + actual[0]["offset"])
                self.assertEqual(stream.read(actual[0]["bytes"]), payload)
            for key, value in (("original_name", original + "_changed"), ("name", "t." + "0" * 56)):
                changed = copy.deepcopy(records)
                changed[0][key] = value
                with self.subTest(field=key), self.assertRaisesRegex(ValueError, "alias/source identity"):
                    contract.verify_gguf(output, metadata, changed)
            changed = copy.deepcopy(records)
            del changed[0]["original_name"]
            with self.assertRaisesRegex(ValueError, "original tensor name"):
                contract.verify_gguf(output, metadata, changed)

    def test_alias_collision_including_omitted_source_buffer_fails_before_output(self):
        original = "module." * 12 + "num_batches_tracked"
        # A literal short source name may equal another source's hash alias.
        # Even though the long original is omitted, the full census must reject
        # that collision rather than silently losing the source correspondence.
        rows = [row(original, (), "int64"), row(contract.gguf_tensor_name(original), (1,), storage="1")]
        metadata = {"crispembed.tensor_name_encoding": contract.TENSOR_NAME_ENCODING}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "collision.gguf"
            with self.assertRaisesRegex(ValueError, "alias collision"):
                contract.write_streamed_gguf(output, None, rows, metadata)
            self.assertFalse(output.exists())

    def test_unsupported_alias_encoding_and_unencoded_long_names_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bad.gguf"
            with self.assertRaisesRegex(ValueError, "unsupported GGUF tensor name encoding"):
                contract.write_streamed_gguf(output, None, [row()], {"crispembed.tensor_name_encoding": "unknown"})
            with self.assertRaisesRegex(ValueError, "long tensor name requires"):
                contract.write_streamed_gguf(output, None, [row("x" * 64)], {})
            self.assertFalse(output.exists())

    def test_fixtures_are_binary_and_independent_of_rng(self):
        for name in contract.CASES:
            rgb, mask = contract.fixture(name)
            self.assertEqual(len(rgb), 512 * 512 * 3)
            self.assertEqual(len(mask), 512 * 512)
            self.assertLessEqual(set(mask), {0, 255})
            self.assertEqual(contract.fixture_identity(name)["fill_pixels"], mask.count(255))
        self.assertEqual(contract.fixture_identity("empty")["fill_pixels"], 0)

    def test_converter_is_importable_and_rejects_overwrite(self):
        spec = importlib.util.spec_from_file_location("latent_converter_test", ROOT / "models/convert-latent-inpaint-to-gguf.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "keep.gguf"
            output.write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "already exists"):
                module.convert("moebius", "unused", output, "unused")
            self.assertEqual(output.read_bytes(), b"keep")

    def test_converter_and_contract_do_not_import_frameworks(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)


if __name__ == "__main__":
    unittest.main()
