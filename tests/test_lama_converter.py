#!/usr/bin/env python3
"""Small, authored Big-LaMa format/schema controls; no framework or model run."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import lama_reference_contract as contract

spec = importlib.util.spec_from_file_location("lama_converter", ROOT / "models/convert-lama-to-gguf.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def text(value):
    raw = value.encode("utf-8")
    return b"X" + struct.pack("<I", len(raw)) + raw


def integer(value):
    return b"J" + struct.pack("<i", value)


def tuple_of(values):
    return b"(" + b"".join(integer(value) for value in values) + b"t"


def state_pickle(name="weight", shape=(2,), stride=(1,), count=2, offset=0, dtype="FloatStorage", nested=False):
    storage = (b"(" + text("storage") + b"ctorch\n" + dtype.encode() + b"\n" + text("0")
               + text("cpu") + integer(count) + b"tQ")
    tensor = (b"ctorch._utils\n_rebuild_tensor_v2\n(" + storage + integer(offset)
              + tuple_of(shape) + tuple_of(stride) + b"\x89ccollections\nOrderedDict\n)RtR")
    state = b"}" + text(name) + tensor + b"s"
    if nested:
        state = b"}" + text("state_dict") + state + b"s" + text("epoch") + integer(42) + b"s"
    return b"\x80\x02" + state + b"."


def checkpoint(path, raw=struct.pack("<2f", 0.5, -0.75), pickle_bytes=None, byteorder=b"little", **kwargs):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle_bytes if pickle_bytes is not None else state_pickle(**kwargs))
        archive.writestr("archive/data/0", raw)
        if byteorder is not None:
            archive.writestr("archive/byteorder", byteorder)
    return contract.sha256(path)


class LamaConverterTest(unittest.TestCase):
    def test_full_schema_count_layout_and_byte_count(self):
        shapes = contract.expected_state_shapes()
        self.assertEqual(len(shapes), 837)
        self.assertEqual(len(contract.expected_state_shapes(True)), 989)
        self.assertEqual(sum(math.prod(shape) * 4 for shape in shapes.values()), 204228108)
        self.assertEqual(shapes["model.24.weight"], (512, 256, 3, 3))
        self.assertEqual(shapes["model.5.conv1.ffc.convg2g.fu.conv_layer.weight"], (384, 384, 1, 1))
        self.assertEqual(shapes["model.4.ffc.convl2g.weight"], (384, 256, 3, 3))
        self.assertFalse(any("lfu" in name for name in shapes))

    def test_canonical_config(self):
        generator = contract.canonical_generator_config()
        self.assertEqual(contract.validate_config({"generator": generator}), generator)

    def test_upstream_hydra_interpolations(self):
        generator = contract.canonical_generator_config()
        generator["downsample_conv_kwargs"]["ratio_gin"] = "${generator.init_conv_kwargs.ratio_gout}"
        generator["downsample_conv_kwargs"]["ratio_gout"] = "${generator.downsample_conv_kwargs.ratio_gin}"
        generator["resnet_conv_kwargs"]["ratio_gout"] = "${generator.resnet_conv_kwargs.ratio_gin}"
        self.assertEqual(contract.validate_config({"generator": generator}), contract.canonical_generator_config())

    def test_wrong_profiles_are_rejected(self):
        for key, value in (("n_blocks", 9), ("add_out_act", "tanh"), ("padding_type", "zero"),
                           ("out_ffc", True), ("norm_layer", "instance"), ("ngf", True)):
            generator = contract.canonical_generator_config()
            generator[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unsupported"):
                contract.validate_config({"generator": generator})
        for key, value in (("enable_lfu", True), ("ratio_gin", 0.5), ("spectral_pos_encoding", True)):
            generator = contract.canonical_generator_config()
            generator["resnet_conv_kwargs"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unsupported"):
                contract.validate_config({"generator": generator})

    def test_cyclic_and_environment_interpolations_are_rejected(self):
        for expression in ("${generator.downsample_conv_kwargs.ratio_gin}", "${env:TOKEN}"):
            generator = contract.canonical_generator_config()
            generator["downsample_conv_kwargs"]["ratio_gin"] = expression
            with self.assertRaisesRegex(ValueError, "interpolation"):
                contract.validate_config({"generator": generator})

    def test_concat_mask_is_required(self):
        with self.assertRaisesRegex(ValueError, "concat_mask"):
            contract.validate_config({"generator": contract.canonical_generator_config(),
                                      "training_model": {"concat_mask": False}})

    def test_json_configuration_needs_no_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"generator": contract.canonical_generator_config()}))
            self.assertEqual(contract.load_config(path), contract.canonical_generator_config())

    def test_metadata_does_not_claim_checkpoint_license(self):
        metadata = contract.model_metadata("a" * 64, "b" * 64)
        self.assertEqual(metadata["general.architecture"], "lama")
        self.assertEqual(metadata["lama.source.license"], "Apache-2.0")
        self.assertEqual(metadata["lama.checkpoint.license"], "user-supplied;not-verified")
        self.assertNotIn("general.license", metadata)

    def test_strict_hash_format(self):
        for value in ("A" * 64, "0" * 63, "g" * 64, 0):
            with self.assertRaisesRegex(ValueError, "lowercase"):
                contract.model_metadata(value, "b" * 64)

    def test_checkpoint_hash_is_checked_before_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            checkpoint(path, pickle_bytes=b"\x80\x02cos\nsystem\n.")
            with mock.patch.object(contract._StateReader, "load") as load:
                with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                    contract.extract_tensors(path, "0" * 64, {"weight": (2,)})
                load.assert_not_called()

    def test_native_contiguous_tensor_bytes(self):
        raw = struct.pack("<2f", 0.5, -0.75)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            digest = checkpoint(path, raw=raw)
            self.assertEqual(contract.extract_tensors(path, digest, {"weight": (2,)}),
                             ([("weight", (2,), raw)], []))

    def test_lightning_generator_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            digest = checkpoint(path, name="generator.weight", nested=True)
            tensors, _ = contract.extract_tensors(path, digest, {"weight": (2,)})
            self.assertEqual(tensors[0][:2], ("weight", (2,)))

    def test_lightning_omegaconf_metadata_remains_inert(self):
        # Actual protocol-2 class keys/NEWOBJ/BUILD/defaultdict patterns authored
        # without importing Lightning, OmegaConf, NumPy or PyTorch.
        cfg = (b"comegaconf.dictconfig\nDictConfig\n)\x81}" + text("_metadata")
               + b"comegaconf.base\nContainerMetadata\n)\x81}"
               + text("object_type") + b"c__builtin__\ndict\ns"
               + text("ref_type") + b"ctyping\nDict\ns"
               + text("resolver_cache") + b"ccollections\ndefaultdict\n(c__builtin__\ndict\ntR"
               + text("cached") + integer(7) + b"ssbsb")
        base = state_pickle(name="generator.weight", nested=True)[:-1]
        program = (base + text("hyper_parameters") + cfg + b"s" + text("callbacks") + b"}"
                   + b"cpytorch_lightning.callbacks.model_checkpoint\nModelCheckpoint\n}" + b"ss.")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            digest = checkpoint(path, pickle_bytes=program)
            provenance = {}
            tensors, _ = contract.extract_tensors(path, digest, {"weight": (2,)}, provenance)
            self.assertEqual(tensors[0][0], "weight")
            self.assertIn("omegaconf.dictconfig.DictConfig", provenance["inert_metadata_globals"])
            self.assertIn("collections.defaultdict", provenance["inert_metadata_globals"])
            self.assertIs(provenance["metadata_globals_executed"], False)
            self.assertNotIn("omegaconf", sys.modules)
            self.assertNotIn("pytorch_lightning", sys.modules)

    def test_metadata_bytes_constructor_cannot_allocate(self):
        program = b"\x80\x02c__builtin__\nbytes\n(" + integer(2**30) + b"tR."
        contract.validate_pickle_program(program)
        import io
        value = contract._StateReader(io.BytesIO(program)).load()
        self.assertIsInstance(value, contract._InertMetadata)
        contract.validate_metadata_graph(value)
        self.assertEqual(value._arguments, (2**30,))

    def test_protocol2_builtin_type_aliases_remain_inert(self):
        import io
        import pickle
        for original in (dict, list, tuple, str, int, float, bool, bytes, object):
            with self.subTest(original=original.__name__):
                program = pickle.dumps(original, protocol=2)
                contract.validate_pickle_program(program)
                value = contract._StateReader(io.BytesIO(program)).load()
                self.assertIs(value, contract.INERT_METADATA_TYPES[("builtins", original.__name__)])
                self.assertIsNot(value, original)

    def test_torch_default_constructors_cannot_override_inert_adapters(self):
        import io
        for module, name in (("builtins", "set"), ("builtins", "bytearray"), ("builtins", "complex"), ("_codecs", "encode")):
            program = b"\x80\x02c" + module.encode() + b"\n" + name.encode() + b"\n."
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsupported pickle global"):
                contract._StateReader(io.BytesIO(program)).load()

    def test_deep_metadata_and_unknown_classes_rejected(self):
        value = None
        for _ in range(contract.MAX_METADATA_DEPTH + 2):
            value = [value]
        with self.assertRaisesRegex(ValueError, "depth"):
            contract.validate_metadata_graph(value)
        import io
        with self.assertRaisesRegex(ValueError, "unsupported pickle global"):
            contract._StateReader(io.BytesIO(b"\x80\x02come_gaconf\nDictConfig\n. ")).load()

    def test_counter_omitted_but_preserved_in_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            name = "bn.num_batches_tracked"
            digest = checkpoint(path, raw=struct.pack("<q", 42), name=name, shape=(), stride=(), count=1, dtype="LongStorage")
            self.assertEqual(contract.extract_tensors(path, digest, {name: ()}), ([], [{"name": name, "value": 42}]))
            digest = checkpoint(path, raw=struct.pack("<q", -1), name=name, shape=(), stride=(), count=1, dtype="LongStorage")
            with self.assertRaisesRegex(ValueError, "BatchNorm counter"):
                contract.extract_tensors(path, digest, {name: ()})

    def test_strided_tensor_and_view_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            raw = struct.pack("<6f", -3, 1, 2, 3, 4, -4)
            digest = checkpoint(path, raw=raw, shape=(2, 2), stride=(1, 2), count=6, offset=1)
            tensors, _ = contract.extract_tensors(path, digest, {"weight": (2, 2)})
            self.assertEqual(tensors[0][2], struct.pack("<4f", 1, 3, 2, 4))

    def test_out_of_storage_view_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            digest = checkpoint(path, offset=1)
            with self.assertRaisesRegex(ValueError, "view exceeds"):
                contract.extract_tensors(path, digest, {"weight": (2,)})

    def test_unknown_pickle_global_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            digest = checkpoint(path, pickle_bytes=b"\x80\x02cos\nsystem\n.")
            with self.assertRaisesRegex(ValueError, "unsupported pickle global"):
                contract.extract_tensors(path, digest, {"weight": (2,)})

    def test_memo_allocation_and_extension_cache_bypasses_rejected(self):
        for program in (b"\x80\x02}r\xff\xff\xff\x7f.", b"\x80\x02\x82\x01.",
                        b"\x80\x02\x83\x01\x00.", b"\x80\x02\x84\x01\x00\x00\x00."):
            with self.assertRaisesRegex(ValueError, "memo index|opcode"):
                contract.validate_pickle_program(program)

    def test_build_mutation_is_revalidated(self):
        tensor = contract._rebuild_tensor(contract._Storage("0", 2, 4), 0, (2,), (1,))
        tensor.__dict__["stride"] = (-1,)
        with self.assertRaisesRegex(ValueError, "tensor metadata"):
            contract._tensor_bytes(tensor, b"\0" * 8)
        tensor.__dict__["stride"] = (1,)
        tensor.storage.__dict__["key"] = "../../data.pkl"
        with self.assertRaisesRegex(ValueError, "storage metadata"):
            contract._tensor_bytes(tensor, b"\0" * 8)

    def test_discarded_training_tensor_views_are_bounded_before_torch(self):
        base = state_pickle(name="generator.weight")[:-1]
        extra = state_pickle(name="discriminator.unused", shape=(200000000,), stride=(0,))[3:-1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            digest = checkpoint(path, pickle_bytes=base + extra + b".")
            with self.assertRaisesRegex(ValueError, "view exceeds"):
                contract.extract_tensors(path, digest, {"weight": (2,)})

    def test_rebuild_rejects_invalid_even_before_object_becomes_reachable(self):
        with self.assertRaisesRegex(ValueError, "view exceeds"):
            contract._rebuild_tensor(contract._Storage("0", 2, 4), 0, (200000000,), (0,))
        tensor = contract._rebuild_tensor(contract._Storage("0", 2, 4), 0, (2,), (1,))
        with self.assertRaisesRegex(ValueError, "BUILD"):
            tensor.__setstate__({"stride": (-1,)})
        with self.assertRaisesRegex(ValueError, "BUILD"):
            tensor.storage.__setstate__({"count": 2**60})

    def test_nonfinite_and_negative_running_var_rejected(self):
        for name, value in (("weight", float("inf")), ("weight", float("nan")), ("bn.running_var", -1.0)):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "tiny.ckpt"
                digest = checkpoint(path, raw=struct.pack("<2f", 1, value), name=name)
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    contract.extract_tensors(path, digest, {name: (2,)})

    def test_wrong_tensor_names_shape_and_dtype_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            digest = checkpoint(path)
            for shapes in ({"other": (2,)}, {"weight": (1,)}):
                with self.assertRaises(ValueError):
                    contract.extract_tensors(path, digest, shapes)
            digest = checkpoint(path, raw=struct.pack("<2q", 1, 2), dtype="LongStorage")
            with self.assertRaisesRegex(ValueError, "shape/type"):
                contract.extract_tensors(path, digest, {"weight": (2,)})

    def test_legacy_no_byteorder_and_big_endian_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            digest = checkpoint(path, byteorder=None)
            contract.extract_tensors(path, digest, {"weight": (2,)})
            digest = checkpoint(path, byteorder=b"big")
            with self.assertRaisesRegex(ValueError, "little-endian"):
                contract.extract_tensors(path, digest, {"weight": (2,)})

    def test_trailing_pickle_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ckpt"
            digest = checkpoint(path, pickle_bytes=state_pickle() + b"extra")
            with self.assertRaisesRegex(ValueError, "trailing"):
                contract.extract_tensors(path, digest, {"weight": (2,)})

    def test_deterministic_gguf_preserves_convtranspose_iohw_bytes(self):
        tensors = [("model.24.weight", (2, 3, 1, 1), struct.pack("<6f", *range(6)))]
        metadata = contract.model_metadata("a" * 64, "b" * 64)
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "one.gguf", Path(directory) / "two.gguf"
            contract.write_f32_gguf(first, tensors, metadata)
            contract.write_f32_gguf(second, tensors, metadata)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            actual_metadata, actual = contract.read_f32_gguf(first)
            self.assertEqual(actual_metadata, metadata)
            self.assertEqual(actual, {name: (shape, raw) for name, shape, raw in tensors})

    def test_existing_output_preserved_and_failed_conversion_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.gguf"
            output.write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                converter.convert("unused", "a" * 64, "unused", output)
            self.assertEqual(output.read_bytes(), b"keep")
            output.unlink()
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"generator": contract.canonical_generator_config()}))
            path = Path(directory) / "tiny.ckpt"
            digest = checkpoint(path)
            with self.assertRaisesRegex(ValueError, "generator keys differ"):
                converter.convert(path, digest, config, output)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".json").exists())

    def test_owned_fixtures_and_trace_inventory(self):
        self.assertEqual(len(contract.expected_stage_names()), 713)
        self.assertEqual(len(set(contract.expected_stage_names())), 713)
        for name in contract.CASES:
            width, height, rgb, mask = contract.fixture(name)
            contract.validate_input(rgb, mask, width, height)
            self.assertEqual(contract.fixture_identity(name)["rgb_sha256"], hashlib.sha256(rgb).hexdigest())
        self.assertEqual(contract.fixture_identity("padded")["padded_width"], 40)

    def test_no_framework_imported(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)


if __name__ == "__main__":
    unittest.main()
