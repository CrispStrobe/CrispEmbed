"""Pure pinned-profile controls; no model libraries or checkpoint payloads."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen35_ovis_profile", ROOT / "models/qwen35_ovis_profile.py")
profile_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile_module)


class OvisProfileControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = json.loads((ROOT / "tests/results/qwen35-ocr/ovisocr2-source-contract.json").read_text())
        cls.config = source["models"][profile_module.MODEL]["config"]
        cls.producer = profile_module._verified_evidence()

    def _loaded_profile(self, directory):
        (directory / "config.json").write_text(json.dumps(self.config))
        with mock.patch.object(profile_module, "_file_sha256",
                               side_effect=lambda path: self.producer["checkpoint_files_sha256"][Path(path).name]):
            return profile_module.load_profile(directory, profile_module.MODEL, profile_module.REVISION)

    def test_reviewed_proof_and_exact_config_allow_only_pinned_mtp(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile = self._loaded_profile(Path(temporary))
        self.assertEqual(profile["stop_ids"], [248044, 248046])
        self.assertEqual(profile["qualification"]["loaded_source_sha256"],
                         self.producer["loaded_source_sha256"])
        self.assertEqual(profile["qualification"]["transformers_version"], "5.16.1")
        self.assertFalse(profile["native_alias_qualified"])
        self.assertFalse(profile["image_max_qualified"])
        self.assertIsNone(profile_module.validate_config(self.config, profile))
        for mutation in (
            lambda p: p.update(proof_marker="forged"),
            lambda p: p.update(source_revision="0" * 40),
            lambda p: p.update(stop_ids=[248046]),
            lambda p: p.update(rendered_prompt="different prompt"),
            lambda p: p["generation"].update(eos_token_id=248046),
            lambda p: p["qualification"].update(loaded_source_sha256={}),
        ):
            changed = copy.deepcopy(profile)
            mutation(changed)
            with self.assertRaises(ValueError):
                profile_module.validate_config(self.config, changed)
        for key, value in (("mtp_num_hidden_layers", 0), ("mtp_num_hidden_layers", 2),
                           ("mtp_use_dedicated_embeddings", True)):
            changed = copy.deepcopy(self.config)
            changed["text_config"][key] = value
            with self.assertRaises(ValueError):
                profile_module.validate_config(changed, profile)
        changed = copy.deepcopy(self.config)
        changed["unaudited_option"] = True
        with self.assertRaises(ValueError):
            profile_module.validate_config(changed, profile)

    def test_source_identity_missing_files_and_source_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text(json.dumps(self.config))
            with self.assertRaises(ValueError):
                profile_module.load_profile(directory, "other/model", profile_module.REVISION)
            with self.assertRaises(ValueError):
                profile_module.load_profile(directory, profile_module.MODEL, "0" * 40)
            (directory / "generation_config.json").write_text("{}")
            with self.assertRaises(ValueError):
                profile_module.load_profile(directory, profile_module.MODEL, profile_module.REVISION)
            (directory / "generation_config.json").unlink()
            (directory / "processor_config.json").write_text("{}")
            with self.assertRaises(ValueError):
                profile_module.load_profile(directory, profile_module.MODEL, profile_module.REVISION)
            (directory / "processor_config.json").unlink()
            with self.assertRaises(FileNotFoundError):
                profile_module.load_profile(directory, profile_module.MODEL, profile_module.REVISION)
            with mock.patch.object(profile_module, "_file_sha256", return_value="0" * 64):
                with self.assertRaises(ValueError):
                    profile_module.load_profile(directory, profile_module.MODEL, profile_module.REVISION)

    def test_tampered_bundled_producer_rejected_before_source_trust(self):
        evidence = json.loads(profile_module._EVIDENCE.read_text())
        evidence["producer"]["resolved_generation"]["eos_token_id"] = 248046
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "evidence.json"
            altered.write_text(json.dumps(evidence))
            with mock.patch.object(profile_module, "_EVIDENCE", altered):
                with self.assertRaisesRegex(ValueError, "producer proof changed"):
                    profile_module._verified_evidence()

    def test_checkpoint_size_and_digest_rejected_without_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            profile = self._loaded_profile(directory)
            small = directory / "model.safetensors"
            small.write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(ValueError, "byte count"):
                profile_module.verify_checkpoint(small, profile)
            with mock.patch.object(profile_module.Path, "stat", return_value=mock.Mock(st_size=profile_module.CHECKPOINT_BYTES)):
                with mock.patch.object(profile_module, "_file_sha256", return_value="0" * 64):
                    with self.assertRaisesRegex(ValueError, "SHA256"):
                        profile_module.verify_checkpoint(small, profile)

    def test_converter_default_rejects_mtp_and_profile_preserves_config_checks(self):
        converter_spec=importlib.util.spec_from_file_location('ovis_converter',ROOT/'models/convert-qwen35-ocr-to-gguf.py')
        converter=importlib.util.module_from_spec(converter_spec);converter_spec.loader.exec_module(converter)
        with self.assertRaisesRegex(ValueError,'MTP checkpoints'):
            converter.validate_config(self.config)
        with tempfile.TemporaryDirectory() as temporary:
            profile=self._loaded_profile(Path(temporary))
        text,vision=converter.validate_config(self.config,profile)
        self.assertEqual(text['mtp_num_hidden_layers'],1)
        self.assertEqual(vision['patch_size'],16)
        with self.assertRaises(ValueError):converter.validate_config(self.config,{**profile,'stop_ids':[248046]})
    def test_absent_generation_only_uses_explicit_derived_metadata(self):
        converter_spec=importlib.util.spec_from_file_location('ovis_converter_tokens',ROOT/'models/convert-qwen35-ocr-to-gguf.py')
        converter=importlib.util.module_from_spec(converter_spec);converter_spec.loader.exec_module(converter)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            tokenizer={'model':{'type':'BPE','vocab':{'<pad>':0,'<end>':1,'x':2},'merges':[]},
                       'added_tokens':[],'normalizer':{'type':'NFC'},
                       'pre_tokenizer':{'pretokenizers':[{'pattern':{'Regex':'x'}}]}}
            (path/'tokenizer.json').write_text(json.dumps(tokenizer))
            (path/'tokenizer_config.json').write_text(json.dumps({'eos_token':'<end>','pad_token':'<pad>'}))
            with self.assertRaises(FileNotFoundError):converter.read_tokenizer_metadata(path,3)
            result=converter.read_tokenizer_metadata(path,3,generation={'eos_token_id':0})
            self.assertEqual(result['generation_eos'],0)
            self.assertEqual(result['tokenizer_eos'],1)
            self.assertFalse((path/'generation_config.json').exists())
            with self.assertRaises(ValueError):converter.read_tokenizer_metadata(path,3,generation={'eos_token_id':[0,1]})
    def test_shard_index_rejected_before_full_payload_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);(path/'model.safetensors.index.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'shard index'):
                profile_module.load_profile(path,profile_module.MODEL,profile_module.REVISION)

    def test_cli_checks_metadata_before_any_weight_download(self):
        import sys,types,contextlib,io
        converter_spec=importlib.util.spec_from_file_location('ovis_converter_cli',ROOT/'models/convert-qwen35-ocr-to-gguf.py')
        converter=importlib.util.module_from_spec(converter_spec);converter_spec.loader.exec_module(converter)
        fake=types.ModuleType('huggingface_hub');fake.snapshot_download=mock.Mock()
        with mock.patch.dict(sys.modules,{'huggingface_hub':fake}),contextlib.redirect_stderr(io.StringIO()):
            for extra in [['--model',profile_module.MODEL,'--revision',profile_module.REVISION],
                          ['--model','other/model','--revision',profile_module.REVISION,'--profile','ovisocr2'],
                          ['--model',profile_module.MODEL,'--revision','0'*40,'--profile','ovisocr2']]:
                with mock.patch.object(sys,'argv',['convert','--output','unused',*extra]),self.assertRaises(SystemExit):converter.main()
            fake.snapshot_download.assert_not_called()
            with tempfile.TemporaryDirectory() as d:
                fake.snapshot_download.return_value=d
                with mock.patch.object(sys,'argv',['convert','--output','unused','--model',profile_module.MODEL,
                                       '--revision',profile_module.REVISION,'--profile','ovisocr2']),self.assertRaises(FileNotFoundError):converter.main()
                self.assertEqual(fake.snapshot_download.call_count,1)
                self.assertNotIn('*.safetensors',fake.snapshot_download.call_args.kwargs['allow_patterns'])
                self.assertEqual(set(fake.snapshot_download.call_args.kwargs['allow_patterns']),set(self.producer['checkpoint_files_sha256']))
                self.assertIn('README.md',fake.snapshot_download.call_args.kwargs['allow_patterns'])

    def test_no_model_imports(self):
        import sys
        for name in ("torch", "transformers", "safetensors", "gguf"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
