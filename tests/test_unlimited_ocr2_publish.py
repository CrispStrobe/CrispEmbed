"""Stdlib tests for tools/ci-heavy/unlimited_ocr2_publish.py (no network, no weights, no Hub)."""
import hashlib
import copy
import json
import sys
import types
from unittest.mock import patch
import importlib.util
import io
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("uocr2_publish", ROOT / "tools/ci-heavy/unlimited_ocr2_publish.py")
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)


class Protocol(unittest.TestCase):
    def setUp(self):
        self.cfg, self.source = p.load()

    def test_pins_and_license(self):
        up = self.cfg["upstream"]
        self.assertEqual(self.cfg["destination"], "cstr/Unlimited-OCR-2-GGUF")
        self.assertEqual(up["revision"], self.source["revision"])
        self.assertEqual(up["license"], "mit")
        self.assertIn(up["license_file"]["commit"], up["license_file"]["url"])

    def test_variants(self):
        self.assertEqual(p.parse_variants("", self.cfg), ["F16", "Q8_0"])
        self.assertEqual(p.parse_variants("q8_0, f16", self.cfg), ["Q8_0", "F16"])
        for bad in ("Q3_K", "F16,F16"):
            with self.assertRaises(ValueError):
                p.parse_variants(bad, self.cfg)

    def test_plan_orders_conversions_first(self):
        steps, drop = p.plan(self.cfg, ["Q8_0", "F16"])
        self.assertEqual([(s[0], s[1]) for s in steps], [("convert", "F32"), ("convert", "F16"), ("quantize", "Q8_0")])
        self.assertEqual(steps[1][3], ["--fp16"])
        self.assertEqual(steps[2][3], ["q8_0"])
        self.assertTrue(drop)
        steps, drop = p.plan(self.cfg, ["F16"])
        self.assertEqual([s[1] for s in steps], ["F16"])
        self.assertFalse(drop)
        steps, drop = p.plan(self.cfg, ["F32", "Q4_K"])
        self.assertEqual([s[1] for s in steps], ["F32", "Q4_K"])
        self.assertFalse(drop)

    def test_receipts_summary(self):
        r = p.receipts(self.cfg)
        self.assertTrue(r["untiled"]["ids_exact"])
        self.assertEqual(r["tiled"]["status"], "token-id-exact")
        self.assertTrue(r["tiled"]["ids_exact"])

    def test_receipts_refuse_regression(self):
        # the superseded text-only tiled receipt must not pass the token-ID gate
        bad = dict(self.cfg, receipts=dict(self.cfg["receipts"], tiled="tests/results/unlimited-ocr2/native-tiled.json"))
        with self.assertRaises(ValueError):
            p.receipts(bad)

    def test_card(self):
        files = [{"variant": "F16", "filename": "unlimited-ocr-2-f16.gguf", "bytes": 6_700_000_000, "sha256": "a" * 64}]
        text = p.card(self.cfg, files, p.receipts(self.cfg), "b" * 40)
        self.assertTrue(text.startswith("---\nlicense: mit\n"))
        self.assertIn("base_model: baidu/Unlimited-OCR-2\n", text)
        self.assertIn(self.cfg["upstream"]["revision"], text)
        self.assertIn("a" * 64, text)
        self.assertIn("Tiled pages", text)
        self.assertIn("Copyright (c) 2026 Baidu", text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("/mnt/", text)

    def test_card_refresh_requires_original_bytes_ids_and_default_tiling(self):
        trial = json.loads((ROOT / "tests/results/unlimited-ocr2/trial.json").read_text())
        rows = []
        for case in trial["reference"]["cases"]:
            if case["id"] in {"table", "receipt", "paragraphs", "german"}:
                rows.append(dict(tag="after", alias="unlimited-ocr2", page=case["id"], exit=0,
                                 ids=case["ids"] if isinstance(case["ids"], list) else json.loads(case["ids"]),
                                 text=case["text"], env={}))
        verified = dict(schema="crispembed-unlimited-ocr2-verify-v1", commit="8ca9d3de2a8160e6f5235156d3550c9719583744",
                        scope="text", tests={name: dict(exit=0) for name in ["test-unlimited-ocr2-text", "test-unlimited-ocr2-tiles"]},
                        page_sha_matches_trial={"table": True}, runs=rows)
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder); directory = root / "tests/results/unlimited-ocr2"; directory.mkdir(parents=True)
            (directory / "trial.json").write_text(json.dumps(trial))
            target = directory / "verify-text-v2.json"; target.write_text(json.dumps(verified))
            self.assertTrue(p.trial_text_receipt(root)["raw_text_bytes_exact"])
            for mutate in (lambda v: v["runs"][0].update(text="changed"),
                           lambda v: v["runs"][0].update(ids=[1]),
                           lambda v: v["runs"][0]["env"].update(CRISPEMBED_UNLIMITED_OCR2_TILES="1"),
                           lambda v: v.update(commit="0" * 40), lambda v: v["runs"].pop()):
                bad = copy.deepcopy(verified); mutate(bad); target.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): p.trial_text_receipt(root)

    def test_card_only_payload_preserves_files_source_and_documents_fallback(self):
        publication = json.loads((ROOT / "tests/results/unlimited-ocr2/publication.json").read_text())
        previous = dict(schema="crispembed-unlimited-ocr2-publication-manifest-v1", destination=self.cfg["destination"],
                        files=publication["remote"]["files"], source_commit="b" * 40)
        proof = dict(source="8ca9d3de2a8160e6f5235156d3550c9719583744")
        files = p.refresh_payload(self.cfg, previous, p.receipts(self.cfg), proof)
        self.assertEqual(set(files), {"README.md", "manifest.json"})
        result = json.loads(files["manifest.json"])
        self.assertEqual(result["files"], previous["files"]); self.assertEqual(result["source_commit"], previous["source_commit"])
        text = files["README.md"].decode(); self.assertIn("128000 base pieces", text); self.assertIn("pinned831", text)
        self.assertIn("published weights", text); self.assertIn("unchanged", text); self.assertIn("Q8_0", text)

    def test_actual_card_refresh_operation_allowlist_parent_and_small_readback(self):
        cfg = copy.deepcopy(self.cfg); parent, new = "a" * 40, "b" * 40
        weights = [dict(variant="F16", filename=cfg["recipes"]["F16"]["filename"], bytes=2_000_000_000, sha256="c" * 64)]
        previous = dict(schema="crispembed-unlimited-ocr2-publication-manifest-v1", destination=cfg["destination"],
                        files=weights, source_commit="d" * 40)
        old = {"manifest.json": json.dumps(previous).encode(), "README.md": b"previous card", "LICENSE": b"license", "UPSTREAM_README.md": b"upstream"}
        hashes = {k: hashlib.sha256(v).hexdigest() for k, v in old.items()}
        cfg["upstream"]["license_file"]["sha256"] = hashes["LICENSE"]; cfg["upstream"]["readme"]["sha256"] = hashes["UPSTREAM_README.md"]
        proof = dict(source="8ca9d3de2a8160e6f5235156d3550c9719583744")
        operations = []; latest = dict(old)
        class Operation:
            def __init__(self, path_in_repo, path_or_fileobj): self.path_in_repo, self.data = path_in_repo, path_or_fileobj
        class API:
            def __init__(self, token): pass
            def model_info(self, *args, **kwargs):
                return types.SimpleNamespace(sha=parent, siblings=[types.SimpleNamespace(rfilename=w["filename"], size=w["bytes"], lfs=types.SimpleNamespace(sha256=w["sha256"])) for w in weights])
            def create_commit(self, **kwargs):
                self_outer.assertEqual(kwargs["parent_commit"], parent)
                operations.extend(kwargs["operations"])
                for row in operations: latest[row.path_in_repo] = row.data
                return types.SimpleNamespace(oid=new)
        self_outer = self
        def opener(url, timeout):
            revision, name = url.split("/resolve/", 1)[1].split("/", 1)
            return io.BytesIO((old if revision == parent else latest)[name])
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder); directory = root / "tests/results/unlimited-ocr2"; directory.mkdir(parents=True)
            (directory / "publication.json").write_text(json.dumps(dict(card_update=dict(hub_commit=parent, **{k: hashes[k] for k in ("README.md", "manifest.json")}), remote=dict(files=weights))))
            client = types.SimpleNamespace(CommitOperationAdd=Operation, HfApi=API)
            with patch.object(p, "trial_text_receipt", return_value=proof), patch.object(p, "receipts", return_value=p.receipts(self.cfg)), patch.object(p.urllib.request, "urlopen", side_effect=opener), patch.dict(sys.modules, huggingface_hub=client), patch.dict(p.os.environ, HF_TOKEN="owned-test-placeholder"):
                result = p.refresh_card(cfg, root / "out", root)
            self.assertEqual([row.path_in_repo for row in operations], ["README.md", "manifest.json"])
            self.assertTrue(result["weights_unchanged"]); self.assertTrue(result["notices_unchanged"])
            self.assertEqual(json.loads(latest["manifest.json"])["source_commit"], previous["source_commit"])

    def test_notices_verify_hashes(self):
        up = self.cfg["upstream"]

        def fake(content):
            return lambda url, timeout: io.BytesIO(content)
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            p.notices(self.cfg, d, opener=fake(b"tampered"))
        good = b"x" * up["license_file"]["bytes"]
        expected = dict(bytes=len(good), sha256=hashlib.sha256(good).hexdigest())
        self.assertEqual(p.fetch("u", expected, opener=fake(good)), good)


if __name__ == "__main__":
    unittest.main()
