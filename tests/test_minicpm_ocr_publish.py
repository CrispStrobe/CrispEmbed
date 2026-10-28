"""Stdlib tests for tools/ci-heavy/minicpm_ocr_publish.py (no network, no weights, no Hub)."""
import copy
import hashlib
import importlib.util
import io
import json
import pathlib
import re
import shutil
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mcpm_publish", ROOT / "tools/ci-heavy/minicpm_ocr_publish.py")
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)


def fake_root(cfg_patch=None, parity_patch=None, bench=None):
    """Temporary repo root holding the protocol, source contract and (patched) receipts."""
    d = pathlib.Path(tempfile.mkdtemp())
    cfg = json.loads((ROOT / p.PROTOCOL).read_text())
    if cfg_patch:
        cfg_patch(cfg)
    for rel in (cfg["source_contract"], cfg["receipts"]["parity"]):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, d / rel)
    (d / p.PROTOCOL).parent.mkdir(parents=True, exist_ok=True)
    (d / p.PROTOCOL).write_text(json.dumps(cfg))
    if parity_patch:
        r = json.loads((d / cfg["receipts"]["parity"]).read_text())
        parity_patch(r)
        (d / cfg["receipts"]["parity"]).write_text(json.dumps(r))
    if bench is not None:
        (d / cfg["recipes"]["Q8_0"]["precision_receipt"]).write_text(json.dumps(bench))
    return d


def bench_receipt(exact):
    cfg,_=p.load()
    frozen=json.loads((ROOT/cfg['receipts']['parity']).read_text())
    cases=[dict(id=c['id'],ids_exact=exact or c['id']!='tall',exit=0,text_exact=True,cer=0,
                first_id_difference=None,n_tokens=len(c['reference_ids']),n_ref_tokens=len(c['reference_ids']))
           for c in frozen['cases']]
    return dict(schema='crispembed-minicpm-ocr-bench-v1',completed=True,
                reference_revision=cfg['upstream']['revision'],max_new_tokens=frozen['max_new_tokens'],commit='c'*40,
                arms=[dict(name=name,dtype='q8_0',cases=copy.deepcopy(cases)) for name in ('q8_0-t4','q8_0-t2')])


class Protocol(unittest.TestCase):
    def setUp(self):
        self.cfg, self.source = p.load()

    def test_pins_and_license(self):
        up = self.cfg["upstream"]
        self.assertEqual(self.cfg["destination"], "cstr/MiniCPM-OCR-1.0-GGUF")
        self.assertEqual(up["repo"], "openbmb/MiniCPM-OCR-1.0")
        self.assertEqual(up["revision"], self.source["revision"])
        self.assertEqual(up["license"], "apache-2.0")
        self.assertIn(up["license_file"]["commit"], up["license_file"]["url"])
        self.assertTrue(up["license_file"]["url"].startswith("https://raw.githubusercontent.com/OpenBMB/MiniCPM-OCR/"))
        self.assertRegex(up["license_file"]["sha256"], "^[0-9a-f]{64}$")
        self.assertEqual(up["readme"]["sha256"], self.source["files"]["README.md"]["sha256"])
        self.assertIn("TODO", self.cfg["qualification"]["registry_alias"])

    def test_variants(self):
        self.assertEqual(p.parse_variants("", self.cfg), ["F16"])
        self.assertEqual(p.parse_variants("q8_0, f16", self.cfg), ["Q8_0", "F16"])
        for bad in ("Q4_K", "F16,F16"):
            with self.assertRaises(ValueError):
                p.parse_variants(bad, self.cfg)

    def test_precision_rejects_partial_wrong_source_or_incomplete_selected_arms(self):
        for key,value in [('completed',False),('schema','other'),('reference_revision','0'*40),
                          ('max_new_tokens',1023),('commit','main')]:
            with self.subTest(key=key):
                report=bench_receipt(True);report[key]=value
                root=fake_root(bench=report);cfg,_=p.load(root)
                with self.assertRaises(ValueError):p.precision(cfg,'Q8_0',root)
        for change in ('missing-arm','duplicate-arm','failed-exit','missing-text','wrong-count','first-difference'):
            with self.subTest(change=change):
                report=bench_receipt(True)
                if change=='missing-arm':report['arms'].pop()
                elif change=='duplicate-arm':report['arms'].append(copy.deepcopy(report['arms'][0]))
                elif change=='failed-exit':report['arms'][1]['cases'][0]['exit']=1
                elif change=='missing-text':report['arms'][1]['cases'][0]['text_exact']=False
                elif change=='wrong-count':report['arms'][1]['cases'][0]['n_tokens']+=1
                else:report['arms'][1]['cases'][0]['first_id_difference']=0
                root=fake_root(bench=report);cfg,_=p.load(root)
                with self.assertRaises(ValueError):p.precision(cfg,'Q8_0',root)

    def test_plan(self):
        steps, drop = p.plan(self.cfg, ["F16"])
        self.assertEqual([(s[0], s[1], s[3]) for s in steps], [("convert", "F16", ["--dtype", "f16"])])
        self.assertFalse(drop)
        steps, drop = p.plan(self.cfg, ["Q8_0", "F16"])
        self.assertEqual([(s[0], s[1]) for s in steps], [("convert", "F32"), ("convert", "F16"), ("quantize", "Q8_0")])
        self.assertEqual(steps[0][3], ["--dtype", "f32"])
        self.assertEqual(steps[2][3], ["q8_0", "--tensor-type", "v.emb.patch_embedding.weight=f16"])
        self.assertTrue(drop)

    def test_parity_receipt(self):
        r = p.receipts(self.cfg)
        self.assertEqual(r["status"], "token-id-exact")
        self.assertEqual(list(r["cases"]), ["line", "para-en", "table", "tall", "tiny"])
        self.assertEqual(r["reference"]["revision"], self.cfg["upstream"]["revision"])

    def test_parity_refuses_regression(self):
        def broken(r):
            r["cases"][3]["arms"]["f16"]["ids_exact"] = False
        root = fake_root(parity_patch=broken)
        cfg, _ = p.load(root)
        with self.assertRaises(ValueError):
            p.receipts(cfg, root)

        def dropped(r):
            r["cases"].pop()
        root = fake_root(parity_patch=dropped)
        with self.assertRaises(ValueError):
            p.receipts(p.load(root)[0], root)

    def test_q8_requires_exact_precision_receipt(self):
        self.assertIsNone(p.precision(self.cfg, "F16"))
        root = fake_root()  # no receipt at all
        with self.assertRaises(ValueError):
            p.precision(p.load(root)[0], "Q8_0", root)
        root = fake_root(bench=bench_receipt(exact=False))
        with self.assertRaises(ValueError):
            p.precision(p.load(root)[0], "Q8_0", root)
        root = fake_root(bench=bench_receipt(exact=True))
        r = p.precision(p.load(root)[0], "Q8_0", root)
        self.assertTrue(r["ids_exact"])
        self.assertEqual(r["arms"], ["q8_0-t4", "q8_0-t2"])

    def test_unapproved_or_rerouted(self):
        for patch in (lambda c: c.update(approved_by_owner=False), lambda c: c.update(destination="cstr/other"),
                      lambda c: c["upstream"].update(license="mit")):
            with self.assertRaises(ValueError):
                p.load(fake_root(cfg_patch=patch))

    def test_card(self):
        files = [{"variant": "F16", "filename": "minicpm-ocr-1.0-f16.gguf", "bytes": 2_700_000_000, "sha256": "a" * 64}]
        text = p.card(self.cfg, files, p.receipts(self.cfg), {"F16": None}, "b" * 40)
        self.assertTrue(text.startswith("---\nlicense: apache-2.0\n"))
        self.assertIn("base_model: openbmb/MiniCPM-OCR-1.0\n", text)
        self.assertIn(self.cfg["upstream"]["revision"], text)
        self.assertIn(self.cfg["upstream"]["license_file"]["sha256"], text)
        self.assertIn("Copyright OpenBMB", text)
        self.assertIn("Statement of changes", text)
        self.assertIn("format-converted", text)
        self.assertIn("Only unquantized precisions are published", text)
        self.assertIn("a" * 64, text)
        self.assertNotIn("quantized**", text)
        for leak in ("/home/", "/mnt/", "/kaggle/"):
            self.assertNotIn(leak, text)

    def test_manifest_lists_every_hash(self):
        files = [{"variant": "F16", "filename": "minicpm-ocr-1.0-f16.gguf", "bytes": 1, "sha256": "a" * 64}]
        m = p.manifest(self.cfg, files, p.receipts(self.cfg), {"F16": None}, "b" * 40, "c" * 64)
        self.assertEqual(m["files"], files)
        self.assertEqual(m["readme_sha256"], "c" * 64)
        self.assertEqual(m["precision"], {})
        self.assertTrue(re.fullmatch("[0-9a-f]{64}", m["protocol_sha256"]))

    def test_notices_verify_hashes(self):
        up = self.cfg["upstream"]

        def fake(content):
            return lambda url, timeout: io.BytesIO(content)
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            p.notices(self.cfg, d, opener=fake(b"tampered"))
        good = b"x" * up["license_file"]["bytes"]
        expected = dict(bytes=len(good), sha256=hashlib.sha256(good).hexdigest())
        self.assertEqual(p.fetch("u", expected, opener=fake(good)), good)

    def test_publish_refuses_without_commit(self):
        with self.assertRaises(ValueError):
            p.publish(copy.deepcopy(self.cfg), ["F16"], "/nonexistent", "/nonexistent", pathlib.Path("."), "HEAD")


if __name__ == "__main__":
    unittest.main()
