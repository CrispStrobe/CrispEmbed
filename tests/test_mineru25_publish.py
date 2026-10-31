"""Stdlib tests for tools/ci-heavy/mineru25_publish.py (no network, no weights, no Hub)."""
import hashlib
import importlib.util
import io
import json
import pathlib
import shutil
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mineru_publish", ROOT / "tools/ci-heavy/mineru25_publish.py")
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)
CFG = json.loads((ROOT / p.PROTOCOL).read_text())
CASES = CFG["parity_cases"]


def parity(exact=True, cases=CASES):
    out = []
    for i, c in enumerate(cases):
        ids = [i, 151645]
        arm = {"exit": 0, "ids_exact": True, "ids": ids}
        out.append({"id": c, "reference_ids": ids, "arms": {"f32": dict(arm), "f32-reuse": dict(arm)}})
    if not exact:
        out[-1]["arms"]["f32-reuse"].update(ids_exact=False, ids=[0], exit=1)
    return {"completed": True, "commit": "a" * 40, "reference_revision": CFG["upstream"]["revision"],
            "reference_versions": {"transformers": "4.57.3", "mineru_vl_utils": "c" * 40},
            "max_new_tokens": 2048, "cases": out}


def precision(dtypes, broken=None):
    return {"commit": "a" * 40, "arms": [
        {"name": f"{dt}-cpu", "dtype": dt, "cases": [{"id": c, "ids_exact": not (dt == broken and c == CASES[0])}
                                                   for c in CASES]} for dt in dtypes]}


def fake_root(parity_receipt=None, precision_receipt=None):
    d = pathlib.Path(tempfile.mkdtemp())
    for rel in (p.PROTOCOL, CFG["source_contract"]):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, d / rel)
    res = d / "tests/results/mineru25"
    res.mkdir(parents=True)
    if parity_receipt is not None:
        (res / "native-parity.json").write_text(json.dumps(parity_receipt))
    if precision_receipt is not None:
        (res / "precision.json").write_text(json.dumps(precision_receipt))
    return d


class Protocol(unittest.TestCase):
    def test_pins_and_license(self):
        cfg, source = p.load()
        up = cfg["upstream"]
        self.assertEqual(cfg["destination"], "cstr/MinerU2.5-Pro-2604-1.2B-GGUF")
        self.assertEqual((up["repo"], up["revision"]), (source["repository"], source["revision"]))
        self.assertEqual(up["license"], "apache-2.0")
        self.assertIn(up["license_file"]["commit"], up["license_file"]["url"])
        lic = (ROOT / "licenses/mineru-vl-utils-LICENSE").read_bytes()
        self.assertEqual(hashlib.sha256(lic).hexdigest(), up["license_file"]["sha256"])
        self.assertEqual(len(lic), up["license_file"]["bytes"])
        self.assertEqual(sorted(cfg["case_inputs"]), sorted(CASES))
        for image, task in cfg["case_inputs"].values():
            self.assertTrue((ROOT / "tests/regression/images" / image).is_file(), image)
            self.assertIn(task, ("layout", "text", "table", "formula", "image"))

    def test_plan_converts_f32_first_and_drops_it(self):
        steps, drop = p.plan(CFG, ["Q4_K", "F16"])
        self.assertEqual([s[:2] for s in steps], [("convert", "F32"), ("convert", "F16"), ("quantize", "Q4_K")])
        self.assertTrue(drop)
        mixed = {s[1]: s[3] for s in p.plan(CFG, ["Q4_K_DQ8", "Q4_K_DF16"])[0]}
        self.assertEqual(mixed["Q4_K_DQ8"], ["q4_k", "--tensor-type", "l.*=q8_0"])
        self.assertEqual(mixed["Q4_K_DF16"], ["q4_k", "--decoder-f16"])
        self.assertEqual(p.plan(CFG, ["F16"]), ([("convert", "F16", "mineru25-pro-2604-f16.gguf", ["--dtype", "f16"])], False))

    def test_variants(self):
        self.assertEqual(p.parse_variants("", CFG), ["F16"])
        with self.assertRaises(ValueError):
            p.parse_variants("Q5_K", CFG)
        with self.assertRaises(ValueError):
            p.parse_variants("F16,f16", CFG)


class Receipts(unittest.TestCase):
    def test_missing_receipt_refuses(self):
        with self.assertRaises(FileNotFoundError):
            p.receipts(CFG, fake_root())

    def test_exact_receipt_passes(self):
        r = p.receipts(CFG, fake_root(parity()))
        self.assertEqual(list(r["cases"]), CASES)

    def test_one_inexact_arm_refuses(self):
        with self.assertRaises(ValueError):
            p.receipts(CFG, fake_root(parity(exact=False)))

    def test_missing_case_refuses(self):
        with self.assertRaises(ValueError):
            p.receipts(CFG, fake_root(parity(cases=CASES[:-1])))

    def test_precision_requires_every_case_exact(self):
        root = fake_root(parity(), precision(["f16", "q8_0"], broken="q8_0"))
        self.assertTrue(p.precision(CFG, "F16", root)["ids_exact"])
        with self.assertRaises(ValueError):
            p.precision(CFG, "Q8_0", root)
        with self.assertRaises(ValueError):  # no q4_k arm at all
            p.precision(CFG, "Q4_K", root)
        self.assertIsNone(p.precision(CFG, "F32", root))


class Notices(unittest.TestCase):
    def test_hash_mismatch_refuses(self):
        bad = lambda url, timeout: io.BytesIO(b"x" * CFG["upstream"]["license_file"]["bytes"])
        with self.assertRaises(ValueError):
            p.notices(CFG, tempfile.mkdtemp(), opener=bad)


class Card(unittest.TestCase):
    def test_card_states_license_changes_and_scope(self):
        par = p.receipts(CFG, fake_root(parity()))
        files = [{"variant": "F16", "filename": "mineru25-pro-2604-f16.gguf", "bytes": 10, "sha256": "0" * 64}]
        text = p.card(CFG, files, par, {"F16": {"receipt": "tests/results/mineru25/precision.json"}}, "b" * 40)
        self.assertTrue(text.startswith("---\nlicense: apache-2.0\n"))
        for needle in ("Statement of changes", "LICENSE.md", CFG["upstream"]["revision"], "CRISPEMBED_MINERU25_PRO=1",
                       "**F16** is token-ID exact", "Not verified"):
            self.assertIn(needle, text)
        self.assertNotIn("MiniCPM", text)


if __name__ == "__main__":
    unittest.main()
