"""Stdlib tests for tools/ci-heavy/ocr_gguf_publish.py (no network, no weights, no Hub)."""
import hashlib
import importlib.util
import io
import json
import pathlib
import shutil
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ocr_publish", ROOT / "tools/ci-heavy/ocr_gguf_publish.py")
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)
PROTOCOLS = {"mineru": "docs/mineru25-pro-publication-v1.json", "teleocr": "docs/teleocr-publication-v1.json",
             "oxr": "docs/oxr-publication-v1.json", "oxr-dfine": "docs/oxr-dfine-publication-v1.json"}


def cfg_of(name):
    return p.load(PROTOCOLS[name])


def parity(cfg, exact=True, cases=None):
    cases = cfg["parity_cases"] if cases is None else cases
    out = []
    for i, c in enumerate(cases):
        ids = [i, 151645]
        arms = {a: {"exit": 0, "ids_exact": True, "ids": list(ids)} for a in cfg["parity_arms"]}
        out.append({"id": c, "reference_ids": ids, "arms": arms})
    if not exact:
        out[-1]["arms"][cfg["parity_arms"][-1]].update(ids_exact=False, ids=[0], exit=1)
    return {"completed": True, "commit": "a" * 40, "reference_revision": cfg["upstream"]["revision"],
            "reference_versions": {"transformers": "4.57"}, "max_new_tokens": 1024, "cases": out}


def precision(cfg, dtypes, broken=None):
    return {"commit": "a" * 40, "arms": [
        {"name": f"{dt}-cpu", "dtype": dt,
         "cases": [{"id": c, "ids_exact": not (dt == broken and c == cfg["parity_cases"][0])}
                   for c in cfg["parity_cases"]]} for dt in dtypes]}


def fake_root(cfg, parity_receipt=None, precision_receipt=None):
    d = pathlib.Path(tempfile.mkdtemp())
    for rel in (cfg["_protocol"], cfg["converter"]["script"]):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, d / rel)
    if parity_receipt is not None:
        path = d / cfg["receipts"]["parity"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(parity_receipt))
    if precision_receipt is not None:
        rel = next(r["precision_receipt"] for r in cfg["recipes"].values() if "precision_receipt" in r)
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(json.dumps(precision_receipt))
    return d


class Protocols(unittest.TestCase):
    def test_pins_license_and_cases(self):
        for name in PROTOCOLS:
            with self.subTest(name=name):
                cfg = cfg_of(name)
                self.assertIn(cfg["destination"], p.ALLOWED_DESTINATIONS)
                self.assertEqual(cfg["upstream"]["license"], {"mineru": "apache-2.0", "teleocr": "qwen-research",
                                                              "oxr": "oxr", "oxr-dfine": "oxr"}[name])
                self.assertTrue(set(cfg["default_variants"]) <= set(cfg["recipes"]))
                self.assertEqual(len(set(cfg["parity_cases"])), len(cfg["parity_cases"]))

    def test_vendored_license_bytes_match_pins(self):
        vendored = {("mineru", "LICENSE"): "licenses/mineru-vl-utils-LICENSE",
                    ("teleocr", "LICENSE"): "models/licenses/qwen-research/LICENSE",
                    ("teleocr", "LICENSE-APACHE-2.0"): "models/licenses/apache-2.0/LICENSE"}
        for (name, shipped), rel in vendored.items():
            with self.subTest(name=name, file=shipped):
                lf = {f["name"]: f for f in p.license_files(cfg_of(name))}[shipped]
                data = (ROOT / rel).read_bytes()
                self.assertEqual((len(data), hashlib.sha256(data).hexdigest()), (lf["bytes"], lf["sha256"]))

    def test_teleocr_qwen_research_obligations(self):
        cfg = cfg_of("teleocr")
        self.assertEqual([f["name"] for f in p.license_files(cfg)], ["LICENSE", "LICENSE-APACHE-2.0"])
        self.assertIn("Qwen/Qwen2.5-VL-3B-Instruct/resolve/", p.license_files(cfg)[0]["url"])
        notice = cfg["notice_text"]
        self.assertIn("Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. "
                      "All Rights Reserved.", notice)  # agreement section 3c, verbatim
        self.assertIn("Built with Qwen", notice)  # section 4b
        self.assertTrue((ROOT / cfg["upstream"]["license_lineage"]["evidence"]).is_file())

    def test_mineru_source_pins_match_contract(self):
        cfg = cfg_of("mineru")
        source = json.loads((ROOT / cfg["source_contract"]).read_text())
        self.assertEqual((cfg["upstream"]["repo"], cfg["upstream"]["revision"]),
                         (source["repository"], source["revision"]))
        for image, task in cfg["case_inputs"].values():
            self.assertTrue((ROOT / "tests/regression/images" / image).is_file(), image)
            self.assertIn(task, ("layout", "text", "table", "formula", "image"))

    def test_plans(self):
        mineru = cfg_of("mineru")
        steps, drop = p.plan(mineru, ["Q8_0", "F16"])
        self.assertEqual([s[:2] for s in steps], [("convert", "F32"), ("convert", "F16"), ("quantize", "Q8_0")])
        self.assertEqual(steps[-1][3], ["mineru25-pro-2604-f32.gguf", "q8_0"])
        self.assertEqual(drop, ["mineru25-pro-2604-f32.gguf"])
        tele = cfg_of("teleocr")
        steps, drop = p.plan(tele, ["Q4_K", "Q8_0"])
        self.assertEqual([s[:2] for s in steps], [("convert", "F16"), ("convert", "Q8_0"), ("quantize", "Q4_K")])
        self.assertEqual(steps[-1][3], ["teleocr-f16.gguf", "q4_k", "--tensor-type", "v.*=f16",
                                        "--tensor-type", "l.embed_tokens.weight=f16"])
        self.assertEqual(drop, ["teleocr-f16.gguf"])

    def test_variants(self):
        cfg = cfg_of("mineru")
        self.assertEqual(p.parse_variants("", cfg), ["F16", "Q8_0"])
        for bad in ("Q5_K", "F16,f16"):
            with self.assertRaises(ValueError):
                p.parse_variants(bad, cfg)


class Receipts(unittest.TestCase):
    def test_parity_gate(self):
        for name in PROTOCOLS:
            if cfg_of(name)["receipts"].get("kind") == "layout":
                continue  # token-ID receipts; layout receipts are covered by OxrPublication
            with self.subTest(name=name):
                cfg = cfg_of(name)
                with self.assertRaises(FileNotFoundError):
                    p.receipts(cfg, fake_root(cfg))
                self.assertEqual(list(p.receipts(cfg, fake_root(cfg, parity(cfg)))["cases"]), cfg["parity_cases"])
                for bad in (parity(cfg, exact=False), parity(cfg, cases=cfg["parity_cases"][:-1])):
                    with self.assertRaises(ValueError):
                        p.receipts(cfg, fake_root(cfg, bad))

    def test_precision_requires_every_case_exact(self):
        cfg = cfg_of("teleocr")
        root = fake_root(cfg, parity(cfg), precision(cfg, ["f16", "q8_0"], broken="q8_0"))
        self.assertTrue(p.precision(cfg, "F16", root)["ids_exact"])
        with self.assertRaises(ValueError):
            p.precision(cfg, "Q8_0", root)
        with self.assertRaises(ValueError):  # no q4_k arm at all
            p.precision(cfg, "Q4_K", root)
        self.assertIsNone(p.precision(cfg, "F32", root))


    def test_cosine_gate_uses_owner_bars(self):
        cfg = cfg_of("teleocr")
        rec = cfg["recipes"]["Q8_0"]
        rec.update(precision_gate="cosine")
        receipt = precision(cfg, ["q8_0"], broken="q8_0")  # one token flip
        for c in receipt["arms"][0]["cases"]:
            c["final_cos_min"] = 0.9988
        root = fake_root(cfg, parity(cfg), receipt)
        q = p.precision(cfg, "Q8_0", root)
        self.assertEqual((q["gate"], q["bar"], q["ids_exact"]), ("cosine", 0.97, False))
        receipt["arms"][0]["cases"][-1]["final_cos_min"] = 0.96  # below the Q8_0 bar
        root = fake_root(cfg, parity(cfg), receipt)
        with self.assertRaises(ValueError):
            p.precision(cfg, "Q8_0", root)
        del receipt["arms"][0]["cases"][-1]["final_cos_min"]  # cosine gate needs every case's cosine
        root = fake_root(cfg, parity(cfg), receipt)
        with self.assertRaises(ValueError):
            p.precision(cfg, "Q8_0", root)


class Notices(unittest.TestCase):
    def test_hash_mismatch_refuses(self):
        cfg = cfg_of("teleocr")
        bad = lambda url, timeout: io.BytesIO(b"x" * cfg["upstream"]["license_file"]["bytes"])
        with self.assertRaises(ValueError):
            p.notices(cfg, tempfile.mkdtemp(), opener=bad)


class Card(unittest.TestCase):
    def test_cards(self):
        for name, gate in (("mineru", "CRISPEMBED_MINERU25_PRO=1"), ("teleocr", "CRISPEMBED_TELEOCR=1")):
            with self.subTest(name=name):
                cfg = cfg_of(name)
                par = p.receipts(cfg, fake_root(cfg, parity(cfg)))
                fname = cfg["recipes"]["F16"]["filename"]
                files = [{"variant": "F16", "filename": fname, "bytes": 10, "sha256": "0" * 64}]
                text = p.card(cfg, files, par, {"F16": {"receipt": "r.json"}}, "b" * 40)
                head = {"mineru": "---\nlicense: apache-2.0\n",
                        "teleocr": "---\nlicense: other\nlicense_name: qwen-research\nlicense_link: LICENSE\n"}[name]
                self.assertTrue(text.startswith(head))
                if name == "teleocr":
                    for needle in ("Non-commercial use only", "Built with Qwen", "CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1",
                                   "qwen-vision-lineage.json"):
                        self.assertIn(needle, text)
                for needle in ("Statement of changes", cfg["upstream"]["revision"], gate, fname,
                               "**F16** is token-ID exact", cfg["_protocol"]):
                    self.assertIn(needle, text)
                self.assertIn("not verified", text.lower())
                self.assertNotIn("{first}", text)




class OxrPublication(unittest.TestCase):
    def test_oxr_license_obligations(self):
        for name in ("oxr", "oxr-dfine"):
            with self.subTest(name=name):
                cfg = cfg_of(name)
                self.assertIn("license_name: oxr", cfg["card"]["license_frontmatter"])
                lic = next(f for f in cfg["license_files"] if f["name"] == "LICENSE")
                self.assertEqual(p.digest(ROOT / "licenses/oxr-LICENSE"), lic["sha256"])
                self.assertIn("CRISPEMBED_ACCEPT_OXR_LICENSE=1", cfg["card"]["usage"])
                self.assertIn("SoMark OXR", cfg["notice_text"])

    def test_committed_receipts_gate(self):
        oxr = p.receipts(cfg_of("oxr"))
        self.assertEqual(oxr["status"], "token-id-exact")
        self.assertEqual(p.precision(cfg_of("oxr"), "F16")["ids_exact"], True)
        layout = p.receipts(cfg_of("oxr-dfine"))
        self.assertEqual(layout["status"], "regions-exact")
        self.assertEqual(layout["cases"]["receipt"], 21)

    def test_layout_receipt_refuses_inexact_page(self):
        cfg = cfg_of("oxr-dfine")
        r = json.loads((ROOT / cfg["receipts"]["parity"]).read_text())
        r["cases"][1]["arms"]["f32"]["regions_exact"] = False
        root = fake_root(cfg, parity_receipt=r)
        with self.assertRaises(ValueError):
            p.receipts(cfg, root=root)


if __name__ == "__main__":
    unittest.main()
