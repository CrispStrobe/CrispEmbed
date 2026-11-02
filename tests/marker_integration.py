#!/usr/bin/env python3
"""Exercise actual native Marker PDF conversion and public clients on owned gold."""
import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from crispembed import CrispMarker


class Text(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.tags = []
    def handle_data(self, data): self.parts.append(data)
    def handle_starttag(self, tag, attrs): self.tags.append(tag)


def normalized(value):
    return " ".join(value.split())


def walk(node):
    yield node
    for child in node.get("children") or []:
        yield from walk(child)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", required=True)
    p.add_argument("--pdfium", required=True)
    p.add_argument("--cli")
    p.add_argument("--server")
    p.add_argument("--out", required=True)
    p.add_argument("--fixtures", default=str(ROOT / "tests/fixtures/marker"))
    args = p.parse_args()
    os.environ["CRISPEMBED_MARKER"] = "1"
    os.environ["CRISPEMBED_OFFLINE"] = "1"
    fixture = Path(args.fixtures).resolve(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gold = json.loads((fixture / "expected.json").read_text())
    checks, cases = [], []

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    def must_fail(name, fn):
        try: fn()
        except (ValueError, RuntimeError) as exc: check(name, True, error=str(exc))
        else: check(name, False)

    opts = dict(mode="text", pdfium_path=args.pdfium, lib_path=args.library)
    for case in gold["cases"]:
        path = fixture / case["pdf"]
        check(case["id"] + ": fixture hash", hashlib.sha256(path.read_bytes()).hexdigest() == case["pdf_sha256"])
        extra = {"password": "marker-fixture"} if case["id"] == "encrypted" else {}
        try:
            with CrispMarker(**opts, **extra) as model:
                start = time.monotonic()
                document = json.loads(model.convert(path, "json"))
                elapsed = time.monotonic() - start
                outputs = {kind: model.render(kind) for kind in ("markdown", "html", "json", "chunks")}
                for kind, text in outputs.items():
                    (out / f'{case["id"]}.{ "md" if kind == "markdown" else kind }').write_text(text, encoding="utf-8")
                metadata = model.metadata
                (out / f'{case["id"]}.meta.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
                nodes = list(walk(document)); leaves = [n for n in nodes if n.get("block_type") not in ("Document", "Page")]
                visible = Text(); visible.feed(outputs["html"])
                content = normalized(" ".join(visible.parts))
                check(case["id"] + ": pages", metadata["page_count"] == case["page_count"])
                check(case["id"] + ": provenance", metadata["backend"] == "pdfium-text" and
                      metadata["configuration"]["surya_model_parity"] == "not_claimed")
                check(case["id"] + ": no model tokens", metadata["generated_tokens"] == 0)
                # Semantic HTML lists carry bullets through <li>, not a literal
                # leading dash. Keep authored gold and test the structure below.
                list_items = case.get("expected_list_items", [])
                missing = [s for s in case["must_contain"] if normalized(s[2:] if s in list_items and s.startswith("- ") else s) not in content]
                check(case["id"] + ": visible content", not missing, missing=missing)
                all_ids = [n["id"] for n in nodes if "id" in n]
                check(case["id"] + ": unique block IDs", len(all_ids) == len(set(all_ids)))
                check(case["id"] + ": finite geometry", all(len(n["bbox"]) == 4 and
                      all(isinstance(v, (int, float)) and abs(v) < 100000 for v in n["bbox"]) for n in leaves))
                chunks = json.loads(outputs["chunks"])
                check(case["id"] + ": chunk pages", len(chunks["page_info"]) == case["page_count"])
                if "expected_heading" in case:
                    check(case["id"] + ": heading", any(n["block_type"] == "SectionHeader" and
                          case["expected_heading"] in n.get("text", "") for n in leaves))
                if "expected_headings" in case:
                    actual = [n.get("text", "") for n in leaves if n["block_type"] == "SectionHeader"]
                    check(case["id"] + ": all headings", actual == case["expected_headings"], actual=actual)
                if list_items:
                    actual = [n.get("text", "") for n in leaves if n["block_type"] == "ListItem"]
                    expected_items = [s[2:] if s.startswith("- ") else s for s in list_items]
                    check(case["id"] + ": semantic list items", actual == expected_items, actual=actual)
                if "expected_code" in case:
                    actual = "\n".join(n.get("text", "") for n in leaves if n["block_type"] == "Code")
                    check(case["id"] + ": exact code indentation", actual == case["expected_code"], actual=actual)
                if case["id"] == "unicode":
                    check("Unicode escaped script", "script" not in visible.tags and "<script>alert(1)</script>" in content)
                    check("Markdown literal escaping", r"\*stars\*" in outputs["markdown"] and r"\[brackets\]" in outputs["markdown"])
                if case["id"] == "multicolumn":
                    positions = [content.find(normalized(s)) for s in case["logical_order"]]
                    check("Column reading order", all(n >= 0 for n in positions) and positions == sorted(positions), positions=positions)
                if "expected_table" in case:
                    table_text = " ".join(n.get("html", "") for n in leaves if n["block_type"] == "Table")
                    check("Digital table structure", "<table" in table_text and table_text.count("<tr") == len(case["expected_table"]))
                    check("Digital table cells", all(html.escape(cell) in table_text for row in case["expected_table"] for cell in row))
                model.save(out / f'{case["id"]}-saved.md')
                check(case["id"] + ": save", (out / f'{case["id"]}-saved.md').read_text() == outputs["markdown"])
                cases.append(dict(id=case["id"], seconds=elapsed, text=content, metadata=metadata))
        except Exception as exc:
            check(case["id"] + ": conversion exception", False, error=repr(exc))

    with CrispMarker(**opts, first_page=1, last_page=1) as model:
        data = json.loads(model.convert(fixture / "multipage.pdf", "json"))
        check("Inclusive page range", len(data["children"]) == 1 and data["children"][0]["id"] == "/page/1")
        must_fail("bad format", lambda: model.render("invalid"))
        check("render recovery", bool(model.render("markdown")))
        must_fail("missing file", lambda: model.convert(fixture / "missing.pdf"))
        must_fail("stale output cleared", lambda: model.render("json"))
        check("conversion recovery", bool(model.convert(fixture / "multipage.pdf")))
    with CrispMarker(**opts) as model:
        must_fail("encrypted needs password", lambda: model.convert(fixture / "encrypted.pdf"))
        must_fail("text mode refuses image", lambda: model.convert(fixture / "simpledigital-page-1.png"))
        must_fail("raw wrong dtype", lambda: model.convert_image([[1.0]]))
        must_fail("NUL path", lambda: model.convert("a\0b"))
        must_fail("NUL format", lambda: model.render("json\0bad"))
    must_fail("use after close", lambda: model.render())
    with CrispMarker(**opts, max_pages=2) as model:
        must_fail("max pages", lambda: model.convert(fixture / "multipage.pdf"))
    with CrispMarker(**opts, first_page=99) as model:
        must_fail("out of range page", lambda: model.convert(fixture / "multipage.pdf"))
    with CrispMarker(mode="text", lib_path=args.library) as model:
        must_fail("missing PDFium", lambda: model.convert(fixture / "simpledigital.pdf"))
    must_fail("negative pixel config", lambda: CrispMarker(**opts, max_pixels=-1))
    os.environ["CRISPEMBED_MARKER"] = "0"
    must_fail("Python disabled gate", lambda: CrispMarker(**opts))
    os.environ["CRISPEMBED_MARKER"] = "1"

    if args.cli:
        for fmt in ("markdown", "html", "json", "chunks"):
            dest = out / ("cli-" + fmt + ".txt")
            command = [args.cli, str(fixture / "simpledigital.pdf"), "--mode", "text", "--pdfium", args.pdfium,
                       "--format", fmt, "--output", str(dest)]
            result = subprocess.run(command, capture_output=True, text=True)
            check("CLI " + fmt, result.returncode == 0 and dest.exists(), stderr=result.stderr)
        result = subprocess.run([args.cli, str(fixture / "simpledigital.pdf"), "--mode", "text", "--pdfium", args.pdfium],
                                capture_output=True, text=True, env={**os.environ, "CRISPEMBED_MARKER": "0"})
        check("CLI disabled gate", result.returncode != 0 and "CRISPEMBED_MARKER" in result.stderr)
    if args.server:
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        log = open(out / "http-server.log", "w")
        process = subprocess.Popen([args.server, "--marker-text", "--marker-pdfium", args.pdfium,
                                    "--image-root", str(fixture), "--host", "127.0.0.1", "--port", str(port), "--offline"],
                                   stdout=log, stderr=log)
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                if process.poll() is not None: raise RuntimeError("HTTP server exited during startup")
                try:
                    urllib.request.urlopen(base + "/health", timeout=1).close(); break
                except OSError: time.sleep(0.05)
            def request(body):
                req = urllib.request.Request(base + "/ocr/marker", json.dumps(body).encode(), {"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as res: return res.status, json.load(res)
                except urllib.error.HTTPError as exc: return exc.code, json.load(exc)
            for fmt in ("markdown", "html", "json", "chunks"):
                code, response = request({"file":str(fixture / "simpledigital.pdf"), "format":fmt})
                check("HTTP " + fmt, code == 200 and "Hello from CrispEmbed." in response.get("output", ""))
                (out / f"http-{fmt}.json").write_text(json.dumps(response, indent=2, ensure_ascii=False) + "\n")
            code, _ = request({"file":str(ROOT / "LICENSE")})
            check("HTTP path confinement", code == 400)
            code, _ = request({"file":str(fixture / "simpledigital.pdf"), "format":"invalid"})
            check("HTTP invalid format", code == 400)
        except Exception as exc:
            check("HTTP execution", False, error=repr(exc))
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            log.close()
    report = dict(schema_version=1, scope="Owned digital-PDF native integration; no Surya or model parity claim",
                  checks=checks, cases=cases, passed=sum(c["passed"] for c in checks), total=len(checks))
    report["all_passed"] = report["passed"] == report["total"]
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k:report[k] for k in ("passed", "total", "all_passed")}))
    for c in checks:
        if not c["passed"]: print("FAIL", c)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
