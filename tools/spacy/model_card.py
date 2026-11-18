#!/usr/bin/env python3
"""Render the Hugging Face model card for the native German spaCy repository.

Input is the tracked `hub_catalog.json` (optionally with additional package
entries from a publication run). Output is deterministic Markdown. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CATALOG = Path(__file__).resolve().parent / "hub_catalog.json"


def _short(value: str | None) -> str:
    return f"`{value[:12]}…`" if value else "—"


def render(catalog: dict) -> str:
    repo = catalog["repository"]
    model = catalog["model"]
    bundle = catalog["bundle"]
    resolve = f"https://huggingface.co/{repo}/resolve"
    lines = [
        "---",
        "license: mit",
        "language: de",
        "library_name: crispembed",
        "tags: [spacy, gguf, crispembed, german, pos-tagging, dependency-parsing, lemmatization]",
        "---",
        "",
        f"# CrispEmbed native German spaCy — {model['name']} {model['version']} ({model['precision']})",
        "",
        "A native C++/ggml reimplementation of the trained spaCy "
        f"`{model['name']}` {model['version']} pipeline "
        f"({', '.join(model['components'])}), converted from the official wheel "
        f"(SHA-256 `{model['source_wheel_sha256']}`). It needs no Python at run time. "
        "Output is tokens with original-text spans, fine/coarse part of speech, "
        "morphology, lemmas, dependency heads/relations and sentences. "
        f"Omitted: {'; '.join(model['omitted'])}.",
        "",
        "Consumers: [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed) "
        "(`tools/spacy`) and the optional Quds Office German analysis "
        "(`packages/quds_proofing_spacy`), which installs these files only after "
        "an explicit user action.",
        "",
        "## Download rules",
        "",
        "- Always download from an **immutable commit** (`/resolve/<40-hex revision>/…`), never `main`.",
        "- Check the complete SHA-256 and length **before** decompressing or parsing.",
        "- Capsules are closed canonical USTAR/gzip trees; every member is pinned by the "
        "embedded `distribution.json`. Install by staging privately and publishing the "
        "content-addressed directory with one no-replace rename "
        "(`tools/spacy/distribution.py` for v1, `tools/spacy/platform_distribution.py` for v2).",
        "- The native library refuses work until the host enables it after an explicit opt-in: "
        "`crispembed_spacy_set_enabled(CRISPEMBED_SPACY_ENABLE_ANALYSIS)` (no environment change) or "
        "`CRISPEMBED_SPACY=1`. Nothing downloads at analysis time.",
        "",
        f"## Model bundle (revision `{bundle['revision']}`)",
        "",
        "| File | Bytes | SHA-256 | Purpose |",
        "| --- | ---: | --- | --- |",
    ]
    for path, pin in bundle["files"].items():
        lines.append(f"| [`{path}`]({resolve}/{bundle['revision']}/{path}) | {pin['bytes']:,} | "
                     f"`{pin['sha256']}` | {pin['purpose']} |")
    lines += [
        "",
        "## Platform packages",
        "",
        "Qualification is bound to one exact native library digest. A rebuilt "
        "library never inherits another build's claims.",
        "",
        "| Platform | Status | Package (revision) | Library | Analysis | Manual grammar |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for package in catalog["packages"]:
        if package.get("path"):
            link = (f"[`{package['path']}`]({resolve}/{package['revision']}/{package['path']}) "
                    f"({_short(package['revision'])}; {package['bytes']:,} B; SHA-256 `{package['sha256']}`)")
        else:
            link = "—"
        lines.append(f"| {package['platform']} | {package['status']} | {link} | "
                     f"{_short(package.get('library_sha256'))} | {package['analysis']} | {package['grammar']} |")
    requirements = [p for p in catalog["packages"] if p.get("requirements")]
    if requirements:
        lines += ["", "Requirements:"]
        lines += [f"- {p['platform']}: {p['requirements']}" for p in requirements]
    receipts = [p for p in catalog["packages"] if p.get("receipt")]
    if receipts:
        lines += ["", "Publication/qualification receipts:"]
        for p in receipts:
            receipt = p['receipt']
            if not receipt.startswith('https://') and p.get('revision'):
                receipt = f"{resolve}/{p['revision']}/{receipt}"
            lines.append(f"- {p['platform']}: {receipt}")
    sdks = catalog.get("sdks") or []
    if sdks:
        lines += [
            "",
            "## Mobile SDKs",
            "",
            "Mobile apps must ship native code inside the signed app; only model files may be "
            "downloaded at run time. These archives are build inputs for app developers.",
            "",
            "| Target | Archive (revision) | Contents | Verification | Manual grammar |",
            "| --- | --- | --- | --- | --- |",
        ]
        for sdk in sdks:
            link = (f"[`{sdk['path']}`]({resolve}/{sdk['revision']}/{sdk['path']}) "
                    f"({_short(sdk['revision'])}; {sdk['bytes']:,} B; SHA-256 `{sdk['sha256']}`)")
            lines.append(f"| {sdk['target']} | {link} | {sdk['contents']} | {sdk['verification']} | "
                         f"{sdk['grammar']} |")
    ner = catalog.get("ner")
    if ner:
        lines += ["", "## Optional NER model", "", f"Status: {ner['status']}. {ner.get('qualification', '')}", ""]
        if ner.get("revision"):
            lines += ["| File | Bytes | SHA-256 |", "| --- | ---: | --- |"]
            for path, pin in ner["files"].items():
                lines.append(f"| [`{path}`]({resolve}/{ner['revision']}/{path}) | {pin['bytes']:,} | `{pin['sha256']}` |")
    lines += [
        "",
        "## Not qualified",
        "",
        *[f"- {item}" for item in catalog["not_qualified"]],
        "",
        "## Licenses and notices",
        "",
        "- Model weights: MIT (ExplosionAI GmbH), `bundle/LICENSE` verbatim from the wheel. "
        "`bundle/LICENSES_SOURCES` retains the training-source notices (TIGER corpus, "
        "Tiger2Dep, WikiNER CC BY 4.0). No training corpus is included and these notices "
        "grant no rights to the original corpora.",
        "- Native code: CrispEmbed (MIT), ggml (MIT), spaCy and Thinc semantics (MIT), "
        "PCRE2 (BSD), CPython-derived Unicode property tables (PSF), Unicode data (Unicode "
        "License). Every capsule carries the full texts under "
        "`native/share/licenses/crispembed-spacy/`.",
        "",
        "## Other files",
        "",
        "- `distribution.json`, `artifacts.json`, `operational-protocol.json`: the frozen Linux v1 "
        "profile, its artifact pins and the operational consumer protocol.",
        "- `packages/<platform>/<source commit>/`: v2 platform capsules with their publication receipt.",
        "- `sdk/<target>/<source commit>/`: mobile SDK archives with their verification receipt.",
        "- `retained-w7-evidence/`, `macos-candidates/`, `quantization-attempts/`: retained evidence; "
        "not installable products.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.catalog.read_text(encoding='utf-8'))), encoding='utf-8', newline='\n')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
