#!/usr/bin/env python3
"""Render the Hugging Face model card of a native spaCy v2 language bundle.

Input is a published `bundle-catalog.json` (schema
crispembed.spacy.language_bundle.v1, written by
tools/ci-heavy/spacy_v2_bundle.py). Output is deterministic Markdown. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PURPOSES = {
    "bundle/model.gguf": "F32 weights (GGUF architecture spacy_pipeline_v2), NER included",
    "bundle/resources.json": "tokenizer rules, NORM tables, lookups, attribute-ruler patterns, labels (format_version 2)",
    "bundle/manifest.json": "converter manifest: per-tensor and per-file SHA-256 pins",
    "bundle/LICENSE": "model license, verbatim from the official wheel",
    "bundle/LICENSES_SOURCES": "training-source notices, verbatim from the official wheel",
}


def _short(value: str | None) -> str:
    return f"`{value[:12]}…`" if value else "—"


def render(catalog: dict) -> str:
    repo = catalog["repository"]
    model = catalog["model"]
    bundle = catalog["bundle"]
    license_ = model["license"]
    resolve = f"https://huggingface.co/{repo}/resolve"
    front = ["---", f"license: {license_['spdx']}"]
    if license_["spdx"] == "other":
        # The Hub requires an absolute https link; pin it to the bundle commit.
        front += ["license_name: lgpl-lr",
                  f"license_link: https://huggingface.co/{repo}/blob/{bundle['revision']}/bundle/LICENSE"]
    front += [
        f"language: {model['language']}",
        "library_name: crispembed",
        f"tags: [spacy, gguf, crispembed, {model['language_name'].lower()}, pos-tagging, dependency-parsing, "
        "lemmatization, named-entity-recognition]",
        "---",
        "",
    ]
    lines = front + [
        f"# CrispEmbed native {model['language_name']} spaCy — {model['name']} {model['version']} (F32)",
        "",
        f"Model files for the native C++/ggml runtime of the trained spaCy `{model['name']}` "
        f"{model['version']} pipeline ({', '.join(model['components'])}), converted with "
        "`models/convert-spacy-pipeline-to-gguf.py` from the official wheel "
        f"([`{model['wheel']['name']}`]({model['wheel']['url']}), {model['wheel']['bytes']:,} B, SHA-256 "
        f"`{model['wheel']['sha256']}`). No Python is needed at run time. Output per text: tokens with "
        "original-text spans, fine/coarse part of speech, morphology, lemmas, dependency heads/relations, "
        f"sentences and named entities ({', '.join(model['entity_labels'])}).",
        "",
        "This repository holds **only the language bundle**. The native library is language-independent: the "
        "same `libcrispembed-spacy` that serves German opens this bundle, dispatching on the GGUF architecture. "
        f"Library capsules are published once, in [`{catalog['library_repository']}`]"
        f"(https://huggingface.co/{catalog['library_repository']}); only builds from CrispEmbed source that "
        "contains the v2 runtime (PR #244) can open this bundle. Older builds reject it.",
        "",
        "## Use",
        "",
        "```c",
        "crispembed_spacy_set_enabled(CRISPEMBED_SPACY_ENABLE_ANALYSIS);  /* explicit opt-in; no setenv */",
        "ctx = crispembed_spacy_create_with_resources(\"bundle/model.gguf\", MODEL_SHA256,",
        "                                             \"bundle/resources.json\", RESOURCES_SHA256, err, sizeof err);",
        "crispembed_spacy_analyze_utf8(ctx, text, len, &doc, NULL, NULL, NULL, NULL, err, sizeof err);",
        "/* tokens: crispembed_spacy_document_token_*; entities: crispembed_spacy_document_entity_* */",
        "```",
        "",
        "Entities come from the NER component inside `model.gguf`. `crispembed_spacy_attach_ner` and the "
        "prepared numeric services are German-only and reject this bundle.",
        "",
        "## Download rules",
        "",
        "- Download from the **immutable commit** below (`/resolve/<40-hex revision>/…`), never `main`.",
        "- Check the complete SHA-256 and length of every file **before** parsing it; the runtime also refuses "
        "a model or resource file whose SHA-256 differs from the one the host passes in.",
        "- Download only after an explicit user action; nothing is fetched at analysis time.",
        "",
        f"## Files (revision `{bundle['revision']}`)",
        "",
        "| File | Bytes | SHA-256 | Purpose |",
        "| --- | ---: | --- | --- |",
    ]
    for path, pin in bundle["files"].items():
        lines.append(f"| [`{path}`]({resolve}/{bundle['revision']}/{path}) | {pin['bytes']:,} | "
                     f"`{pin['sha256']}` | {PURPOSES.get(path, '')} |")
    lines += [
        "",
        f"Converted on hosted Linux x86-64 in the hash-locked reference environment (CrispEmbed "
        f"`{bundle['source_commit'][:12]}`, run {bundle['run']}). `model.gguf`, `resources.json` and both "
        "notices equal the pins recorded by the earlier macOS and Linux qualification runs byte for byte; "
        "the notices are byte-identical to the files inside the wheel.",
        "",
        "## Qualification",
        "",
        *[f"- {item}" for item in model["qualification"]],
        "",
        "Analysis claims only. No grammar-checking claim is made for this language.",
        "",
        "## Native libraries known to load this bundle",
        "",
        "Each row is a hosted run that downloaded this bundle from the immutable revision above and the "
        "published library capsule from its own immutable revision, verified both by SHA-256/length, installed "
        "the capsule, and analyzed text through the C ABI only (Python ctypes, no rebuild). The comparison is "
        f"with original spaCy on the raw text of the {model['treebank']['name']}.",
        "",
        "| Platform | Result | Library SHA-256 | Capsule (revision) | Run | Admission controls | Texts exact | Tokens "
        "| Entities |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in catalog["libraries"]:
        if item.get("capsule_path"):
            capsule = (f"[`{item['capsule_path'].rsplit('/', 1)[-1]}`]"
                       f"(https://huggingface.co/{catalog['library_repository']}/resolve/{item['capsule_revision']}/"
                       f"{item['capsule_path']}) ({_short(item['capsule_revision'])})")
        else:
            capsule = "none found"
        library = f"`{item['library_sha256']}`" if item.get("library_sha256") else "—"
        result = item.get("comparison") or {}
        entities = result.get("entities") or {}
        status = "loads and matches" if item["passed"] else f"FAILED ({(item.get('failure') or '')[:120]})"
        lines.append(
            f"| {item['platform']} | {status} | {library} | {capsule} | {item['run']} | "
            f"{'pass' if item['controls_passed'] else 'fail'} | "
            f"{result.get('exact_cases', '—')}/{result.get('cases', '—')} | {result.get('tokens', '—')} | "
            f"{entities.get('matched', '—')}/{entities.get('oracle', '—')} |")
    if not catalog["libraries"]:
        lines.append("| — | none recorded yet | | | | | | | |")
    lines += [
        "",
        "Admission controls: disabled until enabled, wrong model/resource hash rejected, German resources and "
        "the German v1 model rejected with this bundle, text-only capabilities, numeric service and "
        "`attach_ner` rejected.",
        "",
        "## Not qualified",
        "",
        *[f"- {item}" for item in catalog["not_qualified"]],
        "",
        "## Licenses and notices",
        "",
        f"- **{license_['name']}.** {license_['text']}",
        "- Native code is not in this repository; each library capsule carries its own notices.",
        "- Evaluation input text (UD treebank) is used only as input in CI and is not redistributed here.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.catalog.read_text(encoding="utf-8"))), encoding="utf-8",
                           newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
