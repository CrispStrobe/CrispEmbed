#!/usr/bin/env python3
"""Relabel a published cstr/* re-host whose weights inherit a stricter upstream license.

    python tools/relabel_hf_license.py REPO --license-name qwen-research \
        --license-file models/licenses/qwen-research/LICENSE --notice "..." --note "..."

One Hub commit: README front matter set to license: other / license_name / license_link:
LICENSE, a marked license section prepended to the card body (replaced on re-runs), the
verbatim agreement as LICENSE and the required attribution as NOTICE. GGUF files are never
touched. Evidence: tests/results/licenses/registry-qwen-audit.json.
"""
import argparse
import re
from pathlib import Path

BEGIN, END = "<!-- crispembed-license-notice -->", "<!-- /crispembed-license-notice -->"


def rewrite(card: str, license_name: str, note: str) -> str:
    match = re.match(r"---\n(.*?)\n---\n", card, re.S)
    front, body = (match.group(1), card[match.end():]) if match else ("", card)
    lines = [l for l in front.splitlines() if not re.match(r"(license|license_name|license_link)\s*:", l)]
    front = "\n".join(["license: other", f"license_name: {license_name}", "license_link: LICENSE", *lines])
    body = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n*", "", body, flags=re.S)
    return f"---\n{front}\n---\n{BEGIN}\n{note.strip()}\n{END}\n\n{body.lstrip()}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--license-name", required=True)
    ap.add_argument("--license-file", required=True, type=Path)
    ap.add_argument("--notice", required=True)
    ap.add_argument("--note", required=True)
    args = ap.parse_args()
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi()
    card = Path(hf_hub_download(args.repo, "README.md")).read_text(encoding="utf-8")
    new = rewrite(card, args.license_name, args.note)
    ops = [CommitOperationAdd("README.md", new.encode()),
           CommitOperationAdd("LICENSE", args.license_file.read_bytes()),
           CommitOperationAdd("NOTICE", args.notice.encode())]
    commit = api.create_commit(args.repo, operations=ops,
                               commit_message=f"License: weights inherit {args.license_name} (see NOTICE, LICENSE)")
    print(args.repo, commit.oid)


if __name__ == "__main__":
    main()
