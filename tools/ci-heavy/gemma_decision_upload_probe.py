"""Check the existing CI token's write permission after the publisher created the repo."""

from pathlib import Path
import os
import json
from huggingface_hub import HfApi


def main():
    repo = "cstr/embeddinggemma-2-ft-typed-decision-GGUF"
    api = HfApi()
    before = api.model_info(repo).sha
    payload = json.dumps(
        {
            "workflow_run": os.environ.get("GITHUB_RUN_ID"),
            "source_commit": os.environ.get("GITHUB_SHA"),
            "purpose": "Verify actual write permission; no credential values",
        }
    ).encode()
    result = api.upload_file(
        repo_id=repo,
        path_in_repo="results/upload-permission.json",
        path_or_fileobj=payload,
        commit_message="Record complete reference and binding validation before model upload",
    )
    assert result.oid != before, "No-op commit is not proof of write permission"
    Path(os.environ["HEAVY_OUT"]).mkdir(parents=True, exist_ok=True)
    (Path(os.environ["HEAVY_OUT"]) / "summary.md").write_text(
        "Existing CI token can write the created model repository.\n"
    )
    print("PASS existing CI upload permission", repo, result.oid, flush=True)


if __name__ == "__main__":
    main()
