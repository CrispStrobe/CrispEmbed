"""Check the existing CI token's write permission after the publisher created the repo."""

from pathlib import Path
import os
from huggingface_hub import HfApi


def main():
    repo = "cstr/embeddinggemma-2-ft-typed-decision-GGUF"
    result = HfApi().upload_file(
        repo_id=repo,
        path_in_repo="results/validated-artifacts.json",
        path_or_fileobj="tests/results/gemma-decision/validated-artifacts.json",
        commit_message="Record complete reference and binding validation before model upload",
    )
    Path(os.environ["HEAVY_OUT"]).mkdir(parents=True, exist_ok=True)
    (Path(os.environ["HEAVY_OUT"]) / "summary.md").write_text(
        "Existing CI token can write the created model repository.\n"
    )
    print("PASS existing CI upload permission", repo, result.oid, flush=True)


if __name__ == "__main__":
    main()
