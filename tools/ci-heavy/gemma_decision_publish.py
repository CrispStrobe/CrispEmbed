#!/usr/bin/env python
"""Publish only after the complete original-Python validation passes on this runner."""

import hashlib
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import CommitOperationAdd, HfApi
from gemma_decision_validate import main as validate
from gemma_decision_bindings import main as validate_bindings


def main():
    validate()
    validate_bindings()
    scratch, out = Path(os.environ["HEAVY_SCRATCH"]), Path(os.environ["HEAVY_OUT"])
    repo = "cstr/embeddinggemma-2-ft-typed-decision-GGUF"
    names = {
        "gemma-decision-f32.gguf": "embeddinggemma-2-ft-typed-decision-f32.gguf",
        "gemma-decision-q8_0.gguf": "embeddinggemma-2-ft-typed-decision-q8_0.gguf",
    }
    artifacts = {}
    for local, remote in names.items():
        path = scratch / local
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        artifacts[remote] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    f32 = json.loads((out / "parity-f32.json").read_text())
    q8 = json.loads((out / "parity-q8.json").read_text())
    matched = sum(
        q["native"]["label"] == q["reference"]["label"] for q in q8["questions"]
    )
    total = len(q8["questions"])
    manifest = {
        "source_model": "mrm8488/embeddinggemma-2-ft-typed-decision",
        "source_revision": "fcc165b7d7d97827e241e09752e83fd11165a74b",
        "native_commit": os.environ.get("GITHUB_SHA"),
        "workflow_run": os.environ.get("GITHUB_RUN_ID"),
        "artifacts": artifacts,
        "f32_contracts": f32["contracts"],
        "f32_embeddings": f32["embeddings"],
        "q8_embeddings": q8["embeddings"],
        "q8_reference_labels": {"matched": matched, "total": total},
        "bindings": json.loads((out / "bindings.json").read_text()),
        "note": "Reference agreement on these probes, not downstream task accuracy. CPU only.",
    }
    (out / "publication.json").write_text(json.dumps(manifest, indent=2))
    card = f"""---
license: apache-2.0
base_model: mrm8488/embeddinggemma-2-ft-typed-decision
base_model_relation: quantized
pipeline_tag: text-classification
tags: [gguf, crispembed, embeddinggemma, typed-decisions, calibration]
---

# EmbeddingGemma 2 fine-tuned for typed decisions — CrispEmbed GGUF

Complete encoder plus schema-conditioned head from
[mrm8488/embeddinggemma-2-ft-typed-decision](https://huggingface.co/mrm8488/embeddinggemma-2-ft-typed-decision),
pinned to fcc165b7d7d97827e241e09752e83fd11165a74b. Both this fine-tune and its
[Google base](https://huggingface.co/google/embeddinggemma-2) declare Apache-2.0.
This custom GGUF architecture requires CrispEmbed; stock llama.cpp support is not claimed.

| File | Bytes | Precision |
|---|---:|---|
| embeddinggemma-2-ft-typed-decision-f32.gguf | {artifacts[names["gemma-decision-f32.gguf"]]["bytes"]} | original FP32 weights |
| embeddinggemma-2-ft-typed-decision-q8_0.gguf | {artifacts[names["gemma-decision-q8_0.gguf"]]["bytes"]} | Q8 encoder matrices; FP32 head/norms/scalars/output projection/calibration |

CPU validation: five every-layer comparisons with exact token IDs, norms and
embedding checks against actual original Python FP32, plus API/CLI/HTTP/schema/head
and decoded typed-value contracts. FP32 decision-bank embedding minimum cosine:
{min(e['cosine_min'] for e in f32['embeddings']):.9f}; maximum option probability error:
{max(q['max_probability_error'] for q in f32['questions']):.9g}.
Q8 embedding minimum cosine: {min(e['cosine_min'] for e in q8['embeddings']):.9f}.
Q8 agrees on {matched}/{total} reference labels;
its measured probability changes are in results/parity-q8.json. This is a small
numerical reference suite, not an independent task accuracy benchmark. No GPU or speed claim.
The exporter does not offer FP16 because the publisher reports overflow in that dtype.

Build CrispEmbed at {manifest['native_commit']} or a later commit including this backend, enable the CPU reference path,
and use your downloaded GGUF path:

```sh
CRISPEMBED_GEMMA_DECISION=1 crispembed -m model-f32.gguf --decide '{{"state":{{"task":"Delete old branches","constraint_violations":1}},"questions":{{"review":{{"type":"noul","instructions":"Does this require human review?"}}}}}}'
```

For Q8 also set CRISPEMBED_GEMMA_DECISION_F32_DOT=1 to reproduce the reported
arithmetic. Native inference adds the original classification prompt automatically,
mean-pools the projected token outputs and applies the trained head plus per-type
temperatures. Never substitute base-model embeddings, retrieval prompts or plain
cosine scoring for the full typed-decision pipeline.

Python: `CrispEmbed(model_path).decide(state, questions, workflow="...")`.
Questions support choice, score and noul (boolean); responses preserve string,
integer and boolean values, confidence, calibrated option probabilities and score
expectation. Rust, Dart, the C ABI and POST /decide expose the same pipeline.

[Usage and limits](https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/embeddinggemma2-typed-decision.md).
results/ includes full reports, source/workflow revisions and artifact SHA-256 hashes.
[Python layer references](https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/tree/main/gemma-decision)
are hosted separately. The base model is a multilingual backbone; the head was
trained on English schemas. Reference parity is not validation on your own tasks.
"""
    api = HfApi()
    api.create_repo(repo, exist_ok=True, private=False)
    api.create_commit(
        repo_id=repo,
        commit_message="Document original-Python validated typed-decision GGUFs",
        operations=[
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()),
            CommitOperationAdd(
                path_in_repo="LICENSE",
                path_or_fileobj="models/licenses/apache-2.0/LICENSE",
            ),
            CommitOperationAdd(
                path_in_repo="UPSTREAM_README.md",
                path_or_fileobj=str(scratch / "model/README.md"),
            ),
            *[
                CommitOperationAdd(
                    path_in_repo="results/" + p.name, path_or_fileobj=str(p)
                )
                for p in out.glob("*.json")
            ],
            *[
                CommitOperationAdd(
                    path_in_repo="results/" + p.name, path_or_fileobj=str(p)
                )
                for p in out.glob("diff-*.log")
            ],
        ],
    )
    for local, remote in names.items():
        result = api.upload_file(
            repo_id=repo,
            path_in_repo=remote,
            path_or_fileobj=str(scratch / local),
            commit_message="Publish validated " + remote,
        )
        print("MODEL UPLOADED", remote, result.oid, flush=True)
    shutil.copy("models/licenses/apache-2.0/LICENSE", scratch / "refs/LICENSE")
    (scratch / "refs/README.md").write_text(
        "# Original FP32 typed-decision references\n\n"
        "Source: mrm8488/embeddinggemma-2-ft-typed-decision at "
        "fcc165b7d7d97827e241e09752e83fd11165a74b (Apache-2.0).\n"
        "Generated by tools/dump_gemma_decision_reference.py from the complete "
        "original Python model, including every layer, exact IDs and typed answers. "
        "Five numerical probes and eight questions are not a task accuracy benchmark.\n"
    )
    fixture_commit = api.upload_folder(
        repo_id="cstr/crispembed-regression-fixtures",
        repo_type="dataset",
        folder_path=str(scratch / "refs"),
        path_in_repo="gemma-decision/original-fp32",
        commit_message="Add complete original Python typed-decision layer and answer fixtures",
    )
    info = api.model_info(repo, files_metadata=True)
    verified = set()
    for file in info.siblings:
        if file.rfilename in artifacts:
            assert file.lfs.sha256 == artifacts[file.rfilename]["sha256"]
            assert file.size == artifacts[file.rfilename]["bytes"]
            verified.add(file.rfilename)
    assert verified == set(
        artifacts
    ), "Published artifacts missing from remote manifest"
    manifest["model_repo"] = repo
    manifest["model_revision"] = info.sha
    manifest["fixture_repo"] = "cstr/crispembed-regression-fixtures"
    manifest["fixture_revision"] = fixture_commit.oid
    manifest["fixture_path"] = "gemma-decision/original-fp32"
    (out / "publication.json").write_text(json.dumps(manifest, indent=2))
    api.upload_file(
        repo_id=repo,
        path_in_repo="results/publication.json",
        path_or_fileobj=str(out / "publication.json"),
        commit_message="Record verified artifact revision and remote hashes",
    )
    print("PUBLISHED", repo, info.sha, flush=True)


if __name__ == "__main__":
    main()
