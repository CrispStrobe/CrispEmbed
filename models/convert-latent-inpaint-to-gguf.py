#!/usr/bin/env python3
"""Stream a pinned Moebius, PixelHacker or shared SDXL-VAE checkpoint to F32 GGUF.

The converter reuses CrispEmbed's inert checkpoint-metadata and minimal GGUF
patterns. No PyTorch, upstream Python code, model inference or network call is
executed. Run large checkpoint conversion on a hosted worker, as required by
the repository. Exact official checkpoints only; arbitrary models are rejected.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from latent_inpaint_contract import (Checkpoint, CONTRACT, MODELS, gguf_metadata, load_contract,
                                    model_record, sha256, verify_gguf, verify_source, write_streamed_gguf)


def convert(model, checkpoint, output, source_dir):
    contract = load_contract()
    record = model_record(model, contract)
    output, checkpoint = Path(output), Path(checkpoint)
    receipt_path = output.with_suffix(output.suffix + ".json")
    notices_path = output.with_suffix(output.suffix + ".licenses")
    for path in (output, receipt_path, notices_path):
        if path.exists() or path.is_symlink():
            raise ValueError("output, receipt or licence directory already exists; choose a new destination")
    if not output.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    source = verify_source(source_dir, model, contract)
    metadata = gguf_metadata(model, contract)
    created = []
    with tempfile.TemporaryDirectory(prefix=".latent-convert-", dir=output.parent) as temporary:
        stage = Path(temporary)
        staged_model = stage / "model.gguf"
        with Checkpoint(checkpoint, record) as original:
            omitted = original.validate_omitted(record["state"])
            tensors = write_streamed_gguf(staged_model, original, record["state"], metadata)
        verify_gguf(staged_model, metadata, tensors)
        receipt = {
            "schema": "crispembed-latent-inpaint-conversion-v1",
            "model": model,
            "component": record["component"],
            "source_commit": source["commit"],
            "source_contract_sha256": sha256(CONTRACT),
            "checkpoint_sha256": record["checkpoint"]["sha256"],
            "checkpoint_bytes": record["checkpoint"]["bytes"],
            "checkpoint_revision": record["checkpoint"]["revision"],
            "checkpoint_pickle_sha256": record["checkpoint"]["pickle_sha256"],
            "gguf_sha256": sha256(staged_model),
            "gguf_bytes": staged_model.stat().st_size,
            "original_tensor_count": record["original_tensor_count"],
            "export_tensor_count": len(tensors),
            "metadata": metadata,
            "tensors": tensors,
            "omitted_inference_inert_buffers": omitted,
            "all_f32_tensor_readbacks_exact": True,
            "f16_sources_widened_exactly_to_f32": True,
            "framework_imported": False,
            "inference_performed": False,
            "native_qualified": False,
            "representative_quality_qualified": False,
            "published": False,
            "python": platform.python_version(),
            "converter_sha256": sha256(__file__),
            "helper_sha256": sha256(Path(__file__).resolve().parents[1] / "tools/latent_inpaint_contract.py"),
            "command": [sys.executable, str(Path(__file__).resolve()), "--model", model,
                        "--checkpoint", str(checkpoint), "--source-dir", str(source_dir), "--output", str(output)],
        }
        staged_receipt = stage / "conversion.json"
        staged_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        notices = stage / "licenses"
        notices.mkdir()
        shutil.copyfile(Path(source_dir) / "LICENSE", notices / "UPSTREAM-CODE-LICENSE")
        card = record["model_card"]
        if len(card["text"].encode()) != card["bytes"] or hashlib.sha256(card["text"].encode()).hexdigest() != card["sha256"]:
            raise ValueError("frozen author model card differs")
        (notices / "AUTHOR-WEIGHTS-README.md").write_text(card["text"], encoding="utf-8")
        (notices / "SOURCE.txt").write_text(
            "Model: " + model + "\n"
            "Source: " + source["repository"] + "/tree/" + source["commit"] + "\n"
            "Checkpoint: " + record["checkpoint"]["url"] + "\n"
            "Checkpoint SHA-256: " + record["checkpoint"]["sha256"] + "\n"
            "Author weight model card: " + card["url"] + "\n"
            "Declared weight licence: " + record["license"] + "\n"
            + record["license_note"] + "\n"
            + "Licence evidence:\n" + "\n".join(record["license_evidence"]) + "\n"
            "Conversion: floating weights stored as F32; named singleton-depth axes squeezed;\n"
            "inference-inert BatchNorm training counters omitted with individual hashes in receipt.\n"
            "No inference quality, commercial clearance or native parity result is asserted by conversion.\n",
            encoding="utf-8")
        # Exclusive final creates avoid an overwrite race. The model and receipt
        # stay on the same filesystem, so a hard link publishes completed bytes.
        try:
            os.link(staged_model, output)
            created.append(output)
            os.link(staged_receipt, receipt_path)
            created.append(receipt_path)
            notices_path.mkdir()
            created.append(notices_path)
            for path in notices.iterdir():
                shutil.copyfile(path, notices_path / path.name)
        except BaseException:
            for path in reversed(created):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pinned official checkpoint; complete SHA-256 is checked first")
    parser.add_argument("--source-dir", type=Path, required=True, help="Exact upstream checkout (PixelHacker checkout for shared VAE)")
    parser.add_argument("--output", type=Path, required=True, help="New GGUF path; writes adjacent JSON receipt and licence directory")
    args = parser.parse_args(argv)
    try:
        receipt = convert(args.model, args.checkpoint, args.output, args.source_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, "error: " + str(error) + "\n")
    print(json.dumps({key: receipt[key] for key in ("model", "gguf_sha256", "gguf_bytes", "export_tensor_count", "native_qualified")}, indent=2))


if __name__ == "__main__":
    main()
