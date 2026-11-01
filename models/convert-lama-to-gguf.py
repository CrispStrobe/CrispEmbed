#!/usr/bin/env python3
"""Convert the exact official-linked Big-LaMa archive into generator-only F32 GGUF.

No framework, arbitrary checkpoint code or network is used. Full conversion is
intended for a hosted worker. The pinned outer ZIP is the input identity; the
observed inner checkpoint digest is recorded separately after safe extraction.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lama_contract import (Checkpoint, CONTRACT, WEIGHT_LICENSE_STATUS, generator_rows, gguf_metadata,
                           load_contract, sha256, verify_gguf, verify_source, write_streamed_gguf)


def convert(archive, output, source_dir):
    contract = load_contract()
    output, archive = Path(output), Path(archive)
    receipt_path = output.with_suffix(output.suffix + ".json")
    notices_path = output.with_suffix(output.suffix + ".licenses")
    for path in (output, receipt_path, notices_path):
        if path.exists() or path.is_symlink():
            raise ValueError("output, receipt or licence directory already exists; choose a new destination")
    if not output.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    source = verify_source(source_dir, contract)
    metadata = gguf_metadata(contract)
    rows = generator_rows(contract)
    created = []
    with tempfile.TemporaryDirectory(prefix=".lama-convert-", dir=output.parent) as temporary:
        stage = Path(temporary)
        staged_model = stage / "model.gguf"
        with Checkpoint(archive, contract, work_dir=stage) as original:
            omitted = original.validate_omitted()
            tensors = write_streamed_gguf(staged_model, original, rows, metadata)
            inner_sha256, inner_bytes = original.inner_sha256, original.inner_bytes
        verify_gguf(staged_model, metadata, tensors)
        cp = contract["checkpoint"]
        receipt = {
            "schema": "crispembed-lama-conversion-v1",
            "model": "big_lama",
            "source_commit": source["commit"],
            "source_contract_sha256": sha256(CONTRACT),
            "checkpoint_sha256": cp["sha256"],
            "checkpoint_bytes": cp["bytes"],
            "checkpoint_revision": cp["revision"],
            "checkpoint_container": "official-zip",
            "checkpoint_member": cp["member"]["path"],
            "checkpoint_pickle_sha256": cp["pickle_sha256"],
            "inner_checkpoint_sha256": inner_sha256,
            "inner_checkpoint_bytes": inner_bytes,
            "inner_checkpoint_sha256_independently_published": False,
            "configuration_sha256": cp["config"]["sha256"],
            "gguf_sha256": sha256(staged_model),
            "gguf_bytes": staged_model.stat().st_size,
            "original_tensor_count": contract["original_tensor_count"],
            "generator_tensor_count": contract["generator_tensor_count"],
            "export_tensor_count": len(tensors),
            "export_tensor_bytes": contract["export_tensor_bytes"],
            "metadata": metadata,
            "tensors": tensors,
            "omitted_inference_inert_buffers": omitted,
            "excluded_evaluation_state": [row["name"] for row in contract["state"]
                                           if not row["name"].startswith("generator.")],
            "excluded_auxiliary_state": [row["name"] for row in cp["auxiliary_state"]],
            "all_f32_tensor_readbacks_exact": True,
            "framework_imported": False,
            "inference_performed": False,
            "native_qualified": False,
            "representative_quality_qualified": False,
            "published": False,
            "weights_license_status": WEIGHT_LICENSE_STATUS,
            "python": platform.python_version(),
            "converter_sha256": sha256(__file__),
            "helper_sha256": sha256(Path(__file__).resolve().parents[1] / "tools/lama_contract.py"),
            "shared_helper_sha256": sha256(Path(__file__).resolve().parents[1] / "tools/latent_inpaint_contract.py"),
            "command": [sys.executable, str(Path(__file__).resolve()), "--checkpoint-archive", str(archive),
                        "--source-dir", str(source_dir), "--output", str(output)],
        }
        staged_receipt = stage / "conversion.json"
        staged_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        notices = stage / "licenses"
        notices.mkdir()
        shutil.copyfile(Path(source_dir) / "LICENSE", notices / "UPSTREAM-CODE-LICENSE")
        (notices / "WEIGHTS-LICENSE-EVIDENCE.json").write_text(
            json.dumps(contract["license"], indent=2) + "\n", encoding="utf-8")
        (notices / "SOURCE.txt").write_text(
            "Model: Big-LaMa generator (Places2), experimental CPU/F32\n"
            "Source: " + source["repository"] + "/tree/" + source["commit"] + "\n"
            "Official-linked input archive: " + cp["url"] + "\n"
            "Input archive SHA-256: " + cp["sha256"] + "\n"
            "Extracted checkpoint member: " + cp["member"]["path"] + "\n"
            "Observed inner checkpoint SHA-256: " + inner_sha256 + "\n"
            "The inner digest is observed after verified extraction; no independent upstream inner pin is asserted.\n"
            "Weight licence evidence status: " + WEIGHT_LICENSE_STATUS + "\n"
            + contract["license"]["note"] + "\n"
            "Conversion keeps the original F32 generator tensor bytes and names.\n"
            "Training-only BN counters, evaluation networks and callback scores are excluded with an exact census.\n"
            "Conversion establishes no inference parity, representative image quality or separate weight-licence grant.\n",
            encoding="utf-8")
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
    parser.add_argument("--checkpoint-archive", "--archive", dest="archive", type=Path, required=True,
                        help="Exact official-linked big-lama.zip; --archive is an alias")
    parser.add_argument("--source-dir", type=Path, required=True, help="Pinned original LaMa source checkout")
    parser.add_argument("--output", type=Path, required=True, help="New GGUF path; writes receipt and licence sidecars")
    args = parser.parse_args(argv)
    try:
        receipt = convert(args.archive, args.output, args.source_dir)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, "error: " + str(error) + "\n")
    print(json.dumps({key: receipt[key] for key in ("model", "gguf_sha256", "gguf_bytes", "export_tensor_count",
                                                  "inner_checkpoint_sha256", "native_qualified")}, indent=2))


if __name__ == "__main__":
    main()
