#!/usr/bin/env python3
"""Contract checks for the detector benchmark manifest parser."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("detector_benchmark", ROOT / "tools" / "benchmark_easyocr_detector_paths.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    craft_ref = MODULE.parse_probe(
        "easyocr-craft-reference-benchmark repetitions=10 boxes=106 graph_ms=396.027 canvas=288x544",
        MODULE.CRAFT_REFERENCE,
        "CRAFT reference",
    )
    craft_native = MODULE.parse_probe(
        "easyocr-craft-benchmark repetitions=10 boxes=106 graph_ms=850.018",
        MODULE.CRAFT_NATIVE,
        "CRAFT native",
    )
    record = MODULE.detector_record("CRAFT", craft_ref, craft_native, ["python"], ["native"])
    assert record["output"] == {"box_count_match": True}
    assert abs(record["timing_ms"]["native_over_reference"] - 850.018 / 396.027) < 1e-9
    db_ref = MODULE.parse_probe(
        "dbnet-reference-benchmark device=cpu repetitions=10 graph_ms=1213.450 input=736x1472",
        MODULE.DBNET_REFERENCE,
        "DBNet reference",
    )
    db_native = MODULE.parse_probe(
        "dbnet-benchmark threads=8 repetitions=10 cold_ms=3000.0 warm_ms=2727.3 cold_boxes=96 warm_boxes=96",
        MODULE.DBNET_NATIVE,
        "DBNet native",
    )
    assert MODULE.detector_record("DBNet", db_ref, db_native, [], [])["output"]["box_count_match"] is None
    print("easyocr detector benchmark contract: PASS")


if __name__ == "__main__":
    main()
