"""Frozen eleven-case OvisOCR3 plain-text OCR screen; standard library only."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "tests/results/qwen35-ocr/ovisocr3-quality-panel.json"
PRESERVATION = ROOT / "tests/results/qwen35-ocr/ovisocr2-quality-prepared-preservation.json"
SOURCE_PANEL = ROOT / "tests/results/qwen35-ocr/ovisocr2-quality-panel.json"
PANEL_CANONICAL_SHA256 = "bc1a4bf8754e355c6e93a0a65b0877495e05fd4da9f508a64eff4f4a06668356"
SOURCE_PANEL_CANONICAL_SHA256 = "0bec97a7a04683c7c0aec95063aa01cf9a73bcf78db61f433d7d66d89588f212"
PRESERVATION_SHA256 = "6cf0c8750202b0c11480ac7a22970ed2e9d1dd40121c7ba185cc46b916aef46f"
STOPS = frozenset((248044, 248046))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_panel():
    panel = json.loads(PANEL.read_text())
    prior = json.loads(PRESERVATION.read_text())
    source = json.loads(SOURCE_PANEL.read_text())
    if (canonical_sha(panel) != PANEL_CANONICAL_SHA256
            or file_sha(PRESERVATION) != PRESERVATION_SHA256
            or canonical_sha(source) != SOURCE_PANEL_CANONICAL_SHA256
            or panel["model"] != "ATH-MaaS/OvisOCR3"
            or panel["revision"] != "5f6ce6932efee9ec10c30746db55bd780214b7fd"
            or panel["task"] != "plain_ocr" or panel["policy"] != "ovisocr3-plain-raw-v1"
            or panel["prepared_zip_sha256"] != prior["sha256"]
            or panel["prepared_zip_bytes"] != prior["bytes"]
            or panel["renderer"] != source["renderer"]
            or len(panel["cases"]) != 11
            or [row["id"] for row in panel["cases"][:8]] != [row["id"] for row in source["cases"]]
            or [row["split"] for row in panel["cases"]] != ["development"] * 4 + ["heldout"] * 4 + ["realistic-evaluation"] * 3
            or [row["id"] for row in panel["cases"][8:]] != [row["id"] for row in prior["source"]["case_inputs"]]):
        raise ValueError("Frozen OvisOCR3 plain OCR panel differs")
    for actual, frozen in zip(panel["cases"][:8], source["cases"]):
        if (actual["gold"] != frozen["gold"] or actual["renderer"] != {key: frozen[key] for key in ("x", "y", "font_size")}
                or actual["cap"] != 256 or actual["license"] != "CC0-1.0"):
            raise ValueError("Frozen OvisOCR3 synthetic source/gold differs")
    for actual, frozen, license_row in zip(panel["cases"][8:], prior["source"]["case_inputs"],
                                           prior["source"]["licenses"]):
        if (actual["gold_file"] != license_row["gold"] or file_sha(ROOT / actual["gold_file"]) != frozen["gold_sha256"]
                or actual["gold_sha256"] != frozen["gold_sha256"]
                or actual["png_sha256"] != frozen["png_sha256"]
                or actual["rgb_sha256"] != frozen["rgb_sha256"]
                or actual["license"] != license_row["license"]
                or actual["cap"] != license_row["max_new_tokens"]):
            raise ValueError("Frozen OvisOCR3 licensed document/gold differs")
    return panel


def distance(left, right):
    """Exact Levenshtein edit count with a bounded CPU footprint."""
    if len(left) * len(right) > 25_000_000:
        raise ValueError("Plain OCR score exceeds bounded edit grid")
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, value in enumerate(left, 1):
        current = [index]
        for column, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (value != other)))
        previous = current
    return previous[-1]


def score_case(gold, row, cap):
    if not isinstance(gold, str) or not gold or not isinstance(row, dict) or type(cap) is not int or cap < 1:
        raise ValueError("Plain OCR score source/cap differs")
    text, ids, stop_id = row.get("raw_text"), row.get("token_ids"), row.get("stop_id")
    if (not isinstance(text, str) or len(text) > 16000 or not isinstance(ids, list)
            or not 1 <= len(ids) <= cap
            or any(type(token) is not int or not 0 <= token < 248320 for token in ids)
            or stop_id not in (None, 248044, 248046)
            or (stop_id is None and ids[-1] in STOPS)
            or (stop_id is not None and (ids[-1] != stop_id or stop_id not in STOPS))):
        raise ValueError("Plain OCR output IDs/text/stop differ")
    complete = stop_id in STOPS
    if row.get("stop_kind") != ("eos" if complete else "length"):
        raise ValueError("Plain OCR finish reason contradicts selected IDs")
    characters = distance(gold, text)
    words = distance(gold.split(), text.split())
    return {"raw_cer": characters / len(gold), "raw_wer": words / max(1, len(gold.split())),
            "character_edits": characters, "word_edits": words,
            "complete": complete, "cap_hit": len(ids) == cap,
            "selected_token_count": len(ids), "stop_id": stop_id}


def score_panel(cases, outputs):
    panel = load_panel()
    if (not isinstance(cases, list) or len(cases) != 11
            or [item.get("id") for item in cases] != [item["id"] for item in panel["cases"]]
            or not isinstance(outputs, dict) or set(outputs) != {item["id"] for item in panel["cases"]}):
        raise ValueError("Plain OCR eleven-case coverage/order differs")
    scored = {}
    for spec, case in zip(panel["cases"], cases):
        if (case.get("split") != spec["split"] or case.get("cap") != spec["cap"]
                or not isinstance(case.get("rgb_sha256"), str) or len(case["rgb_sha256"]) != 64
                or not isinstance(case.get("png_sha256"), str) or len(case["png_sha256"]) != 64):
            raise ValueError("Plain OCR frozen case input differs: " + spec["id"])
        if "rgb_sha256" in spec and (case["rgb_sha256"] != spec["rgb_sha256"] or case["png_sha256"] != spec["png_sha256"]):
            raise ValueError("Plain OCR reviewed document bytes differ: " + spec["id"])
        record = outputs[spec["id"]]
        if (record.get("rgb_sha256") != case["rgb_sha256"] or record.get("png_sha256") != case["png_sha256"]
                or record.get("cap") != spec["cap"]):
            raise ValueError("Plain OCR output/source binding differs: " + spec["id"])
        gold = spec["gold"] if "gold" in spec else json.loads((ROOT / spec["gold_file"]).read_text())["text"]
        scored[spec["id"]] = score_case(gold, record, spec["cap"])
    return {"policy": panel["policy"], "cases": scored,
            "all_complete": all(row["complete"] for row in scored.values()),
            "total_character_edits": sum(row["character_edits"] for row in scored.values()),
            "total_word_edits": sum(row["word_edits"] for row in scored.values())}


def paired_screen(reference, candidate):
    expected = {row["id"] for row in load_panel()["cases"]}
    if (not isinstance(reference, dict) or not isinstance(candidate, dict)
            or set(reference) != expected or set(candidate) != expected):
        raise ValueError("Plain OCR paired case coverage differs")
    result = {}
    for name in reference:
        a, b = reference[name], candidate[name]
        if not all(isinstance(row, dict) for row in (a, b)):
            raise ValueError("Plain OCR paired rows differ")
        result[name] = {"ids_exact": a.get("token_ids") == b.get("token_ids"),
                        "raw_text_exact": a.get("raw_text") == b.get("raw_text"),
                        "stop_exact": a.get("stop_id") == b.get("stop_id") and
                                      a.get("stop_kind") == b.get("stop_kind")}
    return {"cases": result, "all_exact": all(all(row.values()) for row in result.values())}
