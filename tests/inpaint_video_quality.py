#!/usr/bin/env python3
"""Owned, deterministic video fixtures and paired raw-stream quality measurements.

Five six-frame clips, no downloaded inputs or implicit model download. The fake
engine mode checks the harness only; it cannot qualify an inpainting model.
Reports retain the declared MI-GAN/load controls and two CPU thread variables;
real-arm values are preserved, while fake mode clears inherited model controls.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


SCHEMA = "crispembed-inpaint-video-quality-v1"
CASES = ("static", "fade", "moving-fine-texture", "subtitle-change", "scene-cut")
WIDTH, HEIGHT, FRAME_COUNT = 96, 64, 6
GATES = ("CRISPEMBED_INPAINT_STREAM", "CRISPEMBED_MIGAN", "CRISPEMBED_MIGAN512",
         "CRISPEMBED_MOEBIUS", "CRISPEMBED_PIXELHACKER", "CRISPEMBED_INPAINT_GRAPH",
         "CRISPEMBED_INPAINT_FAST", "CRISPEMBED_MIGAN_GPU", "CRISPEMBED_MIGAN_CPU_KERNELS",
         "CRISPEMBED_MIGAN_FAST_MATMUL", "CRISPEMBED_MIGAN_BATCH", "CRISPEMBED_MIGAN_DEQUANT_WEIGHTS",
         "CRISPEMBED_MIGAN_QUANT", "CRISPEMBED_INPAINT_SHA256_ACCEL")
RUNTIME_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
SUMMARY = re.compile(r"inpaint: (\d+) frames, (\d+) masked, (\d+) engine;")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


@dataclass(frozen=True)
class Clip:
    name: str
    width: int
    height: int
    clean: tuple
    masks: tuple

    def records(self):
        records = bytearray()
        for rgb, mask in zip(self.clean, self.masks):
            burned = bytearray(rgb)
            for i, value in enumerate(mask):
                if value >= 128:
                    burned[3 * i:3 * i + 3] = b"\xff\xff\xff"
            records.extend(burned)
            records.extend(mask)
        return bytes(records)


def fixture(name):
    """All source pixels and glyphs are authored here; no RNG or external asset."""
    if name not in CASES:
        raise ValueError("unknown video fixture")
    glyphs = (("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
              ("11110", "10001", "10001", "11110", "10001", "10001", "11110"))
    clean, masks = [], []
    for frame in range(FRAME_COUNT):
        rgb, mask = bytearray(), bytearray(WIDTH * HEIGHT)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if name == "static":
                    color = (40, 140, 60)
                elif name == "moving-fine-texture":
                    # Nearly equal luma, very different chroma. One-pixel motion
                    # also preserves many 8x8 means: both old shortcuts are tested.
                    color = (240, 20, 20) if (x + y + frame) % 2 else (20, 132, 20)
                else:
                    level = 45 + 90 * ((x // 6 + y // 5) % 2)
                    color = (level, level + 20, level + 40)
                    if name == "fade":
                        # Adjacent frames are below the old static threshold,
                        # but the cumulative drift reaches 10 by the last frame.
                        color = tuple(channel + frame * 2 for channel in color)
                    if name == "scene-cut" and frame >= FRAME_COUNT // 2:
                        color = (205 - level, 35, 70 + level)
                rgb.extend(color)
        # Clean endpoints give propagation a defined opportunity; the fade,
        # moving pattern and scene cut keep their masks throughout the clip.
        if name not in ("static", "subtitle-change") or frame not in (0, FRAME_COUNT - 1):
            glyph = glyphs[int(name == "subtitle-change" and frame >= FRAME_COUNT // 2)]
            for left in (18, 34, 50, 66):
                for gy, row in enumerate(glyph):
                    for gx, bit in enumerate(row):
                        if bit == "1":
                            for dy in (0, 1):
                                for dx in (0, 1):
                                    mask[(44 + 2 * gy + dy) * WIDTH + left + 2 * gx + dx] = 255
        clean.append(bytes(rgb))
        masks.append(bytes(mask))
    return Clip(name, WIDTH, HEIGHT, tuple(clean), tuple(masks))


def metrics(clip, output):
    frame_bytes = clip.width * clip.height * 3
    if len(output) != frame_bytes * len(clip.clean):
        raise ValueError("output does not contain exactly one RGB8 record per input frame")
    squared = masked_channels = known_errors = known_max = 0
    temporal_abs = temporal_squared = temporal_channels = 0
    previous_error = previous_mask = None
    for t, (clean, mask) in enumerate(zip(clip.clean, clip.masks)):
        actual = output[t * frame_bytes:(t + 1) * frame_bytes]
        error = [a - b for a, b in zip(actual, clean)]
        for i, delta in enumerate(error):
            if mask[i // 3] >= 128:
                squared += delta * delta
                masked_channels += 1
            else:
                known_errors += delta != 0
                known_max = max(known_max, abs(delta))
            if previous_error is not None and (mask[i // 3] >= 128 or previous_mask[i // 3] >= 128):
                change = delta - previous_error[i]
                temporal_abs += abs(change)
                temporal_squared += change * change
                temporal_channels += 1
        previous_error, previous_mask = error, mask
    mse = squared / masked_channels if masked_channels else 0.0
    return {"masked_channels": masked_channels, "masked_mse": mse,
            "masked_psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
            "masked_exact": squared == 0, "known_rgb_exact": known_errors == 0,
            "known_channel_errors": known_errors, "known_max_abs": known_max,
            "temporal_channels": temporal_channels,
            "temporal_error_mae": temporal_abs / temporal_channels if temporal_channels else 0.0,
            "temporal_error_mse": temporal_squared / temporal_channels if temporal_channels else 0.0}


def compare(baseline, temporal, max_mse_increase=None, max_temporal_increase=None):
    mse_change = temporal["masked_mse"] - baseline["masked_mse"]
    flicker_change = temporal["temporal_error_mae"] - baseline["temporal_error_mae"]
    violations = []
    if max_mse_increase is not None and mse_change > max_mse_increase:
        violations.append("masked MSE increase exceeds declared bound")
    if max_temporal_increase is not None and flicker_change > max_temporal_increase:
        violations.append("temporal error MAE increase exceeds declared bound")
    return {"masked_mse_change": mse_change, "temporal_error_mae_change": flicker_change,
            "masked_mse_regressed": mse_change > 0, "temporal_error_mae_regressed": flicker_change > 0,
            "declared_bound_violations": violations}


def run_process(command, input_path, output_path, stderr_path, environment, timeout):
    """Use files to avoid pipe deadlock; wait4 gives this child's exact peak RSS."""
    started = time.monotonic()
    peak_rss = rss_method = None
    timed_out = False
    with input_path.open("rb") as source, output_path.open("wb") as output, stderr_path.open("wb") as errors:
        process = subprocess.Popen(command, stdin=source, stdout=output, stderr=errors, env=environment)
        if hasattr(os, "wait4"):
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    if sys.platform in ("linux", "darwin"):
                        peak_rss = int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
                        rss_method = "wait4.ru_maxrss"
                    break
                if time.monotonic() - started > timeout and not timed_out:
                    # Popen.kill() polls with waitpid on POSIX and can reap a
                    # child that exited after our wait4 poll. Keep wait4 the sole
                    # reaper so its exit status and per-child RSS are retained.
                    try:
                        os.kill(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    timed_out = True
                time.sleep(0.01)
        else:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                timed_out = True
    return {"returncode": process.returncode, "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_rss, "peak_rss_method": rss_method}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_panel(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)  # Never replace an earlier failed run.
    report = {"schema": SCHEMA, "source_sha": args.source_sha,
              "harness_sha256": file_digest(__file__), "fake_engine": args.fake_engine,
              "model_quality_qualified": False, "passed": False, "cases": [],
              "protocol": {"engine": "migan256", "backend": "cpu", "mask_threshold": 128,
                           "width": WIDTH, "height": HEIGHT, "frames_per_case": FRAME_COUNT,
                           "temporal_window": args.window, "threads": args.threads,
                           "max_mse_increase": args.max_mse_increase,
                           "max_temporal_mae_increase": args.max_temporal_mae_increase}}
    try:
        environment = os.environ.copy()
        for key in list(environment):
            if key.startswith("INPAINT_TEST_"):
                environment.pop(key)
        if args.fake_engine:
            model = root / "fake-model.gguf"
            model.write_bytes(b"test-only inpaint video harness model; no real weights\n")
            for key in GATES:
                environment.pop(key, None)
            environment.update(CRISPEMBED_MIGAN="1", CRISPEMBED_INPAINT_STREAM="1")
        else:
            if args.model is None:
                raise ValueError("--model is required unless --fake-engine is explicit")
            if re.fullmatch(r"[0-9a-f]{40}", args.source_sha) is None:
                raise ValueError("real model measurement requires --source-sha with the full source commit")
            model = args.model.resolve()
            # Real engine gates must be explicit in the hosted caller/environment.
            for key in ("CRISPEMBED_INPAINT_STREAM", "CRISPEMBED_MIGAN"):
                if environment.get(key) in (None, "", "0"):
                    raise ValueError(f"real model measurement requires {key}=1")
        binary = args.binary.resolve()
        report.update(binary_sha256=file_digest(binary), model_sha256=file_digest(model),
                      model_bytes=model.stat().st_size,
                      gates={key: environment.get(key) for key in GATES},
                      runtime_environment={key: environment.get(key) for key in RUNTIME_ENVIRONMENT})
        if args.expected_model_sha256 and report["model_sha256"] != args.expected_model_sha256:
            raise ValueError("model SHA-256 differs from the declared model identity")
        successful = True
        for name in CASES:
            clip = fixture(name)
            folder = root / name
            folder.mkdir()
            inputs = {"input.rgbm": clip.records(), "clean.rgb": b"".join(clip.clean),
                      "mask.gray": b"".join(clip.masks)}
            for filename, data in inputs.items():
                (folder / filename).write_bytes(data)
            row = {"case": name, "input_sha256": {key: digest(value) for key, value in inputs.items()},
                   "arms": {}}
            report["cases"].append(row)
            for arm, window in (("independent", 0), ("temporal", args.window)):
                output, stderr = folder / f"{arm}.rgb", folder / f"{arm}.stderr.txt"
                command = [str(binary), "--inpaint-model", str(model), "--inpaint-stream", f"{WIDTH}x{HEIGHT}",
                           "--inpaint-temporal", str(window), "--threads", str(args.threads),
                           "--gpu-backend", "cpu"]
                result = run_process(command, folder / "input.rgbm", output, stderr, environment, args.timeout)
                row["arms"][arm] = result
                result["output_sha256"] = file_digest(output)
                result["stderr_sha256"] = file_digest(stderr)
                match = SUMMARY.search(stderr.read_text(encoding="utf-8", errors="replace"))
                result["counters"] = dict(zip(("frames", "masked_frames", "engine_frames"),
                                               map(int, match.groups()))) if match else None
                result["passed"] = result["returncode"] == 0 and not result["timed_out"]
                try:
                    result["metrics"] = metrics(clip, output.read_bytes())
                    result["passed"] &= result["metrics"]["known_rgb_exact"]
                    masked_frames = sum(any(value >= 128 for value in mask) for mask in clip.masks)
                    result["passed"] &= bool(match and int(match[1]) == FRAME_COUNT
                                             and int(match[2]) == masked_frames
                                             and int(match[3]) <= masked_frames)
                except ValueError as error:
                    result["error"] = str(error)
                    result["passed"] = False
                successful &= result["passed"]
                write_json(root / "report.json", report)  # Keep partial/failure evidence.
            arms = row["arms"]
            if all("metrics" in arm for arm in arms.values()):
                row["comparison"] = compare(arms["independent"]["metrics"], arms["temporal"]["metrics"],
                                             args.max_mse_increase, args.max_temporal_mae_increase)
                successful &= not row["comparison"]["declared_bound_violations"]
        report["passed"] = bool(successful)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
    finally:
        write_json(root / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fake-engine", action="store_true")
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--source-sha", default="unrecorded")
    parser.add_argument("--max-mse-increase", type=float)
    parser.add_argument("--max-temporal-mae-increase", type=float)
    args = parser.parse_args()
    if not 1 <= args.window <= 120 or not 1 <= args.threads <= 64 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("window must be 1..120, threads 1..64, timeout finite and positive")
    for value in (args.max_mse_increase, args.max_temporal_mae_increase):
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error("declared metric increases must be finite and nonnegative")
    report = run_panel(args)
    print(json.dumps({"passed": report["passed"], "fake_engine": report["fake_engine"],
                      "model_quality_qualified": False, "cases": len(report["cases"])}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
