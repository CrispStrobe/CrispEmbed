#!/usr/bin/env python3
"""Run pinned real photographic inputs through the production inpainting CLI.

This is an opt-in model-bearing integration check, not a default unit test.
Supply a manifest from tools/dump_migan_photo_reference.py prepare, the actual
crispembed executable and the selected engine's reviewed F32 GGUF digests.
The existing MI-GAN-256 tile/boundary smoke remains test_inpaint_model.py.

Example (paths and SHA values come from the caller's conversion receipts):
  python tests/test_inpaint_engine_live.py --binary build/crispembed \
    --engine migan512 --model model.gguf --model-sha256 SHA256 \
    --checkpoint-sha256 CHECKPOINT_SHA256 --fixtures photos512/fixtures.json \
    --output-dir new-cli-evidence --graph --cpu-optimized --batch-repeat

--expected-rgb FILE --expected-rgb-sha256 SHA256 additionally binds the decoded
CLI pixels to an independently saved ordinary image API result. Shared native
API agreement is not original-model or perceptual-quality qualification.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import struct
import subprocess
import sys
import time
import zlib

from test_inpaint_cli import PNG_SIGNATURE, decode_png, png_bytes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from dump_migan_photo_reference import load_fixtures  # noqa: E402
from inpaint_hash_evidence import SHA256_GATE, hash_evidence  # noqa: E402

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ENGINES = ("migan256", "migan512", "lama", "moebius", "pixelhacker")
ENGINE_GATES = ("CRISPEMBED_MIGAN", "CRISPEMBED_MIGAN512", "CRISPEMBED_LAMA",
                "CRISPEMBED_MOEBIUS", "CRISPEMBED_PIXELHACKER")
CPU_GATES = ("CRISPEMBED_INPAINT_GEMM_LAYOUT", "CRISPEMBED_INPAINT_REFLECT_GATHER",
             "CRISPEMBED_INPAINT_CPU_FFT", "CRISPEMBED_INPAINT_CPU_GLA",
             "CRISPEMBED_MOEBIUS_RELATIVE_REASSOC", "CRISPEMBED_MIGAN_CPU_KERNELS",
             "CRISPEMBED_INPAINT_CPU_ATTENTION_TILES", "CRISPEMBED_MOEBIUS_LOCAL_REASSOC",
             "CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM", "CRISPEMBED_INPAINT_CPU_DEPTHWISE", SHA256_GATE)
CPU_ENGINE_GATES = {
    "migan256": (CPU_GATES[5],), "migan512": (CPU_GATES[5],),
    "lama": CPU_GATES[:3], "moebius": (CPU_GATES[0], CPU_GATES[4]),
    "pixelhacker": (CPU_GATES[0], CPU_GATES[3]),
}
MAX_MODEL_BYTES = 4 * 1024 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value, name):
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(name + " must be a lowercase SHA-256 digest")
    return value


def regular_file(path, name, maximum):
    path = Path(path).resolve(strict=True)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
        raise ValueError(name + " must be a bounded nonempty regular file")
    return path


def fingerprint(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def file_record(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read_json(path):
    path = regular_file(path, "JSON input", MAX_METADATA_BYTES)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_options(args):
    if args.require_sha256_accel and not args.sha256_accel:
        raise ValueError("--require-sha256-accel requires --sha256-accel")
    if args.engine not in ENGINES or args.backend not in ("cpu", "cuda", "metal"):
        raise ValueError("unsupported engine or explicit backend")
    if not 1 <= args.threads <= 64 or not 1 <= args.timeout <= 7200:
        raise ValueError("threads must be 1..64 and timeout 1..7200 seconds")
    if args.backend != "cpu" and not args.graph:
        raise ValueError("CUDA and Metal require --graph")
    if args.cpu_optimized and (not args.graph or args.backend != "cpu" or args.precision != "f32"):
        raise ValueError("--cpu-optimized requires --graph --backend cpu --precision f32")
    if args.moebius_cpu_depthwise and (args.engine != "moebius" or not args.graph
                                         or args.backend != "cpu" or args.precision != "f32"):
        raise ValueError("--moebius-cpu-depthwise requires --engine moebius --graph --backend cpu --precision f32")
    if args.moebius_local_reassoc or args.moebius_local_pad_gemm:
        if args.engine != "moebius" or not args.graph or args.backend != "cpu" or args.precision != "f32":
            raise ValueError("Moebius local experiments require --engine moebius --graph --backend cpu --precision f32")
        if args.moebius_local_pad_gemm and not args.moebius_local_reassoc:
            raise ValueError("--moebius-local-pad-gemm requires --moebius-local-reassoc")
    if args.precision == "mixed" and (not args.graph or args.backend == "cpu"):
        raise ValueError("mixed precision requires a CUDA or Metal graph")
    if bool(args.expected_rgb) != bool(args.expected_rgb_sha256):
        raise ValueError("--expected-rgb and --expected-rgb-sha256 must be supplied together")
    digest(args.model_sha256, "model digest")
    for name in ("binary_sha256", "checkpoint_sha256", "vae_sha256", "vae_checkpoint_sha256",
                 "fixtures_sha256", "expected_rgb_sha256"):
        if getattr(args, name) is not None:
            digest(getattr(args, name), name)
    latent = args.engine in ("moebius", "pixelhacker")
    if latent:
        if args.vae is None or args.vae_sha256 is None:
            raise ValueError("latent engines require --vae and --vae-sha256")
        if not 2 <= args.steps <= 100 or not 0 <= args.seed < 2**64:
            raise ValueError("invalid latent step count or seed")
    elif any(x is not None for x in (args.vae, args.vae_sha256, args.vae_checkpoint_sha256)):
        raise ValueError("this engine has no VAE")
    elif args.steps != 20 or args.seed != 0:
        raise ValueError("sampler settings require a latent engine")
    if args.engine == "migan512" and args.checkpoint_sha256 is None:
        raise ValueError("MI-GAN-512 requires its independently reviewed checkpoint digest")


def checkpoint_identities(args):
    """Read frozen source identities; never infer an unknown upstream digest."""
    if args.engine == "migan512":
        return {"model_checkpoint_sha256": args.checkpoint_sha256,
                "model_checkpoint_pin_origin": "caller-supplied independently reviewed official checkpoint"}
    if args.engine == "migan256":
        source = ROOT / "tests/results/migan/original-reference.json"
        data = read_json(source)
        expected = data["checkpoint_identity"]["sha256"]
        if args.model_sha256 != data["conversion"]["gguf_sha256"]:
            raise ValueError("MI-GAN-256 model differs from its immutable reviewed GGUF")
    elif args.engine == "lama":
        source = ROOT / "docs/lama-source-contract.json"
        expected = read_json(source)["checkpoint"]["sha256"]
    else:
        source = ROOT / "docs/latent-inpaint-source-contract.json"
        data = read_json(source)
        expected = data["models"][args.engine]["checkpoint"]["sha256"]
    if args.checkpoint_sha256 is not None and args.checkpoint_sha256 != expected:
        raise ValueError("checkpoint digest differs from the frozen source identity")
    result = {"model_checkpoint_sha256": expected,
              "model_checkpoint_pin_origin": str(source.relative_to(ROOT)),
              "source_contract_sha256": sha256(source)}
    if args.engine == "lama":
        result["checkpoint_digest_scope"] = "complete official ZIP, not its inner checkpoint member"
    if args.engine in ("moebius", "pixelhacker"):
        vae = data["models"]["sdxl_vae"]["checkpoint"]["sha256"]
        if args.vae_checkpoint_sha256 is not None and args.vae_checkpoint_sha256 != vae:
            raise ValueError("VAE checkpoint differs from the frozen source identity")
        result["vae_checkpoint_sha256"] = vae
    return result


def controlled_environment(args):
    environment = dict(os.environ)
    for name in tuple(environment):
        if (name.startswith(("CRISPEMBED_INPAINT_", "INPAINT_TEST_")) or name in ENGINE_GATES + CPU_GATES
                or name in ("CRISPEMBED_IMAGE_FORMAT", "CRISPEMBED_C2PA_KEY", "CRISPEMBED_C2PA_CERT")):
            environment.pop(name, None)
    gates = {key: "0" for key in ENGINE_GATES + CPU_GATES}
    gates["CRISPEMBED_MIGAN" if args.engine.startswith("migan") else "CRISPEMBED_" + args.engine.upper()] = "1"
    if args.engine == "migan512":
        gates["CRISPEMBED_MIGAN512"] = "1"
    if args.cpu_optimized:
        gates.update({name: "1" for name in CPU_ENGINE_GATES[args.engine]})
    if args.sha256_accel:
        gates[SHA256_GATE] = "1"
    if args.moebius_cpu_depthwise:
        gates["CRISPEMBED_INPAINT_CPU_DEPTHWISE"] = "1"
    if args.moebius_local_reassoc:
        gates["CRISPEMBED_MOEBIUS_LOCAL_REASSOC"] = "1"
    if args.moebius_local_pad_gemm:
        gates["CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM"] = "1"
    gates.update({"CRISPEMBED_INPAINT_GRAPH": "1" if args.graph else "0",
                  "CRISPEMBED_INPAINT_FAST": "1" if args.precision == "mixed" else "0",
                  "CRISPEMBED_INPAINT_BACKEND": args.backend,
                  "CRISPEMBED_INPAINT_PROFILE": "1", "CRISPEMBED_IMAGE_FORMAT": "png",
                  "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
                  "OPENBLAS_NUM_THREADS": "1"})
    environment.update(gates)
    return environment, gates


def cli_arguments(args, binary, model, vae=None):
    command = [str(binary), "--inpaint-engine", args.engine, "--inpaint-model", str(model),
               "--gpu-backend", args.backend, "--threads", str(args.threads), "--mask-dilate", "0"]
    if args.engine != "migan256":
        command += ["--inpaint-model-sha256", args.model_sha256]
    if args.engine == "migan512":
        command += ["--inpaint-checkpoint-sha256", args.checkpoint_sha256]
    if args.engine in ("moebius", "pixelhacker"):
        command += ["--inpaint-vae", str(vae), "--inpaint-vae-sha256", args.vae_sha256,
                    "--inpaint-steps", str(args.steps), "--inpaint-seed", str(args.seed),
                    "--inpaint-guidance", "2" if args.engine == "moebius" else "4.5",
                    "--inpaint-noise-offset", "0.0357"]
    return command


def expected_metadata(args):
    fields = {"engine": "migan-inpaint" if args.engine == "migan256" else args.engine + "-inpaint"}
    if args.graph:
        fields.update(execution="ggml-graph", backend_request=args.backend,
                      precision_request="mixed-f16" if args.precision == "mixed" else "f32")
    if args.engine != "migan256":
        fields["model_sha256"] = args.model_sha256
    if args.engine == "migan512":
        fields["checkpoint_sha256"] = args.checkpoint_sha256
    if args.engine in ("moebius", "pixelhacker"):
        fields.update(steps=str(args.steps), seed=str(args.seed),
                      guidance="2.000000" if args.engine == "moebius" else "4.500000",
                      noise_offset="0.035700", tiles="0", vae_sha256=args.vae_sha256)
    return fields


def checked_png(data, size, rgb, mask, expected_fields, expected_rgb=None):
    """Decode independently, bounding decompression before the shared test decoder."""
    if (len(data) < 33 or len(data) > 8 * 1024 * 1024 or data[:8] != PNG_SIGNATURE
            or data[12:16] != b"IHDR" or struct.unpack_from(">II", data, 16) != (size, size)):
        raise ValueError("PNG signature, geometry or bounded file size differs")
    offset, compressed, chunks = 8, bytearray(), []
    while offset + 12 <= len(data):
        count = struct.unpack_from(">I", data, offset)[0]
        if count > len(data) - offset - 12:
            raise ValueError("truncated PNG chunk")
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + count]
        if kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"iTXt" and payload.startswith(b"CrispEmbed\0"):
            chunks.append(payload)
        offset += 12 + count
    decoder = zlib.decompressobj()
    expected_bytes = (size * 3 + 1) * size
    try:
        raw = decoder.decompress(compressed, expected_bytes + 1)
    except zlib.error as exc:
        raise ValueError("invalid PNG deflate stream") from exc
    if (len(raw) != expected_bytes or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data):
        raise ValueError("unexpected PNG decompressed extent")
    width, height, pixels, _ = decode_png(data)  # CRCs, filters, RGB8 layout and exact IEND checked here.
    if (width, height) != (size, size) or len(chunks) != 1:
        raise ValueError("expected one CrispEmbed provenance iTXt chunk")
    prefix = b"CrispEmbed\0\0\0\0\0"
    if not chunks[0].startswith(prefix):
        raise ValueError("unexpected compressed or translated provenance")
    text = chunks[0][len(prefix):].decode("utf-8")
    values = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if not sep or key in values:
            raise ValueError("malformed or duplicate PNG provenance field")
        values[key] = value
    if (values.get("generated") != "true" or values.get("software") != "CrispEmbed"
            or values.get("digitalSourceType") != "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicallyEnhanced"):
        raise ValueError("missing production provenance identity")
    segments = values.get("engine", "").split(";")
    actual_fields = {"engine": segments[0]}
    for segment in segments[1:]:
        key, sep, value = segment.partition("=")
        if not sep or key in actual_fields:
            raise ValueError("malformed or duplicate engine provenance")
        actual_fields[key] = value
    if actual_fields != expected_fields:
        raise ValueError("PNG engine/model/sampler provenance differs from the requested invocation")
    if len(rgb) != len(pixels) or len(mask) * 3 != len(pixels):
        raise ValueError("comparison input size differs")
    changed = sum(pixels[3*i:3*i+3] != rgb[3*i:3*i+3] for i, fill in enumerate(mask) if fill == 0)
    if changed:
        raise ValueError(f"CLI changed {changed} known pixels")
    if expected_rgb is not None and pixels != expected_rgb:
        raise ValueError("decoded CLI output differs from the pinned ordinary API output")
    return pixels, {"width": width, "height": height, "png_sha256": hashlib.sha256(data).hexdigest(),
                    "rgb_sha256": hashlib.sha256(pixels).hexdigest(), "metadata": values,
                    "known_pixel_errors": 0, "empty_mask_passthrough": pixels == rgb if not any(mask) else None,
                    "expected_rgb_exact": pixels == expected_rgb if expected_rgb is not None else None}


def backend_evidence(stderr, requested, graph_required, precision="f32", migan_cpu_kernels=False,
                     moebius_cpu_depthwise=False):
    if b"INPAINT_TEST_" in stderr:
        raise ValueError("test-only CLI driver cannot supply live model evidence")
    records = []
    for line in stderr.decode("utf-8", errors="replace").splitlines():
        if not (line.startswith("inpaint_graph label=") or line.startswith("migan graph production:")):
            continue
        backend = re.search(r"(?:^| )backend=(\S+)", line)
        runs = re.search(r"(?:^| )runs=(\d+)(?: |$)", line)
        actual_precision = re.search(r"(?:^| )precision=(\S+)", line)
        if not backend or not runs or not actual_precision:
            raise ValueError("malformed execution profile")
        name = backend.group(1)
        expected = {"cpu": r"cpu", "cuda": r"cuda[0-9]*", "metal": r"(?:metal|mtl)[0-9]*"}[requested]
        if re.fullmatch(expected, name.lower()) is None:
            raise ValueError("actual graph backend differs from explicit request")
        expected_precision = ("f32" if precision == "f32" else
                              "backend-default-may-reduce-f32" if line.startswith("migan graph ") else "mixed-f16")
        if actual_precision.group(1) != expected_precision:
            raise ValueError("actual graph precision differs from explicit request")
        record = {"backend": name, "runs": int(runs.group(1)),
                  "precision": actual_precision.group(1), "line": line}
        if line.startswith("migan graph production:"):
            record["label"] = "migan-production"
            for field in ("cpu_kernels", "persistent", "builds", "allocations"):
                value = re.search(r"(?:^| )" + field + r"=(\d+)(?: |$)", line)
                record[field] = int(value.group(1)) if value else None
            if migan_cpu_kernels and record["cpu_kernels"] != 1:
                raise ValueError("MI graph did not confirm requested cpu_kernels=1")
            if graph_required and record["persistent"] != 1:
                raise ValueError("MI graph did not confirm persistent execution")
        else:
            label = re.search(r"^inpaint_graph label=(.*?) backend=", line)
            record["label"] = label.group(1) if label else None
            count = re.findall(r"(?:^| )cpu_depthwise_nodes=(\S*)", line)
            if count and (len(count) != 1 or re.fullmatch(r"[0-9]+", count[0]) is None):
                raise ValueError("malformed cpu_depthwise_nodes execution profile")
            record["cpu_depthwise_nodes"] = int(count[0]) if count else None
        records.append(record)
    if graph_required and not any(row["runs"] > 0 for row in records):
        raise ValueError("filled graph invocation supplied no actual execution profile")
    if not moebius_cpu_depthwise and any(row.get("cpu_depthwise_nodes") not in (None, 0) for row in records):
        raise ValueError("runtime selected CPU depthwise nodes although the explicit gate is off")
    if moebius_cpu_depthwise and graph_required:
        expected = {"unet-cfg-ddim-batch2": 47, "vae-encoder-batch2": 0, "vae-decoder": 0}
        for label, count in expected.items():
            selected = [row for row in records if row["label"] == label]
            if len(selected) != 1 or selected[0]["runs"] <= 0 or selected[0].get("cpu_depthwise_nodes") != count:
                raise ValueError("requested Moebius CPU depthwise graph must report " + label + " cpu_depthwise_nodes=" + str(count))
    return records


def batch_reuse_evidence(records, engine, graph):
    """Require counters from once-per-image graphs, not a multi-step denoiser."""
    if engine.startswith("migan"):
        used = [row for row in records if row["label"] == "migan-production"]
        if [row["runs"] for row in used] != [1, 2]:
            raise ValueError("MI batch reconstructed its context or did not reuse cumulative image runs")
        if graph and any(row.get("persistent") != 1 or row.get("builds") != 1 or row.get("allocations") != 1 for row in used):
            raise ValueError("MI batch rebuilt or reallocated the persistent graph")
    elif graph:
        labels = ("lama-generator",) if engine == "lama" else ("vae-encoder-batch2", "vae-decoder")
        used = []
        for label in labels:
            selected = [row for row in records if row["label"] == label]
            if len(selected) != 1 or selected[0]["runs"] != 2:
                raise ValueError("batch did not retain one twice-executed " + label + " graph")
            used.extend(selected)
    else:
        return {"verified": False, "reason": "Eager LaMa/latent mode exposes no context-lifetime counter; only same-process output equality is asserted."}
    return {"verified": True, "graphs": [{key: row[key] for key in ("label", "runs", "builds", "allocations") if key in row} for row in used]}


def retained_ddim_steps(engine, requested_steps):
    if engine not in ("moebius", "pixelhacker") or not 2 <= requested_steps <= 100:
        raise ValueError("unsupported latent schedule")
    # The frozen source strengths are 0.99 and 0.999. Integer arithmetic
    # expresses their truncation over the admitted 2..100 requested steps.
    numerator, denominator = (99, 100) if engine == "moebius" else (999, 1000)
    return requested_steps * numerator // denominator


def latent_schedule_evidence(records, engine, requested_steps, filled_images, graph):
    """Verify executed graph counts; profiles do not expose timestep values."""
    retained = retained_ddim_steps(engine, requested_steps)
    if filled_images not in (0, 1, 2):
        raise ValueError("unsupported latent CLI image count")
    expected = {"vae-encoder-batch2": filled_images,
                "unet-cfg-ddim-batch2": filled_images * retained,
                "vae-decoder": filled_images}
    result = {"requested_steps": requested_steps, "retained_steps_per_filled_image": retained,
              "completed_filled_images": filled_images, "expected_graph_runs": expected,
              "stock_twenty_step_profile": requested_steps == 20,
              "timestep_values_profiled": False, "execution_count_verified": False,
              "scope": "Actual encoder/denoiser/decoder graph execution counts; timestep values remain covered by the frozen scheduler component controls."}
    if not graph or filled_images == 0:
        if any(row["runs"] != 0 for row in records):
            raise ValueError("eager or empty-mask latent CLI unexpectedly executed a graph")
        result["execution_count_verified"] = bool(graph)
        result["empty_mask_no_graph_execution"] = filled_images == 0
        return result
    if any(row["label"] not in expected for row in records):
        raise ValueError("latent schedule supplied an unexpected graph label")
    for label, runs in expected.items():
        selected = [row for row in records if row["label"] == label]
        if len(selected) != 1 or selected[0]["runs"] != runs:
            raise ValueError("latent schedule requires one retained " + label + " graph with runs=" + str(runs))
    result["execution_count_verified"] = True
    result["observed_graph_runs"] = {row["label"]: row["runs"] for row in records}
    return result


def source_provenance():
    def git(directory, *args):
        try:
            completed = subprocess.run(["git", "-C", str(directory), *args], stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed.stdout.decode().strip() if completed.returncode == 0 else None
    native = git(ROOT, "rev-parse", "HEAD")
    ggml_root = git(ROOT / "ggml", "rev-parse", "--show-toplevel")
    ggml = git(ROOT / "ggml", "rev-parse", "HEAD") if ggml_root and Path(ggml_root).resolve() == ROOT / "ggml" else None
    names = ("examples/cli/inpaint_cli.cpp", "src/core/image_out.cpp", "tests/test_inpaint_cli.py",
             "tools/dump_migan_photo_reference.py", "tests/test_inpaint_engine_live.py",
             "tools/inpaint_hash_evidence.py")
    return {"native_commit": native, "ggml_commit": ggml,
            "source_files": [{"path": name, "sha256": sha256(ROOT / name)} for name in names],
            "binary_revision_attested": False,
            "note": "Local source identity and binary digest are recorded independently; no build receipt attests their correspondence."}


class LiveRun:
    def __init__(self, args):
        self.args = args
        self.output = args.output_dir.resolve()
        self.identities = {}
        self.report = {"schema": "crispembed-inpaint-engine-live-v1", "engine": args.engine,
                       "complete": False, "passed": False, "commands": [], "outputs": [],
                       "original_model_quality_qualified": False, "perceptual_quality_qualified": False,
                       "temporal_quality_qualified": False, "same_context_repeat_exact": None,
                       "same_process_repeat_exact": None,
                       "fresh_context_and_batch_exact": None,
                       "timing_scope": "Whole CLI subprocess wall time, including model validation/loading and PNG I/O; diagnostic only, not an inference benchmark."}

    def save(self):
        (self.output / "report.json").write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")

    def track(self, path, name, maximum, expected=None):
        path = regular_file(path, name, maximum)
        before = fingerprint(path)
        identity = file_record(path)
        if fingerprint(path) != before or expected is not None and identity["sha256"] != expected:
            raise ValueError(name + " changed or differs from its caller-supplied digest")
        self.identities[path] = (before, identity["sha256"])
        return path, identity

    def check_inputs(self, rehash=False):
        for path, (initial, expected) in self.identities.items():
            if fingerprint(path) != initial or rehash and sha256(path) != expected:
                raise ValueError("pinned input changed during the live check: " + path.name)

    def prepare(self):
        args = self.args
        self.binary, binary = self.track(args.binary, "production executable", 512 * 1024 * 1024, args.binary_sha256)
        self.model, model = self.track(args.model, "model GGUF", MAX_MODEL_BYTES, args.model_sha256)
        with self.model.open("rb") as source:
            if source.read(4) != b"GGUF":
                raise ValueError("model file is not GGUF")
        self.vae, vae = None, None
        if args.vae:
            self.vae, vae = self.track(args.vae, "VAE GGUF", MAX_MODEL_BYTES, args.vae_sha256)
            with self.vae.open("rb") as source:
                if source.read(4) != b"GGUF":
                    raise ValueError("VAE file is not GGUF")
        self.hash_expected = {"migan" if args.engine.startswith("migan") else args.engine:
                              {"sha256": model["sha256"], "bytes": model["bytes"]}}
        if vae is not None:
            self.hash_expected["sdxl_vae"] = {"sha256": vae["sha256"], "bytes": vae["bytes"]}
        identities = checkpoint_identities(args)
        fixtures_path, fixture_identity = self.track(args.fixtures, "photo fixture manifest", MAX_METADATA_BYTES, args.fixtures_sha256)
        fixtures = load_fixtures(fixtures_path)
        self.size = fixtures["size"]
        if args.engine in ("moebius", "pixelhacker") and self.size != 512:
            raise ValueError("latent live checks require the native 512 photo fixtures")
        if args.engine.startswith("migan") and self.size != int(args.engine[-3:]):
            raise ValueError("use the photo manifest matching the selected MI-GAN resolution")
        cases = [case for case in fixtures["cases"] if case["name"] == args.case]
        if len(cases) != 1:
            raise ValueError("photo case is absent or duplicated")
        case = cases[0]
        raw, raw_identity = self.track(fixtures_path.parent / args.case / "input.rgb8", "fixture RGB", 512*512*3, case["rgb_sha256"])
        mask, mask_identity = self.track(fixtures_path.parent / args.case / "fill-mask.u8", "fixture mask", 512*512, case["mask_sha256"])
        self.rgb, self.mask = raw.read_bytes(), mask.read_bytes()
        if (len(self.rgb) != 3*self.size*self.size or len(self.mask) != self.size*self.size
                or set(self.mask) - {0, 255} or not any(self.mask)):
            raise ValueError("fixture is not a nonempty exact binary mask/RGB pair")
        self.expected, expected_identity = None, None
        if args.expected_rgb:
            expected, expected_identity = self.track(args.expected_rgb, "ordinary API RGB", len(self.rgb), args.expected_rgb_sha256)
            self.expected = expected.read_bytes()
            if len(self.expected) != len(self.rgb):
                raise ValueError("ordinary API RGB size differs")
            if any(self.expected[3*i:3*i+3] != self.rgb[3*i:3*i+3] for i, fill in enumerate(self.mask) if not fill):
                raise ValueError("ordinary API comparison already changed known RGB")
        self.environment, gates = controlled_environment(args)
        self.base_command = cli_arguments(args, self.binary, self.model, self.vae)
        self.metadata = expected_metadata(args)
        self.report.update({"binary": binary, "model": model, "vae": vae, "checkpoint_identities": identities,
                            "source": source_provenance(), "requested_backend": args.backend,
                            "graph": args.graph, "precision": args.precision, "cpu_optimized_requested": args.cpu_optimized,
                            "sha256_accel_requested": args.sha256_accel,
                            "sha256_hardware_required": args.require_sha256_accel,
                            "moebius_cpu_depthwise_requested": args.moebius_cpu_depthwise,
                            "moebius_local_reassoc_requested": args.moebius_local_reassoc,
                            "moebius_local_pad_gemm_requested": args.moebius_local_pad_gemm,
                            "optimizer_evidence_scope": "MI cpu_kernels and explicitly selected Moebius CPU depthwise node counts are checked in actual profiles; other optimizer flags remain requested settings, not profiler-confirmed selection.",
                            "threads": args.threads, "controlled_environment": gates,
                            "fixtures": fixture_identity, "case": case, "rgb": raw_identity, "mask": mask_identity,
                            "expected_ordinary_api_rgb": expected_identity,
                            "sampler": {"requested_steps": args.steps, "seed": args.seed,
                                        "expected_retained_ddim_steps": retained_ddim_steps(args.engine, args.steps),
                                        "guidance": 2 if args.engine == "moebius" else 4.5,
                                        "noise_offset": 0.0357, "tiles": False} if args.engine in ("moebius", "pixelhacker") else None,
                            "input_png_policy": "Regenerated from validated raw fixture bytes; no resize, filtering, dilation or mask substitution"})
        for filename, pixels, channels in (("input.png", self.rgb, 3), ("mask.png", self.mask, 1),
                                          ("empty-mask.png", bytes(len(self.mask)), 1)):
            path = self.output / filename
            path.write_bytes(png_bytes(self.size, self.size, pixels, channels))
            self.identities[path] = (fingerprint(path), sha256(path))
        self.report["cli_inputs"] = [{"file": name, "sha256": sha256(self.output / name)} for name in
                                      ("input.png", "mask.png", "empty-mask.png")]
        self.save()

    @staticmethod
    def stop(process):
        if process.poll() is not None:
            return
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)

    def run_command(self, label, command, stdout_name, graph_required):
        self.check_inputs()
        if sha256(self.binary) != self.report["binary"]["sha256"]:
            raise ValueError("production executable changed before invocation")
        stdout_path, stderr_path = self.output / stdout_name, self.output / (label + "-stderr.txt")
        row = {"label": label, "argv": command, "started_unix": time.time(), "complete": False,
               "stdout_file": stdout_path.name, "stderr_file": stderr_path.name, "timeout_seconds": self.args.timeout}
        self.report["commands"].append(row)
        self.save()
        started = time.monotonic()
        process = None
        failure = None
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                process = subprocess.Popen(command, cwd=self.output, env=self.environment, stdout=stdout, stderr=stderr,
                                           start_new_session=os.name == "posix")
                try:
                    process.wait(timeout=self.args.timeout)
                except BaseException as exc:
                    failure = exc
                    self.stop(process)
        finally:
            row.update(wall_seconds_diagnostic=time.monotonic()-started,
                       returncode=process.returncode if process is not None else None,
                       complete=process is not None and process.returncode is not None,
                       timed_out=isinstance(failure, subprocess.TimeoutExpired))
            for name, path in (("stdout", stdout_path), ("stderr", stderr_path)):
                if path.exists():
                    row[name + "_sha256"] = sha256(path)
                    row[name + "_bytes"] = path.stat().st_size
            self.save()
        if failure is not None:
            raise RuntimeError(label + ": process interrupted or timed out; partial evidence retained") from failure
        if row["returncode"] != 0:
            raise RuntimeError(label + ": production CLI failed; see retained stderr")
        self.check_inputs()
        if sha256(self.binary) != self.report["binary"]["sha256"]:
            raise ValueError("production executable changed during invocation")
        row["backend_evidence"] = backend_evidence(stderr_path.read_bytes(), self.args.backend,
                                                   graph_required, self.args.precision,
                                                   self.args.cpu_optimized and self.args.engine.startswith("migan"),
                                                   self.args.moebius_cpu_depthwise)
        if self.args.engine in ("moebius", "pixelhacker"):
            row["latent_schedule_evidence"] = latent_schedule_evidence(
                row["backend_evidence"], self.args.engine, self.args.steps,
                {"single": 1, "empty": 0, "batch": 2}[label], self.args.graph)
        row["model_hash_evidence"] = hash_evidence(
            stderr_path.read_bytes(), self.hash_expected, requested=self.args.sha256_accel,
            require_accel=self.args.require_sha256_accel, require_models=label != "empty")
        row["moebius_cpu_depthwise_runtime_verified"] = bool(self.args.moebius_cpu_depthwise and graph_required)
        self.save()

    def check_output(self, name, mask, expected=None):
        path = regular_file(self.output / name, "CLI PNG", 8 * 1024 * 1024)
        pixels, details = checked_png(path.read_bytes(), self.size, self.rgb, mask, self.metadata, expected)
        details["expected_rgb_kind"] = ("original_input_empty_mask" if not any(mask) else
                                        "pinned_ordinary_image_api" if expected is not None else None)
        self.report["outputs"].append({"file": name, **details})
        self.save()
        return pixels

    def execute(self):
        self.prepare()
        single = self.base_command + ["--inpaint", str(self.output / "input.png"), "--mask", str(self.output / "mask.png")]
        self.run_command("single", single, "single.png", self.args.graph)
        first = self.check_output("single.png", self.mask, self.expected)
        empty = self.base_command + ["--inpaint", str(self.output / "input.png"), "--mask", str(self.output / "empty-mask.png")]
        self.run_command("empty", empty, "empty.png", False)
        self.check_output("empty.png", bytes(len(self.mask)), self.rgb)
        if self.args.batch_repeat:
            manifest = self.output / "frames.tsv"
            manifest.write_text("input.png\tmask.png\tbatch-first.png\n"
                                "input.png\tempty-mask.png\tbatch-empty.png\n"
                                "input.png\tmask.png\tbatch-repeat.png\n")
            self.identities[manifest] = (fingerprint(manifest), sha256(manifest))
            batch = self.base_command + ["--inpaint-list", str(manifest)]
            self.run_command("batch", batch, "batch-stdout.txt", self.args.graph)
            if (self.output / "batch-stdout.txt").stat().st_size:
                raise ValueError("file-output batch contaminated stdout")
            stderr = (self.output / "batch-stderr.txt").read_bytes()
            for index in (1, 2, 3):
                if stderr.count(f"inpaint: {index}/3 ".encode()) != 1:
                    raise ValueError("batch did not complete each expected frame once")
            batch_first = self.check_output("batch-first.png", self.mask, self.expected)
            self.check_output("batch-empty.png", bytes(len(self.mask)), self.rgb)
            batch_repeat = self.check_output("batch-repeat.png", self.mask, self.expected)
            if first != batch_first or first != batch_repeat:
                raise ValueError("fresh/same-context CLI outputs differ")
            reuse = batch_reuse_evidence(self.report["commands"][-1]["backend_evidence"], self.args.engine, self.args.graph)
            self.report.update(batch_context_reuse=reuse, same_process_repeat_exact=True,
                               same_context_repeat_exact=True if reuse["verified"] else None,
                               fresh_context_and_batch_exact=True)
        self.check_inputs(rehash=True)
        self.report.update(complete=True, passed=True, all_pinned_inputs_unchanged=True)
        self.save()


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--binary", type=Path, required=True)
    result.add_argument("--binary-sha256")
    result.add_argument("--engine", choices=ENGINES, required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--model-sha256", required=True)
    result.add_argument("--checkpoint-sha256")
    result.add_argument("--vae", type=Path)
    result.add_argument("--vae-sha256")
    result.add_argument("--vae-checkpoint-sha256")
    result.add_argument("--fixtures", type=Path, required=True)
    result.add_argument("--fixtures-sha256")
    result.add_argument("--case", choices=("formula_photo_bar", "document_photo_center"), default="formula_photo_bar")
    result.add_argument("--expected-rgb", type=Path)
    result.add_argument("--expected-rgb-sha256")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--backend", choices=("cpu", "cuda", "metal"), default="cpu")
    result.add_argument("--graph", action="store_true")
    result.add_argument("--cpu-optimized", action="store_true")
    result.add_argument("--sha256-accel", action="store_true",
                        help="explicit optional accelerated model hashing; honest scalar fallback is allowed")
    result.add_argument("--require-sha256-accel", action="store_true",
                        help="with --sha256-accel, require actual SHA-NI blocks for every loaded model")
    result.add_argument("--moebius-cpu-depthwise", action="store_true",
                        help="Explicit experimental Moebius CPU depthwise kernels; requires CPU/F32 graph and 47 actual UNet nodes")
    result.add_argument("--moebius-local-reassoc", action="store_true",
                        help="Explicit experimental Moebius local reassociation; requires CPU/F32 graph")
    result.add_argument("--moebius-local-pad-gemm", action="store_true",
                        help="Explicit experimental Moebius local padding GEMM; requires --moebius-local-reassoc")
    result.add_argument("--precision", choices=("f32", "mixed"), default="f32")
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--steps", type=int, default=20)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--timeout", type=int, default=1800)
    result.add_argument("--batch-repeat", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    validate_options(args)
    if args.output_dir.is_symlink():
        raise ValueError("evidence output must not be a symlink")
    output = args.output_dir.resolve()
    for value in (args.binary, args.model, args.vae, args.fixtures, args.expected_rgb):
        if value is not None and (output == value.resolve() or output in value.resolve().parents):
            raise ValueError("evidence output cannot contain or replace an input")
    output.mkdir(parents=True, exist_ok=False)
    run = LiveRun(args)
    run.save()
    try:
        run.execute()
    except BaseException as exc:
        run.report.update(complete=True, passed=False, error=f"{type(exc).__name__}: {exc}")
        run.save()
        raise
    print(json.dumps({"schema": run.report["schema"], "engine": args.engine, "passed": True,
                      "outputs": len(run.report["outputs"]), "report": str(output / "report.json"),
                      "original_model_quality_qualified": False}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print("inpainting live check failed: " + str(error), file=sys.stderr)
        sys.exit(1)
