#!/usr/bin/env python3
"""Model-free admission, PNG-integrity and failure-retention controls for the live runner."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import test_inpaint_engine_live as live
from test_inpaint_cli import png_bytes, png_chunk


def arguments(engine="migan512", extra=()):
    argv = ["--binary", "crispembed", "--engine", engine, "--model", "model.gguf",
            "--model-sha256", "a" * 64, "--fixtures", "photos/fixtures.json", "--output-dir", "new-evidence"]
    if engine == "migan512":
        argv += ["--checkpoint-sha256", "b" * 64]
    if engine in ("moebius", "pixelhacker"):
        argv += ["--vae", "vae.gguf", "--vae-sha256", "c" * 64]
    return live.parser().parse_args(argv + list(extra))


def marked_png(size, pixels, fields, duplicate=False):
    value = fields["engine"] + "".join(";" + key + "=" + val for key, val in fields.items() if key != "engine")
    text = ("generated=true\nsoftware=CrispEmbed\nengine=" + value + "\n"
            "digitalSourceType=http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicallyEnhanced\n")
    chunk = png_chunk(b"iTXt", b"CrispEmbed\0\0\0\0\0" + text.encode())
    data = png_bytes(size, size, pixels)
    return data[:33] + chunk * (2 if duplicate else 1) + data[33:]


class InpaintEngineLiveControls(unittest.TestCase):
    def test_all_engine_dispatch_and_pinned_arguments(self):
        for engine in live.ENGINES:
            with self.subTest(engine=engine):
                args = arguments(engine)
                live.validate_options(args)
                command = live.cli_arguments(args, Path("crispembed"), Path("model.gguf"), Path("vae.gguf"))
                self.assertEqual(command[command.index("--inpaint-engine") + 1], engine)
                self.assertEqual("--inpaint-model-sha256" in command, engine != "migan256")
                self.assertEqual("--inpaint-checkpoint-sha256" in command, engine == "migan512")
                self.assertEqual("--inpaint-vae" in command, engine in ("moebius", "pixelhacker"))
                self.assertEqual(command[command.index("--mask-dilate") + 1], "0")

    def test_latent_stock_sampler_defaults_are_explicit(self):
        for engine, guidance in (("moebius", "2"), ("pixelhacker", "4.5")):
            args = arguments(engine)
            command = live.cli_arguments(args, Path("crispembed"), Path("model.gguf"), Path("vae.gguf"))
            for name, expected in (("--inpaint-steps", "20"), ("--inpaint-seed", "0"),
                                   ("--inpaint-guidance", guidance), ("--inpaint-noise-offset", "0.0357")):
                self.assertEqual(command[command.index(name) + 1], expected)
            self.assertNotIn("--inpaint-tiles", command)

    def test_invalid_engine_profile_and_sampler_combinations_rejected(self):
        cases = [arguments(extra=("--backend", "cuda")),
                 arguments(extra=("--cpu-optimized",)),
                 arguments(extra=("--graph", "--cpu-optimized", "--backend", "metal")),
                 arguments(extra=("--graph", "--precision", "mixed")),
                 arguments(extra=("--expected-rgb", "rgb.raw")),
                 arguments(extra=("--expected-rgb-sha256", "a" * 64)),
                 arguments(extra=("--threads", "0")),
                 arguments(extra=("--timeout", "7201")),
                 arguments(extra=("--steps", "19")),
                 arguments("moebius", ("--seed", str(2**64))),
                 arguments("pixelhacker", ("--steps", "1"))]
        cases.append(arguments())
        cases[-1].checkpoint_sha256 = None
        cases.append(arguments("moebius"))
        cases[-1].vae_sha256 = None
        cases.append(arguments("lama", ("--vae", "wrong.gguf")))
        for args in cases:
            with self.subTest(engine=args.engine, args=vars(args)):
                with self.assertRaises(ValueError):
                    live.validate_options(args)

    def test_environment_does_not_inherit_other_profiles_or_signing(self):
        dirty = {name: "1" for name in live.ENGINE_GATES + live.CPU_GATES}
        dirty.update(CRISPEMBED_INPAINT_FAST="1", CRISPEMBED_INPAINT_UNKNOWN="1",
                     CRISPEMBED_C2PA_KEY="unused-test-key", CRISPEMBED_C2PA_CERT="unused-test-cert",
                     INPAINT_TEST_LOAD="1", CRISPEMBED_IMAGE_FORMAT="ppm")
        with mock.patch.dict(os.environ, dirty):
            for engine in live.ENGINES:
                args = arguments(engine, ("--graph", "--cpu-optimized"))
                environment, gates = live.controlled_environment(args)
                for name in live.CPU_GATES:
                    self.assertEqual(gates[name], "1" if name in live.CPU_ENGINE_GATES[engine] else "0")
                for name in ("CRISPEMBED_C2PA_KEY", "CRISPEMBED_C2PA_CERT", "INPAINT_TEST_LOAD",
                             "CRISPEMBED_INPAINT_UNKNOWN"):
                    self.assertNotIn(name, environment)
                self.assertEqual(environment["CRISPEMBED_INPAINT_FAST"], "0")
                self.assertEqual(environment["CRISPEMBED_IMAGE_FORMAT"], "png")
                self.assertEqual(environment["CRISPEMBED_MIGAN512"], "1" if engine == "migan512" else "0")

    def test_moebius_local_experiments_default_off_and_require_explicit_flags(self):
        for optimized in (False, True):
            base = ("--graph",) + (("--cpu-optimized",) if optimized else ())
            for selected in ((), ("--moebius-local-reassoc",),
                             ("--moebius-local-reassoc", "--moebius-local-pad-gemm")):
                args = arguments("moebius", base + selected)
                live.validate_options(args)
                environment, gates = live.controlled_environment(args)
                self.assertEqual(gates["CRISPEMBED_MOEBIUS_LOCAL_REASSOC"],
                                 "1" if "--moebius-local-reassoc" in selected else "0")
                self.assertEqual(gates["CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM"],
                                 "1" if "--moebius-local-pad-gemm" in selected else "0")
                self.assertEqual(gates["CRISPEMBED_MOEBIUS_RELATIVE_REASSOC"], "1" if optimized else "0")
                self.assertEqual(gates["CRISPEMBED_INPAINT_CPU_ATTENTION_TILES"], "0")
                self.assertEqual(environment["CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM"],
                                 gates["CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM"])

    def test_moebius_local_flags_reject_other_engines_profiles_and_missing_dependency(self):
        invalid = [arguments(engine, ("--graph", "--moebius-local-reassoc"))
                   for engine in live.ENGINES if engine != "moebius"]
        invalid += [arguments("moebius", ("--moebius-local-reassoc",)),
                    arguments("moebius", ("--graph", "--moebius-local-pad-gemm")),
                    arguments("moebius", ("--graph", "--backend", "cuda", "--moebius-local-reassoc")),
                    arguments("moebius", ("--graph", "--backend", "metal", "--moebius-local-reassoc")),
                    arguments("moebius", ("--graph", "--precision", "mixed", "--moebius-local-reassoc"))]
        for args in invalid:
            with self.subTest(engine=args.engine, backend=args.backend, graph=args.graph), self.assertRaises(ValueError):
                live.validate_options(args)

    def test_inherited_moebius_local_experiments_cannot_enable_default_profile(self):
        with mock.patch.dict(os.environ, {"CRISPEMBED_MOEBIUS_LOCAL_REASSOC": "1",
                                          "CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM": "1"}):
            args = arguments("moebius", ("--graph", "--cpu-optimized"))
            environment, gates = live.controlled_environment(args)
            for name in ("CRISPEMBED_MOEBIUS_LOCAL_REASSOC", "CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM"):
                self.assertEqual(gates[name], "0")
                self.assertEqual(environment[name], "0")

    def test_frozen_checkpoint_identity_and_unknown512_provenance(self):
        args = arguments("migan256")
        with self.assertRaisesRegex(ValueError, "immutable"):
            live.checkpoint_identities(args)
        reviewed = live.read_json(live.ROOT / "tests/results/migan/original-reference.json")
        args.model_sha256 = reviewed["conversion"]["gguf_sha256"]
        self.assertEqual(live.checkpoint_identities(args)["model_checkpoint_sha256"],
                         reviewed["checkpoint_identity"]["sha256"])
        args.checkpoint_sha256 = "f" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint digest"):
            live.checkpoint_identities(args)
        args = arguments("lama", ("--checkpoint-sha256", "f" * 64))
        with self.assertRaisesRegex(ValueError, "checkpoint digest"):
            live.checkpoint_identities(args)
        identity = live.checkpoint_identities(arguments())
        self.assertEqual(identity["model_checkpoint_sha256"], "b" * 64)
        self.assertIn("caller-supplied", identity["model_checkpoint_pin_origin"])

    def test_png_pixels_and_exact_provenance_for_all_engines(self):
        size = 2
        rgb = bytes(range(12))
        mask = bytes((0, 255, 0, 0))
        output = rgb[:3] + b"\x55\x66\x77" + rgb[6:]
        for engine in live.ENGINES:
            args = arguments(engine, ("--graph",))
            fields = live.expected_metadata(args)
            pixels, record = live.checked_png(marked_png(size, output, fields), size, rgb, mask, fields, output)
            self.assertEqual(pixels, output)
            self.assertEqual(record["known_pixel_errors"], 0)
            self.assertTrue(record["expected_rgb_exact"])

    def test_wrong_model_metadata_and_duplicates_rejected(self):
        rgb = bytes(range(12))
        fields = live.expected_metadata(arguments())
        changed = dict(fields, model_sha256="d" * 64)
        for data in (marked_png(2, rgb, changed), marked_png(2, rgb, fields, duplicate=True)):
            with self.assertRaises(ValueError):
                live.checked_png(data, 2, rgb, bytes(4), fields)
        changed = dict(fields, unexpected="not-in-request")
        with self.assertRaises(ValueError):
            live.checked_png(marked_png(2, rgb, changed), 2, rgb, bytes(4), fields)

    def test_known_pixels_empty_mask_and_expected_api_mismatch(self):
        rgb = bytes(range(12))
        fields = live.expected_metadata(arguments())
        data = marked_png(2, rgb, fields)
        _, checked = live.checked_png(data, 2, rgb, bytes(4), fields, rgb)
        self.assertTrue(checked["empty_mask_passthrough"])
        changed = bytes((255,)) + rgb[1:]
        with self.assertRaisesRegex(ValueError, "known pixels"):
            live.checked_png(marked_png(2, changed, fields), 2, rgb, bytes((0, 255, 0, 0)), fields)
        with self.assertRaisesRegex(ValueError, "ordinary API"):
            live.checked_png(data, 2, rgb, bytes((255, 0, 0, 0)), fields, changed)

    def test_png_corruption_geometry_and_trailing_bytes_rejected(self):
        rgb = bytes(range(12))
        fields = live.expected_metadata(arguments())
        data = marked_png(2, rgb, fields)
        corrupt = bytearray(data)
        corrupt[-5] ^= 1
        for value in (bytes(corrupt), data + b"extra", data[:25]):
            with self.assertRaises((ValueError, AssertionError)):
                live.checked_png(value, 2, rgb, bytes(4), fields)
        with self.assertRaises(ValueError):
            live.checked_png(data, 512, rgb, bytes(4), fields)

    def test_backend_evidence_rejects_fallback_stub_and_empty_execution(self):
        profile = b"inpaint_graph label=generator backend=MTL0 precision=f32 nodes=3 allocation_bytes=16 runs=1 upload_bytes=4 download_bytes=4 compute_ms=1\n"
        self.assertEqual(live.backend_evidence(profile, "metal", True)[0]["backend"], "MTL0")
        for value, backend in ((profile, "cuda"), (profile.replace(b"MTL0", b"cuda-tool"), "cuda"),
                               (profile.replace(b"runs=1", b"runs=0"), "metal"),
                               (profile + b"INPAINT_TEST_LOAD\n", "metal"), (b"", "cpu")):
            with self.assertRaises(ValueError):
                live.backend_evidence(value, backend, True)
        with self.assertRaisesRegex(ValueError, "precision"):
            live.backend_evidence(profile.replace(b"precision=f32", b"precision=mixed-f16"), "metal", True)

    def test_migan_mixed_profile_matches_its_actual_diagnostic(self):
        profile = b"migan graph production: backend=Metal0 persistent=1 precision=backend-default-may-reduce-f32 runs=1\n"
        self.assertEqual(live.backend_evidence(profile, "metal", True, "mixed")[0]["runs"], 1)

    def test_requested_migan_cpu_kernel_must_be_confirmed_by_runtime(self):
        profile = b"migan graph production: backend=CPU persistent=1 precision=f32 cpu_kernels=1 builds=1 allocations=1 runs=1\n"
        self.assertEqual(live.backend_evidence(profile, "cpu", True, "f32", True)[0]["cpu_kernels"], 1)
        for wrong in (profile.replace(b"cpu_kernels=1", b"cpu_kernels=0"),
                      profile.replace(b"cpu_kernels=1 ", b""),
                      profile.replace(b"persistent=1", b"persistent=0")):
            with self.assertRaises(ValueError):
                live.backend_evidence(wrong, "cpu", True, "f32", True)

    def test_batch_migan_requires_one_persistent_context(self):
        prefix = b"migan graph production: backend=CPU persistent=1 precision=f32 cpu_kernels=1 builds=1 allocations=1 runs="
        good = live.backend_evidence(prefix + b"1\n" + prefix + b"2\n", "cpu", True, "f32", True)
        self.assertTrue(live.batch_reuse_evidence(good, "migan512", True)["verified"])
        reset = live.backend_evidence(prefix + b"1\n" + prefix + b"1\n", "cpu", True, "f32", True)
        with self.assertRaisesRegex(ValueError, "reconstructed"):
            live.batch_reuse_evidence(reset, "migan512", True)
        rebuilt = [dict(row, builds=2) for row in good]
        with self.assertRaisesRegex(ValueError, "rebuilt"):
            live.batch_reuse_evidence(rebuilt, "migan512", True)

    def test_batch_latent_proof_uses_vae_once_per_image_not_step_count(self):
        good = [{"label": "vae-encoder-batch2", "runs": 2}, {"label": "vae-decoder", "runs": 2},
                {"label": "unet-cfg-ddim-batch2", "runs": 38}]
        self.assertTrue(live.batch_reuse_evidence(good, "moebius", True)["verified"])
        restarted = [{"label": "vae-encoder-batch2", "runs": 1}, {"label": "vae-decoder", "runs": 1},
                     {"label": "unet-cfg-ddim-batch2", "runs": 19}] * 2
        with self.assertRaisesRegex(ValueError, "retain"):
            live.batch_reuse_evidence(restarted, "moebius", True)
        self.assertFalse(live.batch_reuse_evidence([], "moebius", False)["verified"])
        self.assertTrue(live.batch_reuse_evidence([{"label": "lama-generator", "runs": 2}], "lama", True)["verified"])
        with self.assertRaises(ValueError):
            live.batch_reuse_evidence([{"label": "lama-generator", "runs": 1}] * 2, "lama", True)

    def test_digest_and_mid_run_file_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args = arguments(extra=("--output-dir", str(root)))
            runner = live.LiveRun(args)
            file = root / "input.raw"
            file.write_bytes(b"1234")
            with self.assertRaisesRegex(ValueError, "digest"):
                runner.track(file, "input", 4, "a" * 64)
            runner.track(file, "input", 4, hashlib.sha256(b"1234").hexdigest())
            file.write_bytes(b"4321")
            # Same-size rewrites inside one filesystem timestamp tick (Windows
            # NTFS) keep the cheap fingerprint; the final rehash still rejects.
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.check_inputs(rehash=True)
            file.write_bytes(b"43210")
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.check_inputs()

    def test_existing_output_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            marker = root / "keep"
            marker.write_text("existing evidence")
            argv = ["--binary", "crispembed", "--engine", "migan512", "--model", "model.gguf",
                    "--model-sha256", "a" * 64, "--checkpoint-sha256", "b" * 64,
                    "--fixtures", "fixtures.json", "--output-dir", str(root)]
            with self.assertRaises(FileExistsError):
                live.main(argv)
            self.assertEqual(marker.read_text(), "existing evidence")

    def test_timeout_retains_both_streams_and_stops_only_its_process(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args = arguments(extra=("--output-dir", str(root)))
            runner = live.LiveRun(args)
            runner.binary = root / "not-executed-model-free-control"
            runner.binary.write_bytes(b"not a model executable")
            runner.report["binary"] = live.file_record(runner.binary)
            runner.environment = {}
            process = mock.Mock()
            process.returncode = None
            process.wait.side_effect = subprocess.TimeoutExpired(["not-executed"], 1)

            def fake_popen(*unused, **kwargs):
                kwargs["stdout"].write(b"partial output")
                kwargs["stderr"].write(b"partial diagnostic")
                return process

            def stop_owned(value):
                self.assertIs(value, process)
                value.returncode = -15

            with mock.patch.object(live.subprocess, "Popen", side_effect=fake_popen), \
                    mock.patch.object(runner, "stop", side_effect=stop_owned) as stop:
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    runner.run_command("timeout", [str(runner.binary)], "partial.bin", False)
                stop.assert_called_once_with(process)
            report = json.loads((root / "report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertTrue(report["commands"][0]["timed_out"])
            self.assertEqual((root / "partial.bin").read_bytes(), b"partial output")
            self.assertEqual((root / "timeout-stderr.txt").read_bytes(), b"partial diagnostic")


if __name__ == "__main__":
    unittest.main()
