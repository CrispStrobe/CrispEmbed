#!/usr/bin/env python3
"""Provider admission controls: no frameworks, builds, network or model loads."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mineru_cpu", ROOT / "tools/kaggle/mineru2605-cpu/kernel.py")
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


class Provider(unittest.TestCase):
    def test_actual_four_file_retention_and_shared_space_are_counted(self):
        plan = dict(scratch_bytes_required=10, output_bytes_required=8, available_memory_bytes_required=12)
        f32_plan = dict(scratch_bytes_required=7)
        # Four stored candidates need 16 scratch units, not the old 10.
        with self.assertRaises(ValueError):
            worker.storage_guard(plan, f32_plan, scratch_free=15, output_free=20, shared=False, memory=12)
        with self.assertRaises(ValueError):
            worker.storage_guard(plan, f32_plan, scratch_free=23, output_free=23, shared=True, memory=12)
        self.assertEqual(worker.storage_guard(plan, f32_plan, scratch_free=24, output_free=24,
                                             shared=True, memory=12)["scratch_required_bytes"], 16)
        worker.storage_guard(plan, f32_plan, scratch_free=16, output_free=8, shared=False, memory=12)

    def test_ram_and_output_shortfall_fail_before_any_model(self):
        plan = dict(scratch_bytes_required=10, output_bytes_required=8, available_memory_bytes_required=12)
        for memory, output in ((11, 8), (12, 7)):
            with self.assertRaises(ValueError):
                worker.storage_guard(plan, {"scratch_bytes_required": 7}, scratch_free=100,
                                     output_free=output, shared=False, memory=memory)

    def valid(self):
        return dict(working=True, inputs=True, machine="x86_64", system="linux", cpus=4, nvidia=False)

    def test_all_frozen_cases_on_actual_cpu_provider(self):
        for case in worker.CASES:
            worker.validate_provider(case, {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, **self.valid())

    def test_local_or_missing_mounts_rejected(self):
        for key in ("working", "inputs"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_provider("formula", {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, **dict(self.valid(), **{key: False}))
        with self.assertRaises(ValueError):
            worker.validate_provider("formula", {}, **self.valid())

    def test_github_environment_is_never_faked(self):
        for key in ("GITHUB_ACTIONS", "GITHUB_SHA"):
            for value in ("true", "false", ""):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    worker.validate_provider("formula", {"KAGGLE_KERNEL_RUN_TYPE": "Batch", key: value}, **self.valid())

    def test_gpu_wrong_arch_threads_and_unknown_case_rejected(self):
        for key, value in (("nvidia", True), ("machine", "aarch64"), ("system", "darwin"), ("cpus", 2)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                worker.validate_provider("formula", {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, **dict(self.valid(), **{key: value}))
        with self.assertRaises(ValueError):
            worker.validate_provider("all", {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}, **self.valid())


if __name__ == "__main__":
    unittest.main()
