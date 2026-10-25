"""Source-only OvisOCR3 native review closure, including CMake and ggml."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ovisocr3_source_closure_test", ROOT / "tools/ci-heavy/ovisocr3_source_closure.py")
closure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(closure)
HEAD = "a" * 40
GGML = "b" * 40


def fixture(root=ROOT):
    names = closure.expected_native_sources(root)
    return {"source_commit": HEAD,
            "native_source_sha256": {name: closure.digest(Path(root) / name) for name in names}}


def fake_git(root, *, cmake=None, reviewed=GGML, current=GGML, staged=GGML,
             commit_present=True, seen=None):
    if cmake is None:
        cmake = (Path(root) / "CMakeLists.txt").read_bytes()
    def call(args, *, root):
        if seen is not None:
            seen.append(tuple(args))
        if args == ["cat-file", "-e", HEAD + "^{commit}"]:
            if not commit_present and sum(item == tuple(args) for item in seen) == 1:
                raise subprocess.CalledProcessError(1, args)
            return b""
        if args == ["show", HEAD + ":CMakeLists.txt"]:
            return cmake
        if args == ["ls-tree", HEAD, "ggml"]:
            return f"160000 commit {reviewed}\tggml\n".encode()
        if args == ["ls-tree", "HEAD", "ggml"]:
            return f"160000 commit {current}\tggml\n".encode()
        if args == ["ls-files", "--stage", "ggml"]:
            return f"160000 {staged} 0\tggml\n".encode()
        raise AssertionError("Unexpected git command: " + repr(args))
    return call


class SourceClosureControls(unittest.TestCase):
    def test_full_worker_set_and_reviewed_cmake_gitlink(self):
        producer = fixture()
        self.assertEqual(len(producer["native_source_sha256"]), 29)
        result = closure.require_native_source_closure(producer, HEAD, root=ROOT,
                    call=fake_git(ROOT), fetch=lambda *a, **k: self.fail("unexpected fetch"))
        self.assertEqual(result["file_count"], 29)
        self.assertEqual(result["ggml_gitlink"], GGML)
        self.assertTrue(result["gitlink_index_exact"])

    def test_missing_or_changed_native_source_rejected_before_git(self):
        producer = fixture()
        for change in ("missing", "extra", "norm"):
            bad = {**producer, "native_source_sha256": dict(producer["native_source_sha256"])}
            if change == "missing":
                del bad["native_source_sha256"]["src/qwen35_norm.cpp"]
            elif change == "extra":
                bad["native_source_sha256"]["src/unreviewed.cpp"] = "0" * 64
            else:
                bad["native_source_sha256"]["src/qwen35_norm.cpp"] = "0" * 64
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "source/checker"):
                closure.require_native_source_closure(bad, HEAD, root=ROOT,
                    call=lambda *a, **k: self.fail("git before source check"))
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            for name in closure.expected_native_sources(ROOT):
                target = temp / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            shutil.copyfile(ROOT / "CMakeLists.txt", temp / "CMakeLists.txt")
            source = fixture(temp)
            (temp / "src/qwen35_norm.cpp").write_bytes(b"changed norm")
            with self.assertRaisesRegex(ValueError, "qwen35_norm.cpp"):
                closure.require_native_source_closure(source, HEAD, root=temp,
                    call=lambda *a, **k: self.fail("git before source check"))

    def test_cmake_and_every_gitlink_view_must_match(self):
        producer = fixture()
        cases = (({"cmake": b"changed CMake"}, "CMakeLists"),
                 ({"reviewed": "c" * 40}, "ggml gitlink"),
                 ({"current": "c" * 40}, "ggml gitlink"),
                 ({"staged": "c" * 40}, "ggml gitlink"))
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                closure.require_native_source_closure(producer, HEAD, root=ROOT,
                    call=fake_git(ROOT, **kwargs))
        with self.assertRaisesRegex(ValueError, "source head"):
            closure.require_native_source_closure(producer, "not-a-head", root=ROOT,
                call=lambda *a, **k: self.fail("git before head check"))

    def test_shallow_checkout_fetches_only_reviewed_commit(self):
        producer, seen, fetches = fixture(), [], []
        closure.require_native_source_closure(producer, HEAD, root=ROOT,
            call=fake_git(ROOT, commit_present=False, seen=seen),
            fetch=lambda head, *, root: fetches.append((head, root)))
        self.assertEqual(fetches, [(HEAD, ROOT)])
        self.assertEqual(seen.count(("cat-file", "-e", HEAD + "^{commit}")), 2)


if __name__ == "__main__":
    unittest.main()
