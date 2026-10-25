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
SRC_TREE = "c" * 40


def fixture(root=ROOT):
    names = closure.expected_native_sources(root)
    return {"source_commit": HEAD,
            "native_source_sha256": {name: closure.digest(Path(root) / name) for name in names}}


def fake_git(root, *, cmake=None, reviewed=GGML, current=GGML, staged=GGML,
             checkout=GGML, checkout_root=None, reviewed_src=SRC_TREE,
             current_src=SRC_TREE, src_dirty_command=None, ggml_dirty_command=None,
             commit_present=True, seen=None):
    if cmake is None:
        cmake = (Path(root) / "CMakeLists.txt").read_bytes()
    if checkout_root is None:
        checkout_root = str(Path(root) / "ggml")
    def call(args, *, root):
        if seen is not None:
            seen.append(tuple(args))
        if args == ["cat-file", "-e", HEAD + "^{commit}"]:
            if not commit_present and sum(item == tuple(args) for item in seen) == 1:
                raise subprocess.CalledProcessError(1, args)
            return b""
        if args == ["show", HEAD + ":CMakeLists.txt"]:
            return cmake
        if args == ["rev-parse", HEAD + ":src"]:
            return (reviewed_src + "\n").encode()
        if args == ["rev-parse", "HEAD:src"]:
            return (current_src + "\n").encode()
        src_clean = (
            ["diff", "--name-only", "HEAD", "--", "src"],
            ["diff", "--cached", "--name-only", "HEAD", "--", "src"],
            ["diff", "--name-only", "--", "src"],
            ["ls-files", "--others", "--", "src"],
        )
        if args in src_clean:
            return b"src/core_gguf.cpp\n" if args == src_dirty_command else b""
        if args == ["ls-tree", HEAD, "ggml"]:
            return f"160000 commit {reviewed}\tggml\n".encode()
        if args == ["ls-tree", "HEAD", "ggml"]:
            return f"160000 commit {current}\tggml\n".encode()
        if args == ["ls-files", "--stage", "ggml"]:
            return f"160000 {staged} 0\tggml\n".encode()
        if args == ["-C", str(Path(root) / "ggml"), "rev-parse", "--show-toplevel"]:
            return (checkout_root + "\n").encode()
        if args == ["-C", str(Path(root) / "ggml"), "rev-parse", "HEAD"]:
            return (checkout + "\n").encode()
        ggml_clean = (
            ["-C", str(Path(root) / "ggml"), "diff", "--name-only", "HEAD", "--"],
            ["-C", str(Path(root) / "ggml"), "diff", "--cached", "--name-only", "HEAD", "--"],
            ["-C", str(Path(root) / "ggml"), "diff", "--name-only", "--"],
            ["-C", str(Path(root) / "ggml"), "ls-files", "--others", "--exclude-standard"],
        )
        if args in ggml_clean:
            return b"src/changed.cpp\n" if args == ggml_dirty_command else b""
        raise AssertionError("Unexpected git command: " + repr(args))
    return call


class SourceClosureControls(unittest.TestCase):
    def test_full_worker_set_and_reviewed_cmake_gitlink(self):
        producer = fixture()
        expected = len(closure.expected_native_sources(ROOT))
        self.assertEqual(len(producer["native_source_sha256"]), expected)
        result = closure.require_native_source_closure(producer, HEAD, root=ROOT,
                    call=fake_git(ROOT), fetch=lambda *a, **k: self.fail("unexpected fetch"))
        self.assertEqual(result["file_count"], expected)
        self.assertEqual(result["ggml_gitlink"], GGML)
        self.assertTrue(result["gitlink_index_exact"])
        self.assertTrue(result["ggml_checkout_exact"])
        self.assertEqual(result["src_tree_oid"], SRC_TREE)
        self.assertTrue(result["src_checkout_clean"])
        self.assertTrue(result["ggml_checkout_clean"])

    def test_core_src_tree_and_dirty_checkout_gates(self):
        producer = fixture()
        # A reviewed qwen35 file set can coexist with a changed GGUF loader.
        with self.assertRaisesRegex(ValueError, "src Git tree"):
            closure.require_native_source_closure(producer, HEAD, root=ROOT,
                call=fake_git(ROOT, current_src="d" * 40))
        src_commands = (
            ["diff", "--name-only", "HEAD", "--", "src"],
            ["diff", "--cached", "--name-only", "HEAD", "--", "src"],
            ["diff", "--name-only", "--", "src"],
            ["ls-files", "--others", "--", "src"],
        )
        for args in src_commands:
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "src checkout"):
                closure.require_native_source_closure(producer, HEAD, root=ROOT,
                    call=fake_git(ROOT, src_dirty_command=args))

    def test_dirty_ggml_rejected_even_when_gitlink_and_head_match(self):
        producer = fixture()
        ggml = str(ROOT / "ggml")
        commands = (
            ["-C", ggml, "diff", "--name-only", "HEAD", "--"],
            ["-C", ggml, "diff", "--cached", "--name-only", "HEAD", "--"],
            ["-C", ggml, "diff", "--name-only", "--"],
            ["-C", ggml, "ls-files", "--others", "--exclude-standard"],
        )
        for args in commands:
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "ggml checkout"):
                closure.require_native_source_closure(producer, HEAD, root=ROOT,
                    call=fake_git(ROOT, ggml_dirty_command=args))

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
                 ({"staged": "c" * 40}, "ggml gitlink"),
                 ({"checkout": "c" * 40}, "ggml gitlink"),
                 ({"checkout_root": str(ROOT)}, "ggml gitlink"))
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
