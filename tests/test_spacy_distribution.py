"""Adversarial model-free controls for the fixed spaCy package boundary.

Synthetic fixture bytes replace module constants only within each test. They
are not native inference or new linguistic evidence. The actual admitted input
package and consumer checks are recorded separately by the W7 protocol.
"""

from __future__ import annotations

import contextlib
import ctypes
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/spacy/distribution.py"
SPEC = importlib.util.spec_from_file_location("spacy_distribution", MODULE_PATH)
distribution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(distribution)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pins(files):
    return {name: {"size_bytes": len(data), "sha256": digest(data)}
            for name, data in files.items()}


def writable_tree(root):
    for current, _, _ in os.walk(root):
        os.chmod(current, 0o700)


class DistributionControls(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix=".spacy-distribution-", dir=Path.cwd()))
        self.addCleanup(lambda: (writable_tree(self.root), shutil.rmtree(self.root)))
        self.native_files = {name: ("native fixture: " + name).encode()
                             for name in distribution.NATIVE_FILES}
        self.bundle_files = {name: ("bundle fixture: " + name).encode()
                             for name in distribution.BUNDLE_FILES}
        self.original = distribution._pack(self.native_files)
        self.native = self.root / "native.tar.gz"
        self.native.write_bytes(self.original)
        self.bundle = self.root / "model"
        self.bundle.mkdir()
        for name, data in self.bundle_files.items():
            (self.bundle / name).write_bytes(data)
        overrides = {
            "NATIVE_FILES": pins(self.native_files),
            "BUNDLE_FILES": pins(self.bundle_files),
            "LIBRARY_SHA256": digest(self.native_files["lib/libcrispembed-spacy.so"]),
            "NATIVE_ARCHIVE_SHA256": digest(self.original),
            "NATIVE_ARCHIVE_BYTES": len(self.original),
            "BUNDLE_MANIFEST_SHA256": digest(self.bundle_files["manifest.json"]),
        }
        for name, value in overrides.items():
            patch = mock.patch.object(distribution, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.package = self.root / "package.tar.gz"
        self.receipt = distribution.build(self.native, self.bundle, self.package)
        self.sha = self.receipt["archive_sha256"]
        self.payloads = {
            **{distribution._native_path(name): data for name, data in self.native_files.items()},
            **{"bundle/" + name: data for name, data in self.bundle_files.items()},
            distribution.MANIFEST_NAME: distribution._manifest_bytes(),
        }
        self.cache = self.root / "cache"
        self.counter = 0

    def archive(self, data):
        self.counter += 1
        path = self.root / f"mutant-{self.counter}.tar.gz"
        path.write_bytes(data)
        return path, digest(data)

    def reject(self, data):
        path, sha = self.archive(data)
        with self.assertRaises((distribution.DistributionError, OSError)):
            distribution.verify_archive(path, sha)

    def tar_with(self, *, extra=None, mutate=None, format=tarfile.USTAR_FORMAT):
        target = io.BytesIO()
        with tarfile.open(fileobj=target, mode="w", format=format) as archive:
            for name, payload in sorted(self.payloads.items()):
                entry = tarfile.TarInfo(name)
                entry.mode = 0o644
                entry.size = len(payload)
                if mutate:
                    mutate(entry)
                archive.addfile(entry, io.BytesIO(payload))
            if extra is not None:
                entry, payload = extra
                archive.addfile(entry, io.BytesIO(payload) if payload is not None else None)
        return gzip.compress(target.getvalue(), mtime=0)

    def test_deterministic_build_ignores_source_paths_and_mtimes(self):
        for path in self.bundle.iterdir():
            os.utime(path, (1000000, 1000000))
        second = self.root / "second.tar.gz"
        receipt = distribution.build(self.native, self.bundle, second)
        self.assertEqual(self.package.read_bytes(), second.read_bytes())
        self.assertEqual(receipt, self.receipt)
        self.assertEqual(receipt["payload_file_count"], 20)

    def test_output_never_replaces_existing_regular_file_or_link(self):
        before = self.package.read_bytes()
        with self.assertRaises(FileExistsError):
            distribution.build(self.native, self.bundle, self.package)
        self.assertEqual(self.package.read_bytes(), before)
        link = self.root / "link.tar.gz"
        link.symlink_to(self.package)
        with self.assertRaises(FileExistsError):
            distribution.build(self.native, self.bundle, link)
        self.assertEqual(self.package.read_bytes(), before)

    def test_verify_loads_no_native_library(self):
        with mock.patch.object(ctypes, "CDLL", side_effect=AssertionError("native load")):
            receipt = distribution.verify_archive(self.package, self.sha)
        self.assertFalse(receipt["native_library_loaded"])
        self.assertFalse(receipt["system_dependencies_verified"])
        self.assertEqual(receipt["manifest"]["scope"]["default_enabled"], False)
        self.assertEqual(len(receipt["manifest"]["system_dependencies"]["elf_needed"]), 6)

    def test_external_hash_is_mandatory_and_checked_before_decompression(self):
        for value in (None, "", "A" * 64, "0" * 63, "0" * 64):
            with self.subTest(value=value):
                with mock.patch.object(distribution.zlib, "decompressobj",
                                       side_effect=AssertionError("parsed unauthenticated archive")):
                    with self.assertRaises(distribution.DistributionError):
                        distribution.verify_archive(self.package, value)

    def test_native_archive_pin_checked_before_parsing(self):
        corrupt = bytearray(self.original)
        corrupt[-1] ^= 1
        self.native.write_bytes(corrupt)
        with mock.patch.object(distribution, "_read_tar", side_effect=AssertionError("parsed bad source")):
            with self.assertRaises(distribution.DistributionError):
                distribution.build(self.native, self.bundle, self.root / "bad.tar.gz")

    def test_model_inputs_cannot_repin_themselves(self):
        path = self.bundle / "model.gguf"
        value = path.read_bytes()
        path.write_bytes(bytes([value[0] ^ 1]) + value[1:])
        (self.bundle / "manifest.json").write_text(json.dumps({"files": pins({"model.gguf": path.read_bytes()})}))
        with self.assertRaises(distribution.DistributionError):
            distribution.build(self.native, self.bundle, self.root / "bad.tar.gz")

    def test_missing_extra_and_linked_bundle_files_rejected(self):
        for change in ("missing", "extra", "link"):
            with self.subTest(change=change):
                folder = self.root / change
                shutil.copytree(self.bundle, folder)
                if change == "missing":
                    (folder / "LICENSE").unlink()
                elif change == "extra":
                    (folder / "unexpected").write_bytes(b"x")
                else:
                    (folder / "LICENSE").unlink()
                    (folder / "LICENSE").symlink_to(self.bundle / "LICENSE")
                with self.assertRaises((distribution.DistributionError, OSError)):
                    distribution.build(self.native, folder, self.root / f"{change}.tar.gz")

    def test_archive_path_escape_and_alias_spellings_rejected(self):
        for name in ("../escape", "/escape", "C:/escape", "native\\escape",
                     "native//escape", "./escape", "native/../escape", "native/./escape"):
            with self.subTest(name=name):
                entry = tarfile.TarInfo(name)
                entry.mode, entry.size = 0o644, 1
                self.reject(self.tar_with(extra=(entry, b"x")))
        self.assertFalse((self.root / "escape").exists())

    def test_links_directories_devices_and_fifos_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE,
                     tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE):
            with self.subTest(kind=kind):
                def mutate(entry):
                    if entry.name == "bundle/LICENSE":
                        entry.type = kind
                        if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                            entry.linkname = "../../escape"
                self.reject(self.tar_with(mutate=mutate))

    def test_duplicate_unlisted_and_missing_members_rejected(self):
        payload = self.payloads["bundle/LICENSE"]
        entry = tarfile.TarInfo("bundle/LICENSE")
        entry.mode, entry.size = 0o644, len(payload)
        self.reject(self.tar_with(extra=(entry, payload)))
        self.reject(distribution._pack({**self.payloads, "unlisted": b"x"}))
        self.reject(distribution._pack({name: data for name, data in self.payloads.items()
                                        if name != "bundle/LICENSE"}))

    def test_manifest_cannot_change_pins_target_or_activation(self):
        for key, value in (("schema", "other"), ("files", {}), ("target", {}),
                           ("scope", {"default_enabled": True})):
            with self.subTest(key=key):
                manifest = distribution.manifest()
                manifest[key] = value
                self.reject(distribution._pack({**self.payloads,
                    distribution.MANIFEST_NAME: json.dumps(manifest).encode()}))

    def test_changed_payload_rejected_even_with_valid_transport_pin(self):
        payload = bytearray(self.payloads["bundle/model.gguf"])
        payload[0] ^= 1
        self.reject(distribution._pack({**self.payloads, "bundle/model.gguf": bytes(payload)}))

    def test_member_size_and_archive_expansion_are_bounded(self):
        entry = tarfile.TarInfo("bundle/LICENSE")
        entry.mode = 0o644
        entry.size = distribution.MAX_EXPANDED_BYTES + 1
        self.reject(gzip.compress(entry.tobuf(format=tarfile.USTAR_FORMAT) + b"\0" * 1024, mtime=0))
        with mock.patch.object(distribution, "MAX_EXPANDED_BYTES", 1024):
            with self.assertRaises(distribution.DistributionError):
                distribution.verify_archive(self.package, self.sha)
        with mock.patch.object(distribution, "MAX_ARCHIVE_BYTES", 10):
            with self.assertRaises(distribution.DistributionError):
                distribution.verify_archive(self.package, self.sha)

    def test_truncated_concatenated_and_trailing_gzip_rejected(self):
        value = self.package.read_bytes()
        for bad in (value[:-1], value + b"junk", value + gzip.compress(b"", mtime=0), b"not gzip"):
            with self.subTest(size=len(bad)):
                self.reject(bad)

    def test_tar_hidden_metadata_and_trailing_records_rejected(self):
        self.reject(self.tar_with(mutate=lambda entry: setattr(entry, "mtime", 1)))
        self.reject(self.tar_with(mutate=lambda entry: setattr(entry, "pax_headers", {"comment": "x"}),
                                  format=tarfile.PAX_FORMAT))
        raw = gzip.decompress(self.package.read_bytes())
        self.reject(gzip.compress(raw + b"\0" * 512, mtime=0))
        self.reject(gzip.compress(raw + distribution._tar_bytes({"hidden": b"x"}), mtime=0))

    def test_install_is_atomic_readonly_idempotent_and_matches_loader_layout(self):
        result = distribution.install(self.package, self.sha, self.cache)
        root = Path(result["root"])
        library = Path(result["entrypoints"]["library"])
        self.assertEqual(root.name, self.sha)
        self.assertEqual(library.parent.name, distribution.LIBRARY_SHA256)
        self.assertEqual(stat.S_IMODE(library.stat().st_mode), 0o444)
        self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o555)
        self.assertTrue(result["installed"])
        self.assertTrue(result["transport_verified"])
        before = {str(path.relative_to(root)): path.stat().st_ino for path in root.rglob("*")}
        second = distribution.install(self.package, self.sha, self.cache)
        self.assertFalse(second["installed"])
        self.assertEqual(before, {str(path.relative_to(root)): path.stat().st_ino for path in root.rglob("*")})
        self.assertEqual(list(self.cache.iterdir()), [root])
        with mock.patch.object(ctypes, "CDLL", side_effect=AssertionError("native load")):
            self.assertEqual(distribution.verify_tree(root, self.sha)["status"], "verified")

    def test_corrupt_existing_install_is_never_repaired_or_replaced(self):
        result = distribution.install(self.package, self.sha, self.cache)
        path = Path(result["entrypoints"]["bundle"]) / "model.gguf"
        path.chmod(0o600)
        wrong = b"x" * path.stat().st_size
        path.write_bytes(wrong)
        path.chmod(0o444)
        for operation in (lambda: distribution.verify_tree(result["root"], self.sha),
                          lambda: distribution.install(self.package, self.sha, self.cache)):
            with self.assertRaises(distribution.DistributionError):
                operation()
        self.assertEqual(path.read_bytes(), wrong)
        self.assertFalse(any(self.cache.glob(".spacy-stage-*")))

    def test_existing_empty_destination_or_file_is_not_replaced(self):
        self.cache.mkdir(mode=0o700)
        destination = self.cache / self.sha
        for kind in ("directory", "file"):
            with self.subTest(kind=kind):
                if kind == "directory":
                    destination.mkdir()
                else:
                    destination.write_bytes(b"existing")
                inode = destination.stat().st_ino
                with self.assertRaises((distribution.DistributionError, OSError)):
                    distribution.install(self.package, self.sha, self.cache)
                self.assertEqual(destination.stat().st_ino, inode)
                destination.rmdir() if kind == "directory" else destination.unlink()

    def test_racing_empty_destination_is_not_overwritten(self):
        original = distribution._publish_directory
        inodes = []
        def race(stage, destination):
            destination.mkdir()
            inodes.append(destination.stat().st_ino)
            return original(stage, destination)
        with mock.patch.object(distribution, "_publish_directory", side_effect=race):
            with self.assertRaises(distribution.DistributionError):
                distribution.install(self.package, self.sha, self.cache)
        destination = self.cache / self.sha
        self.assertEqual(destination.stat().st_ino, inodes[0])
        self.assertEqual(list(destination.iterdir()), [])
        self.assertFalse(any(self.cache.glob(".spacy-stage-*")))

    def test_interrupted_staging_leaves_no_visible_partial_tree(self):
        original = distribution._write_new_file
        writes = []
        def fail(path, data, mode):
            writes.append(path)
            if len(writes) == 2:
                raise OSError("simulated disk failure")
            original(path, data, mode)
        with mock.patch.object(distribution, "_write_new_file", side_effect=fail):
            with self.assertRaises(OSError):
                distribution.install(self.package, self.sha, self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_cache_ancestor_links_and_shared_write_permissions_rejected(self):
        real = self.root / "real"
        real.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(real, target_is_directory=True)
        with self.assertRaises(distribution.DistributionError):
            distribution.install(self.package, self.sha, alias / "cache")
        self.cache.mkdir(mode=0o700)
        self.cache.chmod(0o777)
        with self.assertRaises(distribution.DistributionError):
            distribution.install(self.package, self.sha, self.cache)

    def test_installed_extra_files_directories_and_links_rejected(self):
        result = distribution.install(self.package, self.sha, self.cache)
        root = Path(result["root"])
        root.chmod(0o755)
        extra = root / "extra"
        for kind in ("file", "directory", "symlink"):
            with self.subTest(kind=kind):
                if kind == "file":
                    extra.write_bytes(b"x")
                    extra.chmod(0o444)
                elif kind == "directory":
                    extra.mkdir(mode=0o555)
                else:
                    extra.symlink_to(self.bundle, target_is_directory=True)
                root.chmod(0o555)
                with self.assertRaises(distribution.DistributionError):
                    distribution.verify_tree(root, self.sha)
                root.chmod(0o755)
                extra.rmdir() if kind == "directory" else extra.unlink()

    def test_installed_hardlinked_or_missing_payload_rejected(self):
        result = distribution.install(self.package, self.sha, self.cache)
        root = Path(result["root"])
        member = root / "bundle/LICENSE"
        witness = self.root / "witness"
        os.link(member, witness)
        with self.assertRaises(distribution.DistributionError):
            distribution.verify_tree(root, self.sha)
        witness.unlink()
        member.parent.chmod(0o755)
        member.unlink()
        member.parent.chmod(0o555)
        with self.assertRaises(distribution.DistributionError):
            distribution.verify_tree(root, self.sha)

    def test_wrong_tree_identity_and_unsupported_installer_host_rejected(self):
        result = distribution.install(self.package, self.sha, self.cache)
        with self.assertRaises(distribution.DistributionError):
            distribution.verify_tree(result["root"], "0" * 64)
        with mock.patch.object(distribution.sys, "platform", "darwin"):
            with self.assertRaises(distribution.DistributionError):
                distribution.install(self.package, self.sha, self.root / "other")

    def test_cli_requires_external_hash_and_reports_rejection(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                distribution.main(["verify", "--archive", str(self.package)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(distribution.main(["verify", "--archive", str(self.package),
                                                 "--sha256", "0" * 64]), 1)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(distribution.main(["inspect", "--archive", str(self.package),
                                                 "--sha256", self.sha]), 0)
        self.assertEqual(json.loads(output.getvalue())["archive_sha256"], self.sha)


if __name__ == "__main__":
    unittest.main(verbosity=2)
