#!/usr/bin/env python3
"""Closed spaCy capsules for additional native platforms (schema v2).

The frozen Linux W7 capsule (`distribution.py`, schema v1) is unchanged. This
module packages a freshly built native install tree (CMake `--install` prefix)
with the same five pinned model-bundle files into the same closed format:
canonical USTAR, gzip level 9, mode 0644, uid/gid/mtime 0, lexical order, and a
`distribution.json` that pins every member. The library is placed under
`native/<library sha256>/` so a loader path can never name different bytes.

Trust: a v2 capsule's member pins come from its embedded manifest, which is
authenticated only by an externally trusted archive SHA-256 (for example the
digest compiled into the Quds catalog or recorded in a publication receipt).
The bundle pins are fixed in `distribution.BUNDLE_FILES` and cannot be changed
by an archive. A new build gets new pins; it never inherits a qualification.

Stdlib only. Never loads the native library.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tarfile
import tempfile
import zlib

_SPEC = importlib.util.spec_from_file_location(
    "spacy_distribution_v1", Path(__file__).resolve().parent / "distribution.py")
v1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v1)

DistributionError = v1.DistributionError
SCHEMA = "crispembed.spacy.distribution.v2"
MANIFEST_NAME = v1.MANIFEST_NAME
MODEL = "de_core_news_sm-3.8.0"

# Declared targets. `library` is the CMake install name under `library_dir`
# (lib/ for shared objects; bin/ for a Windows DLL, whose import library is
# not shipped because consumers load the DLL at run time).
PLATFORMS = {
    "linux-x86_64": {"system": "Linux", "machine": "x86_64", "library": "libcrispembed-spacy.so",
                     "library_dir": "lib"},
    "linux-arm64": {"system": "Linux", "machine": "aarch64", "library": "libcrispembed-spacy.so",
                    "library_dir": "lib"},
    "macos-arm64": {"system": "Darwin", "machine": "arm64", "library": "libcrispembed-spacy.dylib",
                    "library_dir": "lib"},
    "windows-x86_64": {"system": "Windows", "machine": "AMD64", "library": "crispembed-spacy.dll",
                       "library_dir": "bin"},
}
WINDOWS = os.name == "nt"

# Every notice shipped by the accepted Linux capsule must be present.
REQUIRED_NOTICES = sorted(
    name for name in v1.NATIVE_FILES if name.startswith("share/licenses/"))


def archive_name(platform: str) -> str:
    _platform(platform)
    return f"spacy-{MODEL}-{platform}-f32.tar.gz"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pin(data: bytes) -> dict:
    return {"sha256": _sha(data), "size_bytes": len(data)}


def _platform(name: str) -> dict:
    if name not in PLATFORMS:
        raise DistributionError(f"Undeclared platform {name!r}.")
    return PLATFORMS[name]


def _library_path(target: dict) -> str:
    return f"{target['library_dir']}/{target['library']}"


# --- File primitives -------------------------------------------------------
# POSIX hosts use the frozen v1 primitives unchanged. Windows has no O_NOFOLLOW,
# fchmod or hard-link publication, and os.open defaults to text mode, so it
# gets binary-safe equivalents: reparse points are refused, new files are
# created exclusively, and read-only is the only permission bit Windows keeps.

def _reject_reparse(path: Path) -> os.stat_result:
    info = path.lstat()
    if getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        raise DistributionError("An input is a link or reparse point.")
    return info


def _read(path, *, maximum: int, pin: dict | None = None) -> bytes:
    if not WINDOWS:
        return v1._read_regular(path, maximum=maximum, pin=pin)
    path = Path(os.path.abspath(path))
    for parent in reversed(path.parents):
        if parent.exists():
            _reject_reparse(parent)
    info = _reject_reparse(path)
    if not stat.S_ISREG(info.st_mode):
        raise DistributionError("An input is not a regular file.")
    if info.st_size < 1 or info.st_size > maximum:
        raise DistributionError("An input exceeds its byte bound or is empty.")
    if pin is not None and info.st_size != pin["size_bytes"]:
        raise DistributionError("A pinned input has the wrong size.")
    with open(path, "rb") as source:
        data = source.read(maximum + 1)
    if len(data) != info.st_size or path.lstat().st_mtime_ns != info.st_mtime_ns:
        raise DistributionError("An input changed while being read.")
    if pin is not None and _sha(data) != pin["sha256"]:
        raise DistributionError("A pinned input failed SHA-256 verification.")
    return data


def _write_new(path: Path, data: bytes, mode: int) -> None:
    if not WINDOWS:
        return v1._write_new_file(path, data, mode)
    with open(path, "xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(path, mode)


def _publish(path: Path, data: bytes) -> None:
    if not WINDOWS:
        return v1._publish_file(path, data)
    _write_new(path, data, 0o644)  # exclusive create: never replaces an output


def _remove(directory: Path) -> None:
    if not WINDOWS:
        return v1._remove_staging(directory)
    for current, dirs, names in os.walk(directory):
        for name in dirs + names:
            os.chmod(Path(current) / name, stat.S_IWRITE | stat.S_IREAD)
    os.chmod(directory, stat.S_IWRITE | stat.S_IREAD)
    shutil.rmtree(directory)


def collect_native(install_prefix: os.PathLike | str, platform: str) -> dict[str, bytes]:
    """Read a CMake install tree: exactly the library, headers and notices."""
    prefix = Path(os.path.abspath(install_prefix))
    target = _platform(platform)
    library = _library_path(target)
    files: dict[str, bytes] = {}
    for path in sorted(prefix.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise DistributionError("Native install tree contains a link.")
        if stat.S_ISDIR(info.st_mode):
            continue
        relative = path.relative_to(prefix).as_posix()
        if target["system"] == "Windows" and relative == "lib/" + Path(target["library"]).stem + ".lib":
            continue  # import library: not needed by run-time loaders
        if not (relative == library or relative.startswith("include/")
                or relative.startswith("share/licenses/")):
            raise DistributionError(f"Unexpected native install member {relative}.")
        files[relative] = _read(path, maximum=64 * 1024 * 1024)
    if library not in files:
        raise DistributionError("Native install tree lacks the library.")
    missing = [name for name in REQUIRED_NOTICES if name not in files]
    if missing:
        raise DistributionError(f"Native install tree lacks notices: {missing}")
    return files


def manifest(platform: str, native: dict[str, bytes], *, source_commit: str,
             build: dict, system_dependencies: dict) -> dict:
    target = _platform(platform)
    library = native[_library_path(target)]
    library_sha = _sha(library)

    def place(name: str) -> str:
        if name == _library_path(target):
            return f"native/{library_sha}/{target['library']}"
        return "native/" + name

    files = {place(name): _pin(data) for name, data in native.items()}
    files.update({"bundle/" + name: dict(pin) for name, pin in v1.BUNDLE_FILES.items()})
    return {
        "schema": SCHEMA,
        "profile": f"{MODEL}-f32-{platform}",
        "target": {"platform": platform, "system": target["system"],
                   "machine": target["machine"], "precision": "F32", "locale": "de-DE"},
        "native": {"source_commit": source_commit, "library_sha256": library_sha,
                   "library_size_bytes": len(library), "abi_version": 1,
                   "text_abi_version": 1, "build": build},
        "bundle_manifest_sha256": v1.BUNDLE_MANIFEST_SHA256,
        "entrypoints": {"library": place(_library_path(target)), "bundle": "bundle"},
        "files": dict(sorted(files.items())),
        "archive_format": {"tar": "USTAR", "compression": "gzip-level-9", "timestamp": 0,
                           "uid": 0, "gid": 0, "file_mode": "0644", "order": "lexical path"},
        "system_dependencies": system_dependencies,
        "scope": {
            "default_enabled": False,
            "automatic_download": False,
            "requires_explicit_environment_gate": "CRISPEMBED_SPACY=1",
            "qualification": ("Per exact library digest only, as recorded by its "
                              "publication receipt. Grammar qualification is never "
                              "inherited from another build."),
        },
    }


def _manifest_bytes(data: dict) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=2) + "\n").encode("utf-8")


def build(platform: str, install_prefix, bundle, output, *, source_commit: str,
          build_info: dict, system_dependencies: dict) -> dict:
    """Create one deterministic capsule; an existing output is never replaced."""
    native = collect_native(install_prefix, platform)
    bundle = Path(os.path.abspath(bundle))
    if {item.name for item in bundle.iterdir()} != v1.BUNDLE_FILES.keys():
        raise DistributionError("The bundle must contain exactly five pinned files.")
    data = manifest(platform, native, source_commit=source_commit, build=build_info,
                    system_dependencies=system_dependencies)
    files = {path: None for path in data["files"]}
    target = _platform(platform)
    for name, payload in native.items():
        key = (data["entrypoints"]["library"] if name == _library_path(target)
               else "native/" + name)
        files[key] = payload
    for name, pin in v1.BUNDLE_FILES.items():
        files["bundle/" + name] = _read(bundle / name, maximum=pin["size_bytes"], pin=pin)
    files[MANIFEST_NAME] = _manifest_bytes(data)
    packed = v1._pack(files)
    _read_capsule(packed)  # exercise the exact reader before publication
    _publish(Path(os.path.abspath(output)), packed)
    return {"archive_sha256": _sha(packed), "archive_size_bytes": len(packed),
            "manifest_sha256": _sha(files[MANIFEST_NAME]),
            "manifest_size_bytes": len(files[MANIFEST_NAME]),
            "library_sha256": data["native"]["library_sha256"],
            "library_size_bytes": data["native"]["library_size_bytes"],
            "entrypoints": data["entrypoints"], "payload_file_count": len(files) - 1,
            "profile": data["profile"]}


def _read_capsule(data: bytes) -> tuple[dict, dict[str, bytes]]:
    """Parse a closed v2 capsule whose transport was already authenticated."""
    if len(data) > v1.MAX_ARCHIVE_BYTES:
        raise DistributionError("Archive exceeds the compressed byte bound.")
    try:
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        raw = decompressor.decompress(data, v1.MAX_EXPANDED_BYTES + 1)
        if len(raw) > v1.MAX_EXPANDED_BYTES or not decompressor.eof or decompressor.unused_data:
            raise DistributionError("Invalid, trailing or oversized gzip stream.")
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            entry = archive.getmember(MANIFEST_NAME)
            stream = archive.extractfile(entry)
            manifest_bytes = stream.read(256 * 1024 + 1) if stream else b""
    except (KeyError, tarfile.TarError, zlib.error, EOFError) as error:
        raise DistributionError("Malformed capsule or missing manifest.") from error
    try:
        data_manifest = json.loads(manifest_bytes)
    except ValueError as error:
        raise DistributionError("Unreadable capsule manifest.") from error
    _check_manifest(data_manifest)
    pins = {**data_manifest["files"], MANIFEST_NAME: _pin(manifest_bytes)}
    files = v1._read_tar(data, pins, canonical=True)
    return data_manifest, files


def _check_manifest(data: dict) -> None:
    try:
        target = _platform(data["target"]["platform"])
        library_sha = data["native"]["library_sha256"]
        entry = data["entrypoints"]["library"]
        files = data["files"]
    except (KeyError, TypeError) as error:
        raise DistributionError("Capsule manifest is incomplete.") from error
    if (data.get("schema") != SCHEMA
            or data.get("bundle_manifest_sha256") != v1.BUNDLE_MANIFEST_SHA256
            or entry != f"native/{library_sha}/{target['library']}"
            or data["entrypoints"].get("bundle") != "bundle"
            or files.get(entry, {}).get("sha256") != library_sha
            or MANIFEST_NAME in files):
        raise DistributionError("Capsule manifest has the wrong profile.")
    bundle = {name[len("bundle/"):]: pin for name, pin in files.items() if name.startswith("bundle/")}
    if bundle != v1.BUNDLE_FILES:
        raise DistributionError("Capsule bundle differs from the fixed model pins.")
    for name in files:
        if not v1._safe_member(name):
            raise DistributionError("Unsafe capsule member path.")
    for notice in REQUIRED_NOTICES:
        if "native/" + notice not in files:
            raise DistributionError("Capsule lacks a required notice.")


def verify_archive(path, trusted_sha256: str) -> dict:
    expected = v1._trusted_hash(trusted_sha256)
    data = _read(path, maximum=v1.MAX_ARCHIVE_BYTES)
    if _sha(data) != expected:  # authenticate before any parsing
        raise DistributionError("Archive differs from the externally trusted SHA-256.")
    data_manifest, files = _read_capsule(data)
    return {"status": "verified", "archive_sha256": expected, "archive_size_bytes": len(data),
            "manifest_sha256": _sha(files[MANIFEST_NAME]), "payload_file_count": len(files) - 1,
            "entrypoints": data_manifest["entrypoints"], "profile": data_manifest["profile"],
            "native_library_loaded": False}


def _publish_directory(staging: Path, destination: Path) -> bool:
    if sys.platform.startswith("linux"):
        return v1._publish_directory(staging, destination)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(os.fsencode(staging), os.fsencode(destination), 0x00000004) == 0:  # RENAME_EXCL
            return True
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            return False
        raise OSError(code, os.strerror(code))
    if WINDOWS:
        # MoveFileExW without MOVEFILE_REPLACE_EXISTING: an existing
        # destination fails the rename instead of being replaced.
        try:
            os.rename(staging, destination)
            return True
        except FileExistsError:
            return False
    raise DistributionError("No atomic no-replace directory rename on this host.")


def verify_tree(directory, trusted_sha256: str, *, named: bool = True) -> dict:
    directory = Path(os.path.abspath(directory))
    v1._trusted_hash(trusted_sha256)
    if named and directory.name != trusted_sha256:
        raise DistributionError("Installation directory must be the trusted archive SHA-256.")
    root_info = directory.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or (named and stat.S_IMODE(root_info.st_mode) != 0o555):
        raise DistributionError("Installation root must be a regular read-only directory.")
    manifest_bytes = _read(directory / MANIFEST_NAME, maximum=256 * 1024)
    data = json.loads(manifest_bytes)
    _check_manifest(data)
    pins = {**data["files"], MANIFEST_NAME: _pin(manifest_bytes)}
    # Member names are POSIX paths; compare them as such on every host.
    allowed_dirs = {p.as_posix() for name in pins for p in PurePosixPath(name).parents if str(p) != "."}
    found_files, found_dirs = set(), set()
    for current, dirs, names in os.walk(directory, followlinks=False):
        parent = Path(current)
        for name in dirs:
            relative = (parent / name).relative_to(directory).as_posix()
            info = (parent / name).lstat()
            if relative not in allowed_dirs or not stat.S_ISDIR(info.st_mode) \
                    or stat.S_IMODE(info.st_mode) != 0o555:
                raise DistributionError("Unexpected, linked or writable installation directory.")
            found_dirs.add(relative)
        for name in names:
            relative = (parent / name).relative_to(directory).as_posix()
            if relative not in pins or stat.S_IMODE((parent / name).lstat().st_mode) != 0o444:
                raise DistributionError("Unexpected or writable installation file.")
            _read(parent / name, maximum=pins[relative]["size_bytes"], pin=pins[relative])
            found_files.add(relative)
    if found_files != pins.keys() or found_dirs != allowed_dirs:
        raise DistributionError("Installation is missing required files or directories.")
    return {"status": "verified", "archive_sha256": trusted_sha256,
            "payload_file_count": len(pins) - 1, "entrypoints": data["entrypoints"],
            "native_library_loaded": False}


def install(archive, trusted_sha256: str, cache) -> dict:
    """Authenticate, stage, verify, then publish atomically without replacement."""
    expected = v1._trusted_hash(trusted_sha256)
    data = _read(archive, maximum=v1.MAX_ARCHIVE_BYTES)
    if _sha(data) != expected:
        raise DistributionError("Archive differs from the externally trusted SHA-256.")
    _, files = _read_capsule(data)
    cache = Path(os.path.abspath(cache))
    cache.mkdir(mode=0o700, exist_ok=True)
    info = cache.lstat()
    # Windows has no POSIX group/other bits; its per-user profile ACLs apply.
    if not stat.S_ISDIR(info.st_mode) or (not WINDOWS and stat.S_IMODE(info.st_mode) & 0o022):
        raise DistributionError("Installation requires a private cache directory.")
    destination = cache / expected
    if os.path.lexists(destination):
        return {**verify_tree(destination, expected), "installed": False}
    # Stage beside the destination: the no-replace rename needs one parent.
    staging = Path(tempfile.mkdtemp(prefix=".spacy-stage-", dir=cache))
    try:
        for name, payload in sorted(files.items()):
            path = staging / name
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _write_new(path, payload, 0o444)
        for current, _, _ in os.walk(staging, topdown=False):
            os.chmod(current, 0o555)
        verify_tree(staging, expected, named=False)
        # BSD/macOS renames of a directory need write access to it; reopen
        # only the private staging root for the rename, then seal it again.
        os.chmod(staging, 0o700)
        published = _publish_directory(staging, destination)
        if published:
            staging = None
            os.chmod(destination, 0o555)
        return {**verify_tree(destination, expected), "installed": published}
    finally:
        if staging is not None and staging.exists():
            _remove(staging)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "install"):
        command = commands.add_parser(name)
        command.add_argument("--archive", required=True, type=Path)
        command.add_argument("--sha256", required=True)
        if name == "install":
            command.add_argument("--cache", required=True, type=Path)
    check = commands.add_parser("verify-install")
    check.add_argument("--directory", required=True, type=Path)
    check.add_argument("--sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "install":
            result = install(args.archive, args.sha256, args.cache)
        elif args.command == "verify-install":
            result = verify_tree(args.directory, args.sha256)
        else:
            result = verify_archive(args.archive, args.sha256)
    except (DistributionError, OSError, ValueError) as error:
        print(json.dumps({"status": "rejected", "reason": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
