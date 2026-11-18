#!/usr/bin/env python3
"""Package and verify the unchanged Linux spaCy validation candidate.

This stdlib-only developer utility never downloads or loads the spaCy library.
Archive verification needs a trusted SHA-256 obtained outside the archive and
also enforces the original native/model pins compiled into this module. An
installed tree is payload-verified; only verify_archive authenticates transport
bytes. Installation is Linux-only, uses a private cache, and publishes with
renameat2(RENAME_NOREPLACE). It neither checks nor installs system dependencies.
"""

from __future__ import annotations

import argparse
import errno
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import zlib

SCHEMA = "crispembed.spacy.distribution.v1"
MANIFEST_NAME = "distribution.json"
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_EXPANDED_BYTES = 32 * 1024 * 1024
NATIVE_ARCHIVE_SHA256 = "b0cd3ae4d47af442d32a6d2382143c36f8b019470cff7c509f97e4682b67489b"
NATIVE_ARCHIVE_BYTES = 825745
LIBRARY_SHA256 = "ea89842ebf22c08508aff1cfbf18d78a107118af320523009616ff62b6e42977"
BUNDLE_MANIFEST_SHA256 = "d599768375de3dc56abcca5cf1d49da0015857f9d3a10640aebb5317ef5d427e"

# Original acceptance-candidate-linux-x86_64.json and five W6 bundle files.
# No command-line or archive-controlled override of these pins exists.
NATIVE_FILES = {'include/spacy_pipeline.h': {'sha256': '756c25b664b38d26d1da13264d47417d9914f8dd805c588a539fe6f3681f7e62',
                              'size_bytes': 10045},
 'lib/libcrispembed-spacy.so': {'sha256': 'ea89842ebf22c08508aff1cfbf18d78a107118af320523009616ff62b6e42977',
                                'size_bytes': 1959256},
 'share/licenses/crispembed-spacy/CrispEmbed-LICENSE': {'sha256': 'a70d0ab5fb2dbab83d1a63071776891e6dfc7ba6e3063b3940c21274c1225a3e',
                                                        'size_bytes': 4739},
 'share/licenses/crispembed-spacy/THIRD_PARTY_NOTICES': {'sha256': '9386f6a62a3220da9c082dfceea4a018d2a73c1a0dbd03e28e1804912cf0115f',
                                                         'size_bytes': 2188},
 'share/licenses/crispembed-spacy/ggml-LICENSE': {'sha256': '94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d',
                                                  'size_bytes': 1078},
 'share/licenses/crispembed-spacy/notices/CPython-LICENSE': {'sha256': '3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf',
                                                             'size_bytes': 13936},
 'share/licenses/crispembed-spacy/notices/PCRE2-LICENCE.md': {'sha256': '9cf7ac6976099a1d856826d3ef1b093bd6b84489dc6100628ac79e740cf9885a',
                                                              'size_bytes': 3867},
 'share/licenses/crispembed-spacy/notices/Thinc-LICENSE': {'sha256': 'd7192459f33a5d8c66358e2115ccbd3265478750f0b9d15f0c22e68e3ce6cc1e',
                                                           'size_bytes': 1123},
 'share/licenses/crispembed-spacy/notices/Unicode-15.0.0-ReadMe.txt': {'sha256': '53672c0d0b5185e3cf04c8e970d544c3af81ae7c8eeba0b9cf6d355aa954ae1f',
                                                                       'size_bytes': 635},
 'share/licenses/crispembed-spacy/notices/Unicode-LICENSE.txt': {'sha256': 'e7a93b009565cfce55919a381437ac4db883e9da2126fa28b91d12732bc53d96',
                                                                 'size_bytes': 1995},
 'share/licenses/crispembed-spacy/notices/continuation-notice-sources.json': {'sha256': '2f71d905ddbb8ea162a13c89e4e1d4e1f9858fcb4f82c967911912c79c4bf62f',
                                                                              'size_bytes': 2408},
 'share/licenses/crispembed-spacy/notices/de_core_news_sm-LICENSE': {'sha256': '3933c176979b68bc6d0bcc902c7d6c130f1d127f476f17ba5cdba8d99cfd0012',
                                                                     'size_bytes': 1056},
 'share/licenses/crispembed-spacy/notices/de_core_news_sm-LICENSES_SOURCES': {'sha256': '0c3ab78745569d848b7c83a486183bce42b9643561084d6d6f67eb0229c79869',
                                                                              'size_bytes': 13470},
 'share/licenses/crispembed-spacy/notices/ggml-LICENSE': {'sha256': '94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d',
                                                          'size_bytes': 1078},
 'share/licenses/crispembed-spacy/notices/spaCy-LICENSE': {'sha256': '3f4378e67708fecacd3e27651c8bc0f977cf20789e19da0d81a2840e7cf573a2',
                                                           'size_bytes': 1128}}

BUNDLE_FILES = {'LICENSE': {'sha256': '3933c176979b68bc6d0bcc902c7d6c130f1d127f476f17ba5cdba8d99cfd0012',
             'size_bytes': 1056},
 'LICENSES_SOURCES': {'sha256': '0c3ab78745569d848b7c83a486183bce42b9643561084d6d6f67eb0229c79869',
                      'size_bytes': 13470},
 'manifest.json': {'sha256': 'd599768375de3dc56abcca5cf1d49da0015857f9d3a10640aebb5317ef5d427e',
                   'size_bytes': 23762},
 'model.gguf': {'sha256': 'fae2e6ab8d6c611d7a9dddd4b3259a871a1d2f95cacc8259f49c1e4ff31ff6cd',
                'size_bytes': 7383104},
 'resources.json': {'sha256': 'b42496cc373ada57a67ed2f4c2c25f8d22f80c08db5841a7cdb91c8a6e18562a',
                    'size_bytes': 1404948}}

QUALIFIED_SYSTEM_DEPENDENCIES = [{'bytes': 225672,
  'sha256': 'c8438e4fde1934e61c88311633f00949ff645d5c04cdb8671fa3d78164d2f307',
  'soname': 'ld-linux-x86-64.so.2'},
 {'bytes': 1995216,
  'sha256': '9792e3cbb541c8f44c7acf5f14f4022ea62998ecc787d326bed4d8b6547dfd92',
  'soname': 'libc.so.6'},
 {'bytes': 182856,
  'sha256': '30c61ab012a4241bed033725a09b61f5fdd3bb7df95ee852d0b096520524c7af',
  'soname': 'libgcc_s.so.1'},
 {'bytes': 977112,
  'sha256': '6d567d53e895273ca14a1f9dc164fc6c8d39aed2f60aa46a733c2784228915f3',
  'soname': 'libm.so.6'},
 {'bytes': 711216,
  'sha256': '92e9b4167f9bbc237ca8f5526d82ee99ec9f8dbd0270ef1dd0ab24e2217f48e3',
  'soname': 'libpcre2-8.so.0'},
 {'bytes': 2497768,
  'sha256': '972bb2a18b71140dab0240f8a1f68ab3fb1d56bcd4c4f824a91b70888faf5a00',
  'soname': 'libstdc++.so.6'}]


class DistributionError(ValueError):
    """An input or installation fails the closed distribution contract."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _trusted_hash(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise DistributionError("A trusted lowercase SHA-256 is required.")
    return value


def _native_path(source: str) -> str:
    if source == "lib/libcrispembed-spacy.so":
        return f"native/{LIBRARY_SHA256}/libcrispembed-spacy.so"
    return "native/" + source


def _payload_pins() -> dict[str, dict]:
    return {
        **{_native_path(name): dict(pin) for name, pin in NATIVE_FILES.items()},
        **{"bundle/" + name: dict(pin) for name, pin in BUNDLE_FILES.items()},
    }


def manifest() -> dict:
    """Return the fixed profile; an archive cannot replace its asset pins."""
    return {
        "schema": SCHEMA,
        "profile": "de_core_news_sm-3.8.0-f32-linux-x86_64-avx2-validation",
        "target": {
            "system": "Linux",
            "machine": "x86_64",
            "cpu_baseline": "AVX2",
            "precision": "F32",
            "locale": "de-DE",
        },
        "native": {
            "source_commit": "51c2ac328ef40b00f18eb9b03130fb67c3f11313",
            "acceptance_commit": "2176c896dc6bb8c832a9d03b04fd07c7c991fdc5",
            "archive_sha256": NATIVE_ARCHIVE_SHA256,
            "archive_size_bytes": NATIVE_ARCHIVE_BYTES,
            "library_sha256": LIBRARY_SHA256,
            "abi_version": 1,
            "text_abi_version": 1,
        },
        "bundle_manifest_sha256": BUNDLE_MANIFEST_SHA256,
        "entrypoints": {
            "library": _native_path("lib/libcrispembed-spacy.so"),
            "bundle": "bundle",
        },
        "files": dict(sorted(_payload_pins().items())),
        "archive_format": {
            "tar": "USTAR",
            "compression": "gzip-level-9",
            "timestamp": 0,
            "uid": 0,
            "gid": 0,
            "file_mode": "0644",
            "order": "lexical path",
        },
        "installed_permissions": {"files": "0444", "directories": "0555"},
        "system_dependencies": {
            "bundled": False,
            "rpath": None,
            "runpath": None,
            "elf_needed": [
                "libpcre2-8.so.0", "libstdc++.so.6", "libm.so.6",
                "libgcc_s.so.1", "libc.so.6", "ld-linux-x86-64.so.2",
            ],
            "direct_symbol_version_floors": {
                "GLIBC": "2.38", "GLIBCXX": "3.4.29", "CXXABI": "1.3.9",
            },
            "pcre2_version": "10.46",
            "qualified_resolved_files": QUALIFIED_SYSTEM_DEPENDENCIES,
            "qualification": (
                "Only the recorded dependency bytes are qualified. Symbol "
                "version floors alone do not establish portability. The host "
                "must verify actual loader resolution separately."
            ),
        },
        "scope": {
            "runtime_release": False,
            "model_alias": False,
            "production_activation": False,
            "default_enabled": False,
            "automatic_download": False,
            "requires_explicit_environment_gate": "CRISPEMBED_SPACY=1",
            "qualification": "unchanged bounded Linux validation candidate",
        },
    }


def _manifest_bytes() -> bytes:
    return (json.dumps(manifest(), sort_keys=True, indent=2) + "\n").encode("utf-8")


def _all_pins() -> dict[str, dict]:
    data = _manifest_bytes()
    return {
        **_payload_pins(),
        MANIFEST_NAME: {"sha256": _sha(data), "size_bytes": len(data)},
    }


def _absolute(path: os.PathLike | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _parents_are_directories(path: Path) -> None:
    # Never resolve through a symlink and then treat the result as a trusted
    # original path. The cache is private to the caller, not a shared-writer
    # security boundary; the owner must keep it immutable during use.
    for parent in reversed(path.parents):
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise DistributionError("A path ancestor is linked or not a directory.")


def _fingerprint(info: os.stat_result) -> tuple:
    return (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
        info.st_ctime_ns, info.st_nlink,
    )


def _read_regular(path: os.PathLike | str, *, maximum: int,
                  pin: dict | None = None) -> bytes:
    path = _absolute(path)
    _parents_are_directories(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise DistributionError("An input is linked or not a regular file.")
        if before.st_size < 1 or before.st_size > maximum:
            raise DistributionError("An input exceeds its byte bound or is empty.")
        if pin is not None and before.st_size != pin["size_bytes"]:
            raise DistributionError("A pinned input has the wrong size.")
        data = source.read(maximum + 1)
        if (len(data) != before.st_size
                or _fingerprint(os.fstat(source.fileno())) != _fingerprint(before)
                or _fingerprint(path.lstat()) != _fingerprint(before)):
            raise DistributionError("An input changed while being read.")
    if pin is not None and _sha(data) != pin["sha256"]:
        raise DistributionError("A pinned input failed SHA-256 verification.")
    return data


def _safe_member(name: str) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and len(name) <= 240
        and not name.startswith("/")
        and "\\" not in name
        and ":" not in name
        and all(part not in ("", ".", "..") for part in name.split("/"))
        and all(32 <= ord(char) < 127 for char in name)
    )


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, data in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            entry.mtime = 0
            entry.uid = entry.gid = 0
            entry.uname = entry.gname = ""
            archive.addfile(entry, io.BytesIO(data))
    return raw.getvalue()


def _pack(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, filename="", mode="wb",
                       compresslevel=9, mtime=0) as archive:
        archive.write(_tar_bytes(files))
    return output.getvalue()


def _read_tar(data: bytes, pins: dict[str, dict], *, canonical: bool) -> dict[str, bytes]:
    if len(data) > MAX_ARCHIVE_BYTES:
        raise DistributionError("Archive exceeds the compressed byte bound.")
    try:
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        raw = decompressor.decompress(data, MAX_EXPANDED_BYTES + 1)
        if (len(raw) > MAX_EXPANDED_BYTES or not decompressor.eof
                or decompressor.unused_data or decompressor.unconsumed_tail):
            raise DistributionError("Invalid, trailing or oversized gzip stream.")
        result = {}
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            for entry in archive:
                name = entry.name
                if not _safe_member(name):
                    raise DistributionError("Unsafe archive member path.")
                if name in result:
                    raise DistributionError("Duplicate archive member.")
                if name not in pins:
                    raise DistributionError("Unlisted archive member.")
                if (entry.type != tarfile.REGTYPE or entry.linkname
                        or entry.pax_headers or entry.sparse is not None):
                    raise DistributionError("Only ordinary regular members are allowed.")
                if (entry.mode != 0o644 or entry.uid != 0 or entry.gid != 0
                        or entry.mtime != 0 or entry.uname or entry.gname):
                    raise DistributionError("Unexpected archive metadata.")
                pin = pins[name]
                if entry.size != pin["size_bytes"]:
                    raise DistributionError("Archive member has the wrong size.")
                stream = archive.extractfile(entry)
                if stream is None:
                    raise DistributionError("Archive member has no payload.")
                with stream:
                    payload = stream.read(entry.size + 1)
                if len(payload) != entry.size or _sha(payload) != pin["sha256"]:
                    raise DistributionError("Archive member failed its fixed hash.")
                result[name] = payload
        if result.keys() != pins.keys():
            raise DistributionError("Archive is missing a required member.")
        # This also rejects GNU/PAX headers hidden by tarfile, concatenated
        # archives, nonzero padding, extra records and alternate path spellings.
        if canonical and raw != _tar_bytes(result):
            raise DistributionError("Archive is not the canonical closed USTAR tree.")
        return result
    except (tarfile.TarError, zlib.error, EOFError) as error:
        raise DistributionError("Malformed compressed archive.") from error


def _verified_archive(path: os.PathLike | str, trusted_sha256: str) -> tuple[bytes, dict]:
    expected = _trusted_hash(trusted_sha256)
    data = _read_regular(path, maximum=MAX_ARCHIVE_BYTES)
    # Authenticate compressed bytes before the first decompression or parse.
    if _sha(data) != expected:
        raise DistributionError("Archive differs from the externally trusted SHA-256.")
    return data, _read_tar(data, _all_pins(), canonical=True)


def _archive_result(sha256: str, size: int) -> dict:
    return {
        "status": "verified",
        "verification_scope": "archive identity and fixed payload bytes",
        "archive_sha256": sha256,
        "archive_size_bytes": size,
        "manifest_sha256": _sha(_manifest_bytes()),
        "payload_file_count": len(_payload_pins()),
        "entrypoints": manifest()["entrypoints"],
        "system_dependencies_verified": False,
        "native_library_loaded": False,
        "manifest": manifest(),
    }


def verify_archive(path: os.PathLike | str, trusted_sha256: str) -> dict:
    """Authenticate transport and every fixed payload byte without native loading."""
    data, _ = _verified_archive(path, trusted_sha256)
    return _archive_result(trusted_sha256, len(data))


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new_file(path: Path, data: bytes, mode: int) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                           0o600), "wb") as output:
        output.write(data)
        os.fchmod(output.fileno(), mode)
        output.flush()
        os.fsync(output.fileno())


def _publish_file(path: Path, data: bytes) -> None:
    _parents_are_directories(path)
    fd, temporary = tempfile.mkstemp(prefix=".spacy-package-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            os.fchmod(output.fileno(), 0o644)
            output.flush()
            os.fsync(output.fileno())
        # Hard-linking a regular staging file atomically fails if ANY destination
        # already exists. Unlink the private staging name immediately afterwards.
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def build(native_archive: os.PathLike | str, bundle: os.PathLike | str,
          output: os.PathLike | str) -> dict:
    """Create one deterministic capsule; existing output paths are never replaced."""
    original = _read_regular(native_archive, maximum=NATIVE_ARCHIVE_BYTES, pin={
        "sha256": NATIVE_ARCHIVE_SHA256, "size_bytes": NATIVE_ARCHIVE_BYTES,
    })
    native = _read_tar(original, NATIVE_FILES, canonical=False)
    bundle = _absolute(bundle)
    _parents_are_directories(bundle)
    if not stat.S_ISDIR(bundle.lstat().st_mode):
        raise DistributionError("Bundle root is linked or not a directory.")
    if {item.name for item in bundle.iterdir()} != BUNDLE_FILES.keys():
        raise DistributionError("The bundle must contain exactly five pinned files.")
    files = {_native_path(name): data for name, data in native.items()}
    files.update({
        "bundle/" + name: _read_regular(bundle / name, maximum=pin["size_bytes"], pin=pin)
        for name, pin in BUNDLE_FILES.items()
    })
    files[MANIFEST_NAME] = _manifest_bytes()
    packed = _pack(files)
    # Exercise the exact reader before making the completed archive visible.
    _read_tar(packed, _all_pins(), canonical=True)
    _publish_file(_absolute(output), packed)
    return _archive_result(_sha(packed), len(packed))


def _directories(pins: dict) -> set[str]:
    return {str(parent) for name in pins for parent in Path(name).parents if str(parent) != "."}


def _verify_tree(directory: Path, trusted_sha256: str, *, named: bool) -> dict:
    trusted_sha256 = _trusted_hash(trusted_sha256)
    _parents_are_directories(directory)
    if named and directory.name != trusted_sha256:
        raise DistributionError("Installation directory must be the trusted archive SHA-256.")
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o555:
        raise DistributionError("Installation root must be a regular read-only directory.")
    pins = _all_pins()
    allowed_dirs = _directories(pins)
    found_files, found_dirs = set(), set()
    for current, dirs, files in os.walk(directory, followlinks=False):
        parent = Path(current)
        for name in dirs:
            child = parent / name
            relative = child.relative_to(directory).as_posix()
            child_info = child.lstat()
            if (relative not in allowed_dirs or not stat.S_ISDIR(child_info.st_mode)
                    or stat.S_IMODE(child_info.st_mode) != 0o555):
                raise DistributionError("Unexpected, linked or writable installation directory.")
            found_dirs.add(relative)
        for name in files:
            child = parent / name
            relative = child.relative_to(directory).as_posix()
            if relative not in pins or stat.S_IMODE(child.lstat().st_mode) != 0o444:
                raise DistributionError("Unexpected or writable installation file.")
            pin = pins[relative]
            _read_regular(child, maximum=pin["size_bytes"], pin=pin)
            found_files.add(relative)
    if found_files != pins.keys() or found_dirs != allowed_dirs:
        raise DistributionError("Installation is missing required files or directories.")
    return {
        "status": "verified",
        "verification_scope": "fixed installed payload bytes; transport requires verify_archive",
        "archive_sha256": trusted_sha256,
        "manifest_sha256": pins[MANIFEST_NAME]["sha256"],
        "payload_file_count": len(pins) - 1,
        "root": str(directory),
        "entrypoints": {key: str(directory / path)
                        for key, path in manifest()["entrypoints"].items()},
        "system_dependencies_verified": False,
        "native_library_loaded": False,
    }


def verify_tree(directory: os.PathLike | str, trusted_sha256: str) -> dict:
    """Verify the closed installed tree; this does not reauthenticate transport."""
    return _verify_tree(_absolute(directory), trusted_sha256, named=True)


def _remove_staging(directory: Path) -> None:
    # Only a freshly created private staging directory enters this function.
    # Never chmod or delete an existing content-addressed destination.
    for current, _, _ in os.walk(directory):
        os.chmod(current, 0o700)
    shutil.rmtree(directory)


def _publish_directory(staging: Path, destination: Path) -> bool:
    """Linux atomic directory publication, including an empty-destination race."""
    # ctypes is used only for the host's no-replace rename primitive. No spaCy
    # or dependency library is ever loaded by the distribution utility.
    import ctypes

    process = ctypes.CDLL(None, use_errno=True)
    try:
        rename = process.renameat2
    except AttributeError as error:
        raise DistributionError("Atomic no-replace directory rename is unavailable.") from error
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                       ctypes.c_uint]
    rename.restype = ctypes.c_int
    parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if rename(parent_fd, os.fsencode(staging.name), parent_fd,
                  os.fsencode(destination.name), 1) == 0:  # RENAME_NOREPLACE
            os.fsync(parent_fd)
            return True
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            return False
        raise OSError(code, os.strerror(code))
    finally:
        os.close(parent_fd)


def install(archive: os.PathLike | str, trusted_sha256: str,
            cache: os.PathLike | str) -> dict:
    """Authenticate, stage, verify, then publish atomically without replacement."""
    if sys.platform != "linux":
        raise DistributionError("This installation utility is qualified only on Linux.")
    _, files = _verified_archive(archive, trusted_sha256)
    cache = _absolute(cache)
    _parents_are_directories(cache)
    try:
        cache.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = cache.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o022):
        raise DistributionError("Installation requires a caller-owned private cache directory.")
    destination = cache / trusted_sha256
    if os.path.lexists(destination):
        return {**verify_tree(destination, trusted_sha256), "installed": False,
                "transport_verified": True}
    staging = Path(tempfile.mkdtemp(prefix=".spacy-stage-", dir=cache))
    try:
        for name, data in sorted(files.items()):
            path = staging / name
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _write_new_file(path, data, 0o444)
        for current, _, _ in os.walk(staging, topdown=False):
            os.chmod(current, 0o555)
            _fsync_directory(Path(current))
        _verify_tree(staging, trusted_sha256, named=False)
        published = _publish_directory(staging, destination)
        if published:
            staging = None
        result = verify_tree(destination, trusted_sha256)
        return {**result, "installed": published, "transport_verified": True}
    finally:
        if staging is not None and staging.exists():
            _remove_staging(staging)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build", help="Build from the exact admitted inputs.")
    create.add_argument("--native-archive", required=True, type=Path)
    create.add_argument("--bundle", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    for name in ("inspect", "verify", "install"):
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
        if args.command == "build":
            result = build(args.native_archive, args.bundle, args.output)
        elif args.command == "install":
            result = install(args.archive, args.sha256, args.cache)
        elif args.command == "verify-install":
            result = verify_tree(args.directory, args.sha256)
        else:
            result = verify_archive(args.archive, args.sha256)
    except (DistributionError, OSError) as error:
        print(json.dumps({"status": "rejected", "reason": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
