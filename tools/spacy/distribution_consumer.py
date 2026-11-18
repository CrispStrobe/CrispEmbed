#!/usr/bin/env python3
"""Bounded installed-package checks for the unchanged Linux spaCy candidate.

Developer tooling only: no network, model conversion, linguistic gold, frozen
qualification inputs, or production activation. Each actual native run belongs
to a timeout-bounded subprocess after transport, payload, ELF and dependency
admission. Sampling RSS and descriptors is a bounded observation, not leak proof.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
import time

import distribution

SCHEMA = "crispembed.spacy.distribution_consumer.v1"
CASES = (
    ("empty", ""),
    ("unicode_emoji", "🧭 Über dem Hügel öffnet sich ein Café."),
    ("embedded_nul", "Text\0zwischen zwei Markierungen."),
    ("multiple_sentences", "Die Fähre wartet. Danach gehen wir an Bord."),
    ("whitespace", " \t\n  \r\n"),
    ("repeated_word", "Wort Wort Wort, Wort."),
    ("quotation_punctuation", "„Maja“, sagte Timo, „packt den Koffer!“"),
    ("moderate_paragraph", "Am frühen Abend liegt der Hafen ruhig da. "
     "Eine Nachbarin zählt die bunten Boote. "
     "Hinter dem Kran fährt ein Radfahrer langsam vorbei. "
     "Morgen beginnt dort ein kleines Fest."),
)
CYCLE_TEXT = "Am Kai wartet ein kleines Boot."
ENABLED_PROCESSES = 3
DISABLED_PROCESSES = 1
CYCLE_COUNT = 25
CHILD_TIMEOUT_SECONDS = 60
RSS_GROWTH_BUDGET_BYTES = 32 * 1024 * 1024
REQUIRED_CPU_FLAGS = ("avx", "avx2", "bmi2", "f16c", "fma", "sse4_2")
ELF_FLOORS = {"GLIBC": "2.38", "GLIBCXX": "3.4.29", "CXXABI": "1.3.9", "GCC": "3.0"}


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def protocol():
    return {
        "schema": "crispembed.spacy.w7.consumer_protocol.v1",
        "scope": "Authored operational loading, source-coordinate, repeat and resource controls; no linguistic-quality comparison",
        "cases": [{"id": name, "text": text,
                   "utf8_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                  for name, text in CASES],
        "enabled_fresh_processes": ENABLED_PROCESSES,
        "disabled_fresh_processes": DISABLED_PROCESSES,
        "cases_per_enabled_process": len(CASES),
        "warmup_create_analyze_free_cycles": 1,
        "measured_create_analyze_free_cycles": CYCLE_COUNT,
        "cycle_text": CYCLE_TEXT,
        "cycle_utf8_sha256": hashlib.sha256(CYCLE_TEXT.encode("utf-8")).hexdigest(),
        "child_timeout_seconds": CHILD_TIMEOUT_SECONDS,
        "fd_final_maximum": "post-warmup baseline",
        "maximum_sampled_post_free_rss_growth_bytes": RSS_GROWTH_BUDGET_BYTES,
        "platform": {"system": "Linux", "machine": "x86_64",
                     "required_cpu_flags": list(REQUIRED_CPU_FLAGS)},
        "required_elf_symbol_floors": ELF_FLOORS,
        "dependencies": distribution.QUALIFIED_SYSTEM_DEPENDENCIES,
        "repeat_policy": "Exact copied token, annotation, sentence and original-coordinate equality across fresh processes and all short cycles",
        "ownership_policy": "Free the creating context before reading the owned document; copy counted fields, free the document, then validate copied data",
        "resource_interpretation": "Sampled process RSS/FD/thread observations only; no universal leak or target-device budget claim",
        "native_acceptance_rerun": False,
        "grammar_qualification_rerun": False,
        "ci_queried_or_waited": False,
    }


class ConsumerError(ValueError):
    """A closed operational admission or observation failed."""


def require(condition, message):
    if not condition:
        raise ConsumerError(message)


def digest(path):
    """Hash one stable regular file; do not accept replacement during hashing."""
    path = Path(path)
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        require(stat.S_ISREG(before.st_mode), "Only regular files may enter a hash check")
        hasher = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
        after = os.fstat(source.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    signature = lambda value: tuple(getattr(value, field) for field in fields)
    require(signature(before) == signature(after) == signature(path.stat()),
            "A file changed while being hashed")
    return hasher.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: digest(root / name)
            for name in ("distribution.py", "distribution_consumer.py")}


def check_environment(environ=None, *, system=None, machine=None, cpuinfo=None):
    environ = os.environ if environ is None else environ
    forbidden = ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_ORIGIN_PATH")
    require(not any(name in environ for name in forbidden),
            "Loader environment overrides are forbidden for this qualification")
    require((system or platform.system()) == "Linux"
            and (machine or platform.machine()) == "x86_64",
            "Only the pinned Linux x86_64 target is admitted")
    cpuinfo = Path("/proc/cpuinfo").read_text() if cpuinfo is None else cpuinfo
    rows = re.findall(r"^flags\s*:\s*(.+)$", cpuinfo, flags=re.MULTILINE)
    require(rows and all(set(REQUIRED_CPU_FLAGS) <= set(row.split()) for row in rows),
            "The AVX2 baseline CPU flags are unavailable")
    return {"system": "Linux", "machine": "x86_64",
            "required_cpu_flags_present": list(REQUIRED_CPU_FLAGS),
            "loader_overrides_absent": True}


def verify_protocol(path, trusted_sha256):
    require(re.fullmatch(r"[0-9a-f]{64}", trusted_sha256 or ""),
            "An external protocol SHA-256 is required")
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == trusted_sha256,
            "Protocol differs from its externally trusted hash")
    require(data == canonical(protocol()), "Protocol differs from this frozen consumer contract")
    return trusted_sha256


def parse_elf(header, dynamic, versions):
    require(re.search(r"Class:\s+ELF64\b", header)
            and re.search(r"Data:\s+2's complement, little endian", header)
            and re.search(r"Machine:\s+Advanced Micro Devices X86-64", header),
            "Unexpected native ELF architecture")
    require("(RPATH)" not in dynamic and "(RUNPATH)" not in dynamic,
            "The accepted native library must have no RPATH/RUNPATH")
    needed = re.findall(r"\(NEEDED\).*\[(.+?)\]", dynamic)
    expected = {row["soname"] for row in distribution.QUALIFIED_SYSTEM_DEPENDENCIES}
    require(len(needed) == len(expected) and set(needed) == expected,
            "Unexpected native ELF dependency inventory")
    require(re.findall(r"\(SONAME\).*\[(.+?)\]", dynamic) == ["libcrispembed-spacy.so"],
            "Unexpected native library SONAME")
    require("Version needs section" in versions, "Missing ELF version requirements")
    needs = versions.split("Version needs section", 1)[1]
    observed = {}
    for tag in ELF_FLOORS:
        values = set(re.findall(r"Name: " + tag + r"_([\d.]+)", needs))
        require(values, "Missing required ELF symbol-version family")
        observed[tag] = max(values, key=lambda value: tuple(map(int, value.split("."))))
    require(observed == ELF_FLOORS, "Native ELF symbol floors differ from the accepted bytes")
    return {"class": "ELF64", "architecture": "x86_64", "needed": sorted(needed),
            "rpath": None, "runpath": None, "required_symbol_floors": observed,
            "interpretation": "Observed requirements, not cross-distribution qualification"}


def tool_output(command):
    # These are fixed host tools, not executables selected by a package manifest.
    result = subprocess.run(command, check=False, capture_output=True, text=True,
                            timeout=10, env={**os.environ, "LC_ALL": "C"})
    require(result.returncode == 0, "A required host ELF/dependency inspection failed")
    return result.stdout


def inspect_elf(library):
    return parse_elf(
        tool_output(["/usr/bin/readelf", "-h", str(library)]),
        tool_output(["/usr/bin/readelf", "-d", str(library)]),
        tool_output(["/usr/bin/readelf", "--version-info", str(library)]),
    )


def parse_ldd(output):
    """Reject incomplete, duplicated, non-absolute or unexpected resolution."""
    require("not found" not in output, "A required system dependency is missing")
    paths = {}
    for line in output.splitlines():
        if not line.strip() or re.match(r"\s*linux-vdso\.so\.1\s+\(", line):
            continue
        matched = re.fullmatch(r"\s*(\S+)\s+=>\s+(/\S+)\s+\(0x[0-9a-fA-F]+\)\s*", line)
        if matched:
            name, path = matched.groups()
        else:
            matched = re.fullmatch(r"\s*(/\S+)\s+\(0x[0-9a-fA-F]+\)\s*", line)
            require(matched, "Unrecognized dependency-loader output")
            path = matched.group(1)
            name = Path(path).name
        require(name not in paths, "Duplicate dependency resolution")
        paths[name] = Path(path)
    require(set(paths) == {row["soname"] for row in distribution.QUALIFIED_SYSTEM_DEPENDENCIES},
            "Resolved dependencies do not match the frozen six-file closure")
    return paths


def verify_dependencies(paths):
    rows, resolved = [], {}
    for pin in distribution.QUALIFIED_SYSTEM_DEPENDENCIES:
        name = pin["soname"]
        path = Path(paths[name]).resolve(strict=True)
        require(path.is_file() and path.stat().st_size == pin["bytes"],
                "A resolved system dependency differs in size")
        actual = digest(path)
        require(actual == pin["sha256"], "A resolved system dependency differs from its frozen hash")
        rows.append({**pin, "resolved_filename": path.name, "matches_frozen": True})
        resolved[name] = path
    return rows, resolved


def proc_maps(text=None):
    text = Path("/proc/self/maps").read_text() if text is None else text
    rows = []
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6 or not parts[5].startswith("/"):
            continue
        require(not parts[5].endswith(" (deleted)"), "A mapped file has been deleted")
        require(re.fullmatch(r"[0-9a-f]+:[0-9a-f]+", parts[3]), "Malformed mapped device")
        major, minor = (int(value, 16) for value in parts[3].split(":"))
        rows.append({"path": Path(parts[5]), "device": (major, minor), "inode": int(parts[4])})
    return rows


def mapped_closure(resolved, *, library=None, require_all=True, mappings=None):
    mappings = proc_maps() if mappings is None else mappings
    targets = dict(resolved)
    if library is not None:
        targets["libcrispembed-spacy.so"] = Path(library).resolve(strict=True)
    results = []
    for name, path in sorted(targets.items()):
        # Match versioned filenames as well as the stable SONAME. Another
        # already-loaded copy must not silently supersede the ldd resolution.
        prefix = name.split(".so", 1)[0] + ".so"
        relevant = [row for row in mappings if row["path"].name == name
                    or row["path"].name.startswith(prefix + ".")]
        if not relevant:
            require(not require_all, "A required dependency is absent from actual process mappings")
            continue
        info = path.stat()
        identity = ((os.major(info.st_dev), os.minor(info.st_dev)), info.st_ino)
        for row in relevant:
            require(row["path"].resolve(strict=True) == path
                    and (row["device"], row["inode"]) == identity,
                    "Actual loaded dependency differs from preflight resolution")
        expected = (distribution.LIBRARY_SHA256 if name == "libcrispembed-spacy.so"
                    else next(pin["sha256"] for pin in distribution.QUALIFIED_SYSTEM_DEPENDENCIES
                              if pin["soname"] == name))
        require(digest(path) == expected, "A mapped dependency changed after native load")
        results.append({"soname": name, "resolved_filename": path.name,
                        "sha256": expected, "mapped_segments": len(relevant),
                        "mapped_inode_matches_hashed_file": True})
    return results


def preflight(archive, trusted_sha256, installed):
    target = check_environment()
    transport = distribution.verify_archive(archive, trusted_sha256)
    tree = distribution.verify_tree(installed, trusted_sha256)
    library, bundle = (Path(tree["entrypoints"][key]) for key in ("library", "bundle"))
    elf = inspect_elf(library)
    dependencies, resolved = verify_dependencies(
        parse_ldd(tool_output(["/usr/bin/ldd", str(library)])))
    existing = mapped_closure(resolved, require_all=False)
    return {
        "target": target, "archive_sha256": trusted_sha256,
        "archive_size_bytes": transport["archive_size_bytes"],
        "manifest_sha256": transport["manifest_sha256"],
        "payload_file_count": transport["payload_file_count"],
        "transport_and_installed_payloads_verified": True,
        "elf": elf, "resolved_dependencies": dependencies,
        "already_mapped_verified_dependencies": existing,
    }, library, bundle, resolved


TEXT_FIELDS = ("text", "norm", "prefix", "suffix", "shape", "lemma",
               "tag", "pos", "morph", "dependency")
SPAN_FIELDS = ("byte_start", "byte_end", "scalar_start", "scalar_end", "utf16_start", "utf16_end")


class TextView(C.Structure):
    _fields_ = [("data", C.c_void_p), ("length", C.c_size_t)]


class TokenView(C.Structure):
    _fields_ = ([(name, TextView) for name in TEXT_FIELDS]
                + [(name, C.c_size_t) for name in SPAN_FIELDS]
                + [("orth_id", C.c_uint64), ("features", C.c_uint64 * 6),
                   ("head", C.c_int32), ("flags", C.c_uint32)])


def validate_annotations(text, tokens):
    """Validate original coordinates/ownership copies without linguistic gold."""
    raw = text.encode("utf-8")
    cursor = 0
    sentences, sentence_start = [], 0
    for index, token in enumerate(tokens):
        start, end = token["byte_start"], token["byte_end"]
        require(start == cursor and start < end <= len(raw), "Token bytes do not partition the source")
        require(token["text"].encode("utf-8") == raw[start:end], "Token surface differs from original source")
        for boundary in ("start", "end"):
            before = raw[:token["byte_" + boundary]].decode("utf-8", "strict")
            require(token["scalar_" + boundary] == len(before)
                    and token["utf16_" + boundary] == len(before.encode("utf-16-le")) // 2,
                    "Token scalar/UTF-16 coordinates differ from original bytes")
        flags = token["flags"]
        require(flags & ~15 == 0, "Unknown native token flags")
        cursor = end
        if flags & 1:
            require(raw[cursor:cursor + 1] == b" ", "Native trailing-space flag is inconsistent")
            cursor += 1
        require(token["features"][4] == str(int(bool(flags & 1)))
                and token["features"][5] == str(int(bool(flags & 2))),
                "Native lexical flags and features disagree")
        require(0 <= token["head"] < len(tokens), "Native dependency head is outside the document")
        require(bool(flags & 4) == (index == sentence_start), "Native sentence start is inconsistent")
        if flags & 8:
            stop = index + 1
            section = tokens[sentence_start:stop]
            require(sum(token["head"] == position for position, token
                        in enumerate(section, sentence_start)) == 1,
                    "Each sentence needs one self-headed root")
            for position in range(sentence_start, stop):
                visited, current = set(), position
                while True:
                    require(sentence_start <= current < stop,
                            "Native dependency crosses a sentence boundary")
                    require(current not in visited, "Native dependency cycle")
                    visited.add(current)
                    head = tokens[current]["head"]
                    if head == current:
                        break
                    current = head
            first = tokens[sentence_start]
            sentences.append({
                "start_token": sentence_start, "end_token": stop,
                "byte_start": first["byte_start"], "byte_end": end,
                "scalar_start": first["scalar_start"], "scalar_end": token["scalar_end"],
                "utf16_start": first["utf16_start"], "utf16_end": token["utf16_end"],
                "text": raw[first["byte_start"]:end].decode("utf-8"),
            })
            sentence_start = stop
    require(cursor == len(raw) and sentence_start == len(tokens),
            "Native tokens/sentences do not cover the complete original source")
    return {"tokens": tokens, "sentences": sentences}


class NativeConsumer:
    """Only construct after all preflight checks; all counters follow real calls."""

    def __init__(self, library, bundle):
        self.bundle = Path(bundle)
        self.counts = {"contexts_created": 0, "contexts_freed": 0,
                       "documents_created": 0, "documents_freed": 0}
        self.lib = lib = C.CDLL(str(library), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        for name in ("abi_version", "text_abi_version", "capabilities"):
            function = getattr(lib, "crispembed_spacy_" + name)
            function.argtypes, function.restype = [], C.c_uint32
        lib.crispembed_spacy_context_capabilities.argtypes = [C.c_void_p]
        lib.crispembed_spacy_context_capabilities.restype = C.c_uint32
        lib.crispembed_spacy_create_with_resources.argtypes = [C.c_char_p] * 4 + [C.c_void_p, C.c_size_t]
        lib.crispembed_spacy_create_with_resources.restype = C.c_void_p
        lib.crispembed_spacy_free.argtypes, lib.crispembed_spacy_free.restype = [C.c_void_p], None
        lib.crispembed_spacy_analyze_utf8.argtypes = [
            C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p),
            C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t]
        lib.crispembed_spacy_analyze_utf8.restype = C.c_int
        lib.crispembed_spacy_document_token_count.argtypes = [C.c_void_p]
        lib.crispembed_spacy_document_token_count.restype = C.c_size_t
        lib.crispembed_spacy_document_is_analyzed.argtypes = [C.c_void_p]
        lib.crispembed_spacy_document_is_analyzed.restype = C.c_int
        lib.crispembed_spacy_document_token.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(TokenView)]
        lib.crispembed_spacy_document_token.restype = C.c_int
        lib.crispembed_spacy_document_free.argtypes, lib.crispembed_spacy_document_free.restype = [C.c_void_p], None
        require(lib.crispembed_spacy_abi_version() == 1
                and lib.crispembed_spacy_text_abi_version() == 1
                and lib.crispembed_spacy_capabilities() == 3
                and lib.crispembed_spacy_context_capabilities(None) == 0,
                "The admitted native library exposes an unsupported ABI")

    def open(self):
        error = C.create_string_buffer(1024)
        context = self.lib.crispembed_spacy_create_with_resources(
            os.fsencode(self.bundle / "model.gguf"),
            distribution.BUNDLE_FILES["model.gguf"]["sha256"].encode("ascii"),
            os.fsencode(self.bundle / "resources.json"),
            distribution.BUNDLE_FILES["resources.json"]["sha256"].encode("ascii"),
            error, len(error))
        if context:
            self.counts["contexts_created"] += 1
        return context, error.value.decode("utf-8", "strict")

    def free_context(self, context):
        if context:
            self.lib.crispembed_spacy_free(context)
            self.counts["contexts_freed"] += 1

    def copied_tokens(self, document):
        require(self.lib.crispembed_spacy_document_is_analyzed(document) == 1,
                "Native result is not a fully analyzed document")
        count = self.lib.crispembed_spacy_document_token_count(document)
        require(count <= 2048, "Native token count exceeds the ABI bound")
        tokens = []
        for index in range(count):
            view = TokenView()
            require(self.lib.crispembed_spacy_document_token(document, index, C.byref(view)) == 0,
                    "Native token accessor rejected an owned document")
            token = {}
            for name in TEXT_FIELDS:
                field = getattr(view, name)
                require(field.length <= 1024 * 1024 and (field.data or field.length == 0),
                        "Invalid native counted-text extent")
                token[name] = (C.string_at(field.data, field.length).decode("utf-8", "strict")
                               if field.length else "")
            token.update({name: int(getattr(view, name)) for name in SPAN_FIELDS})
            token.update(orth_id=str(view.orth_id), features=[str(value) for value in view.features],
                         head=int(view.head), flags=int(view.flags))
            tokens.append(token)
        return tokens

    def analyze(self, text):
        context, document = None, C.c_void_p()
        try:
            context, error = self.open()
            require(context, "Native context admission failed: " + error)
            require(self.lib.crispembed_spacy_context_capabilities(context) == 15,
                    "Native context lacks complete text capabilities")
            data = text.encode("utf-8")
            buffer = C.create_string_buffer(data)
            message = C.create_string_buffer(1024)
            code = self.lib.crispembed_spacy_analyze_utf8(
                context, buffer, len(data), C.byref(document), None, None, None, None,
                message, len(message))
            if document.value:
                self.counts["documents_created"] += 1
            require(code == 0 and document.value,
                    "Native analysis failed: " + message.value.decode("utf-8", "strict"))
            # The published ABI promises the owned document outlives its
            # creating context. Exercise that lifetime before reading views.
            self.free_context(context)
            context = None
            copied = self.copied_tokens(document)
        finally:
            if document.value:
                self.lib.crispembed_spacy_document_free(document)
                self.counts["documents_freed"] += 1
            self.free_context(context)
        # This validation and digest use only owned Python data, after both
        # native resources have actually been freed.
        annotations = validate_annotations(text, copied)
        return {"input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "input_utf8_bytes": len(text.encode("utf-8")),
                "annotations": annotations,
                "annotations_sha256": hashlib.sha256(canonical(annotations)).hexdigest(),
                "native_context_freed_before_view_read": True,
                "native_document_freed_before_validation": True}

    def assert_drained(self):
        require(self.counts["contexts_created"] == self.counts["contexts_freed"]
                and self.counts["documents_created"] == self.counts["documents_freed"],
                "Native ownership counters did not drain")


def resource_sample():
    status = Path("/proc/self/status").read_text()
    rss = re.search(r"^VmRSS:\s+(\d+) kB$", status, re.MULTILINE)
    threads = re.search(r"^Threads:\s+(\d+)$", status, re.MULTILINE)
    require(rss and threads, "Process resource counters are unavailable")
    # List first, then inspect after the directory descriptor is closed.
    descriptors = os.listdir("/proc/self/fd")
    count = 0
    for name in descriptors:
        try:
            os.readlink("/proc/self/fd/" + name)
            count += 1
        except FileNotFoundError:
            pass
    return {"rss_bytes": int(rss.group(1)) * 1024,
            "open_file_descriptors": count, "threads": int(threads.group(1))}


def resource_verdict(baseline, samples):
    require(len(samples) == CYCLE_COUNT, "Incomplete measured resource cycles")
    growth = max(sample["rss_bytes"] for sample in samples) - baseline["rss_bytes"]
    final = samples[-1]
    return {"post_warmup_baseline": baseline, "final": final,
            "maximum_sampled_rss_growth_bytes": growth,
            "rss_growth_budget_bytes": RSS_GROWTH_BUDGET_BYTES,
            "fd_final_within_baseline": final["open_file_descriptors"] <= baseline["open_file_descriptors"],
            "rss_within_budget": growth <= RSS_GROWTH_BUDGET_BYTES,
            "interpretation": "Post-free samples for these bounded cycles; not a leak proof or device benchmark"}


class Report:
    """Reserve a new result and retain flushed checkpoints, including failures."""

    def __init__(self, path):
        self.path = Path(path).absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("xb") as output:
            output.write(canonical({"schema": SCHEMA, "status": "reserved", "passed": False}))
            output.flush()
            os.fsync(output.fileno())

    def write(self, value):
        descriptor, name = tempfile.mkstemp(prefix=".spacy-consumer-", dir=self.path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(canonical(value))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


def redact(message, args):
    text = str(message)
    replacements = [(str(Path(__file__).resolve().parents[2]), "CONSUMER_SOURCE")]
    for key in ("installed", "archive", "protocol", "report"):
        value = getattr(args, key, None)
        if value is not None:
            replacements.append((str(Path(value).absolute()), key.upper()))
    for actual, label in sorted(replacements, key=lambda pair: len(pair[0]), reverse=True):
        text = text.replace(actual, label)
    return text


def run_child(args, sink):
    before = source_hashes()
    result = {"schema": SCHEMA, "role": "fresh_process", "mode": args.child,
              "index": args.child_index, "status": "running", "passed": False,
              "archive_sha256": args.sha256,
              "protocol_sha256": args.protocol_sha256, "source_hashes_before": before,
              "cases": [], "cycles": [], "native_loaded": False,
              "native_acceptance_rerun": False, "grammar_qualification_rerun": False}
    api = None
    try:
        verify_protocol(args.protocol, args.protocol_sha256)
        require(os.environ.get("CRISPEMBED_SPACY") == ("1" if args.child == "enabled" else "0"),
                "The child gate does not match its declared arm")
        admission, library, bundle, dependencies = preflight(args.archive, args.sha256, args.installed)
        result["admission"] = admission
        result["stage"] = "preflight_passed"
        sink.write(result)
        api = NativeConsumer(library, bundle)
        result["native_loaded"] = True
        result["actual_loaded_closure"] = mapped_closure(dependencies, library=library)
        result["abi"] = {"native": 1, "text": 1, "legacy_capabilities": 3}
        if args.child == "disabled":
            context, error = api.open()
            try:
                require(not context and error and "disabled" in error.lower(),
                        "Disabled gate did not explicitly refuse a native context")
            finally:
                api.free_context(context)
            result.update(explicitly_unavailable=True, diagnostic=error,
                          inference_executed=False, resource_sample=resource_sample())
        else:
            for name, text in CASES:
                result["stage"] = "operational_case:" + name
                sink.write(result)
                started = time.monotonic()
                case = api.analyze(text)
                result["cases"].append({"id": name, **case, "elapsed_seconds": time.monotonic() - started})
                result["ownership"] = dict(api.counts)
                sink.write(result)
            result["stage"] = "cycle_warmup"
            sink.write(result)
            warmup = api.analyze(CYCLE_TEXT)
            api.assert_drained()
            result["warmup"] = warmup
            baseline = resource_sample()
            samples = []
            for index in range(CYCLE_COUNT):
                started = time.monotonic()
                cycle = api.analyze(CYCLE_TEXT)
                require(cycle["annotations"] == warmup["annotations"],
                        "Native annotations changed between identical operational cycles")
                api.assert_drained()
                sample = resource_sample()
                samples.append(sample)
                result["cycles"].append({"index": index,
                    "annotations_sha256": cycle["annotations_sha256"],
                    "elapsed_seconds": time.monotonic() - started, "post_free_resources": sample})
                result["ownership"] = dict(api.counts)
                result["stage"] = "cycle_completed:" + str(index + 1)
                sink.write(result)
            result["resources"] = resource_verdict(baseline, samples)
            require(result["resources"]["fd_final_within_baseline"]
                    and result["resources"]["rss_within_budget"],
                    "The bounded post-warmup resource budget failed")
            result["actual_loaded_closure_after_cycles"] = mapped_closure(dependencies, library=library)
            expected = len(CASES) + 1 + CYCLE_COUNT
            require(all(value == expected for value in api.counts.values()),
                    "The operational ownership denominator is incomplete")
            result.update(context_capabilities=15, inference_executed=True)
        api.assert_drained()
        result["ownership"] = dict(api.counts)
        # Authenticate installed bytes again after execution. This does not
        # substitute for the separate actual mapped-file identity checks.
        distribution.verify_tree(args.installed, args.sha256)
        result["installed_payloads_unchanged_after_execution"] = True
        result["source_hashes_after"] = source_hashes()
        require(result["source_hashes_after"] == before, "Consumer source changed during execution")
        result.update(status="passed", passed=True, stage="complete")
    except Exception as error:
        result.update(status="failed", passed=False, error=redact(error, args))
        if api is not None:
            result["ownership"] = dict(api.counts)
    sink.write(result)
    return result


def child_identity(child, *, mode, index, archive_sha256, protocol_sha256,
                   parent_sources, parent_admission):
    """Bind retained child evidence to the exact process that was launched.

    This returns a parent-side review, leaving the original child report bytes
    unchanged. Per-child before/after equality alone would admit mixed sources
    changed between processes and restored before the driver finished.
    """
    expected = {
        "schema": SCHEMA, "role": "fresh_process", "mode": mode, "index": index,
        "archive_sha256": archive_sha256, "protocol_sha256": protocol_sha256,
        "source_hashes_before": parent_sources, "source_hashes_after": parent_sources,
        "admission.archive_sha256": archive_sha256,
        "admission.manifest_sha256": parent_admission.get("manifest_sha256"),
        "admission.payload_file_count": parent_admission.get("payload_file_count"),
        "admission.transport_and_installed_payloads_verified": True,
    }
    mismatches = []
    for field, wanted in expected.items():
        observed = child
        for component in field.split("."):
            observed = observed.get(component) if isinstance(observed, dict) else None
        if observed != wanted or (field == "index" and type(observed) is not int):
            mismatches.append(field)
    if type(child.get("passed")) is not bool:
        mismatches.append("passed.type")
    return {"passed": not mismatches, "mismatched_fields": mismatches,
            "comparison": "Exact launched identity and initial driver source hashes"}


def repeat_verdict(children):
    enabled = [child for child in children if child.get("mode") == "enabled"]
    disabled = [child for child in children if child.get("mode") == "disabled"]
    complete = (len(enabled) == ENABLED_PROCESSES and len(disabled) == DISABLED_PROCESSES
                and all(child.get("passed") for child in children))
    if not complete:
        return {"complete": False, "annotations_exact_across_processes": False,
                "cycle_annotations_exact_across_processes": False}
    for child in enabled:
        require([case["id"] for case in child["cases"]] == [name for name, _ in CASES]
                and len(child["cycles"]) == CYCLE_COUNT,
                "A successful child omitted operational controls")
    reference = [(case["input_sha256"], case["annotations_sha256"], case["annotations"])
                 for case in enabled[0]["cases"]]
    exact = all([(case["input_sha256"], case["annotations_sha256"], case["annotations"])
                 for case in child["cases"]] == reference for child in enabled[1:])
    cycle = enabled[0]["warmup"]["annotations_sha256"]
    cycles_exact = all(child["warmup"]["annotations_sha256"] == cycle
                       and all(item["annotations_sha256"] == cycle for item in child["cycles"])
                       for child in enabled)
    return {"complete": True, "annotations_exact_across_processes": exact,
            "cycle_annotations_exact_across_processes": cycles_exact,
            "case_input_types": len(CASES),
            "unique_text_inputs_including_cycle": len({text for _, text in CASES} | {CYCLE_TEXT}),
            "case_executions": len(CASES) * ENABLED_PROCESSES,
            "short_cycle_executions_including_warmup": (1 + CYCLE_COUNT) * ENABLED_PROCESSES,
            "native_contexts_created_and_freed": (len(CASES) + 1 + CYCLE_COUNT) * ENABLED_PROCESSES,
            "native_documents_created_and_freed": (len(CASES) + 1 + CYCLE_COUNT) * ENABLED_PROCESSES,
            "independent_linguistic_observations": False}


def run_driver(args, sink):
    before = source_hashes()
    result = {"schema": SCHEMA, "role": "subprocess_driver", "status": "running", "passed": False,
              "archive_sha256": args.sha256, "protocol_sha256": args.protocol_sha256,
              "source_hashes_before": before, "children": [], "native_loaded_in_driver": False,
              "native_acceptance_rerun": False, "grammar_qualification_rerun": False,
              "ci_queried_or_waited": False}
    try:
        verify_protocol(args.protocol, args.protocol_sha256)
        result["admission"], _, _, _ = preflight(args.archive, args.sha256, args.installed)
        child_directory = Path(args.report).absolute().with_name(Path(args.report).stem + "-children")
        child_directory.mkdir(mode=0o700)
        sink.write(result)
        for index, mode in enumerate(["disabled"] * DISABLED_PROCESSES + ["enabled"] * ENABLED_PROCESSES):
            label = str(index) + "-" + mode
            report = child_directory / (label + ".json")
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--archive", str(Path(args.archive).absolute()), "--sha256", args.sha256,
                       "--installed", str(Path(args.installed).absolute()),
                       "--protocol", str(Path(args.protocol).absolute()),
                       "--protocol-sha256", args.protocol_sha256,
                       "--report", str(report), "--child", mode, "--child-index", str(index)]
            started = time.monotonic()
            timeout = False
            try:
                process = subprocess.run(command, capture_output=True, text=True,
                                         timeout=CHILD_TIMEOUT_SECONDS,
                                         env={**os.environ, "CRISPEMBED_SPACY": "1" if mode == "enabled" else "0",
                                              "PYTHONDONTWRITEBYTECODE": "1"})
                code, stdout, stderr = process.returncode, process.stdout, process.stderr
            except subprocess.TimeoutExpired as error:
                timeout, code = True, None
                stdout, stderr = error.stdout or "", error.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", "replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", "replace")
            for suffix, value in (("stdout", stdout), ("stderr", stderr)):
                (child_directory / (label + "." + suffix + ".txt")).write_text(redact(value, args), encoding="utf-8")
            if report.is_file():
                child = json.loads(report.read_text(encoding="utf-8"))
                require(isinstance(child, dict), "Child report is not an object")
            else:
                child = {"mode": mode, "index": index, "passed": False,
                         "error": "Child exited without a retained report"}
            # Preserve the child's declaration even if parent-side process or
            # provenance checks reject it. The original file remains immutable.
            child["reported_passed"] = child.get("passed")
            child["reported_status"] = child.get("status")
            child["identity_check"] = child_identity(
                child, mode=mode, index=index, archive_sha256=args.sha256,
                protocol_sha256=args.protocol_sha256, parent_sources=before,
                parent_admission=result["admission"])
            child["process"] = {"returncode": code, "timed_out": timeout,
                                "timeout_seconds": CHILD_TIMEOUT_SECONDS,
                                "elapsed_seconds": time.monotonic() - started,
                                "report": child_directory.name + "/" + report.name,
                                "report_sha256": digest(report) if report.is_file() else None}
            if timeout or code != 0 or not child["identity_check"]["passed"]:
                child["passed"] = False
                child["status"] = "failed"
            result["children"].append(child)
            sink.write(result)
        result["repeat_checks"] = repeat_verdict(result["children"])
        result["source_hashes_after"] = source_hashes()
        require(result["source_hashes_after"] == before, "Consumer sources changed during subprocess execution")
        distribution.verify_archive(args.archive, args.sha256)
        distribution.verify_tree(args.installed, args.sha256)
        result["archive_and_installed_payloads_unchanged"] = True
        require(result["repeat_checks"]["complete"]
                and result["repeat_checks"]["annotations_exact_across_processes"]
                and result["repeat_checks"]["cycle_annotations_exact_across_processes"],
                "Fresh-process operational controls failed")
        result.update(status="passed", passed=True)
    except Exception as error:
        result.update(status="failed", passed=False, error=redact(error, args))
    sink.write(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-protocol", action="store_true")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--installed", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--child", choices=("enabled", "disabled"), help=argparse.SUPPRESS)
    parser.add_argument("--child-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.print_protocol:
        sys.stdout.buffer.write(canonical(protocol()))
        return 0
    for name in ("archive", "sha256", "installed", "protocol", "protocol_sha256", "report"):
        if getattr(args, name) is None:
            parser.error("--" + name.replace("_", "-") + " is required")
    try:
        sink = Report(args.report)
    except FileExistsError:
        print("Refusing to overwrite a retained consumer report", file=sys.stderr)
        return 2
    result = run_child(args, sink) if args.child else run_driver(args, sink)
    print(json.dumps({key: result[key] for key in ("status", "passed", "error", "repeat_checks") if key in result}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
