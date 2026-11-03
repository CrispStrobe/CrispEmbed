#!/usr/bin/env python3
"""Stable x86 Linux CPU identity for CI caches containing -march=native code."""

import hashlib
import json
from pathlib import Path
import sys


def cpu_signature(cpuinfo: str) -> str | None:
    # Native Rust/sibling jobs currently run on Linux x86_64. Be conservative
    # on other hosts or heterogeneous cores: without a complete uniform CPU
    # identity the caller disables compiler caching instead of sharing one.
    required = ("vendor_id", "cpu family", "model", "stepping", "flags")
    identities = set()
    for record in cpuinfo.strip().split("\n\n"):
        fields = {}
        for line in record.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip()] = value.strip()
        if not fields:
            continue
        if "processor" not in fields or any(not fields.get(key) for key in required):
            return None
        # CPU index, frequency, bogomips and cache counters are intentionally
        # absent. A reordered flags list describes the same instruction set.
        fields["flags"] = " ".join(sorted(set(fields["flags"].split())))
        identities.add(tuple(fields[key] for key in required))
    if len(identities) != 1:
        return None
    payload = json.dumps(sorted(identities), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    try:
        signature = cpu_signature(Path("/proc/cpuinfo").read_text(encoding="ascii"))
    except (OSError, UnicodeError):
        signature = None
    if not signature:
        return 1
    print(signature)
    return 0


if __name__ == "__main__":
    sys.exit(main())
