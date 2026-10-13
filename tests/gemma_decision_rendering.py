#!/usr/bin/env python
"""Model-free: compare native float rendering with the Python str used by the blueprint."""

import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unicodedata


def main():
    assert (
        unicodedata.unidata_version == "15.0.0"
    ), "Use CPython 3.12, matching the model reference"
    values = [
        0.0,
        -0.0,
        1e7,
        1e-5,
        1e-4,
        1e15,
        1e16,
        5e-324,
        1.7976931348623157e308,
        0.84551240822557006,
    ]
    for exponent in range(-323, 309):
        value = float(f"1e{exponent}")
        values.extend(
            [value, math.nextafter(value, 0), math.nextafter(value, math.inf), -value]
        )
    rng = random.Random(0xDEC1510)
    for _ in range(3000):
        value = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if math.isfinite(value):
            values.append(value)
    values = [v for v in values if math.isfinite(v)]
    state = {f"f{i}": v for i, v in enumerate(values)}
    state["large integer"] = 10**400
    state["negative large integer"] = -(10**400)
    state["strings"] = [
        "can't",
        "quote \" and apostrophe' together",
        "slash\\",
        "\n\r\t",
        "日本語 🧪",
        True,
        None,
        "\x85\xa0\xad\u2028\u200d\ue000\u0378\U0001fae8\u2ffc",
    ]
    codepoints = [rng.randrange(0x110000) for _ in range(1500)]
    state["unicode"] = [chr(cp) for cp in codepoints if not 0xD800 <= cp <= 0xDFFF]
    request = {
        "workflow": "format",
        "state": state,
        "questions": {"check": {"type": "noul", "instructions": "Check?"}},
    }
    expected = "workflow: format\n" + "\n".join(f"{k}: {v}" for k, v in state.items())
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "request.json"
        path.write_text(json.dumps(request, ensure_ascii=False, allow_nan=False))
        bank = json.loads(subprocess.check_output([sys.argv[1], str(path)], text=True))
        for state_error in ("", " \t\r\n", "\u00a0{}", "\x1e{}"):
            bad = dict(request, state=state_error)
            path.write_text(json.dumps(bad, ensure_ascii=False))
            result = subprocess.run([sys.argv[1], str(path)], capture_output=True)
            assert (
                result.returncode != 0
            ), f"Accepted state rejected by Python: {state_error!r}"
    if bank[0] != expected:
        for index, (native, reference) in enumerate(
            zip(bank[0].splitlines(), expected.splitlines())
        ):
            if native != reference:
                raise AssertionError(
                    f"line {index}: native={native!r}, Python={reference!r}"
                )
        raise AssertionError("rendered line counts differ")
    print(
        f"PASS Python renderer: {len(values)} finite doubles, scientific boundaries, subnormals, signed zero and string/list repr"
    )


if __name__ == "__main__":
    main()
