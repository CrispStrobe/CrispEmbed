#!/usr/bin/env python3
"""Static contract for the process-wide offline model-resolution switch."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(path, snippets):
    text = (ROOT / path).read_text()
    missing = [snippet for snippet in snippets if snippet not in text]
    if missing:
        raise AssertionError(f"{path} is missing: {missing}")


require(
    "examples/cli/model_mgr.cpp",
    [
        "bool offline()",
        'core_env::on("CRISPEMBED_OFFLINE")',
        "hf_hub_offline_env()",
        "if (offline())",
        "offline mode — not downloading",
    ],
)
require(
    "src/crispembed.cpp",
    ["crispembed_set_offline", "crispembed_is_offline", "crispembed_resolve_model"],
)
require("examples/cli/main.cpp", ['strcmp(argv[i], "--offline")'])
require("examples/server/server.cpp", ['strcmp(argv[i], "--offline")'])
require(
    "python/crispembed/_binding.py",
    [
        "def set_offline(",
        "def is_offline(",
        "offline: Optional[bool] = None",
        # An explicit library path must reach both state configuration and
        # resolution, otherwise ctypes can load two independent copies and the
        # resolver is online despite CrispEmbed(..., offline=True).
        "self.resolve_model(model_path, auto_download=auto_download, lib_path=lib_path)",
    ],
)

print("Offline mode surfaces OK")
