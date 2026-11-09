#!/usr/bin/env python3
"""Static contract: per-call licence acceptance is wired through every surface.

The behaviour is checked natively by tests/test_license_consent.cpp. This
keeps the C API, the Rust/Dart/Python bindings and the docs from drifting,
and asserts that no binding implements consent by setting
CRISPEMBED_ACCEPT_LICENSE (process-wide, racy, and it would accept every model
sharing the tag).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def text(path):
    return (ROOT / path).read_text(encoding="utf-8")


def require(path, snippets):
    body = text(path)
    missing = [s for s in snippets if s not in body]
    if missing:
        raise AssertionError(f"{path} is missing: {missing}")


def forbid(path, snippets):
    body = text(path)
    present = [s for s in snippets if s in body]
    if present:
        raise AssertionError(f"{path} must not contain: {present}")


require(
    "src/crispembed.h",
    [
        "crispembed_resolve_model_accepting(const char * arg, int auto_download,",
        "crispembed_model_license_info(const char * name,",
        # The legacy declaration is unchanged.
        "CRISPEMBED_API const char * crispembed_resolve_model(const char * arg, int auto_download);",
    ],
)
require(
    "src/crispembed.cpp",
    ["crispembed_mgr::resolve_model_accepting(", "crispembed_mgr::model_license_info("],
)
require(
    "examples/cli/model_mgr.cpp",
    [
        '#include "core/license_gate.h"',
        "core_license::is_blanket(accepted_spdx)",
        "return resolve_model_impl(arg, auto_download, accepted_license, true);",
        "return resolve_model_impl(arg, auto_download, accepted_spdx, false);",
        'std::getenv("CRISPEMBED_ACCEPT_LICENSE")',
    ],
)
require(
    "src/core/license_gate.h",
    ['accepted == "all" || accepted == "*"', "if (is_blanket(accepted)) return allow_blanket;"],
)
require(
    "crispembed-sys/src/lib.rs",
    ["pub fn crispembed_resolve_model_accepting(", "pub fn crispembed_model_license_info("],
)
require(
    "crispembed/src/lib.rs",
    ["pub fn resolve_model_accepting(", "pub fn model_license_info(", "pub struct ModelLicenseInfo"],
)
require(
    "python/crispembed/_binding.py",
    ["def resolve_model_accepting(", "def model_license_info(", "crispembed_resolve_model_accepting"],
)
require(
    "flutter/crispembed/lib/src/crispembed.dart",
    ["static String resolveModelAccepting(", "static ModelLicenseInfo? modelLicenseInfo(", "class ModelLicenseInfo"],
)
require(
    "flutter/crispembed/lib/src/crispembed_bindings.dart",
    ["CrispembedResolveModelAcceptingNative", "CrispembedModelLicenseInfoNative"],
)
for binding in (
    "crispembed/src/lib.rs",
    "python/crispembed/_binding.py",
    "flutter/crispembed/lib/src/crispembed.dart",
):
    forbid(binding, ["CRISPEMBED_ACCEPT_LICENSE"])
require("README.md", ["crispembed_resolve_model_accepting", "crispembed_model_license_info"])
require(
    "CMakeLists.txt",
    ["add_executable(test-license-consent tests/test_license_consent.cpp)"],
)
require(".github/workflows/build.yml", ["./build/test-license-consent"])

print("License consent surfaces OK")
