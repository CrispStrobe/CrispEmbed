"""Shared pytest configuration and prerequisite-skip helpers.

The suite has to give a meaningful result on a plain environment (pytest,
numpy, pillow, gguf, sentencepiece, tokenizers; no model files, no HF token,
possibly no network, possibly a shallow clone, no native build). Tests whose
prerequisites are genuinely absent skip with a specific reason through the
helpers below; everything else must pass or fail for a real reason.

Helpers (each usable as a decorator on a test function/class, or as
``pytestmark = helper(...)`` at module level):

    requires_model(path_or_env, ...)   model/data file present
    requires_network(host=...)         outbound network reachable
    requires_binary(name_or_path, ...) built native binary / shared library
    requires_torch(extra=(...))        torch (and optional extra modules)
    requires_module(name, ...)         any optional python module
    requires_git_commit(rev, ...)      commit object present in the clone
"""
from __future__ import annotations

import functools
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Retained evidence snapshots (tests/results/**/source-snapshot/tests/...) are
# frozen copies of older test files kept as provenance. They are not part of
# the live suite and share basenames with live tests.
collect_ignore_glob = ["results/*"]


# ---------------------------------------------------------------------------
# Module-name isolation between test files.
#
# Several tool directories contain helper modules with the same basename
# (tools/*/contract.py, panel.py, ...). Test files put their own tool
# directory on sys.path and ``import contract``; in one pytest session the
# first import wins and later test files get another tool's module. After
# each test module is collected, drop newly imported repository modules whose
# basename is ambiguous, so the next test module resolves its own copy. The
# collected test module keeps its references, so its tests are unaffected.
# ---------------------------------------------------------------------------
def _ambiguous_basenames() -> frozenset[str]:
    counts: Counter[str] = Counter()
    for top in ("tools", "models", "scripts"):
        base = ROOT / top
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            counts[path.stem] += 1
    return frozenset(name for name, n in counts.items() if n > 1)


_AMBIGUOUS = _ambiguous_basenames()


def _repo_module(mod) -> bool:
    f = getattr(mod, "__file__", None)
    if not f:
        return False
    try:
        Path(f).resolve().relative_to(ROOT)
    except ValueError:
        return False
    return True


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    if not isinstance(collector, pytest.Module):
        return (yield)
    before = set(sys.modules)
    try:
        return (yield)
    finally:
        for name in set(sys.modules) - before:
            if "." not in name and name in _AMBIGUOUS and _repo_module(sys.modules[name]):
                del sys.modules[name]


# ---------------------------------------------------------------------------
# Prerequisite helpers
# ---------------------------------------------------------------------------
def _resolve_model(spec: str | os.PathLike) -> Path | None:
    """``spec`` is a path (absolute or repo-relative) or ``$ENV_VAR``."""
    s = str(spec)
    if s.startswith("$"):
        val = os.environ.get(s[1:], "").strip()
        return Path(val) if val else None
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def requires_model(*specs, reason: str | None = None):
    """Skip unless every model/data file exists. ``$VAR`` reads a path from the env."""
    missing = []
    for spec in specs:
        p = _resolve_model(spec)
        if p is None:
            missing.append(f"{spec} (env var unset)")
        elif not p.exists():
            missing.append(str(spec))
    return pytest.mark.skipif(
        bool(missing),
        reason=reason or f"requires model file(s): {', '.join(missing)}",
    )


@functools.lru_cache(maxsize=None)
def network_available(host: str = "huggingface.co", port: int = 443, timeout: float = 3.0) -> bool:
    if os.environ.get("CRISPEMBED_TEST_OFFLINE", "") not in ("", "0"):
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def requires_network(host: str = "huggingface.co", port: int = 443):
    return pytest.mark.skipif(
        not network_available(host, port),
        reason=f"requires network access to {host}:{port}",
    )


def _find_binary(spec: str) -> Path | None:
    p = Path(spec)
    if p.is_absolute():
        return p if p.exists() else None
    if "/" in spec:
        q = ROOT / p
        return q if q.exists() else None
    found = shutil.which(spec)
    if found:
        return Path(found)
    for d in ("build", "build/bin", "build-cpu", "build/Release"):
        q = ROOT / d / spec
        if q.exists():
            return q
    return None


def requires_binary(*specs, reason: str | None = None):
    """Skip unless every native binary/library exists (repo-relative path, or name on PATH/build dirs)."""
    missing = [s for s in specs if _find_binary(s) is None]
    return pytest.mark.skipif(
        bool(missing),
        reason=reason or f"requires built native binary: {', '.join(missing)}",
    )


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def requires_module(*names, reason: str | None = None):
    missing = [n for n in names if not module_available(n)]
    return pytest.mark.skipif(
        bool(missing),
        reason=reason or f"requires python module(s): {', '.join(missing)}",
    )


def requires_torch(extra: tuple[str, ...] = ()):
    return requires_module("torch", *extra)


@functools.lru_cache(maxsize=None)
def git_commit_available(rev: str) -> bool:
    try:
        r = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", f"{rev}^{{commit}}"],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def requires_git_commit(*revs, reason: str | None = None):
    """Skip unless each commit is in the local clone (shallow clones lack history)."""
    missing = [r for r in revs if not git_commit_available(r)]
    return pytest.mark.skipif(
        bool(missing),
        reason=reason or f"requires git commit(s) not in this clone (shallow?): {', '.join(m[:12] for m in missing)}",
    )


# ---------------------------------------------------------------------------
# Fresh-process tests.
#
# These controls assert properties of a *fresh* interpreter, e.g. that
# importing a tool does not pull numpy/gguf/torch/PIL into sys.modules, or
# they do runtime imports of a generic tool module name (``contract``,
# ``reference``) that another test file already bound. They are written for
# per-file ``python -m unittest`` runs (as CI does) and are correct there; in
# one shared pytest session other test files have legitimately imported those
# modules first. Each listed test is therefore executed in its own pytest
# subprocess; its pass/fail is reported unchanged.
# ---------------------------------------------------------------------------
FRESH_PROCESS_TESTS = frozenset({
    "tests/test_gguf_file_type.py::Controls::test_writer_has_explicit_standard_and_legacy_policy",
    "tests/test_josev2b_head_schema.py::HeadControls::test_missing_original_rejects_before_payload_or_optional_imports",
    "tests/test_josev_mixed_recipes.py::Controls::test_predictions_and_exact_c_quantizer_only",
    "tests/test_josev_q4k_packer.py::Controls::test_host_barrier_precedes_source_file_access",
    "tests/test_latent_inpaint_contract.py::LatentInpaintContractTest::test_converter_and_contract_do_not_import_frameworks",
    "tests/test_migan_native_driver.py::NativeDriverTest::test_no_framework_or_model_dependency",
    "tests/test_migan_reference_contract.py::MiganReferenceContractTest::test_no_local_model_execution_dependency",
    "tests/test_minicpm_ocr_graph_adapter.py::Recipe::test_actual_ref_api_schema_fixed_budgets_and_no_model_import",
    "tests/test_minicpm_retained_analysis_v2.py::Controls::test_no_optional_framework_imported_by_controls",
    "tests/test_minicpm_retained_v3.py::Controls::test_all_entrypoints_help_reject_optional_framework_imports",
    "tests/test_minicpm_seven_v2.py::Controls::test_optional_packages_never_imported_by_source_controls",
    "tests/test_moedern_colbert_contract.py::ContractTests::test_direct_forward_capture_is_neutral_and_restores_class_lookup",
    "tests/test_moedern_colbert_contract.py::ContractTests::test_direct_forward_mapping_observer_restores_and_is_neutral",
    "tests/test_moedern_colbert_contract.py::ContractTests::test_direct_forward_restores_after_original_exception",
    "tests/test_moedern_colbert_contract.py::ContractTests::test_direct_forward_restores_existing_override_after_exception",
    "tests/test_moedern_q4k.py::Controls::test_host_barrier_precedes_payload_and_frameworks",
    "tests/test_moedern_scifact_gha_adapter.py::AdapterControls::test_entrypoint_imports_before_any_optional_packages_or_payloads",
    "tests/test_moedern_text_composed.py::Controls::test_actual_cli_pending_negative_creates_evidence_without_api",
    "tests/test_moedern_unigram.py::Controls::test_dependency_budget_rejection_before_download",
    "tests/test_ovisocr2_export_profile.py::OvisProfileControls::test_no_model_imports",
    "tests/test_ovisocr2_frontend_driver.py::FrontendDriverControls::test_import_does_not_load_model_frameworks",
    "tests/test_ovisocr2_lighton_fixture.py::LightOnFixtureControls::test_preflight_stops_before_packages_and_checkpoint",
    "tests/test_ovisocr2_position_report.py::RetainedReport::test_no_framework_dependencies",
    "tests/test_ovisocr2_quality_contract.py::OvisQualityContract::test_no_model_framework_import",
    "tests/test_ovisocr2_quality_run.py::QualityDriverControls::test_no_framework_import",
    "tests/test_ovisocr2_vision_audit.py::OvisVisionAuditControls::test_no_framework_import",
    "tests/test_ovisocr2_vllm_quality_reference.py::QualityGPUControls::test_no_model_framework_import",
    "tests/test_ovisocr2_vllm_recipe.py::RecipeControls::test_import_does_not_load_model_or_cuda_frameworks",
    "tests/test_prepare_migan_inpaint.py::PrepareMiganTest::test_no_framework_import",
})
_FRESH_ENV = "CRISPEMBED_PYTEST_FRESH_CHILD"


def _rel_nodeid(item) -> str:
    path = Path(str(item.path)).resolve().relative_to(ROOT).as_posix()
    return path + item.nodeid[item.nodeid.index("::"):]


def _run_in_fresh_process(item):
    nodeid = _rel_nodeid(item)

    def runtest():
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", nodeid],
            cwd=ROOT, env={**os.environ, _FRESH_ENV: "1"},
            capture_output=True, text=True, timeout=900,
        )
        if r.returncode != 0:
            pytest.fail(f"fresh-process run of {nodeid} failed (exit {r.returncode}):\n"
                        f"{r.stdout[-6000:]}\n{r.stderr[-2000:]}", pytrace=False)

    item.runtest = runtest


# ---------------------------------------------------------------------------
# Test files that are standalone CLI scripts or need CLI-supplied binaries.
# ---------------------------------------------------------------------------
# Scripts whose ``test_*`` functions take CLI arguments (binary, model path)
# rather than fixtures; run them directly as documented in their docstrings.
STANDALONE_SCRIPTS = {
    "tests/test_all_parity.py": "standalone CLI parity script: needs a built crispembed binary and downloaded models "
                                "(python tests/test_all_parity.py --binary ...)",
    "tests/test_bgem3.py": "standalone CLI script: needs the built crispembed shared library and a BGE-M3 GGUF "
                           "(python tests/test_bgem3.py --binary ...)",
    "tests/test_siglip_converter.py": "standalone CLI script: needs a converted SigLIP GGUF and the HF model "
                                      "(python tests/test_siglip_converter.py ...)",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        try:
            rel = Path(str(item.path)).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            continue
        reason = STANDALONE_SCRIPTS.get(rel)
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
        _apply_git_commit_markers(item, _rel_nodeid(item))
        if not os.environ.get(_FRESH_ENV) and _rel_nodeid(item) in FRESH_PROCESS_TESTS:
            _run_in_fresh_process(item)


@pytest.fixture(autouse=True)
def _cli_supplied_binaries(request):
    """Binaries that the per-file CI jobs pass on the command line or via env."""
    mod = getattr(request, "module", None)
    name = getattr(mod, "__name__", "")
    if name == "test_inpaint_cli" and mod.BINARY is None:
        env = os.environ.get("CRISPEMBED_INPAINT_CLI_BIN", "").strip()
        if env:
            mod.BINARY = Path(env).resolve()
            real = os.environ.get("CRISPEMBED_INPAINT_CLI_REAL_BIN", "").strip()
            mod.REAL_BINARY = Path(real).resolve() if real else None
        else:
            original = mod.InpaintCliTests.run_cli

            def run_cli(self, args, *a, binary=None, **kw):
                if binary is None and mod.BINARY is None:
                    pytest.skip("requires built native binary: inpaint CLI driver "
                                "(set CRISPEMBED_INPAINT_CLI_BIN or run the file with --binary)")
                return original(self, args, *a, binary=binary, **kw)

            mp = pytest.MonkeyPatch()
            mp.setattr(mod.InpaintCliTests, "run_cli", run_cli)
            request.addfinalizer(mp.undo)
    if name == "test_rust_prebuilt_config" and not os.environ.get("CRISPEMBED_BUILD_SCRIPT_TEST_BIN"):
        pytest.skip("requires built native binary: Cargo build-script test executable "
                    "(CRISPEMBED_BUILD_SCRIPT_TEST_BIN, built by rust-prebuilt-contract CI)")
    yield


# ---------------------------------------------------------------------------
# Git history prerequisites.
#
# Many frozen-source guards compare the tree against blobs at pinned commits
# (``git show <sha>:path``, ``git ls-tree <sha>``, ``git worktree add <sha>``).
# In a shallow clone those commits are absent. A test that fails *only*
# because a git subprocess named a commit that is not in this clone is
# reported as skipped (the requires_git_commit condition, detected at run
# time because the pinned revs live in tool/contract data). Any other
# failure, including a git failure on a commit that is present, still fails.
# ---------------------------------------------------------------------------
import re as _re

_HEX_REV = _re.compile(r"^[0-9a-f]{7,40}$")

# Tests that probe a pinned commit through a git return code instead of an
# exception; mapped to the commit they need.
GIT_COMMIT_TESTS = {
    "tests/test_josev2b_task_quality_native_v1.py::Controls::test_actual_cmake_gitobject_exists_old_tools_cmake_does_not":
        "9c475d0dac70264adc123a63b6c1827b4b7280d5",
}


def _missing_git_revs(exc) -> list[str]:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            cmd = exc.cmd if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd).split()
            if cmd and Path(str(cmd[0])).name == "git":
                revs = []
                for arg in map(str, cmd[1:]):
                    rev = arg.split(":", 1)[0]
                    if _HEX_REV.match(rev):
                        revs.append(rev)
                missing = [r for r in revs if not git_commit_available(r)]
                if missing:
                    return missing
        exc = exc.__cause__ or exc.__context__
    return []


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.failed and call.excinfo is not None:
        missing = _missing_git_revs(call.excinfo.value)
        if missing:
            report.outcome = "skipped"
            report.longrepr = (
                str(item.path), (item.location[1] or 0) + 1,
                "Skipped: requires git commit(s) not in this clone (shallow?): "
                + ", ".join(m[:12] for m in missing),
            )
    return report


def _apply_git_commit_markers(item, rel_nodeid: str) -> None:
    rev = GIT_COMMIT_TESTS.get(rel_nodeid)
    if rev:
        item.add_marker(requires_git_commit(rev))
