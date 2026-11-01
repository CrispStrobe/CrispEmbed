#!/usr/bin/env python3
"""Explicit import routing for the unchanged, pinned Moebius CPU student.

The upstream model_lib initializer also imports the unused PixelHacker teacher,
whose Triton autotuner needs a GPU during import. The removal.v1_2 initializer
also imports a training dataset. The opt-in CPU profile skips those two package
initializers, then imports the exact student, block, removal, and pipeline source
modules through their original module names. No layer, tensor, or FLA operation
is replaced. Normal upstream package imports remain the default profile.
"""

import importlib
import importlib.machinery
from pathlib import Path
import sys
import types

from latent_inpaint_contract import sha256, verify_source

STANDARD_PROFILE = "upstream-package-imports-v1"
CPU_STUDENT_PROFILE = "moebius-student-only-cpu-v1"
PROFILES = (STANDARD_PROFILE, CPU_STUDENT_PROFILE)
SKIPPED_INITIALIZERS = ("model_lib/__init__.py", "removal/v1_2/__init__.py")
STUDENT_MODULE = "model_lib.nets.unet_lambda_prune_lite"
STUDENT_CLASS = "UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8"
BLOCK_MODULE = "model_lib.nets.unet_lambda_dwconv_blocks"
PACKAGE_ROOTS = ("model_lib", "removal", "utils_infer")


def profile_metadata(profile, model, device):
    if profile not in PROFILES:
        raise ValueError("unknown original import profile")
    if profile == CPU_STUDENT_PROFILE and (model != "moebius" or device != "cpu"):
        raise ValueError("student-only import profile requires Moebius on the CPU")
    return {"name": profile,
            "package_initializers_skipped": list(SKIPPED_INITIALIZERS) if profile == CPU_STUDENT_PROFILE else [],
            "numerical_sources_modified": False}


def _is_source_package(name):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in PACKAGE_ROOTS)


def verify_imported_moebius_modules(source_dir, contract=None):
    """Bind the objects actually imported to the verified numerical files.

    Hashing files alone cannot detect a cached same-named module from another
    tree. Require both the loaded module's path and its actual file digest,
    including the top-level utils_infer helper used by the original pipeline.
    """
    source_dir = Path(source_dir).resolve()
    record = verify_source(source_dir, "moebius", contract)
    required = (STUDENT_MODULE, BLOCK_MODULE, "removal.v1_2.removal_model", "removal.v1_2.pipeline", "utils_infer")
    if any(name not in sys.modules for name in required):
        raise ValueError("original Moebius numerical import closure is incomplete")
    imported = []
    for name, module in sorted(tuple(sys.modules.items())):
        if not _is_source_package(name):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            expected = source_dir.joinpath(*name.split("."))
            paths = [Path(path).resolve() for path in getattr(module, "__path__", ())]
            if paths != [expected]:
                raise ValueError("original namespace search path differs: " + name)
            continue
        path = Path(filename)
        if path.is_symlink():
            raise ValueError("original imported module is a symlink: " + name)
        try:
            relative = path.resolve().relative_to(source_dir).as_posix()
        except ValueError as error:
            raise ValueError("original imported module is outside the verified source: " + name) from error
        expected = record["files"].get(relative)
        actual = sha256(path)
        if expected is None or path.stat().st_size != expected["bytes"] or actual != expected["sha256"]:
            raise ValueError("original imported module identity differs: " + name)
        imported.append({"module": name, "file": relative, "sha256": actual})
    return imported


def prepare_moebius_cpu_imports(source_dir, contract=None):
    """Admit verified files before installing isolated package search paths.

    A fresh process is required: pre-imported upstream packages could otherwise
    hide a source/version mismatch. A failed import removes every package shell
    and child created here, so callers cannot continue with a partial model.
    """
    source_dir = Path(source_dir).resolve()
    verify_source(source_dir, "moebius", contract)
    if any(_is_source_package(name) for name in sys.modules):
        raise RuntimeError("original Moebius packages were already imported; use a fresh process")
    try:
        for name in ("model_lib", "removal", "removal.v1_2"):
            directory = source_dir.joinpath(*name.split("."))
            if not directory.is_dir():
                raise ValueError("missing original package directory: " + name)
            package = types.ModuleType(name)
            package.__package__ = name
            package.__path__ = [str(directory)]
            package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            package.__spec__.submodule_search_locations = package.__path__
            sys.modules[name] = package
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(sys.modules[parent], child, package)
        student = importlib.import_module(STUDENT_MODULE)
        constructor = getattr(student, STUDENT_CLASS)
        if constructor.__module__ != STUDENT_MODULE:
            raise ValueError("student constructor is not defined by the pinned numerical module")
        package = sys.modules["model_lib"]
        setattr(package, STUDENT_CLASS, constructor)
        # Reproduce the initializer's student block exports using the imported
        # original objects. The teacher export is the only omitted model.
        blocks = importlib.import_module(BLOCK_MODULE)
        exports = getattr(blocks, "__all__", [name for name in vars(blocks) if not name.startswith("_")])
        for name in exports:
            setattr(package, name, getattr(blocks, name))
        return constructor
    except BaseException:
        for name in tuple(sys.modules):
            if _is_source_package(name):
                del sys.modules[name]
        raise
