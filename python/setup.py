"""Minimal setup.py — exists only to mark this distribution as having
*platform-specific* content for setuptools' wheel builder.

The package itself is configured entirely via pyproject.toml; this file
only overrides `Distribution.has_ext_modules()` so the wheel filename
includes a platform tag (e.g. `cp311-cp311-manylinux_2_28_x86_64.whl`)
instead of the pure-Python `py3-none-any.whl`.

Without this, cibuildwheel rejects every wheel we build with:

    Build failed because a pure Python wheel was generated.
    If you intend to build a pure-Python wheel, you don't need
    cibuildwheel - use `pip wheel -w DEST_DIR .` instead.

…because setuptools looks at `ext_modules` (empty for us — we don't
compile anything during pip install, we bundle prebuilt .so files
staged by the CI workflow) and concludes the wheel is portable.

Without ext_modules, setuptools also picks the wrong Python ABI tag
(`py3` instead of `cp311`), which would let a wheel built on
CPython 3.11 install on PyPy/CPython 3.12/etc. and dlopen a .so that
the loader can't actually use. Forcing has_ext_modules() = True fixes
both the platform and the ABI tag in one shot.

It also stages POLICY.md and the native components' Apache notices into the
package. Source distributions retain those staged files so rebuilding a wheel
does not require the original repository parent directory.
"""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.dist import Distribution

_HERE = Path(__file__).resolve().parent


def _stage_policy() -> None:
    """Copy the repo-root POLICY.md into the package so it ships in the wheel.

    A `pip install crispembed` user gets accept_biometric_use() and the whole
    face pipeline. Leaving the acceptable-use terms behind in a git repo they
    may never visit puts the EU AI Act Art. 5 prohibitions out of reach of the
    people who need them. Copied at build time rather than checked in twice,
    so the root file stays the single source of truth.

    Best-effort: an sdist unpacked without the parent repo has no ../POLICY.md,
    and that must not break the build. README.md carries the summary and the
    canonical URL either way.
    """
    src = _HERE.parent / "POLICY.md"
    if src.is_file():
        shutil.copyfile(src, _HERE / "crispembed" / "POLICY.md")


def _stage_native_notices() -> None:
    """Retain source attributions in every wheel that bundles native code.

    The repository copies are authoritative when available. An unpacked sdist
    instead uses the copies retained in its package directory. Binding-only
    builds may proceed without a repository parent, but a native binary must
    never be packaged without all three attribution/license files.
    """
    package = _HERE / "crispembed"
    notices = {
        "NOTICE.marker": _HERE.parent / "NOTICE.marker",
        "NOTICE.teleocr": _HERE.parent / "NOTICE.teleocr",
        "Apache-2.0.txt": _HERE.parent / "models" / "licenses" / "apache-2.0" / "LICENSE",
    }
    for name, source in notices.items():
        if source.is_file():
            shutil.copyfile(source, package / name)

    native = any(
        path.is_file()
        for pattern in ("*.so", "*.so.*", "*.dylib", "*.dll", "*.pyd")
        for path in package.rglob(pattern)
    )
    missing = [
        name for name in notices
        if not (package / name).is_file() or (package / name).stat().st_size == 0
    ]
    if native and missing:
        raise RuntimeError(
            "Cannot bundle native CrispEmbed libraries without their Apache notices: "
            + ", ".join(missing)
            + ". Build from a complete checkout or an sdist retaining the staged notice files."
        )


_stage_policy()
_stage_native_notices()


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:  # type: ignore[override]
        return True


setup(distclass=BinaryDistribution)
