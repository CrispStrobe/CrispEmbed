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
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:  # type: ignore[override]
        return True


setup(distclass=BinaryDistribution)
