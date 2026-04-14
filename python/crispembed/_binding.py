"""CrispEmbed Python wrapper via ctypes."""

import ctypes
import os
import platform
import numpy as np
from pathlib import Path
from typing import List, Optional, Union


def _find_lib():
    """Find the crispembed shared library."""
    names = {
        "Linux": "libcrispembed.so",
        "Darwin": "libcrispembed.dylib",
        "Windows": "crispembed.dll",
    }
    lib_name = names.get(platform.system(), "libcrispembed.so")

    # Search paths
    search = [
        Path(__file__).parent,
        Path(__file__).parent.parent.parent / "build",
        Path(__file__).parent.parent.parent / "build" / "lib",
        Path.cwd() / "build",
    ]
    for d in search:
        p = d / lib_name
        if p.exists():
            return str(p)

    # Fall back to system search
    return lib_name


class CrispEmbed:
    """Text embedding model using ggml inference.

    Usage:
        model = CrispEmbed("all-MiniLM-L6-v2.gguf")
        vectors = model.encode(["Hello world", "Goodbye world"])
        print(vectors.shape)  # (2, 384)
    """

    def __init__(self, model_path: str, n_threads: int = 4, lib_path: Optional[str] = None):
        self._lib = ctypes.CDLL(lib_path or _find_lib())

        # Define function signatures
        self._lib.crispembed_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.crispembed_init.restype = ctypes.c_void_p

        self._lib.crispembed_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)
        ]
        self._lib.crispembed_encode.restype = ctypes.POINTER(ctypes.c_float)

        self._lib.crispembed_free.argtypes = [ctypes.c_void_p]
        self._lib.crispembed_free.restype = None

        self._lib.crispembed_get_hparams.argtypes = [ctypes.c_void_p]
        self._lib.crispembed_get_hparams.restype = ctypes.c_void_p

        # Init model
        self._ctx = self._lib.crispembed_init(
            model_path.encode("utf-8"), n_threads
        )
        if not self._ctx:
            raise RuntimeError(f"Failed to load model: {model_path}")

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode text(s) to embedding vectors.

        Args:
            texts: Single string or list of strings.
            normalize: L2-normalize (default True, already done in C).

        Returns:
            np.ndarray of shape (n_texts, dim) or (dim,) for single text.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        results = []
        for text in texts:
            dim = ctypes.c_int(0)
            ptr = self._lib.crispembed_encode(
                self._ctx, text.encode("utf-8"), ctypes.byref(dim)
            )
            if not ptr:
                raise RuntimeError(f"Encoding failed for: {text[:50]}")
            vec = np.ctypeslib.as_array(ptr, shape=(dim.value,)).copy()
            results.append(vec)

        out = np.stack(results)
        return out[0] if single else out

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        # Read from first encode or hparams
        dim = ctypes.c_int(0)
        ptr = self._lib.crispembed_encode(
            self._ctx, b"test", ctypes.byref(dim)
        )
        return dim.value

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.crispembed_free(self._ctx)
            self._ctx = None
