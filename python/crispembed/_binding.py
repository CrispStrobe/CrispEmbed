"""CrispEmbed Python wrapper via ctypes.

Supports dense, sparse (BGE-M3/SPLADE), ColBERT multi-vector, and
cross-encoder reranking — all via a single shared library.
"""

import ctypes
import glob
import os
import platform
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


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
        Path(__file__).parent.parent.parent / "build-cuda",
        Path(__file__).parent.parent.parent / "build-vulkan",
        Path(__file__).parent.parent.parent / "build" / "lib",
        Path.cwd() / "build",
        Path.cwd() / "build-cuda",
        Path.cwd() / "build-vulkan",
    ]
    for d in search:
        p = d / lib_name
        if p.exists():
            return str(p)

    # Fall back to system search
    return lib_name


def _load_library(lib_path: Optional[str] = None):
    path = lib_path or _find_lib()
    if platform.system() == "Windows":
        dll_dir = Path(path).resolve().parent
        extra_dirs = [dll_dir, dll_dir / "bin"]
        for cuda_ver in ("v13.0", "v12.6", "v12.4", "v12.1", "v12.0", "v11.8"):
            base = Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{cuda_ver}")
            if base.is_dir():
                extra_dirs.append(base / "bin")
                if (base / "bin" / "x64").is_dir():
                    extra_dirs.append(base / "bin" / "x64")
                break
        for driver_store in glob.glob("C:/Windows/System32/DriverStore/FileRepository/nvdm*.inf_amd64_*/"):
            p = Path(driver_store)
            if p.is_dir():
                extra_dirs.append(p)
        os.environ["PATH"] = os.pathsep.join(str(p) for p in extra_dirs) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            for entry in extra_dirs:
                try:
                    os.add_dll_directory(str(entry))
                except OSError:
                    pass
    return ctypes.CDLL(path)


class _CrispEmbedHparams(ctypes.Structure):
    _fields_ = [
        ("n_vocab", ctypes.c_int32),
        ("n_max_tokens", ctypes.c_int32),
        ("n_embd", ctypes.c_int32),
        ("n_head", ctypes.c_int32),
        ("n_layer", ctypes.c_int32),
        ("n_intermediate", ctypes.c_int32),
        ("n_output", ctypes.c_int32),
        ("layer_norm_eps", ctypes.c_float),
    ]


class CrispEmbed:
    """Text embedding model using ggml inference.

    Supports dense embeddings, sparse retrieval (BGE-M3/SPLADE), ColBERT
    multi-vector, cross-encoder reranking, and bi-encoder reranking.

    Usage:
        model = CrispEmbed("all-MiniLM-L6-v2.gguf")
        vectors = model.encode(["Hello world", "Goodbye world"])
        print(vectors.shape)  # (2, 384)

        # Sparse (BGE-M3)
        model = CrispEmbed("bge-m3.gguf")
        if model.has_sparse:
            sparse = model.encode_sparse("Hello world")  # {token_id: weight}

        # ColBERT multi-vector
        if model.has_colbert:
            multi = model.encode_multivec("Hello world")  # (n_tokens, colbert_dim)

        # Cross-encoder reranking
        reranker = CrispEmbed("bge-reranker-v2-m3.gguf")
        score = reranker.rerank("query", "document")  # raw logit

        # Bi-encoder reranking (any embedding model)
        results = model.rerank_biencoder("query", ["doc1", "doc2"], top_n=5)
    """

    def __init__(
        self,
        model_path: str,
        n_threads: int = 4,
        lib_path: Optional[str] = None,
        auto_download: Optional[bool] = None,
    ):
        self._lib = _load_library(lib_path)
        self._setup_signatures()

        resolved = self.resolve_model(model_path, auto_download=auto_download)

        # Init model
        self._ctx = self._lib.crispembed_init(
            resolved.encode("utf-8"), n_threads
        )
        if not self._ctx:
            raise RuntimeError(f"Failed to load model: {resolved}")

    def _setup_signatures(self):
        """Define ctypes function signatures for all C API functions."""
        lib = self._lib

        # --- Lifecycle ---
        lib.crispembed_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.crispembed_init.restype = ctypes.c_void_p

        lib.crispembed_free.argtypes = [ctypes.c_void_p]
        lib.crispembed_free.restype = None

        # --- Configuration ---
        lib.crispembed_set_dim.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.crispembed_set_dim.restype = None

        lib.crispembed_get_hparams.argtypes = [ctypes.c_void_p]
        lib.crispembed_get_hparams.restype = ctypes.POINTER(_CrispEmbedHparams)

        # --- Dense encoding ---
        lib.crispembed_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)
        ]
        lib.crispembed_encode.restype = ctypes.POINTER(ctypes.c_float)

        lib.crispembed_encode_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.crispembed_encode_batch.restype = ctypes.POINTER(ctypes.c_float)

        # --- Capability queries ---
        lib.crispembed_has_sparse.argtypes = [ctypes.c_void_p]
        lib.crispembed_has_sparse.restype = ctypes.c_int

        lib.crispembed_has_colbert.argtypes = [ctypes.c_void_p]
        lib.crispembed_has_colbert.restype = ctypes.c_int

        lib.crispembed_is_reranker.argtypes = [ctypes.c_void_p]
        lib.crispembed_is_reranker.restype = ctypes.c_int

        # --- Sparse encoding ---
        lib.crispembed_encode_sparse.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ]
        lib.crispembed_encode_sparse.restype = ctypes.c_int

        # --- ColBERT multi-vector encoding ---
        lib.crispembed_encode_multivec.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.crispembed_encode_multivec.restype = ctypes.POINTER(ctypes.c_float)

        # --- Reranker ---
        lib.crispembed_rerank.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        lib.crispembed_rerank.restype = ctypes.c_float

        # --- Audio encoding (BidirLM-Omni etc.) ---
        # Symbols may be missing from older builds — guard each lookup.
        if hasattr(lib, "crispembed_has_audio"):
            lib.crispembed_has_audio.argtypes = [ctypes.c_void_p]
            lib.crispembed_has_audio.restype = ctypes.c_int
        if hasattr(lib, "crispembed_encode_audio"):
            lib.crispembed_encode_audio.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.crispembed_encode_audio.restype = ctypes.POINTER(ctypes.c_float)

        # --- Image encoding (BidirLM-Omni vision tower) ---
        if hasattr(lib, "crispembed_has_vision"):
            lib.crispembed_has_vision.argtypes = [ctypes.c_void_p]
            lib.crispembed_has_vision.restype = ctypes.c_int
        if hasattr(lib, "crispembed_encode_image"):
            lib.crispembed_encode_image.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.crispembed_encode_image.restype = ctypes.POINTER(ctypes.c_float)
        if hasattr(lib, "crispembed_encode_image_raw"):
            lib.crispembed_encode_image_raw.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.crispembed_encode_image_raw.restype = ctypes.POINTER(ctypes.c_float)
        if hasattr(lib, "crispembed_encode_text_with_image"):
            lib.crispembed_encode_text_with_image.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.crispembed_encode_text_with_image.restype = ctypes.POINTER(ctypes.c_float)
        if hasattr(lib, "crispembed_encode_with_image_ids"):
            lib.crispembed_encode_with_image_ids.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.crispembed_encode_with_image_ids.restype = ctypes.POINTER(ctypes.c_float)
        if hasattr(lib, "crispembed_encode_image_file"):
            lib.crispembed_encode_image_file.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
            ]
            lib.crispembed_encode_image_file.restype = ctypes.POINTER(ctypes.c_float)
        if hasattr(lib, "crispembed_encode_text_with_image_file"):
            lib.crispembed_encode_text_with_image_file.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.crispembed_encode_text_with_image_file.restype = ctypes.POINTER(ctypes.c_float)
        if hasattr(lib, "crispembed_preprocess_image"):
            lib.crispembed_preprocess_image.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.c_int32 * 3,
            ]
            lib.crispembed_preprocess_image.restype = ctypes.POINTER(ctypes.c_float)
        if hasattr(lib, "crispembed_preprocess_image_rgb"):
            lib.crispembed_preprocess_image_rgb.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.c_int32 * 3,
            ]
            lib.crispembed_preprocess_image_rgb.restype = ctypes.POINTER(ctypes.c_float)

        # --- Prefix ---
        lib.crispembed_set_prefix.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.crispembed_set_prefix.restype = None

        lib.crispembed_get_prefix.argtypes = [ctypes.c_void_p]
        lib.crispembed_get_prefix.restype = ctypes.c_char_p

        # --- Ctx prefix (GGUF metadata) ---
        if hasattr(lib, "crispembed_ctx_query_prefix"):
            lib.crispembed_ctx_query_prefix.argtypes = [ctypes.c_void_p]
            lib.crispembed_ctx_query_prefix.restype = ctypes.c_char_p
        if hasattr(lib, "crispembed_ctx_passage_prefix"):
            lib.crispembed_ctx_passage_prefix.argtypes = [ctypes.c_void_p]
            lib.crispembed_ctx_passage_prefix.restype = ctypes.c_char_p

        lib.crispembed_cache_dir.argtypes = []
        lib.crispembed_cache_dir.restype = ctypes.c_char_p

        lib.crispembed_resolve_model.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.crispembed_resolve_model.restype = ctypes.c_char_p

        lib.crispembed_n_models.argtypes = []
        lib.crispembed_n_models.restype = ctypes.c_int

        lib.crispembed_model_name.argtypes = [ctypes.c_int]
        lib.crispembed_model_name.restype = ctypes.c_char_p

        lib.crispembed_model_desc.argtypes = [ctypes.c_int]
        lib.crispembed_model_desc.restype = ctypes.c_char_p

        lib.crispembed_model_filename.argtypes = [ctypes.c_int]
        lib.crispembed_model_filename.restype = ctypes.c_char_p

        lib.crispembed_model_size.argtypes = [ctypes.c_int]
        lib.crispembed_model_size.restype = ctypes.c_char_p

    # ------------------------------------------------------------------
    # Dense embedding
    # ------------------------------------------------------------------

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

        n = len(texts)

        if n == 1:
            dim = ctypes.c_int(0)
            ptr = self._lib.crispembed_encode(
                self._ctx, texts[0].encode("utf-8"), ctypes.byref(dim)
            )
            if not ptr:
                raise RuntimeError(f"Encoding failed for: {texts[0][:50]}")
            out = np.ctypeslib.as_array(ptr, shape=(dim.value,)).copy()
            return out if single else out.reshape(1, -1)

        c_texts = (ctypes.c_char_p * n)(*(t.encode("utf-8") for t in texts))
        dim = ctypes.c_int(0)
        ptr = self._lib.crispembed_encode_batch(
            self._ctx, c_texts, n, ctypes.byref(dim)
        )
        if not ptr:
            raise RuntimeError("Batch encoding failed")
        out = np.ctypeslib.as_array(ptr, shape=(n * dim.value,)).copy()
        return out.reshape(n, dim.value)

    # ------------------------------------------------------------------
    # Sparse retrieval (BGE-M3 / SPLADE)
    # ------------------------------------------------------------------

    def encode_sparse(self, text: str) -> Dict[int, float]:
        """Encode text to sparse term-weight vector.

        Returns:
            Dict mapping vocab token IDs to positive weights.
            Empty dict if the model has no sparse head.
        """
        out_indices = ctypes.POINTER(ctypes.c_int32)()
        out_values = ctypes.POINTER(ctypes.c_float)()
        n = self._lib.crispembed_encode_sparse(
            self._ctx,
            text.encode("utf-8"),
            ctypes.byref(out_indices),
            ctypes.byref(out_values),
        )
        if n <= 0:
            return {}
        return {int(out_indices[i]): float(out_values[i]) for i in range(n)}

    # ------------------------------------------------------------------
    # ColBERT multi-vector
    # ------------------------------------------------------------------

    def encode_multivec(self, text: str) -> np.ndarray:
        """Encode text to per-token L2-normalized embeddings (ColBERT).

        Returns:
            np.ndarray of shape (n_tokens, colbert_dim).
            Empty array (0, 0) if the model has no ColBERT head.
        """
        n_tokens = ctypes.c_int(0)
        colbert_dim = ctypes.c_int(0)
        ptr = self._lib.crispembed_encode_multivec(
            self._ctx,
            text.encode("utf-8"),
            ctypes.byref(n_tokens),
            ctypes.byref(colbert_dim),
        )
        if not ptr or n_tokens.value <= 0:
            return np.empty((0, 0), dtype=np.float32)
        flat = np.ctypeslib.as_array(
            ptr, shape=(n_tokens.value * colbert_dim.value,)
        ).copy()
        return flat.reshape(n_tokens.value, colbert_dim.value)

    # ------------------------------------------------------------------
    # Audio encoding (BidirLM-Omni etc.)
    # ------------------------------------------------------------------

    def encode_audio(self, pcm: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Encode raw audio into the model's shared embedding space.

        For omnimodal models (BidirLM-Omni) the result is in the same 2048-d
        space as ``encode(text)`` so cosine similarity is meaningful across
        modalities.

        Args:
            pcm: 1-D float32 array of mono PCM samples.
            sr:  Sample rate. Audio is resampled to 16 kHz if needed.

        Returns:
            np.ndarray of shape (output_dim,), L2-normalized.
            Empty array (0,) if the model lacks an audio tower.
        """
        if not hasattr(self._lib, "crispembed_encode_audio"):
            return np.empty((0,), dtype=np.float32)
        a = np.ascontiguousarray(pcm, dtype=np.float32)
        if a.ndim != 1:
            raise ValueError("encode_audio expects 1-D mono PCM")
        if sr != 16000:
            # Light dependency: only required when caller passes non-16k audio.
            try:
                import librosa  # type: ignore
                a = librosa.resample(a, orig_sr=sr, target_sr=16000).astype(np.float32)
            except ImportError as e:
                raise RuntimeError(
                    f"sr={sr} != 16000 and librosa is unavailable ({e}). "
                    "Resample upstream or pip install librosa."
                )
        out_dim = ctypes.c_int(0)
        ptr = self._lib.crispembed_encode_audio(
            self._ctx,
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(a.size),
            ctypes.byref(out_dim),
        )
        if not ptr or out_dim.value <= 0:
            return np.empty((0,), dtype=np.float32)
        vec = np.ctypeslib.as_array(ptr, shape=(out_dim.value,)).copy()
        return vec

    @property
    def has_audio(self) -> bool:
        if not hasattr(self._lib, "crispembed_has_audio"):
            return False
        return bool(self._lib.crispembed_has_audio(self._ctx))

    # ------------------------------------------------------------------
    # Image encoding (BidirLM-Omni vision tower)
    # ------------------------------------------------------------------

    def encode_image(self, image, *, processor=None, model_name: Optional[str] = None) -> np.ndarray:
        """Encode an image into the model's shared embedding space.

        Mean-pools the vision tower output across merged tokens and
        L2-normalizes — yields a single vector cosine-comparable to
        ``encode(text)``.

        Args:
            image: PIL.Image, file path, or numpy array (H,W,3) uint8.
            processor: Optional pre-loaded HF image processor.
            model_name: HF repo id of the image processor (defaults to
              the BidirLM-Omni processor).

        Returns:
            np.ndarray of shape (output_dim,), L2-normalized. Empty if
            the model lacks a vision tower.
        """
        if not hasattr(self._lib, "crispembed_encode_image"):
            return np.empty((0,), dtype=np.float32)
        from .image import preprocess_image  # local import — heavy deps
        kw = {}
        if processor is not None:
            kw["processor"] = processor
        if model_name is not None:
            kw["model_name"] = model_name
        pixel_values, grid_thw = preprocess_image(image, **kw)

        n_patches = int(pixel_values.shape[0])
        n_images = int(grid_thw.shape[0])
        out_dim = ctypes.c_int(0)
        ptr = self._lib.crispembed_encode_image(
            self._ctx,
            pixel_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n_patches),
            grid_thw.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(n_images),
            ctypes.byref(out_dim),
        )
        if not ptr or out_dim.value <= 0:
            return np.empty((0,), dtype=np.float32)
        return np.ctypeslib.as_array(ptr, shape=(out_dim.value,)).copy()

    def encode_image_raw(self, image, *, processor=None, model_name: Optional[str] = None):
        """Run the vision tower without pooling — returns (image_embeds, deepstack_features).

        ``image_embeds``: np.ndarray (n_merged, dim).
        ``deepstack_features``: list of np.ndarrays each (n_merged, dim).

        Used by tests/test_bidirlm_vision.py for parity comparisons against HF.
        """
        if not hasattr(self._lib, "crispembed_encode_image_raw"):
            return np.empty((0, 0), dtype=np.float32), []
        from .image import preprocess_image
        kw = {}
        if processor is not None:
            kw["processor"] = processor
        if model_name is not None:
            kw["model_name"] = model_name
        pixel_values, grid_thw = preprocess_image(image, **kw)

        n_patches = int(pixel_values.shape[0])
        n_images = int(grid_thw.shape[0])
        n_merged = ctypes.c_int(0)
        out_dim = ctypes.c_int(0)
        n_deepstack = ctypes.c_int(0)
        ptr = self._lib.crispembed_encode_image_raw(
            self._ctx,
            pixel_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n_patches),
            grid_thw.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(n_images),
            ctypes.byref(n_merged),
            ctypes.byref(out_dim),
            ctypes.byref(n_deepstack),
        )
        if not ptr or n_merged.value <= 0:
            return np.empty((0, 0), dtype=np.float32), []
        per_slab = n_merged.value * out_dim.value
        total = (1 + n_deepstack.value) * per_slab
        flat = np.ctypeslib.as_array(ptr, shape=(total,)).copy()
        image_embeds = flat[:per_slab].reshape(n_merged.value, out_dim.value)
        deepstack = []
        for k in range(n_deepstack.value):
            beg = (1 + k) * per_slab
            end = beg + per_slab
            deepstack.append(flat[beg:end].reshape(n_merged.value, out_dim.value))
        return image_embeds, deepstack

    def encode_with_image_ids(
        self,
        token_ids,
        pixel_patches: np.ndarray,
        grid_thw: np.ndarray,
    ) -> np.ndarray:
        """Lower-level: image-conditioned embedding from pre-tokenized ids.

        Skips the C++ BPE tokenizer entirely — useful when you need
        byte-identical parity with an external tokenizer (e.g. HF) and
        want to remove tokenizer-round-trip risk from a parity test.

        Args:
            token_ids: 1-D int32 array (or list) of token ids; must contain
                the right number of image_token_id placeholders.
            pixel_patches: float32 (n_patches, 1536) — output of
                ``crispembed.image.preprocess_image``.
            grid_thw: int32 (n_images, 3) with rows ``(t, h_patches, w_patches)``.

        Returns:
            np.ndarray of shape (output_dim,), L2-normalized. Empty on
            placeholder/dim mismatch.
        """
        if not hasattr(self._lib, "crispembed_encode_with_image_ids"):
            return np.empty((0,), dtype=np.float32)
        ids = np.ascontiguousarray(token_ids, dtype=np.int32).reshape(-1)
        pv = np.ascontiguousarray(pixel_patches, dtype=np.float32)
        gt = np.ascontiguousarray(grid_thw, dtype=np.int32)
        out_dim = ctypes.c_int(0)
        ptr = self._lib.crispembed_encode_with_image_ids(
            self._ctx,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(int(ids.size)),
            pv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(int(pv.shape[0])),
            gt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(int(gt.shape[0])),
            ctypes.byref(out_dim),
        )
        if not ptr or out_dim.value <= 0:
            return np.empty((0,), dtype=np.float32)
        return np.ctypeslib.as_array(ptr, shape=(out_dim.value,)).copy()

    def encode_image_file(self, path: str) -> np.ndarray:
        """In-process image embedding (no `transformers` dependency).

        Loads and preprocesses the image with CrispEmbed's C++ pipeline
        (smart_resize + Catmull-Rom bicubic + OpenAI CLIP normalize +
        Qwen2VL patchify), then runs the vision tower. Empirical cosine
        against the HF processor is ≈ 0.97 on real photographs (the
        residual gap is sub-pixel resize divergence). For tight HF parity
        use ``encode_image()`` (which calls the Python preprocessor).
        """
        if not hasattr(self._lib, "crispembed_encode_image_file"):
            return np.empty((0,), dtype=np.float32)
        out_dim = ctypes.c_int(0)
        ptr = self._lib.crispembed_encode_image_file(
            self._ctx, path.encode("utf-8"), ctypes.byref(out_dim))
        if not ptr or out_dim.value <= 0:
            return np.empty((0,), dtype=np.float32)
        return np.ctypeslib.as_array(ptr, shape=(out_dim.value,)).copy()

    def encode_text_with_image_file(self, text: str, path: str) -> np.ndarray:
        """In-process image-conditioned text embedding (no `transformers`)."""
        if not hasattr(self._lib, "crispembed_encode_text_with_image_file"):
            return np.empty((0,), dtype=np.float32)
        out_dim = ctypes.c_int(0)
        ptr = self._lib.crispembed_encode_text_with_image_file(
            self._ctx, text.encode("utf-8"), path.encode("utf-8"),
            ctypes.byref(out_dim))
        if not ptr or out_dim.value <= 0:
            return np.empty((0,), dtype=np.float32)
        return np.ctypeslib.as_array(ptr, shape=(out_dim.value,)).copy()

    def preprocess_image_file(self, path: str):
        """Run CrispEmbed's C++ image preprocessor and return (pixel_patches, grid_thw).

        Returns:
            pixel_patches: float32 (n_patches, 1536)
            grid_thw:      int32 (1, 3) with row (t, h_patches, w_patches)
        """
        if not hasattr(self._lib, "crispembed_preprocess_image"):
            return np.empty((0, 0), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
        n_p = ctypes.c_int(0); rd = ctypes.c_int(0)
        grid = (ctypes.c_int32 * 3)(0, 0, 0)
        ptr = self._lib.crispembed_preprocess_image(
            self._ctx, path.encode("utf-8"),
            ctypes.byref(n_p), ctypes.byref(rd), grid)
        if not ptr or n_p.value <= 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0, 3), dtype=np.int32)
        pv = np.ctypeslib.as_array(ptr, shape=(n_p.value, rd.value)).copy()
        gt = np.asarray([[grid[0], grid[1], grid[2]]], dtype=np.int32)
        return pv, gt

    def encode_text_with_image(
        self,
        text: str,
        image,
        *,
        processor=None,
        model_name: Optional[str] = None,
    ) -> np.ndarray:
        """Encode text conditioned on an image (BidirLM-Omni DeepStack).

        The decoder runs over `text` with the vision tower's image_embeds
        spliced into the token embeddings at every image_token_id placeholder
        and DeepStack features added at the first 3 layers. Result is L2-
        normalized in the model's shared embedding space.

        `text` must contain the right number of image-pad placeholder tokens
        (e.g. ``"<|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>...""``)
        — typically you build it via the HF chat template.

        Args:
            text: Prompt with image-pad placeholders.
            image: PIL.Image, file path, or numpy array (H,W,3) uint8.
            processor: Optional pre-loaded HF image processor.
            model_name: HF repo id of the image processor (defaults to
              the BidirLM-Omni processor).

        Returns:
            np.ndarray of shape (output_dim,), L2-normalized. Empty if
            the model lacks a vision tower or placeholders don't match.
        """
        if not hasattr(self._lib, "crispembed_encode_text_with_image"):
            return np.empty((0,), dtype=np.float32)
        from .image import preprocess_image
        kw = {}
        if processor is not None:
            kw["processor"] = processor
        if model_name is not None:
            kw["model_name"] = model_name
        pixel_values, grid_thw = preprocess_image(image, **kw)

        n_patches = int(pixel_values.shape[0])
        n_images = int(grid_thw.shape[0])
        out_dim = ctypes.c_int(0)
        ptr = self._lib.crispembed_encode_text_with_image(
            self._ctx, text.encode("utf-8"),
            pixel_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n_patches),
            grid_thw.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(n_images),
            ctypes.byref(out_dim),
        )
        if not ptr or out_dim.value <= 0:
            return np.empty((0,), dtype=np.float32)
        return np.ctypeslib.as_array(ptr, shape=(out_dim.value,)).copy()

    @property
    def has_vision(self) -> bool:
        if not hasattr(self._lib, "crispembed_has_vision"):
            return False
        return bool(self._lib.crispembed_has_vision(self._ctx))

    # ------------------------------------------------------------------
    # Cross-encoder reranking
    # ------------------------------------------------------------------

    def rerank(self, query: str, document: str) -> float:
        """Score a (query, document) pair with a cross-encoder.

        Args:
            query: Query text.
            document: Document text.

        Returns:
            Raw logit score (higher = more relevant).

        Raises:
            RuntimeError: If the model is not a reranker.
        """
        if not self.is_reranker:
            raise RuntimeError("Model is not a reranker (no classifier head)")
        return float(self._lib.crispembed_rerank(
            self._ctx,
            query.encode("utf-8"),
            document.encode("utf-8"),
        ))

    # ------------------------------------------------------------------
    # Bi-encoder reranking (cosine similarity of embeddings)
    # ------------------------------------------------------------------

    def rerank_biencoder(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: bool = True,
    ) -> List[Dict]:
        """Rank documents by cosine similarity to query embedding.

        Encodes query and all documents, computes dot product of
        L2-normalized embeddings (= cosine similarity), returns
        results sorted by score descending.

        Args:
            query: Query text.
            documents: List of document texts.
            top_n: Return only top N results (None = all).
            return_documents: Include document text in results.

        Returns:
            List of dicts with keys: index, score, document (optional).
        """
        all_texts = [query] + list(documents)
        embeddings = self.encode(all_texts)
        query_vec = embeddings[0]
        doc_vecs = embeddings[1:]

        scores = doc_vecs @ query_vec  # dot product of L2-normalized = cosine sim

        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        if top_n is not None:
            ranked = ranked[:top_n]

        results = []
        for idx, score in ranked:
            entry = {"index": idx, "score": float(score)}
            if return_documents:
                entry["document"] = documents[idx]
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_dim(self, dim: int) -> None:
        """Set Matryoshka output dimension.

        The embedding is truncated and re-normalized to the specified
        dimension. Set to 0 to restore the model's native dimension.

        Args:
            dim: Target dimension (must be <= model's native dimension).
        """
        self._lib.crispembed_set_dim(self._ctx, dim)

    def set_prefix(self, prefix: Optional[str] = None) -> None:
        """Set a text prefix prepended to all inputs before tokenization.

        Typical values:
            "query: "                   (E5, Jina v5)
            "search_query: "            (Nomic, for queries)
            "search_document: "         (Nomic, for documents)
            "Represent this sentence for searching relevant passages: " (BGE)

        Pass None or "" to clear.
        """
        raw = prefix.encode("utf-8") if prefix else b""
        self._lib.crispembed_set_prefix(self._ctx, raw)

    @property
    def prefix(self) -> str:
        """Current text prefix (empty string if none)."""
        raw = self._lib.crispembed_get_prefix(self._ctx)
        return raw.decode("utf-8") if raw else ""

    @property
    def ctx_query_prefix(self) -> Optional[str]:
        """Query prefix from GGUF metadata (colbert.query_prefix), or None."""
        if not hasattr(self._lib, "crispembed_ctx_query_prefix"):
            return None
        p = self._lib.crispembed_ctx_query_prefix(self._ctx)
        return p.decode("utf-8") if p else None

    @property
    def ctx_passage_prefix(self) -> Optional[str]:
        """Passage/document prefix from GGUF metadata, or None."""
        if not hasattr(self._lib, "crispembed_ctx_passage_prefix"):
            return None
        p = self._lib.crispembed_ctx_passage_prefix(self._ctx)
        return p.decode("utf-8") if p else None

    # ------------------------------------------------------------------
    # LoRA adapter hot-swap
    # ------------------------------------------------------------------

    def set_lora(self, adapter_name: Optional[str] = None) -> bool:
        """Switch LoRA adapter at runtime (decoder models with per-task LoRA).

        Args:
            adapter_name: Adapter name (e.g. ``"retrieval"``, ``"classification"``).
                          Pass ``None`` or ``""`` to deactivate (restore base weights).

        Returns:
            True on success, False on failure (no such adapter, not a decoder,
            model has no LoRA).
        """
        if not hasattr(self._lib, "crispembed_set_lora"):
            return False
        self._lib.crispembed_set_lora.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispembed_set_lora.restype = ctypes.c_int
        raw = adapter_name.encode("utf-8") if adapter_name else b""
        return bool(self._lib.crispembed_set_lora(self._ctx, raw))

    @property
    def lora(self) -> str:
        """Currently active LoRA adapter name (empty if none)."""
        if not hasattr(self._lib, "crispembed_get_lora"):
            return ""
        self._lib.crispembed_get_lora.argtypes = [ctypes.c_void_p]
        self._lib.crispembed_get_lora.restype = ctypes.c_char_p
        raw = self._lib.crispembed_get_lora(self._ctx)
        return raw.decode("utf-8") if raw else ""

    def list_lora(self) -> List[str]:
        """List available LoRA adapters in this model.

        Returns:
            List of adapter name strings. Empty if model has no LoRA.
        """
        if not hasattr(self._lib, "crispembed_list_lora"):
            return []
        self._lib.crispembed_list_lora.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.crispembed_list_lora.restype = ctypes.c_int
        names_ptr = ctypes.POINTER(ctypes.c_char_p)()
        count = ctypes.c_int(0)
        ok = self._lib.crispembed_list_lora(self._ctx, ctypes.byref(names_ptr), ctypes.byref(count))
        if not ok or count.value <= 0:
            return []
        return [names_ptr[i].decode("utf-8") for i in range(count.value)]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        hp = self._lib.crispembed_get_hparams(self._ctx)
        if not hp:
            return 0
        return int(hp.contents.n_output or hp.contents.n_embd)

    @property
    def has_sparse(self) -> bool:
        """True if the model has a sparse projection head (BGE-M3/SPLADE)."""
        return bool(self._lib.crispembed_has_sparse(self._ctx))

    @property
    def has_colbert(self) -> bool:
        """True if the model has a ColBERT projection head."""
        return bool(self._lib.crispembed_has_colbert(self._ctx))

    @property
    def is_reranker(self) -> bool:
        """True if the model is a cross-encoder reranker."""
        return bool(self._lib.crispembed_is_reranker(self._ctx))

    @staticmethod
    def cache_dir(lib_path: Optional[str] = None) -> str:
        lib = _load_library(lib_path)
        lib.crispembed_cache_dir.argtypes = []
        lib.crispembed_cache_dir.restype = ctypes.c_char_p
        raw = lib.crispembed_cache_dir()
        return raw.decode("utf-8") if raw else ""

    @staticmethod
    def resolve_model(
        model_path: str,
        auto_download: Optional[bool] = None,
        lib_path: Optional[str] = None,
    ) -> str:
        lib = _load_library(lib_path)
        lib.crispembed_resolve_model.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.crispembed_resolve_model.restype = ctypes.c_char_p
        if auto_download is None:
            auto_download = (
                ".gguf" not in model_path and "/" not in model_path and "\\" not in model_path
            )
        raw = lib.crispembed_resolve_model(model_path.encode("utf-8"), int(auto_download))
        resolved = raw.decode("utf-8") if raw else ""
        if not resolved:
            raise RuntimeError(f"Could not resolve model: {model_path}")
        return resolved

    @staticmethod
    def list_models(lib_path: Optional[str] = None) -> List[Dict[str, str]]:
        """List supported models with descriptions.

        Returns a list of dicts with keys: name, desc, filename, size.
        """
        lib = _load_library(lib_path)
        lib.crispembed_n_models.argtypes = []
        lib.crispembed_n_models.restype = ctypes.c_int
        lib.crispembed_model_name.argtypes = [ctypes.c_int]
        lib.crispembed_model_name.restype = ctypes.c_char_p
        lib.crispembed_model_desc.argtypes = [ctypes.c_int]
        lib.crispembed_model_desc.restype = ctypes.c_char_p
        lib.crispembed_model_filename.argtypes = [ctypes.c_int]
        lib.crispembed_model_filename.restype = ctypes.c_char_p
        lib.crispembed_model_size.argtypes = [ctypes.c_int]
        lib.crispembed_model_size.restype = ctypes.c_char_p

        models = []
        for i in range(lib.crispembed_n_models()):
            models.append({
                "name": (lib.crispembed_model_name(i) or b"").decode("utf-8"),
                "desc": (lib.crispembed_model_desc(i) or b"").decode("utf-8"),
                "filename": (lib.crispembed_model_filename(i) or b"").decode("utf-8"),
                "size": (lib.crispembed_model_size(i) or b"").decode("utf-8"),
            })
        return models

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.crispembed_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# Face pipeline ctypes structures
# ---------------------------------------------------------------------------

class _FaceDetection(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("w", ctypes.c_float),
        ("h", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("landmarks", ctypes.c_float * 10),
    ]


class _FaceResult(ctypes.Structure):
    _fields_ = [
        ("det", _FaceDetection),
        ("embedding", ctypes.POINTER(ctypes.c_float)),
        ("embedding_dim", ctypes.c_int),
    ]


def _setup_face_signatures(lib):
    """Register ctypes signatures for all crispembed_face_* functions."""
    lib.crispembed_face_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_face_init.restype = ctypes.c_void_p

    lib.crispembed_face_dim.argtypes = [ctypes.c_void_p]
    lib.crispembed_face_dim.restype = ctypes.c_int

    lib.crispembed_face_type.argtypes = [ctypes.c_void_p]
    lib.crispembed_face_type.restype = ctypes.c_char_p

    lib.crispembed_detect_faces.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_detect_faces.restype = ctypes.POINTER(_FaceDetection)

    lib.crispembed_encode_face.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_encode_face.restype = ctypes.POINTER(ctypes.c_float)

    lib.crispembed_face_pipeline.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_face_pipeline.restype = ctypes.POINTER(_FaceResult)

    lib.crispembed_face_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_face_free.restype = None


def _det_to_dict(det: _FaceDetection) -> dict:
    """Convert a _FaceDetection struct to a plain Python dict."""
    return {
        "x": float(det.x),
        "y": float(det.y),
        "w": float(det.w),
        "h": float(det.h),
        "confidence": float(det.confidence),
        "landmarks": [float(det.landmarks[i]) for i in range(10)],
    }


# ---------------------------------------------------------------------------
# CrispFace — detection + recognition
# ---------------------------------------------------------------------------

class CrispFace:
    """Face detection and recognition via the crispembed_face_* C API.

    A single instance can be used for either detection (SCRFD-style models)
    or recognition (SFace / ArcFace-style models) depending on what
    ``model_path`` points to.

    Usage::

        # Detection
        det = CrispFace("scrfd_10g.gguf")
        faces = det.detect("photo.jpg", conf=0.5)
        # [{"x": ..., "y": ..., "w": ..., "h": ..., "confidence": ..., "landmarks": [...]}]

        # Recognition
        rec = CrispFace("sface.gguf")
        emb = rec.encode("photo.jpg", landmarks=faces[0]["landmarks"])
        # np.ndarray of shape (rec.dim,)
    """

    def __init__(
        self,
        model_path: str,
        n_threads: int = 4,
        lib_path: Optional[str] = None,
    ):
        self._lib = _load_library(lib_path)
        _setup_face_signatures(self._lib)
        self._ctx = self._lib.crispembed_face_init(
            model_path.encode("utf-8"), n_threads
        )
        if not self._ctx:
            raise RuntimeError(f"Failed to load face model: {model_path}")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, image_path: str, conf: float = 0.5, det_size: int = 0) -> List[dict]:
        """Detect faces in *image_path*.

        Args:
            image_path: Path to an image file.
            conf:       Minimum confidence threshold (0–1).
            det_size:   Detection input resolution (0 = default 640).

        Returns:
            List of dicts with keys ``x, y, w, h, confidence, landmarks``.
            ``landmarks`` is a flat list of 10 floats
            (five (x, y) keypoint pairs: left-eye, right-eye, nose,
            left-mouth, right-mouth).
        """
        n_faces = ctypes.c_int(0)
        ptr = self._lib.crispembed_detect_faces(
            self._ctx,
            image_path.encode("utf-8"),
            ctypes.c_float(conf),
            ctypes.c_int(det_size),
            ctypes.byref(n_faces),
        )
        if not ptr or n_faces.value <= 0:
            return []
        return [_det_to_dict(ptr[i]) for i in range(n_faces.value)]

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def encode(self, image_path: str, landmarks) -> np.ndarray:
        """Encode a face crop into an embedding vector.

        Args:
            image_path: Path to the image containing the face.
            landmarks:  10 floats (5 keypoints × 2 coords) used to align
                        the crop before embedding.  Pass
                        ``face["landmarks"]`` from :meth:`detect`.

        Returns:
            np.ndarray of shape ``(dim,)``, L2-normalized.
        """
        lm = list(landmarks)
        if len(lm) != 10:
            raise ValueError(f"landmarks must have exactly 10 values, got {len(lm)}")
        c_lm = (ctypes.c_float * 10)(*lm)
        out_dim = ctypes.c_int(0)
        ptr = self._lib.crispembed_encode_face(
            self._ctx,
            image_path.encode("utf-8"),
            c_lm,
            ctypes.byref(out_dim),
        )
        if not ptr or out_dim.value <= 0:
            return np.empty((0,), dtype=np.float32)
        return np.ctypeslib.as_array(ptr, shape=(out_dim.value,)).copy()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        name: str,
        n_threads: int = 4,
        lib_path: Optional[str] = None,
    ) -> "CrispFace":
        """Create a CrispFace from a registry name (auto-downloads if needed).

        Args:
            name:      Registry model name (e.g. ``"yunet"``, ``"scrfd-det-10g"``,
                       ``"auraface-v1"``, ``"sface"``).
            n_threads: CPU threads for inference.
            lib_path:  Optional path to the shared library.

        Returns:
            A :class:`CrispFace` instance backed by the resolved GGUF.
        """
        resolved = CrispEmbed.resolve_model(name, auto_download=True, lib_path=lib_path)
        return cls(resolved, n_threads=n_threads, lib_path=lib_path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Embedding dimension of this model (0 for pure detectors)."""
        return int(self._lib.crispembed_face_dim(self._ctx))

    @property
    def model_type(self) -> str:
        """Model type string returned by the C library (e.g. ``"scrfd"`` or ``"sface"``)."""
        raw = self._lib.crispembed_face_type(self._ctx)
        return raw.decode("utf-8") if raw else ""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.crispembed_face_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# CrispFacePipeline — joint detect + embed in one call
# ---------------------------------------------------------------------------

class CrispFacePipeline:
    """End-to-end face pipeline: detect all faces then embed each one.

    Wraps ``crispembed_face_pipeline`` which runs detection and recognition
    in a single C call, avoiding redundant image decoding.

    Usage::

        pipe = CrispFacePipeline("scrfd_10g.gguf", "sface.gguf")
        results = pipe.run("photo.jpg", conf=0.5)
        # [{"det": {...}, "embedding": np.ndarray}, ...]

        sim = pipe.match(results[0]["embedding"], results[1]["embedding"])
        # float in [-1, 1]
    """

    def __init__(
        self,
        det_model: str,
        rec_model: str,
        n_threads: int = 4,
        lib_path: Optional[str] = None,
    ):
        self._lib = _load_library(lib_path)
        _setup_face_signatures(self._lib)

        self._det_ctx = self._lib.crispembed_face_init(
            det_model.encode("utf-8"), n_threads
        )
        if not self._det_ctx:
            raise RuntimeError(f"Failed to load face detector: {det_model}")

        self._rec_ctx = self._lib.crispembed_face_init(
            rec_model.encode("utf-8"), n_threads
        )
        if not self._rec_ctx:
            self._lib.crispembed_face_free(self._det_ctx)
            self._det_ctx = None
            raise RuntimeError(f"Failed to load face recognizer: {rec_model}")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        det_name: str = "yunet",
        rec_name: str = "auraface-v1",
        n_threads: int = 4,
        lib_path: Optional[str] = None,
    ) -> "CrispFacePipeline":
        """Create a CrispFacePipeline from registry names (auto-downloads).

        Args:
            det_name:  Detection model registry name (default ``"yunet"``).
            rec_name:  Recognition model registry name (default ``"auraface-v1"``).
            n_threads: CPU threads for inference.
            lib_path:  Optional path to the shared library.

        Returns:
            A :class:`CrispFacePipeline` instance.
        """
        det_path = CrispEmbed.resolve_model(det_name, auto_download=True, lib_path=lib_path)
        rec_path = CrispEmbed.resolve_model(rec_name, auto_download=True, lib_path=lib_path)
        return cls(det_path, rec_path, n_threads=n_threads, lib_path=lib_path)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(self, image_path: str, conf: float = 0.5, det_size: int = 0) -> List[dict]:
        """Detect and embed all faces in *image_path*.

        Args:
            image_path: Path to an image file.
            conf:       Minimum detection confidence (0–1).
            det_size:   Detection input resolution (0 = default 640).

        Returns:
            List of dicts with keys:

            * ``"det"`` — detection dict (``x, y, w, h, confidence,
              landmarks``).
            * ``"embedding"`` — ``np.ndarray`` of shape ``(rec_dim,)``,
              L2-normalized face embedding.
        """
        n_faces = ctypes.c_int(0)
        ptr = self._lib.crispembed_face_pipeline(
            self._det_ctx,
            self._rec_ctx,
            image_path.encode("utf-8"),
            ctypes.c_float(conf),
            ctypes.c_int(det_size),
            ctypes.byref(n_faces),
        )
        if not ptr or n_faces.value <= 0:
            return []

        results = []
        for i in range(n_faces.value):
            fr = ptr[i]
            det = _det_to_dict(fr.det)
            if fr.embedding and fr.embedding_dim > 0:
                emb = np.ctypeslib.as_array(
                    fr.embedding, shape=(fr.embedding_dim,)
                ).copy()
            else:
                emb = np.empty((0,), dtype=np.float32)
            results.append({"det": det, "embedding": emb})
        return results

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    @staticmethod
    def match(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two face embeddings.

        Both embeddings are expected to be L2-normalized (as returned by
        :meth:`run`).  The dot product therefore equals the cosine
        similarity directly.

        Args:
            emb1: First face embedding, shape ``(dim,)``.
            emb2: Second face embedding, shape ``(dim,)``.

        Returns:
            float in ``[-1, 1]``.  Values above ~0.3 typically indicate
            the same person (threshold is model-dependent).
        """
        a = np.asarray(emb1, dtype=np.float32).ravel()
        b = np.asarray(emb2, dtype=np.float32).ravel()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        if hasattr(self, "_det_ctx") and self._det_ctx:
            self._lib.crispembed_face_free(self._det_ctx)
            self._det_ctx = None
        if hasattr(self, "_rec_ctx") and self._rec_ctx:
            self._lib.crispembed_face_free(self._rec_ctx)
            self._rec_ctx = None


# ---------------------------------------------------------------------------
# CrispVit — standalone ViT image embedding (SigLIP, CLIP)
# ---------------------------------------------------------------------------

def _setup_vit_signatures(lib):
    """Register ctypes signatures for all crispembed_vit_* functions."""
    lib.crispembed_vit_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_vit_init.restype = ctypes.c_void_p

    lib.crispembed_vit_dim.argtypes = [ctypes.c_void_p]
    lib.crispembed_vit_dim.restype = ctypes.c_int

    lib.crispembed_vit_encode_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.crispembed_vit_encode_file.restype = ctypes.POINTER(ctypes.c_float)

    lib.crispembed_vit_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_vit_free.restype = None


class CrispVit:
    """Standalone ViT image embedding (SigLIP, CLIP).

    Loads a GGUF ViT model and encodes image files to embedding vectors.
    Handles resize and normalization internally using the model's stored
    image_mean / image_std metadata.

    Usage::

        vit = CrispVit("siglip-base.gguf")
        emb = vit.encode_file("photo.jpg")
        # np.ndarray of shape (vit.dim,)
    """

    def __init__(self, model_path: str, n_threads: int = 0, lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_vit_signatures(self._lib)
        self._ctx = self._lib.crispembed_vit_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load ViT model: {model_path}")

    @property
    def dim(self) -> int:
        """Embedding dimension of this ViT model."""
        return self._lib.crispembed_vit_dim(self._ctx)

    def encode_file(self, image_path: str) -> np.ndarray:
        """Encode an image file to an embedding vector.

        Loads the image from disk (JPG/PNG/BMP), resizes it to the model's
        expected resolution, normalizes with per-channel mean/std, runs the
        full ViT forward pass, and returns the resulting embedding.

        Args:
            image_path: Path to the image file.

        Returns:
            np.ndarray of shape ``(dim,)``.  Empty array on failure.
        """
        out_dim = ctypes.c_int(0)
        ptr = self._lib.crispembed_vit_encode_file(
            self._ctx, image_path.encode("utf-8"), ctypes.byref(out_dim))
        if not ptr or out_dim.value <= 0:
            return np.array([], dtype=np.float32)
        return np.ctypeslib.as_array(ptr, shape=(out_dim.value,)).copy()

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_vit_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# CrispClipText — CLIP text encoding
# ---------------------------------------------------------------------------

def _setup_clip_text_signatures(lib):
    """Register ctypes signatures for crispembed_clip_text_* functions."""
    lib.crispembed_clip_text_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_clip_text_init.restype = ctypes.c_void_p

    lib.crispembed_clip_text_dim.argtypes = [ctypes.c_void_p]
    lib.crispembed_clip_text_dim.restype = ctypes.c_int

    lib.crispembed_clip_text_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.crispembed_clip_text_encode.restype = ctypes.POINTER(ctypes.c_float)

    lib.crispembed_clip_text_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_clip_text_free.restype = None


class CrispClipText:
    """CLIP text encoder for cross-modal text-image retrieval.

    Loads a CLIP text GGUF and encodes text strings to embedding vectors
    in the same space as CLIP vision embeddings. BPE tokenizer is embedded
    in the GGUF file.

    Usage::

        enc = CrispClipText("clip-text-base.gguf")
        emb = enc.encode("a photo of a cat")
        # np.ndarray of shape (512,), L2-normalized
    """

    def __init__(self, model_path: str, n_threads: int = 0, lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_clip_text_signatures(self._lib)
        self._ctx = self._lib.crispembed_clip_text_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load CLIP text model: {model_path}")

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._lib.crispembed_clip_text_dim(self._ctx)

    def encode(self, text: str) -> np.ndarray:
        """Encode text to a CLIP embedding vector.

        Args:
            text: Input text string.

        Returns:
            np.ndarray of shape ``(dim,)``, L2-normalized. Empty on failure.
        """
        out_dim = ctypes.c_int(0)
        ptr = self._lib.crispembed_clip_text_encode(
            self._ctx, text.encode("utf-8"), ctypes.byref(out_dim))
        if not ptr or out_dim.value <= 0:
            return np.array([], dtype=np.float32)
        return np.ctypeslib.as_array(ptr, shape=(out_dim.value,)).copy()

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_clip_text_free(self._ctx)
            self._ctx = None


def _setup_math_ocr_signatures(lib):
    lib.crispembed_math_ocr_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_math_ocr_init.restype = ctypes.c_void_p

    lib.crispembed_math_ocr_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_math_ocr_free.restype = None

    lib.crispembed_math_ocr_recognize.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_math_ocr_recognize.restype = ctypes.c_char_p

    lib.crispembed_math_ocr_recognize_gray.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_math_ocr_recognize_gray.restype = ctypes.c_char_p


class CrispMathOcr:
    """Math/document OCR — recognizes LaTeX or text from images.

    Supports pix2tex (printed math), PP-FormulaNet (printed math),
    PP-FormulaNet-L (printed math, best), Texo-Distill (printed math, small),
    HMER (handwritten math), BTTR (handwritten math), PosFormer (handwritten),
    MixTex (Chinese+English LaTeX), PARSeq (scene text),
    Qwen2.5-VL (VLM, German docs), InternVL2.5/2-1B (VLM, EN+DE),
    GLM-OCR (VLM, 8 langs, OmniDocBench #1).
    Auto-detects architecture from GGUF metadata.

    Usage::

        # Math formula OCR
        ocr = CrispMathOcr("ppformulanet-l-q8_0.gguf")
        latex = ocr.recognize("formula.png")
        # "\\frac{a}{b}"

        # Document OCR (VLM)
        ocr = CrispMathOcr("internvl2.5-2b-q4_k.gguf")
        text = ocr.recognize("document.png")
    """

    def __init__(self, model_path: str, n_threads: int = 4, lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_math_ocr_signatures(self._lib)
        self._ctx = self._lib.crispembed_math_ocr_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load math OCR model: {model_path}")

    def recognize(self, image) -> str:
        """Recognize LaTeX from an image.

        Args:
            image: File path (str/Path), PIL.Image, or numpy array (H, W, C) uint8.

        Returns:
            Recognized LaTeX string.
        """
        if isinstance(image, (str, Path)):
            from PIL import Image
            image = np.array(Image.open(str(image)).convert("RGB"))
        elif hasattr(image, 'convert'):  # PIL Image
            image = np.array(image.convert("RGB"))
        arr = np.ascontiguousarray(image, dtype=np.uint8)
        h, w = arr.shape[:2]
        ch = arr.shape[2] if arr.ndim == 3 else 1
        out_len = ctypes.c_int(0)
        result = self._lib.crispembed_math_ocr_recognize(
            self._ctx,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(w), ctypes.c_int(h), ctypes.c_int(ch),
            ctypes.byref(out_len),
        )
        return result.decode("utf-8") if result else ""

    def recognize_gray(self, pixels: np.ndarray) -> str:
        """Recognize LaTeX from a float32 grayscale image.

        Args:
            pixels: numpy array (H, W) with values in [0, 1].

        Returns:
            Recognized LaTeX string.
        """
        arr = np.ascontiguousarray(pixels, dtype=np.float32)
        h, w = arr.shape[:2]
        out_len = ctypes.c_int(0)
        result = self._lib.crispembed_math_ocr_recognize_gray(
            self._ctx,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(w), ctypes.c_int(h),
            ctypes.byref(out_len),
        )
        return result.decode("utf-8") if result else ""

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_math_ocr_free(self._ctx)
            self._ctx = None


class _LayoutRegion(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("score", ctypes.c_float),
        ("label", ctypes.c_int),
        ("label_name", ctypes.c_char_p),
    ]


def _setup_layout_signatures(lib):
    lib.crispembed_layout_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_layout_init.restype = ctypes.c_void_p

    lib.crispembed_layout_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_layout_free.restype = None

    lib.crispembed_layout_detect.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_layout_detect.restype = ctypes.POINTER(_LayoutRegion)


class CrispLayout:
    """Document layout detection via RT-DETRv2.

    Detects 17 region classes: text, title, table, figure, formula, caption,
    section_header, list_item, footnote, page_header, page_footer, code,
    document_index, checkbox_selected, checkbox_unselected, form, key_value_region.

    Usage::

        layout = CrispLayout("rt-detrv2-layout-q8_0.gguf")
        regions = layout.detect("page.png")
        for r in regions:
            print(f"{r['label']} ({r['score']:.2f}): [{r['x1']:.0f},{r['y1']:.0f},{r['x2']:.0f},{r['y2']:.0f}]")
    """

    def __init__(self, model_path: str, n_threads: int = 4, lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_layout_signatures(self._lib)
        self._ctx = self._lib.crispembed_layout_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load layout model: {model_path}")

    def detect(self, image_path: str, threshold: float = 0.3) -> list:
        """Detect layout regions in an image file.

        Args:
            image_path: Path to image file (JPG/PNG).
            threshold: Minimum confidence score (default 0.3).

        Returns:
            List of dicts with keys: label, score, x1, y1, x2, y2.
        """
        n = ctypes.c_int(0)
        ptr = self._lib.crispembed_layout_detect(
            self._ctx, str(image_path).encode("utf-8"),
            ctypes.c_float(threshold), ctypes.byref(n))
        results = []
        for i in range(n.value):
            r = ptr[i]
            results.append({
                "label": r.label_name.decode("utf-8") if r.label_name else "",
                "score": r.score,
                "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
            })
        return results

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_layout_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# General OCR Pipeline (text detection + recognition)
# ---------------------------------------------------------------------------

class _CrispOcrResult(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("w", ctypes.c_float),
        ("h", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("text", ctypes.c_char_p),
        ("text_len", ctypes.c_int),
    ]


def _setup_ocr_pipeline_signatures(lib):
    lib.crispembed_ocr_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_ocr_init.restype = ctypes.c_void_p

    lib.crispembed_ocr_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_ocr_free.restype = None

    lib.crispembed_ocr.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_ocr.restype = ctypes.POINTER(_CrispOcrResult)

    lib.crispembed_ocr_recognize.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_ocr_recognize.restype = ctypes.c_char_p


class CrispOcrPipeline:
    """General OCR pipeline — text detection (DBNet) + recognition (TrOCR).

    Detects text regions in a document image, then recognizes each crop.

    Usage::

        ocr = CrispOcrPipeline("dbnet-det", "trocr-printed")
        results = ocr.run("document.png")
        for r in results:
            print(f"[{r['confidence']:.2f}] ({r['x']:.0f},{r['y']:.0f}) \"{r['text']}\"")
    """

    def __init__(self, det_model: str, rec_model: str, n_threads: int = 4,
                 lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_ocr_pipeline_signatures(self._lib)
        self._ctx = self._lib.crispembed_ocr_init(
            det_model.encode("utf-8"), rec_model.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load OCR pipeline: {det_model} + {rec_model}")

    def run(self, image_path: str) -> list:
        """Detect and recognize text in an image.

        Args:
            image_path: Path to image file (JPG/PNG).

        Returns:
            List of dicts with keys: text, x, y, w, h, confidence.
        """
        n = ctypes.c_int(0)
        ptr = self._lib.crispembed_ocr(
            self._ctx, str(image_path).encode("utf-8"), ctypes.byref(n))
        results = []
        for i in range(n.value):
            r = ptr[i]
            results.append({
                "text": r.text.decode("utf-8") if r.text else "",
                "x": r.x, "y": r.y, "w": r.w, "h": r.h,
                "confidence": r.confidence,
            })
        return results

    def recognize(self, image_path: str) -> str:
        """Recognize text from a single image crop (no detection).

        Args:
            image_path: Path to a cropped text region image.

        Returns:
            Recognized text string.
        """
        out_len = ctypes.c_int(0)
        result = self._lib.crispembed_ocr_recognize(
            self._ctx, str(image_path).encode("utf-8"), ctypes.byref(out_len))
        return result.decode("utf-8") if result else ""

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_ocr_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# Surya Text Detection (EfficientViT segformer, 91 languages)
# ---------------------------------------------------------------------------

class _TextDetResult(ctypes.Structure):
    _fields_ = [
        ("x0", ctypes.c_float),
        ("y0", ctypes.c_float),
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("confidence", ctypes.c_float),
    ]


def _setup_text_det_signatures(lib):
    lib.crispembed_text_det_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_text_det_init.restype = ctypes.c_void_p

    lib.crispembed_text_det_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_text_det_free.restype = None

    lib.crispembed_text_det.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_text_det.restype = ctypes.POINTER(_TextDetResult)

    lib.crispembed_text_det_heatmap.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_text_det_heatmap.restype = ctypes.POINTER(ctypes.c_float)


class CrispTextDetect:
    """Surya text line detection (EfficientViT segformer, 91 languages).

    Segmentation-based text line detection. Returns bounding boxes with
    confidence scores.

    Usage::

        det = CrispTextDetect("surya-det-f16.gguf")
        boxes = det.detect("page.png")
        for b in boxes:
            print(f"({b['x0']:.0f},{b['y0']:.0f})-({b['x1']:.0f},{b['y1']:.0f}) conf={b['confidence']:.3f}")
    """

    def __init__(self, model_path: str, n_threads: int = 4, lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_text_det_signatures(self._lib)
        self._ctx = self._lib.crispembed_text_det_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load text detection model: {model_path}")

    def detect(self, image, text_threshold: float = 0.6,
               low_threshold: float = 0.35) -> list:
        """Detect text lines in an image.

        Args:
            image: File path (str/Path), PIL.Image, or numpy array (H, W, C) uint8.
            text_threshold: Confidence threshold for text regions (default 0.6).
            low_threshold: Binary threshold for connected components (default 0.35).

        Returns:
            List of dicts with keys: x0, y0, x1, y1, confidence.
        """
        if isinstance(image, (str, Path)):
            from PIL import Image
            image = np.array(Image.open(str(image)).convert("RGB"))
        elif hasattr(image, 'convert'):
            image = np.array(image.convert("RGB"))
        arr = np.ascontiguousarray(image, dtype=np.uint8)
        h, w = arr.shape[:2]
        ch = arr.shape[2] if arr.ndim == 3 else 1
        n = ctypes.c_int(0)
        ptr = self._lib.crispembed_text_det(
            self._ctx,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(w), ctypes.c_int(h), ctypes.c_int(ch),
            ctypes.c_float(text_threshold), ctypes.c_float(low_threshold),
            ctypes.byref(n))
        results = []
        for i in range(n.value):
            r = ptr[i]
            results.append({
                "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1,
                "confidence": r.confidence,
            })
        return results

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_text_det_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# Named Entity Recognition (GLiNER zero-shot NER)
# ---------------------------------------------------------------------------

class _NEREntity(ctypes.Structure):
    _fields_ = [
        ("start_char", ctypes.c_int),
        ("end_char", ctypes.c_int),
        ("text", ctypes.c_char_p),
        ("label", ctypes.c_char_p),
        ("score", ctypes.c_float),
    ]


def _setup_ner_signatures(lib):
    lib.crispembed_ner_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_ner_init.restype = ctypes.c_void_p

    lib.crispembed_ner_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_ner_free.restype = None

    lib.crispembed_ner_extract.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.POINTER(_NEREntity)),
    ]
    lib.crispembed_ner_extract.restype = ctypes.c_int


class CrispNER:
    """Zero-shot Named Entity Recognition via GLiNER.

    Detects arbitrary entity types specified at inference time — no
    retraining needed. Uses an LFM2.5 bidirectional backbone with a
    GLiNER span-matching head.

    Usage::

        ner = CrispNER("gliner-lfm-f32.gguf")
        entities = ner.extract(
            "Maria Schmidt arbeitet bei Siemens in München",
            labels=["person", "organization", "location"],
        )
        for e in entities:
            print(f"{e['text']} => {e['label']} ({e['score']:.2f})")
    """

    def __init__(self, model_path: str, n_threads: int = 4, lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_ner_signatures(self._lib)
        self._ctx = self._lib.crispembed_ner_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load NER model: {model_path}")

    def extract(self, text: str, labels: list, threshold: float = 0.5) -> list:
        """Extract named entities from text.

        Args:
            text: Input text string.
            labels: List of entity type strings (e.g. ["person", "org"]).
            threshold: Minimum confidence (0.0-1.0, default 0.5).

        Returns:
            List of dicts: [{"text", "label", "start", "end", "score"}, ...]
        """
        # Build C array of label strings
        label_bytes = [l.encode("utf-8") for l in labels]
        label_arr = (ctypes.c_char_p * len(labels))(*label_bytes)

        entities_ptr = ctypes.POINTER(_NEREntity)()
        n = self._lib.crispembed_ner_extract(
            self._ctx,
            text.encode("utf-8"),
            label_arr, len(labels),
            ctypes.c_float(threshold),
            ctypes.byref(entities_ptr),
        )

        results = []
        for i in range(n):
            e = entities_ptr[i]
            results.append({
                "text": e.text.decode("utf-8") if e.text else "",
                "label": e.label.decode("utf-8") if e.label else "",
                "start": e.start_char,
                "end": e.end_char,
                "score": float(e.score),
            })
        return results

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_ner_free(self._ctx)
            self._ctx = None


# ── LiLT — Language-independent Layout Transformer ──────────────────

class _LiLTToken(ctypes.Structure):
    _fields_ = [
        ("token_id", ctypes.c_int),
        ("label_id", ctypes.c_int),
        ("label", ctypes.c_char_p),
        ("score", ctypes.c_float),
    ]


def _setup_lilt_signatures(lib):
    lib.crispembed_lilt_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_lilt_init.restype = ctypes.c_void_p

    lib.crispembed_lilt_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_lilt_free.restype = None

    lib.crispembed_lilt_classify.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_lilt_classify.restype = ctypes.POINTER(_LiLTToken)

    lib.crispembed_lilt_num_labels.argtypes = [ctypes.c_void_p]
    lib.crispembed_lilt_num_labels.restype = ctypes.c_int

    lib.crispembed_lilt_label_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.crispembed_lilt_label_name.restype = ctypes.c_char_p


class CrispLiLT:
    """LiLT — Language-independent Layout Transformer for document understanding.

    Dual-stream encoder (RoBERTa text + layout transformer with BiACM)
    for token classification. Supports form understanding (FUNSD) with
    question/answer/header labeling.

    Usage::

        lilt = CrispLiLT("lilt-funsd-f32.gguf")
        tokens = lilt.classify(
            input_ids=[0, 10566, 35, 2],
            bbox=[[0,0,0,0], [10,50,90,80], [90,50,110,80], [0,0,0,0]],
        )
        for t in tokens:
            print(f"token={t['token_id']} label={t['label']} score={t['score']:.2f}")
    """

    def __init__(self, model_path: str, n_threads: int = 4, lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_lilt_signatures(self._lib)
        self._ctx = self._lib.crispembed_lilt_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load LiLT model: {model_path}")

    @property
    def num_labels(self) -> int:
        return self._lib.crispembed_lilt_num_labels(self._ctx)

    def label_name(self, label_id: int) -> str:
        r = self._lib.crispembed_lilt_label_name(self._ctx, label_id)
        return r.decode("utf-8") if r else ""

    def classify(self, input_ids: list, bbox: list) -> list:
        """Run token classification.

        Args:
            input_ids: List of int token IDs (including BOS/EOS).
            bbox: List of [x0, y0, x1, y1] per token (in [0, 1000] range).

        Returns:
            List of dicts: [{"token_id", "label_id", "label", "score"}, ...]
        """
        T = len(input_ids)
        ids_arr = (ctypes.c_int32 * T)(*input_ids)
        bbox_flat = []
        for b in bbox:
            bbox_flat.extend(b[:4] if len(b) >= 4 else b + [0] * (4 - len(b)))
        bbox_arr = (ctypes.c_int32 * (T * 4))(*bbox_flat)

        out_n = ctypes.c_int(0)
        result_ptr = self._lib.crispembed_lilt_classify(
            self._ctx, ids_arr, bbox_arr, T, ctypes.byref(out_n))

        results = []
        for i in range(out_n.value):
            t = result_ptr[i]
            results.append({
                "token_id": t.token_id,
                "label_id": t.label_id,
                "label": t.label.decode("utf-8") if t.label else "",
                "score": float(t.score),
            })
        return results

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_lilt_free(self._ctx)
            self._ctx = None


# ── Key Information Extraction (KIE) ─────────────────────────────────

class _KIEField(ctypes.Structure):
    _fields_ = [
        ("label", ctypes.c_char_p),
        ("value", ctypes.c_char_p),
        ("score", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("w", ctypes.c_float),
        ("h", ctypes.c_float),
    ]


class _KIEResult(ctypes.Structure):
    _fields_ = [
        ("fields", ctypes.POINTER(_KIEField)),
        ("n_fields", ctypes.c_int),
        ("ocr_text", ctypes.c_char_p),
        ("ocr_confidence", ctypes.c_float),
        ("n_ocr_regions", ctypes.c_int),
    ]


def _setup_kie_signatures(lib):
    lib.crispembed_kie_init.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int
    ]
    lib.crispembed_kie_init.restype = ctypes.c_void_p

    lib.crispembed_kie_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_kie_free.restype = None

    lib.crispembed_kie_extract.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
        ctypes.c_float,
    ]
    lib.crispembed_kie_extract.restype = _KIEResult


class CrispKIE:
    """Key Information Extraction — OCR + NER pipeline.

    Chains text detection + recognition with GLiNER zero-shot NER to
    extract structured key-value fields from document images (receipts,
    invoices, forms, business cards).

    Usage::

        kie = CrispKIE(
            ocr_det_model="dbnet-det-f16.gguf",
            ocr_rec_model="trocr-printed-f16.gguf",
            ner_model="gliner-lfm-f32.gguf",
        )
        result = kie.extract("receipt.png", labels=["total", "date", "vendor"])
        for f in result["fields"]:
            print(f"{f['label']} = {f['value']} ({f['score']:.2f})")
    """

    def __init__(self, ocr_det_model: str, ocr_rec_model: str, ner_model: str,
                 n_threads: int = 4, lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_kie_signatures(self._lib)
        self._ctx = self._lib.crispembed_kie_init(
            ocr_det_model.encode("utf-8"),
            ocr_rec_model.encode("utf-8"),
            ner_model.encode("utf-8"),
            n_threads)
        if not self._ctx:
            raise RuntimeError("Failed to init KIE pipeline")

    def extract(self, image_path: str, labels: list,
                threshold: float = 0.5) -> dict:
        """Extract structured fields from a document image.

        Args:
            image_path: Path to document image (JPG/PNG/BMP).
            labels: List of field names (e.g. ["total", "date", "vendor"]).
            threshold: Minimum NER confidence (0.0-1.0, default 0.5).

        Returns:
            Dict with keys: "fields" (list of dicts), "ocr_text",
            "ocr_confidence", "n_ocr_regions".
        """
        label_bytes = [l.encode("utf-8") for l in labels]
        label_arr = (ctypes.c_char_p * len(labels))(*label_bytes)

        res = self._lib.crispembed_kie_extract(
            self._ctx,
            image_path.encode("utf-8"),
            label_arr, len(labels),
            ctypes.c_float(threshold),
        )

        fields = []
        for i in range(res.n_fields):
            f = res.fields[i]
            fields.append({
                "label": f.label.decode("utf-8") if f.label else "",
                "value": f.value.decode("utf-8") if f.value else "",
                "score": float(f.score),
                "bbox": [float(f.x), float(f.y), float(f.w), float(f.h)],
            })

        return {
            "fields": fields,
            "ocr_text": res.ocr_text.decode("utf-8") if res.ocr_text else "",
            "ocr_confidence": float(res.ocr_confidence),
            "n_ocr_regions": res.n_ocr_regions,
        }

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_kie_free(self._ctx)
            self._ctx = None


# ── Scan cleanup ─────────────────────────────────────────────────────

class _ScanCleanupParams(ctypes.Structure):
    _fields_ = [
        ("deskew", ctypes.c_int),
        ("crop_borders", ctypes.c_int),
        ("whiten_background", ctypes.c_int),
        ("binarize", ctypes.c_int),
        ("binarize_method", ctypes.c_int),
        ("sauvola_k", ctypes.c_float),
        ("sauvola_window", ctypes.c_int),
        ("morph_kernel", ctypes.c_int),
        ("border_threshold", ctypes.c_float),
        ("deskew_max_angle", ctypes.c_float),
    ]


def _setup_scan_cleanup_signatures(lib):
    lib.crispembed_scan_cleanup_defaults.argtypes = []
    lib.crispembed_scan_cleanup_defaults.restype = _ScanCleanupParams

    lib.crispembed_scan_cleanup_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_scan_cleanup_init.restype = ctypes.c_void_p

    lib.crispembed_scan_cleanup_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_scan_cleanup_free.restype = None

    lib.crispembed_scan_cleanup_process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        _ScanCleanupParams,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_scan_cleanup_process.restype = ctypes.c_int

    lib.crispembed_scan_cleanup_free_image.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.crispembed_scan_cleanup_free_image.restype = None


class CrispScanCleanup:
    """Document scan preprocessing (deskew, crop, whiten, binarize).

    No model needed — pure classical image processing.

    Usage::

        cleanup = CrispScanCleanup()
        cleaned = cleanup.process("scan.png")  # returns numpy uint8 RGB array
        # Or with PIL:
        cleaned = cleanup.process(pil_image)
    """

    def __init__(self, model_path: str = None, n_threads: int = 4,
                 lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_scan_cleanup_signatures(self._lib)
        mp = model_path.encode("utf-8") if model_path else None
        self._ctx = self._lib.crispembed_scan_cleanup_init(mp, n_threads)
        if not self._ctx:
            raise RuntimeError("Failed to init scan cleanup")

    def process(self, image, deskew=True, crop_borders=True,
                whiten_background=True, binarize=False,
                binarize_method=0, sauvola_k=0.2, sauvola_window=25,
                morph_kernel=51, border_threshold=0.15,
                deskew_max_angle=15.0):
        """Process a scan image.

        Args:
            image: file path (str), PIL Image, or numpy uint8 array (H, W, C).
            deskew: correct skew angle (default True).
            crop_borders: remove dark scanner borders (default True).
            whiten_background: flatten uneven lighting (default True).
            binarize: adaptive thresholding (default False).
            binarize_method: 0 = Otsu, 1 = Sauvola.

        Returns:
            numpy uint8 array (H, W, 3) — cleaned RGB image.
        """
        import numpy as np

        # Load image from various sources
        if isinstance(image, str):
            from PIL import Image
            img = np.array(Image.open(image).convert("RGB"))
        elif hasattr(image, "convert"):
            img = np.array(image.convert("RGB"))
        else:
            img = np.asarray(image)

        if img.ndim == 2:
            h, w = img.shape
            ch = 1
        else:
            h, w, ch = img.shape

        pixels = img.flatten().astype(np.uint8)
        px_ptr = pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        params = self._lib.crispembed_scan_cleanup_defaults()
        params.deskew = int(deskew)
        params.crop_borders = int(crop_borders)
        params.whiten_background = int(whiten_background)
        params.binarize = int(binarize)
        params.binarize_method = binarize_method
        params.sauvola_k = sauvola_k
        params.sauvola_window = sauvola_window
        params.morph_kernel = morph_kernel
        params.border_threshold = border_threshold
        params.deskew_max_angle = deskew_max_angle

        out_ptr = ctypes.POINTER(ctypes.c_uint8)()
        out_w = ctypes.c_int(0)
        out_h = ctypes.c_int(0)

        rc = self._lib.crispembed_scan_cleanup_process(
            self._ctx, px_ptr, w, h, ch, params,
            ctypes.byref(out_ptr), ctypes.byref(out_w), ctypes.byref(out_h),
        )
        if rc != 0 or not out_ptr:
            raise RuntimeError("Scan cleanup failed")

        ow, oh = out_w.value, out_h.value
        buf = np.ctypeslib.as_array(out_ptr, shape=(oh * ow * 3,)).copy()
        self._lib.crispembed_scan_cleanup_free_image(out_ptr)
        return buf.reshape(oh, ow, 3)

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_scan_cleanup_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# Text Super-Resolution (NAFNet / ESRGAN-style upscaler)
# ---------------------------------------------------------------------------

def _setup_text_sr_signatures(lib):
    lib.crispembed_text_sr_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_text_sr_init.restype = ctypes.c_void_p

    lib.crispembed_text_sr_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_text_sr_free.restype = None

    lib.crispembed_text_sr_upscale_factor.argtypes = [ctypes.c_void_p]
    lib.crispembed_text_sr_upscale_factor.restype = ctypes.c_int

    lib.crispembed_text_sr_process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_text_sr_process.restype = ctypes.c_int

    lib.crispembed_text_sr_free_image.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.crispembed_text_sr_free_image.restype = None


class CrispTextSr:
    """Text super-resolution — upscale low-resolution document images.

    Usage::

        sr = CrispTextSr("text-sr.gguf")
        print(sr.upscale_factor)  # e.g. 4
        out = sr.process(pixels, width, height)  # returns (ndarray, out_w, out_h)
    """

    def __init__(self, model_path: str, n_threads: int = 4,
                 lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_text_sr_signatures(self._lib)
        self._ctx = self._lib.crispembed_text_sr_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load text SR model: {model_path}")

    @property
    def upscale_factor(self) -> int:
        """Return the upscale factor (e.g. 2, 4) reported by the model."""
        return self._lib.crispembed_text_sr_upscale_factor(self._ctx)

    def process(self, pixels: np.ndarray, width: int, height: int,
                tile_size: int = 0, tile_overlap: int = 0
                ) -> Tuple[np.ndarray, int, int]:
        """Upscale an image.

        Args:
            pixels: uint8 numpy array, flattened or shaped (H, W, C).
            width: source image width in pixels.
            height: source image height in pixels.
            tile_size: tile size for tiled inference (0 = full image).
            tile_overlap: overlap between tiles in pixels.

        Returns:
            Tuple of (output_ndarray uint8 shape (out_h, out_w, 3), out_w, out_h).
        """
        flat = np.asarray(pixels, dtype=np.uint8).flatten()
        px_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        out_ptr = ctypes.POINTER(ctypes.c_uint8)()
        out_w = ctypes.c_int(0)
        out_h = ctypes.c_int(0)

        rc = self._lib.crispembed_text_sr_process(
            self._ctx, px_ptr, width, height,
            tile_size, tile_overlap,
            ctypes.byref(out_ptr), ctypes.byref(out_w), ctypes.byref(out_h),
        )
        if rc != 0 or not out_ptr:
            raise RuntimeError("Text SR processing failed")

        ow, oh = out_w.value, out_h.value
        buf = np.ctypeslib.as_array(out_ptr, shape=(oh * ow * 3,)).copy()
        self._lib.crispembed_text_sr_free_image(out_ptr)
        return buf.reshape(oh, ow, 3), ow, oh

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_text_sr_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# TBSRN text-line Super-Resolution
# ---------------------------------------------------------------------------

def _setup_tbsrn_sr_signatures(lib):
    lib.crispembed_tbsrn_sr_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_tbsrn_sr_init.restype = ctypes.c_void_p

    lib.crispembed_tbsrn_sr_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_tbsrn_sr_free.restype = None

    lib.crispembed_tbsrn_sr_process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_tbsrn_sr_process.restype = ctypes.c_int

    lib.crispembed_tbsrn_sr_free_image.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.crispembed_tbsrn_sr_free_image.restype = None


class CrispTbsrnSr:
    """TBSRN text-line super-resolution (PaddleOCR Telescope, 1.1M params).

    Designed for text-line crops rather than whole images.

    Usage::

        sr = CrispTbsrnSr("tbsrn-telescope.gguf")
        out = sr.process(pixels, width, height)  # returns (ndarray, out_w, out_h)
    """

    def __init__(self, model_path: str, n_threads: int = 4,
                 lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_tbsrn_sr_signatures(self._lib)
        self._ctx = self._lib.crispembed_tbsrn_sr_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load TBSRN SR model: {model_path}")

    def process(self, pixels: np.ndarray, width: int, height: int
                ) -> Tuple[np.ndarray, int, int]:
        """Upscale a text-line crop.

        Args:
            pixels: uint8 numpy array, flattened or shaped (H, W, C).
            width: source image width in pixels.
            height: source image height in pixels.

        Returns:
            Tuple of (output_ndarray uint8 shape (out_h, out_w, 3), out_w, out_h).
        """
        flat = np.asarray(pixels, dtype=np.uint8).flatten()
        px_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        out_ptr = ctypes.POINTER(ctypes.c_uint8)()
        out_w = ctypes.c_int(0)
        out_h = ctypes.c_int(0)

        rc = self._lib.crispembed_tbsrn_sr_process(
            self._ctx, px_ptr, width, height,
            ctypes.byref(out_ptr), ctypes.byref(out_w), ctypes.byref(out_h),
        )
        if rc != 0 or not out_ptr:
            raise RuntimeError("TBSRN SR processing failed")

        ow, oh = out_w.value, out_h.value
        buf = np.ctypeslib.as_array(out_ptr, shape=(oh * ow * 3,)).copy()
        self._lib.crispembed_tbsrn_sr_free_image(out_ptr)
        return buf.reshape(oh, ow, 3), ow, oh

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_tbsrn_sr_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# PAN Super-Resolution
# ---------------------------------------------------------------------------

def _setup_pan_sr_signatures(lib):
    lib.crispembed_pan_sr_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_pan_sr_init.restype = ctypes.c_void_p

    lib.crispembed_pan_sr_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_pan_sr_free.restype = None

    lib.crispembed_pan_sr_scale.argtypes = [ctypes.c_void_p]
    lib.crispembed_pan_sr_scale.restype = ctypes.c_int

    lib.crispembed_pan_sr_process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_pan_sr_process.restype = ctypes.c_int

    lib.crispembed_pan_sr_free_image.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.crispembed_pan_sr_free_image.restype = None


class CrispPanSr:
    """PAN super-resolution — upscale low-resolution document images.

    Usage::

        sr = CrispPanSr("pan-sr.gguf")
        print(sr.scale)  # e.g. 4
        out = sr.process(pixels, width, height)  # returns (ndarray, out_w, out_h)
    """

    def __init__(self, model_path: str, n_threads: int = 4,
                 lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_pan_sr_signatures(self._lib)
        self._ctx = self._lib.crispembed_pan_sr_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load PAN SR model: {model_path}")

    @property
    def scale(self) -> int:
        """Return the scale factor (e.g. 2, 4) reported by the model."""
        return self._lib.crispembed_pan_sr_scale(self._ctx)

    def process(self, pixels: np.ndarray, width: int, height: int,
                tile_size: int = 0, tile_overlap: int = 0
                ) -> Tuple[np.ndarray, int, int]:
        """Upscale an image.

        Args:
            pixels: uint8 numpy array, flattened or shaped (H, W, C).
            width: source image width in pixels.
            height: source image height in pixels.
            tile_size: tile size for tiled inference (0 = full image).
            tile_overlap: overlap between tiles in pixels.

        Returns:
            Tuple of (output_ndarray uint8 shape (out_h, out_w, 3), out_w, out_h).
        """
        flat = np.asarray(pixels, dtype=np.uint8).flatten()
        px_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        out_ptr = ctypes.POINTER(ctypes.c_uint8)()
        out_w = ctypes.c_int(0)
        out_h = ctypes.c_int(0)

        rc = self._lib.crispembed_pan_sr_process(
            self._ctx, px_ptr, width, height,
            tile_size, tile_overlap,
            ctypes.byref(out_ptr), ctypes.byref(out_w), ctypes.byref(out_h),
        )
        if rc != 0 or not out_ptr:
            raise RuntimeError("PAN SR processing failed")

        ow, oh = out_w.value, out_h.value
        buf = np.ctypeslib.as_array(out_ptr, shape=(oh * ow * 3,)).copy()
        self._lib.crispembed_pan_sr_free_image(out_ptr)
        return buf.reshape(oh, ow, 3), ow, oh

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_pan_sr_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# TBSRN Super-Resolution (always 2×, 16×64 → 32×128)
# ---------------------------------------------------------------------------

def _setup_tbsrn_sr_signatures(lib):
    lib.crispembed_tbsrn_sr_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_tbsrn_sr_init.restype = ctypes.c_void_p

    lib.crispembed_tbsrn_sr_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_tbsrn_sr_free.restype = None

    lib.crispembed_tbsrn_sr_process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
# SAFMN Whole-Image Super-Resolution (SAFM+CCM AttBlocks, Apache-2.0)
# ---------------------------------------------------------------------------

def _setup_safmn_sr_signatures(lib):
    lib.crispembed_safmn_sr_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_safmn_sr_init.restype = ctypes.c_void_p

    lib.crispembed_safmn_sr_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_safmn_sr_free.restype = None

    lib.crispembed_safmn_sr_scale.argtypes = [ctypes.c_void_p]
    lib.crispembed_safmn_sr_scale.restype = ctypes.c_int

    lib.crispembed_safmn_sr_process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_tbsrn_sr_process.restype = ctypes.c_int

    lib.crispembed_tbsrn_sr_free_image.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.crispembed_tbsrn_sr_free_image.restype = None


class CrispTbsrnSr:
    """TBSRN super-resolution — upscale text-image crops (always 2×, 16×64 → 32×128).

    Usage::

        sr = CrispTbsrnSr("tbsrn-sr.gguf")
        out = sr.process(pixels, width, height)  # returns (ndarray, out_w, out_h)
    """

    scale: int = 2  # TBSRN is always 2×

    def __init__(self, model_path: str, n_threads: int = 4,
                 lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_tbsrn_sr_signatures(self._lib)
        self._ctx = self._lib.crispembed_tbsrn_sr_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load TBSRN SR model: {model_path}")

    def process(self, pixels: np.ndarray, width: int, height: int,
                ) -> Tuple[np.ndarray, int, int]:
        """Upscale a text-image crop (always 2×).
    lib.crispembed_safmn_sr_process.restype = ctypes.c_int

    lib.crispembed_safmn_sr_free_image.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.crispembed_safmn_sr_free_image.restype = None


class CrispSafmnSr:
    """SAFMN super-resolution — lightweight upscaling with SAFM+CCM AttBlocks.

    Usage::

        sr = CrispSafmnSr("safmn-sr-x4.gguf")
        print(sr.scale)  # e.g. 4
        out = sr.process(pixels, width, height)  # returns (ndarray, out_w, out_h)
    """

    def __init__(self, model_path: str, n_threads: int = 4,
                 lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_safmn_sr_signatures(self._lib)
        self._ctx = self._lib.crispembed_safmn_sr_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load SAFMN SR model: {model_path}")

    @property
    def scale(self) -> int:
        """Return the scale factor (e.g. 2, 4) reported by the model."""
        return self._lib.crispembed_safmn_sr_scale(self._ctx)

    def process(self, pixels: np.ndarray, width: int, height: int,
                tile_size: int = 0, tile_overlap: int = 0
                ) -> Tuple[np.ndarray, int, int]:
        """Upscale an image.

        Args:
            pixels: uint8 numpy array, flattened or shaped (H, W, C).
            width: source image width in pixels.
            height: source image height in pixels.
            tile_size: tile size for tiled inference (0 = full image).
            tile_overlap: overlap between tiles in pixels.

        Returns:
            Tuple of (output_ndarray uint8 shape (out_h, out_w, 3), out_w, out_h).
        """
        flat = np.asarray(pixels, dtype=np.uint8).flatten()
        px_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        out_ptr = ctypes.POINTER(ctypes.c_uint8)()
        out_w = ctypes.c_int(0)
        out_h = ctypes.c_int(0)

        rc = self._lib.crispembed_tbsrn_sr_process(
            self._ctx, px_ptr, width, height,
            ctypes.byref(out_ptr), ctypes.byref(out_w), ctypes.byref(out_h),
        )
        if rc != 0 or not out_ptr:
            raise RuntimeError("TBSRN SR processing failed")

        ow, oh = out_w.value, out_h.value
        buf = np.ctypeslib.as_array(out_ptr, shape=(oh * ow * 3,)).copy()
        self._lib.crispembed_tbsrn_sr_free_image(out_ptr)
        rc = self._lib.crispembed_safmn_sr_process(
            self._ctx, px_ptr, width, height,
            tile_size, tile_overlap,
            ctypes.byref(out_ptr), ctypes.byref(out_w), ctypes.byref(out_h),
        )
        if rc != 0 or not out_ptr:
            raise RuntimeError("SAFMN SR processing failed")

        ow, oh = out_w.value, out_h.value
        buf = np.ctypeslib.as_array(out_ptr, shape=(oh * ow * 3,)).copy()
        self._lib.crispembed_safmn_sr_free_image(out_ptr)
        return buf.reshape(oh, ow, 3), ow, oh

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_tbsrn_sr_free(self._ctx)
            self._lib.crispembed_safmn_sr_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# Table Structure Recognition
# ---------------------------------------------------------------------------

def _setup_table_parse_signatures(lib):
    lib.crispembed_table_parse_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_table_parse_init.restype = ctypes.c_void_p

    lib.crispembed_table_parse_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_table_parse_free.restype = None

    lib.crispembed_table_parse_to_html.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
    ]
    lib.crispembed_table_parse_to_html.restype = ctypes.c_void_p

    lib.crispembed_table_parse_free_string.argtypes = [ctypes.c_void_p]
    lib.crispembed_table_parse_free_string.restype = None


class CrispTableParse:
    """Table structure recognition — extracts HTML from a table image.

    Uses morphological line detection and grid intersection analysis to
    produce an HTML ``<table>`` element. Optional built-in cell OCR via
    a Tesseract LSTM GGUF model.

    Usage::

        tp = CrispTableParse()                        # no OCR
        tp = CrispTableParse("tesseract.gguf")        # with OCR
        html = tp.to_html(gray_pixels, width, height) # returns str
    """

    def __init__(self, ocr_model_path: Optional[str] = None, n_threads: int = 4,
                 lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_table_parse_signatures(self._lib)
        path_bytes = ocr_model_path.encode("utf-8") if ocr_model_path else None
        self._ctx = self._lib.crispembed_table_parse_init(path_bytes, n_threads)
        if not self._ctx:
            raise RuntimeError(
                f"Failed to init table parser"
                + (f" with OCR model: {ocr_model_path}" if ocr_model_path else "")
            )

    def to_html(self, gray_pixels: np.ndarray, width: int, height: int) -> str:
        """Parse a grayscale table image and return an HTML string.

        Args:
            gray_pixels: uint8 numpy array of grayscale pixel values,
                         flattened or shaped (height, width).
            width: image width in pixels.
            height: image height in pixels.

        Returns:
            HTML string containing a ``<table>`` element.

        Raises:
            RuntimeError: if parsing fails.
        """
        flat = np.asarray(gray_pixels, dtype=np.uint8).flatten()
        px_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        raw = self._lib.crispembed_table_parse_to_html(self._ctx, px_ptr, width, height)
        if not raw:
            raise RuntimeError("Table parsing failed")

        html = ctypes.cast(raw, ctypes.c_char_p).value.decode("utf-8")
        self._lib.crispembed_table_parse_free_string(raw)
        return html

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_table_parse_free(self._ctx)
# Real-ESRGAN Whole-Image Super-Resolution (SRVGGNetCompact, BSD-3)
# ---------------------------------------------------------------------------

def _setup_esrgan_sr_signatures(lib):
    lib.crispembed_esrgan_sr_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_esrgan_sr_init.restype = ctypes.c_void_p

    lib.crispembed_esrgan_sr_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_esrgan_sr_free.restype = None

    lib.crispembed_esrgan_sr_scale.argtypes = [ctypes.c_void_p]
    lib.crispembed_esrgan_sr_scale.restype = ctypes.c_int

    lib.crispembed_esrgan_sr_process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_esrgan_sr_process.restype = ctypes.c_int

    lib.crispembed_esrgan_sr_free_image.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.crispembed_esrgan_sr_free_image.restype = None


class CrispEsrganSr:
    """Real-ESRGAN super-resolution — SRVGGNetCompact whole-image upscaling.

    Usage::

        sr = CrispEsrganSr("esrgan-x4.gguf")
        print(sr.scale)  # e.g. 4
        out = sr.process(pixels, width, height)  # returns (ndarray, out_w, out_h)
    """

    def __init__(self, model_path: str, n_threads: int = 4,
                 lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_esrgan_sr_signatures(self._lib)
        self._ctx = self._lib.crispembed_esrgan_sr_init(
            model_path.encode("utf-8"), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load Real-ESRGAN SR model: {model_path}")

    @property
    def scale(self) -> int:
        """Return the scale factor (e.g. 4) reported by the model."""
        return self._lib.crispembed_esrgan_sr_scale(self._ctx)

    def process(self, pixels: np.ndarray, width: int, height: int,
                tile_size: int = 0, tile_overlap: int = 0
                ) -> Tuple[np.ndarray, int, int]:
        """Upscale an image.

        Args:
            pixels: uint8 numpy array, flattened or shaped (H, W, C).
            width: source image width in pixels.
            height: source image height in pixels.
            tile_size: tile size for tiled inference (0 = full image).
            tile_overlap: overlap between tiles in pixels.

        Returns:
            Tuple of (output_ndarray uint8 shape (out_h, out_w, 3), out_w, out_h).
        """
        flat = np.asarray(pixels, dtype=np.uint8).flatten()
        px_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        out_ptr = ctypes.POINTER(ctypes.c_uint8)()
        out_w = ctypes.c_int(0)
        out_h = ctypes.c_int(0)

        rc = self._lib.crispembed_esrgan_sr_process(
            self._ctx, px_ptr, width, height,
            tile_size, tile_overlap,
            ctypes.byref(out_ptr), ctypes.byref(out_w), ctypes.byref(out_h),
        )
        if rc != 0 or not out_ptr:
            raise RuntimeError("Real-ESRGAN SR processing failed")

        ow, oh = out_w.value, out_h.value
        buf = np.ctypeslib.as_array(out_ptr, shape=(oh * ow * 3,)).copy()
        self._lib.crispembed_esrgan_sr_free_image(out_ptr)
        return buf.reshape(oh, ow, 3), ow, oh

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_esrgan_sr_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# OCR Orchestrator (source-type routing + cleanup + accept-gate)
# ---------------------------------------------------------------------------

class _CrispOcrPipelineParams(ctypes.Structure):
    _fields_ = [
        ("router", ctypes.c_int),
        ("cleanup_enabled", ctypes.c_int),
        ("min_chars", ctypes.c_int),
        ("min_confidence", ctypes.c_float),
        ("det_model", ctypes.c_char_p),
        ("rec_model", ctypes.c_char_p),
        ("nafnet_model", ctypes.c_char_p),
        ("sr_model", ctypes.c_char_p),
        ("vlm_model", ctypes.c_char_p),
        ("vlm_engine", ctypes.c_int),
        ("punct_model", ctypes.c_char_p),
    ]


def _setup_ocr_orchestrator_signatures(lib):
    lib.crispembed_ocr_pipeline_defaults.argtypes = []
    lib.crispembed_ocr_pipeline_defaults.restype = _CrispOcrPipelineParams

    lib.crispembed_ocr_pipeline_init.argtypes = [
        ctypes.POINTER(_CrispOcrPipelineParams), ctypes.c_int]
    lib.crispembed_ocr_pipeline_init.restype = ctypes.c_void_p

    lib.crispembed_ocr_pipeline_run.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.crispembed_ocr_pipeline_run.restype = ctypes.POINTER(_CrispOcrResult)

    lib.crispembed_ocr_pipeline_free.argtypes = [ctypes.c_void_p]
    lib.crispembed_ocr_pipeline_free.restype = None


class CrispOcrOrchestrator:
    """OCR orchestrator with source-type routing, cleanup, and accept-gate.

    Composes detection + recognition engines with automatic source-type
    classification (screenshot vs scan vs photo), per-stage cleanup, and
    cascading with confidence-based accept gates.

    Usage::

        orch = CrispOcrOrchestrator("surya-det", "qwen2vl-3b")
        result = orch.run("document.png")
        print(result["text"])
        print(f"  {result['n_regions']} regions, confidence={result['mean_confidence']:.2f}")
    """

    def __init__(self, det_model: str, rec_model: str, *,
                 router: bool = True, cleanup: bool = True,
                 min_chars: int = 8, min_confidence: float = 0.5,
                 nafnet_model: Optional[str] = None,
                 sr_model: Optional[str] = None,
                 vlm_model: Optional[str] = None, vlm_engine: int = 0,
                 punct_model: Optional[str] = None,
                 n_threads: int = 4, lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_ocr_orchestrator_signatures(self._lib)
        _setup_ocr_pipeline_signatures(self._lib)  # for _CrispOcrResult

        params = self._lib.crispembed_ocr_pipeline_defaults()
        params.router = int(router)
        params.cleanup_enabled = int(cleanup)
        params.min_chars = min_chars
        params.min_confidence = min_confidence
        params.det_model = det_model.encode("utf-8")
        params.rec_model = rec_model.encode("utf-8")
        if nafnet_model:
            params.nafnet_model = nafnet_model.encode("utf-8")
        if sr_model:
            params.sr_model = sr_model.encode("utf-8")
        if vlm_model:
            params.vlm_model = vlm_model.encode("utf-8")
            params.vlm_engine = vlm_engine
        if punct_model:
            params.punct_model = punct_model.encode("utf-8")

        self._ctx = self._lib.crispembed_ocr_pipeline_init(
            ctypes.byref(params), n_threads)
        if not self._ctx:
            raise RuntimeError("Failed to init OCR orchestrator")

    def run(self, image_path: str) -> dict:
        """Run the full orchestrator pipeline on an image.

        Returns:
            Dict with keys: text, n_regions, mean_confidence, regions.
        """
        n = ctypes.c_int(0)
        full_text = ctypes.c_char_p()
        mean_conf = ctypes.c_float(0.0)
        ptr = self._lib.crispembed_ocr_pipeline_run(
            self._ctx, str(image_path).encode("utf-8"),
            ctypes.byref(n), ctypes.byref(full_text), ctypes.byref(mean_conf))

        regions = []
        for i in range(n.value):
            r = ptr[i]
            regions.append({
                "text": r.text.decode("utf-8") if r.text else "",
                "x": r.x, "y": r.y, "w": r.w, "h": r.h,
                "confidence": r.confidence,
            })
        return {
            "text": full_text.value.decode("utf-8") if full_text.value else "",
            "n_regions": n.value,
            "mean_confidence": mean_conf.value,
            "regions": regions,
        }

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.crispembed_ocr_pipeline_free(self._ctx)
            self._ctx = None


# ---------------------------------------------------------------------------
# Classical Preprocessing (model-free, CPU-only)
# ---------------------------------------------------------------------------

def _setup_preproc_signatures(lib):
    lib.crispembed_pdf_page_dpi.argtypes = [
        ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)]
    lib.crispembed_pdf_page_dpi.restype = ctypes.c_int

    lib.crispembed_dewarp.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int)]
    lib.crispembed_dewarp.restype = ctypes.c_int

    lib.crispembed_tps_auto_dewarp.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8)]
    lib.crispembed_tps_auto_dewarp.restype = ctypes.c_int

    lib.crispembed_find_skew.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    lib.crispembed_find_skew.restype = ctypes.c_int

    lib.crispembed_adaptive_binarize.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8)]
    lib.crispembed_adaptive_binarize.restype = None

    lib.crispembed_background_norm.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8)]
    lib.crispembed_background_norm.restype = None

    lib.crispembed_despeckle.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8)]
    lib.crispembed_despeckle.restype = None

    lib.crispembed_ocr_render.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p]
    lib.crispembed_ocr_render.restype = ctypes.c_void_p


class CrispPreprocess:
    """Classical document preprocessing — model-free, CPU-only.

    Provides dewarp, deskew, adaptive binarization, background normalization,
    and despeckle on grayscale uint8 images.

    Usage::

        pp = CrispPreprocess()
        angle, conf = pp.find_skew(gray_image, w, h)
        dewarped = pp.dewarp(gray_image, w, h)
    """

    def __init__(self, lib_path: Optional[str] = None):
        self._lib = _load_library(lib_path)
        _setup_preproc_signatures(self._lib)

    def pdf_page_dpi(self, path: str, page: int = 0) -> Tuple[float, int]:
        """Get the DPI and image count for a PDF page.

        Args:
            path: path to the PDF file.
            page: zero-based page index (default 0).

        Returns:
            (dpi, n_images) tuple. dpi is the mean DPI across embedded
            raster images on that page; n_images is the image count.
        """
        dpi = ctypes.c_float(0)
        n_images = ctypes.c_int(0)
        ret = self._lib.crispembed_pdf_page_dpi(
            path.encode('utf-8'), page,
            ctypes.byref(dpi), ctypes.byref(n_images))
        if ret != 0:
            return (0.0, 0)
        return (dpi.value, n_images.value)

    def dewarp(self, gray: np.ndarray, w: int, h: int) -> np.ndarray:
        """Dewarp a grayscale page image. Returns dewarped uint8 array."""
        out = np.zeros(w * h, dtype=np.uint8)
        ow, oh = ctypes.c_int(0), ctypes.c_int(0)
        ret = self._lib.crispembed_dewarp(
            gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w, h,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(ow), ctypes.byref(oh))
        if ret != 0:
            return gray.copy()
        return out.reshape(oh.value, ow.value)

    def tps_dewarp(self, gray: np.ndarray, w: int, h: int, model_path: str) -> np.ndarray:
        """TPS auto-dewarp using a learned localizer model (GGUF). Returns dewarped uint8 array."""
        out = np.zeros(w * h, dtype=np.uint8)
        ret = self._lib.crispembed_tps_auto_dewarp(
            gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w, h,
            model_path.encode('utf-8'),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        if ret != 0:
            return gray.copy()
        return out.reshape(h, w)

    def find_skew(self, gray: np.ndarray, w: int, h: int) -> tuple:
        """Find skew angle in degrees. Returns (angle, confidence)."""
        angle, conf = ctypes.c_float(0), ctypes.c_float(0)
        self._lib.crispembed_find_skew(
            gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w, h, ctypes.byref(angle), ctypes.byref(conf))
        return angle.value, conf.value

    def adaptive_binarize(self, gray: np.ndarray, w: int, h: int) -> np.ndarray:
        """Adaptive Otsu binarization. Returns uint8 array (0/255)."""
        out = np.zeros(w * h, dtype=np.uint8)
        self._lib.crispembed_adaptive_binarize(
            gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w, h, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return out.reshape(h, w)

    def background_norm(self, gray: np.ndarray, w: int, h: int) -> np.ndarray:
        """Normalize background (handle gradients/shadows). Returns uint8."""
        out = np.zeros(w * h, dtype=np.uint8)
        self._lib.crispembed_background_norm(
            gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w, h, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return out.reshape(h, w)

    def despeckle(self, gray: np.ndarray, w: int, h: int,
                  max_w: int = 5, max_h: int = 5) -> np.ndarray:
        """Remove small noise components. Returns uint8 (0/255)."""
        out = np.zeros(w * h, dtype=np.uint8)
        self._lib.crispembed_despeckle(
            gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w, h, max_w, max_h,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return out.reshape(h, w)

    def cc_detect(self, gray: np.ndarray, w: int, h: int) -> list:
        """Detect text lines using connected components (model-free).
        Returns list of dicts with x, y, w, h keys."""
        self._lib.crispembed_cc_detect.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int)]
        self._lib.crispembed_cc_detect.restype = ctypes.POINTER(_CrispOcrResult)
        n = ctypes.c_int(0)
        ptr = self._lib.crispembed_cc_detect(
            gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w, h, ctypes.byref(n))
        results = []
        for i in range(n.value):
            r = ptr[i]
            results.append({"x": r.x, "y": r.y, "w": r.w, "h": r.h})
        if ptr and n.value > 0:
            ctypes.cdll.LoadLibrary("libc.so.6").free(ptr)
        return results


def _setup_ocr_render_signatures(lib):
    lib.crispembed_ocr_render.argtypes = [
        ctypes.POINTER(_CrispOcrResult), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    lib.crispembed_ocr_render.restype = ctypes.c_void_p


def ocr_render(results: list, page_w: int, page_h: int, format: str = "text",
               lib_path: Optional[str] = None) -> str:
    """Render OCR results to text/hOCR/ALTO/PDF format.

    Args:
        results: list of dicts with keys: text, x, y, w, h, confidence.
        page_w: page width in pixels.
        page_h: page height in pixels.
        format: "text", "hocr", "alto", or "pdf".

    Returns:
        Rendered string (or PDF bytes as string for "pdf" format).
    """
    lib = _load_library(lib_path)
    _setup_ocr_render_signatures(lib)
    _setup_ocr_pipeline_signatures(lib)  # for _CrispOcrResult

    n = len(results)
    if n == 0:
        return ""

    # Build C-compatible result array
    arr = (_CrispOcrResult * n)()
    text_bufs = []  # keep alive
    for i, r in enumerate(results):
        arr[i].x = r.get("x", 0)
        arr[i].y = r.get("y", 0)
        arr[i].w = r.get("w", 0)
        arr[i].h = r.get("h", 0)
        arr[i].confidence = r.get("confidence", 1.0)
        text_bytes = r.get("text", "").encode("utf-8")
        text_bufs.append(ctypes.c_char_p(text_bytes))
        arr[i].text = text_bufs[-1]
        arr[i].text_len = len(text_bytes)

    fmt = format.encode("utf-8")
    ptr = lib.crispembed_ocr_render(arr, n, page_w, page_h, fmt)
    if not ptr:
        return ""
    result = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", errors="replace")
    ctypes.cdll.LoadLibrary("libc.so.6").free(ptr)
    return result
