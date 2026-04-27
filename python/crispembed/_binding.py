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

        # --- Prefix ---
        lib.crispembed_set_prefix.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.crispembed_set_prefix.restype = None

        lib.crispembed_get_prefix.argtypes = [ctypes.c_void_p]
        lib.crispembed_get_prefix.restype = ctypes.c_char_p

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
