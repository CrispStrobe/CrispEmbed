"""Image preprocessing for BidirLM-Omni vision tower.

This is a thin wrapper around ``Qwen2VLImageProcessorFast`` from
transformers — BidirLM-Omni uses the same processor unchanged. The wrapper
returns the two flat buffers the C ABI consumes:

* ``pixel_patches`` — float32, shape ``(n_patches, in_channels * temporal_patch_size *
  patch_size * patch_size)`` = ``(n_patches, 1536)`` for a 16×16 RGB patch.
* ``image_grid_thw`` — int32, shape ``(n_images, 3)`` with columns
  ``(temporal, height_in_patches, width_in_patches)``.

For still images the processor pre-replicates the frame ``temporal_patch_size``
times along T so a single ``Conv3d(stride=(2,16,16))`` kernel covers it. The
returned ``grid_thw`` row is ``(1, h, w)`` in that case.

Heavy dependency on ``transformers`` is intentional: matching HF preprocessing
byte-for-byte is the only way the parity test (``tests/test_bidirlm_vision.py``)
will pass. A C++ port of smart_resize + normalize + patchify is future work.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np


_PROCESSOR_CACHE = {}


def _get_processor(model_name: str = "BidirLM/BidirLM-Omni-2.5B-Embedding"):
    """Lazy-load and cache the HF Qwen2VLImageProcessorFast."""
    if model_name in _PROCESSOR_CACHE:
        return _PROCESSOR_CACHE[model_name]
    try:
        from transformers import AutoImageProcessor
    except ImportError as e:
        raise RuntimeError(
            "encode_image requires transformers; install with `pip install transformers pillow`"
        ) from e
    proc = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    _PROCESSOR_CACHE[model_name] = proc
    return proc


def preprocess_image(
    image,
    *,
    model_name: str = "BidirLM/BidirLM-Omni-2.5B-Embedding",
    processor=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert PIL.Image (or path string) into (pixel_patches, image_grid_thw).

    Args:
        image: PIL.Image, file path, or numpy array (H, W, 3) uint8.
        model_name: HF model ID for the image processor.
        processor: Optional pre-loaded image processor (skips load).

    Returns:
        pixel_patches: float32 (n_patches, 1536)
        image_grid_thw: int32 (1, 3)  — (t, h_patches, w_patches)
    """
    if processor is None:
        processor = _get_processor(model_name)

    img = _coerce_image(image)
    inputs = processor(images=img, return_tensors="np")
    pixel_values = np.ascontiguousarray(inputs["pixel_values"], dtype=np.float32)
    grid_thw = np.ascontiguousarray(inputs["image_grid_thw"], dtype=np.int32)

    if pixel_values.ndim != 2:
        raise RuntimeError(
            f"unexpected pixel_values shape from HF processor: {pixel_values.shape}"
        )
    return pixel_values, grid_thw


def preprocess_images(
    images: Sequence,
    *,
    model_name: str = "BidirLM/BidirLM-Omni-2.5B-Embedding",
    processor=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch variant of preprocess_image. Concatenates patches across images."""
    if processor is None:
        processor = _get_processor(model_name)
    coerced = [_coerce_image(im) for im in images]
    inputs = processor(images=coerced, return_tensors="np")
    pixel_values = np.ascontiguousarray(inputs["pixel_values"], dtype=np.float32)
    grid_thw = np.ascontiguousarray(inputs["image_grid_thw"], dtype=np.int32)
    return pixel_values, grid_thw


def _coerce_image(image):
    """Accept PIL.Image, file path string, or numpy array (H,W,3) uint8."""
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "encode_image requires Pillow; install with `pip install pillow`"
        ) from e
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB") if hasattr(image, "convert") else image
