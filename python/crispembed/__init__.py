"""CrispEmbed — lightweight text embedding via ggml."""

from ._binding import CrispEmbed

__all__ = ["CrispEmbed"]
# Tracks /VERSION (the C library version). Wheel CI copies the freshly built
# libcrispembed.{so,dylib,dll} + ggml siblings alongside this file.
__version__ = "0.3.2"
