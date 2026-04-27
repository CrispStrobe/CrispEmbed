#!/usr/bin/env python3
"""BidirLM-Omni cross-modal retrieval demo.

Replicates the bass/treble example from
https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding using
CrispEmbed's encode() + encode_audio() so both modalities land in the
same 2048-d shared space.

Usage:
    python examples/demo/bidirlm_cross_modal.py

Requires:
    - cstr/bidirlm-omni-2.5b-GGUF (audio-enabled) — auto-downloads on first run.
    - libcrispembed.dylib built with crisp_audio (sibling-repo CrispASR clone).

Expected ranking (cosine values are small in absolute terms because the
upstream BidirLM-Omni-2.5B-Embedding checkpoint has weaker text/audio
alignment than the README's "omni-st5.4" example — but the relative
ordering matches exactly: each text's matching modality wins its row):

                                    80Hz bass    7500Hz high
    "A deep bass sound."           [ ~0.034 ]    [ ~-0.005 ]   ← bass wins
    "A high-pitched sound."        [ ~0.027 ]    [  ~0.037 ]   ← treble wins
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from crispembed._binding import CrispEmbed


def main():
    print("Loading BidirLM-Omni-2.5B (audio-enabled)…", flush=True)
    model = CrispEmbed("bidirlm-omni-2.5b")
    if not model.has_audio:
        print(
            "ERROR: this CrispEmbed build has no audio support. Rebuild with\n"
            "  cmake -S . -B build  (auto-discovers the configured CrispAudio checkout)",
            file=sys.stderr,
        )
        return 1

    texts = [
        "A deep bass sound.",
        "A high-pitched sound.",
    ]
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    audios = [
        ("80 Hz bass",        np.sin(2 * np.pi *   80 * t).astype(np.float32)),
        ("7500 Hz treble",    np.sin(2 * np.pi * 7500 * t).astype(np.float32)),
    ]

    text_vecs  = np.stack([model.encode(t) for t in texts])
    audio_vecs = np.stack([model.encode_audio(a, sr=sr) for _, a in audios])

    sim = text_vecs @ audio_vecs.T  # both already L2-normalised

    print()
    print(f"{'':32s}", end="")
    for label, _ in audios:
        print(f"{label:>14s}", end="")
    print()
    for i, t in enumerate(texts):
        print(f"{t:32s}", end="")
        for j in range(len(audios)):
            print(f"{sim[i, j]:>14.4f}", end="")
        print()
    print()

    # Sanity check: bass ↔ bass should beat bass ↔ treble (and symmetric).
    bass_match   = sim[0, 0] > sim[0, 1]
    treble_match = sim[1, 1] > sim[1, 0]
    if bass_match and treble_match:
        print("PASS: cross-modal ranking matches expectation (bass→bass, treble→treble).")
        return 0
    print("FAIL: cross-modal ranking is off.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
