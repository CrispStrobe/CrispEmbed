#!/usr/bin/env python3
"""BidirLM-Omni audio-path parity test.

Loads only BidirLMOmniAudioEncoder + WhisperFeatureExtractor from HF
(bypassing the full omni SentenceTransformer pipeline) and compares
against CrispEmbed's crispembed_encode_audio() output.

    python tests/test_bidirlm_audio.py \
        --model BidirLM/BidirLM-Omni-2.5B-Embedding \
        --gguf  ~/.cache/crispembed/bidirlm-omni-2.5b-q8_0.gguf \
        --pcm   samples/jfk.raw   # f32le 16 kHz mono

The HF reference path:
  PCM → WhisperFeatureExtractor (128 mel bins, log10 + (x+4)/4 norm)
      → BidirLMOmniAudioEncoder (conv stem + 24 encoder layers + proj)
      → (n_frames, 2048) per-frame embeddings
      → mean over n_frames + L2 norm
      → 2048-d vector

CrispEmbed path:
  PCM → crispembed_encode_audio() (uses crisp_audio internally)
      → 2048-d vector

Pass criterion: cosine similarity ≥ 0.99 against HF reference.
"""

import argparse
import sys
import numpy as np


def hf_audio_embed(model_id: str, pcm: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute the HF reference audio embedding for one PCM clip."""
    import torch
    from transformers import AutoConfig, WhisperFeatureExtractor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import json

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Load just the audio encoder.
    audio_cls = get_class_from_dynamic_module(
        "modeling_bidirlm_omni.BidirLMOmniAudioEncoder", model_id,
    )
    audio = audio_cls(config.audio_config).to(torch.bfloat16)
    audio.eval()

    # Pull the audio_tower.* slice from safetensors and strip the prefix.
    sd = {}
    try:
        st_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        raw = load_file(st_path)
    except Exception:
        idx_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)["weight_map"]
        shards = {v for v in idx.values()}
        raw = {}
        for shard in shards:
            sp = hf_hub_download(repo_id=model_id, filename=shard)
            raw.update(load_file(sp))
    for k, v in raw.items():
        if k.startswith("audio_tower."):
            sd[k[len("audio_tower."):]] = v.to(torch.bfloat16)
    missing, unexpected = audio.load_state_dict(sd, strict=False)
    if missing:
        print(f"WARN: {len(missing)} missing audio tensors, e.g. {missing[:3]}")

    # Compute log-mel via the model's preprocessor.
    fe = WhisperFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
    feats = fe(pcm, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
    input_features = feats["input_features"][0]              # (n_mels=128, T_padded=3000)
    # WhisperFeatureExtractor pads to nb_max_frames=3000; the true audio
    # length is recovered from attention_mask. crisp_audio's compute_mel
    # returns the unpadded mel of length ~T_mel = ceil(samples / hop), so
    # we must pass that same length to HF's encoder for fair comparison.
    feature_lens = feats["attention_mask"][0].sum().unsqueeze(0).to(torch.long)
    # Trim input_features to match.
    input_features = input_features[:, : int(feature_lens.item())]

    with torch.no_grad():
        # forward expects (input_features, feature_lens=None) — see
        # modeling_bidirlm_omni.BidirLMOmniAudioEncoder.forward.
        out = audio(input_features=input_features.to(torch.bfloat16),
                    feature_lens=feature_lens)
        h = out.last_hidden_state.float()                    # (N_frames, output_dim)

    # Mean pool + L2 normalize — matches sentence-transformers Pooling+similarity.
    pooled = h.mean(dim=0)
    return torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()


def crispembed_audio_embed(gguf: str, pcm: np.ndarray, lib_path: str | None = None) -> np.ndarray:
    """Compute the CrispEmbed audio embedding for one PCM clip."""
    sys.path.insert(0, "python")
    from crispembed._binding import CrispEmbed

    ce = CrispEmbed(gguf, lib_path=lib_path) if lib_path else CrispEmbed(gguf)
    if not ce.has_audio:
        raise RuntimeError("Loaded CrispEmbed build has no audio support — "
                           "rebuild with crisp_audio (CRISP_AUDIO_DIR found at configure time).")
    return ce.encode_audio(pcm.astype(np.float32), sr=16000)


def load_pcm(path: str) -> np.ndarray:
    """Load f32le 16 kHz mono PCM (.raw) or .wav."""
    if path.endswith(".wav"):
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32")
        except ImportError:
            raise RuntimeError("install soundfile to read .wav, or convert to .raw via "
                               "ffmpeg -i in.wav -ar 16000 -ac 1 -f f32le out.raw")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000).astype(np.float32)
        return data
    return np.fromfile(path, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model ID")
    p.add_argument("--gguf",  required=True, help="GGUF path")
    p.add_argument("--pcm",   required=True, help="16 kHz mono float32 PCM (.raw or .wav)")
    p.add_argument("--lib",   default=None, help="libcrispembed.{dylib,so} path")
    args = p.parse_args()

    print(f"Audio: {args.pcm}")
    pcm = load_pcm(args.pcm)
    print(f"  {len(pcm)} samples ({len(pcm) / 16000:.2f} s)")

    print("HF reference (audio-only path)...")
    hf = hf_audio_embed(args.model, pcm)
    print(f"  HF dim: {hf.shape}")

    print("CrispEmbed (crisp_audio)...")
    ce = crispembed_audio_embed(args.gguf, pcm, args.lib)
    print(f"  CE dim: {ce.shape}")

    if hf.shape != ce.shape:
        print(f"FAIL: shape mismatch ({hf.shape} vs {ce.shape})")
        return 1

    cos = float(np.dot(hf, ce) / (np.linalg.norm(hf) * np.linalg.norm(ce) + 1e-12))
    max_diff = float(np.max(np.abs(hf - ce)))
    ok = cos > 0.99
    print()
    print(f"cosine = {cos:.6f}   max_diff = {max_diff:.6f}   {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
