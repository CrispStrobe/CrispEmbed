#!/usr/bin/env python3
"""Dump LFM2.5-Embedding per-layer reference activations for crispembed_diff parity testing.

Captures:
  post_embed    after embedding lookup (before any layers)
  layer_N       output of LFM2 layer N  (0 .. n_layers-1)
  final_norm    after embedding_norm RMSNorm
  cls_raw       position-0 (CLS) vector before L2 normalisation
  cls_norm      L2-normalised CLS vector (final model output)

Usage:
  PYTHONPATH=python HF_HOME=... TRANSFORMERS_OFFLINE=1 \\
  python tools/dump_lfm2_reference.py \\
      --model /tmp/lfm2-embed-model \\
      --text "hello world" \\
      --output /tmp/lfm2-ref.gguf
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "ggml" / "scripts"))
try:
    import gguf
except ImportError:
    print("pip install gguf", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Local directory for LFM2.5-Embedding-350M")
    parser.add_argument("--text", default="hello world",
                        help="Input text (without prefix)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--texts-file", help="JSON list of texts; --output becomes a reference directory")
    parser.add_argument("--long-context", action="store_true", help="Add a >512-token case to a reference suite")
    parser.add_argument("--gguf-weights", type=Path,
                        help="Replace HF weights with exactly dequantized GGUF weights; isolate runtime arithmetic")
    args = parser.parse_args()

    # Load tokenizer + model via AutoModel (needs trust_remote_code for
    # Lfm2BidirectionalModel class registered in modeling_lfm2_bidirectional.py)
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
    torch.set_num_threads(args.threads)
    print(f"Loading tokenizer from {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model from {args.model} ...")
    config = json.loads((Path(args.model) / "config.json").read_text())
    is_mlm = "Lfm2BidirectionalForMaskedLM" in config.get("architectures", [])
    if args.gguf_weights and not is_mlm:
        parser.error("--gguf-weights requires the masked encoder checkpoint")
    model = (AutoModelForMaskedLM if is_mlm else AutoModel).from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    if args.gguf_weights:
        from lfm2_gguf_reference_weights import load_gguf_weights
        load_gguf_weights(model, args.gguf_weights)

    # Tokenise

    # ----------------------------------------------------------------
    # Forward hooks
    # ----------------------------------------------------------------
    intermediates = {}

    def hook_embed(module, inp, out):
        intermediates["post_embed"] = out.detach().float().cpu().squeeze(0).numpy()

    # The backbone is model itself (Lfm2BidirectionalModel / Lfm2Model).
    # embed_tokens + layers + embedding_norm live directly on it.
    backbone = model.lfm2 if is_mlm else model

    embed_hook = backbone.embed_tokens.register_forward_hook(hook_embed)

    layer_hooks = []
    for i, layer in enumerate(backbone.layers):
        def make_hook(idx):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                intermediates[f"layer_{idx}"] = out.detach().float().cpu().squeeze(0).numpy()
            return hook
        layer_hooks.append(layer.register_forward_hook(make_hook(i)))

    if hasattr(backbone, "embedding_norm"):
        def hook_fn(module, inp, out):
            intermediates["final_norm"] = out.detach().float().cpu().squeeze(0).numpy()
        norm_hook = backbone.embedding_norm.register_forward_hook(hook_fn)
    else:
        norm_hook = None

    # ----------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------
    texts = json.loads(Path(args.texts_file).read_text()) if args.texts_file else [args.text]
    if args.long_context:
        if not args.texts_file:
            raise ValueError("--long-context requires --texts-file")
        texts.append("The quick brown fox jumps over the lazy dog. " * 64 + "The capital of France is <|mask|>.")
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        raise ValueError("texts-file must contain a nonempty JSON string list")
    if args.texts_file:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "index.json").write_text(json.dumps(texts, ensure_ascii=False, indent=2))
        # The LM matrix is padded beyond tokenizer length (64402 -> 65536).
        # HF decodes out-of-tokenizer vocabulary IDs to an empty string.
        decoded = [tok.decode([i], clean_up_tokenization_spaces=False) for i in range(config["vocab_size"])]
        (Path(args.output) / "decoded_vocab.json").write_text(json.dumps(decoded, ensure_ascii=False))
        import transformers
        (Path(args.output) / "versions.json").write_text(json.dumps({
            "torch": torch.__version__, "transformers": transformers.__version__,
            "architecture": config["architectures"], "attention": "eager", "dtype": "float32",
            "gguf_weights": args.gguf_weights.name if args.gguf_weights else None}, indent=2))
    for case, text in enumerate(texts):
        output_path = str(Path(args.output) / f"{case:03d}.gguf") if args.texts_file else args.output
        intermediates.clear()
        enc = tok(text, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        print(f"Text: {text!r}\nToken IDs: {input_ids.tolist()}")
        with torch.no_grad():
            # Keep the official backbone and tied LM head, but project only masked
            # rows rather than allocating sequence_length * vocabulary logits.
            out = backbone(**enc, use_cache=False)

        lhs = out.last_hidden_state[0].float()  # (T, H)

        # CLS = position 0
        cls_raw = lhs[0].numpy().astype(np.float32)
        cls_norm = cls_raw / (np.linalg.norm(cls_raw) + 1e-12)

        intermediates["cls_raw"]  = cls_raw
        intermediates["cls_norm"] = cls_norm
        mask_positions = (input_ids == tok.mask_token_id).nonzero().flatten() if is_mlm else torch.empty(0, dtype=torch.long)
        if len(mask_positions):
            with torch.no_grad():
                logits = model.lm_head(lhs[mask_positions])
            intermediates["masked_logits"] = logits.float().numpy()
            for pos, row in zip(mask_positions.tolist(), logits):
                values, ids = row.topk(5)
                print(f"Mask {pos}: {[(int(i), tok.decode([i]), float(v)) for i, v in zip(ids, values)]}")

        print(f"\nCaptured: {sorted(intermediates.keys())}")

        # ----------------------------------------------------------------
        # Write GGUF reference archive
        # ----------------------------------------------------------------
        writer = gguf.GGUFWriter(output_path, arch="lfm2_ref")
        # NB: GGUFWriter(arch=...) already writes general.architecture; adding it again
        # raises "Duplicated key name 'general.architecture'" on newer gguf.
        writer.add_string("ref.text", text)
        writer.add_array("ref.input_ids", input_ids.tolist())
        if len(mask_positions):
            writer.add_array("ref.mask_positions", mask_positions.tolist())

        for name in sorted(intermediates.keys()):
            arr = np.ascontiguousarray(intermediates[name], dtype=np.float32)
            # Shapes: post_embed / layer_N / final_norm are (T, H); cls_* are (H,)
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
            print(f"  {name}: shape={arr.shape}, mean={arr.mean():.6f}")

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        import os
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nWrote: {output_path} ({size_mb:.1f} MB)")

    # Cleanup hooks
    embed_hook.remove()
    for h in layer_hooks:
        h.remove()
    if norm_hook:
        norm_hook.remove()



if __name__ == "__main__":
    main()
