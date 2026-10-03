#!/usr/bin/env python3
"""Check the production C ABI/Python wrapper against official HF reference dumps.

Generate references once with tools/dump_lfm2_reference.py --texts-file
tests/lfm2_encoder_cases.json --output REFS --model HF_DIRECTORY. Then run
this script --model MODEL.gguf --refs REFS --lib build/libcrispembed.so.
No torch/transformers needed to replay. Quant thresholds are explicit; F16
port parity and quantization quality must be assessed separately.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys

import numpy as np
from gguf import GGUFReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from crispembed import CrispEmbed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--max-relative-error", type=float, default=0.02)
    ap.add_argument("--require-top1", action="store_true")
    ap.add_argument("--measure-only", action="store_true", help="Report quantization damage without calling it parity")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    os.environ["CRISPEMBED_LFM2_ENCODER"] = "1"
    texts = json.loads((args.refs / "index.json").read_text())
    model = CrispEmbed(args.model, n_threads=args.threads, lib_path=args.lib)
    assert model.has_masked_lm and not model.has_colbert
    hp = model._lib.crispembed_get_hparams(model._ctx).contents
    assert (hp.n_vocab, hp.n_embd, hp.n_head, hp.n_layer, hp.n_intermediate, hp.n_max_tokens) == (65536, 1024, 16, 14, 2560, 128000)
    decoded = json.loads((args.refs / "decoded_vocab.json").read_text())
    for token_id, expected in enumerate(decoded):
        assert model.token_bytes(token_id).decode("utf-8", errors="replace") == expected, ("token decode", token_id)
    print(f"PASS: all {len(decoded)} vocabulary pieces decode exactly like Hugging Face")
    summary = []
    top1_matches = masks_total = 0

    def compare(label, mine, reference):
        assert mine.shape == reference.shape, (label, mine.shape, reference.shape)
        assert np.isfinite(mine).all(), label
        a, b = mine.astype(np.float64), reference.astype(np.float64)
        rows_a, rows_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        cosine = np.sum(rows_a * rows_b, axis=1) / (np.linalg.norm(rows_a, axis=1) * np.linalg.norm(rows_b, axis=1))
        relative = float(np.linalg.norm(a - b) / np.linalg.norm(b))
        norms = (float(np.linalg.norm(a)), float(np.linalg.norm(b)))
        print(f"{label}: cos_min={cosine.min():.8f} relative_error={relative:.6f} |mine|={norms[0]:.6f} |ref|={norms[1]:.6f}")
        if not args.measure_only:
            assert cosine.min() >= args.min_cos, label
            assert relative <= args.max_relative_error, label
        return {"cos_min": float(cosine.min()), "relative_error": relative, "mine_norm": norms[0], "ref_norm": norms[1]}

    for case, text in enumerate(texts):
        ref = GGUFReader(args.refs / f"{case:03d}.gguf")
        tensors = {t.name: np.array(t.data, copy=True) for t in ref.tensors}
        expected_ids = np.asarray(ref.fields["ref.input_ids"].contents(), dtype=np.int32)
        ids, raw = model.encode_tokens(text, normalize=False)
        np.testing.assert_array_equal(ids, expected_ids)
        entry = {"text": text, "tokens": len(ids), "hidden": compare(f"case {case} hidden", raw, tensors["final_norm"])}
        ids2, normalized = model.encode_tokens(text)
        np.testing.assert_array_equal(ids, ids2)
        np.testing.assert_allclose(normalized, raw / np.linalg.norm(raw, axis=1, keepdims=True), atol=2e-6, rtol=2e-5)
        # Alternating graph shapes/outputs must not corrupt scheduler reuse.
        dense = model.encode(text)
        np.testing.assert_allclose(np.asarray(dense).reshape(-1), normalized[0], atol=2e-6, rtol=2e-5)
        if "masked_logits" in tensors:
            positions, logits = model.masked_logits(text)
            np.testing.assert_array_equal(positions, ref.fields["ref.mask_positions"].contents())
            entry["logits"] = compare(f"case {case} logits", logits, tensors["masked_logits"])
            predicted, expected = logits.argmax(1), tensors["masked_logits"].argmax(1)
            matches = int(np.sum(predicted == expected))
            top1_matches += matches
            masks_total += len(positions)
            entry["top1_matches"] = matches
            entry["masks"] = len(positions)
            if args.require_top1:
                np.testing.assert_array_equal(predicted, expected)
            filled = model.fill_mask(text.replace("<|mask|>", "[MASK]"))
            for i, result in enumerate(filled):
                assert result["position"] == int(positions[i])
                winner = result["predictions"][0]
                assert winner["token_id"] == int(predicted[i])
                assert winner["token"] == decoded[int(predicted[i])]
                assert winner["logit"] == float(logits[i, predicted[i]])
                probabilities = np.exp(logits[i].astype(np.float64) - logits[i].max())
                probabilities /= probabilities.sum()
                assert abs(winner["score"] - probabilities[predicted[i]]) < 1e-12
                print(f"  mask {positions[i]}: {winner['token']!r} (ID {winner['token_id']}, reference {expected[i]})")
            _, repeated = model.masked_logits(text)
            np.testing.assert_array_equal(logits, repeated)
        summary.append(entry)

    # Failed calls clear output counts/pointers rather than exposing stale state.
    n, vocab = ctypes.c_int(99), ctypes.c_int(99)
    positions = ctypes.cast(ctypes.c_void_p(1), ctypes.POINTER(ctypes.c_int32))
    ptr = model._lib.crispembed_masked_logits(model._ctx, b"no mask here", ctypes.byref(n), ctypes.byref(vocab), ctypes.byref(positions))
    assert not ptr and n.value == 0 and vocab.value == 0 and not positions
    n, dim = ctypes.c_int(99), ctypes.c_int(99)
    ptr = model._lib.crispembed_encode_tokens(model._ctx, None, ctypes.byref(n), ctypes.byref(dim))
    assert not ptr and n.value == 0 and dim.value == 0
    assert not model._lib.crispembed_last_token_ids(model._ctx)
    for k in (0, -1, True, 65537):
        try:
            model.fill_mask("<|mask|>", top_k=k)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid top_k={k}")
    assert model.token_bytes(5242) == b" Paris"
    assert model.token_bytes(16) == b"<|mask|>"
    for token_id in (-1, 65536):
        try:
            model.token_bytes(token_id)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted invalid token ID")

    os.environ["CRISPEMBED_LFM2_ENCODER"] = "0"
    assert not model.has_masked_lm
    n, dim = ctypes.c_int(99), ctypes.c_int(99)
    ptr = model._lib.crispembed_encode_tokens_raw(model._ctx, b"hello", ctypes.byref(n), ctypes.byref(dim))
    assert not ptr and n.value == 0 and dim.value == 0
    assert not model._lib.crispembed_last_token_ids(model._ctx)
    assert np.isfinite(model.encode("hello world")).all()
    os.environ["CRISPEMBED_LFM2_ENCODER"] = "1"
    report = {"model": Path(args.model).name, "cases": summary, "top1_matches": top1_matches, "masks": masks_total}
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    verdict = "MEASURED quantization" if args.measure_only else "PASS reference parity"
    print(f"{verdict}: {len(texts)} cases; exact token IDs; normalized/pooled consistency; {top1_matches}/{masks_total} mask predictions; repeated calls and failure contracts")


if __name__ == "__main__":
    main()
