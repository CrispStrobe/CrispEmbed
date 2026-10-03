#!/usr/bin/env python3
"""Separate checkpoint rounding, activation arithmetic, and upstream-runtime drift.

Upstream raw output is generated with tools/lfm2_llama_reference.cpp using the
binary input archive from --write-inputs. All comparisons use exact reference IDs.
"""

import argparse, json, struct, sys, os
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from crispembed import CrispEmbed


def tensors(path):
    r = GGUFReader(path)
    return r, {t.name: np.array(t.data, copy=True) for t in r.tensors}


def metric(a, b):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    a, b = a.astype(np.float64), b.astype(np.float64)
    x, y = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dict(
        cos_min=float(cos.min()),
        cos_global=float((a * b).sum() / (na * nb)),
        relative_error=float(np.linalg.norm(a - b) / nb),
        mine_norm=float(na),
        ref_norm=float(nb),
    )


def logits_metrics(a, b, decoded):
    r = metric(a, b)
    p, e = a.argmax(1), b.argmax(1)
    r["top1_matches"] = int((p == e).sum())
    r["masks"] = len(e)
    r["predictions"] = []
    for x, y, pi, ei in zip(a, b, p, e):
        top = np.sort(y.astype(np.float64))[-2:]
        r["predictions"].append(
            dict(
                predicted=int(pi),
                reference=int(ei),
                token=decoded[int(pi)],
                reference_token=decoded[int(ei)],
                reference_margin=float(top[1] - top[0]),
                reference_winner_rank=int((x > x[ei]).sum() + 1),
            )
        )
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--same-weights-refs", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--lib", default="build/libcrispembed.so")
    ap.add_argument("--upstream-prefix", type=Path)
    ap.add_argument("--write-inputs", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    texts = json.loads((args.refs / "index.json").read_text())
    if args.write_inputs:
        with args.write_inputs.open("wb") as out:
            out.write(struct.pack("<I", len(texts)))
            for i, text in enumerate(texts):
                ref = GGUFReader(args.refs / f"{i:03}.gguf")
                ids = np.asarray(ref.fields["ref.input_ids"].contents(), dtype="<i4")
                data = text.encode("utf-8")
                out.write(struct.pack("<I", len(data)))
                out.write(data)
                out.write(struct.pack("<I", len(ids)))
                out.write(ids.tobytes())
        return
    if not args.model or not args.output:
        ap.error("--model and --output required for comparison")
    os.environ["CRISPEMBED_LFM2_ENCODER"] = "1"
    os.environ.pop("CRISPEMBED_IMATRIX_OUT", None)
    model = CrispEmbed(args.model, n_threads=4, lib_path=args.lib)
    gguf = GGUFReader(args.model)
    emb = next(
        t
        for t in gguf.tensors
        if t.name in ("token_embd.weight", "lfm.embed_tokens.weight")
    )
    weight = dequantize(emb.data, emb.tensor_type).astype(np.float32)
    decoded = json.loads((args.refs / "decoded_vocab.json").read_text())
    rows = []
    for i, text in enumerate(texts):
        ref, t = tensors(args.refs / f"{i:03}.gguf")
        ids, raw = model.encode_tokens(text, normalize=False)
        np.testing.assert_array_equal(ids, ref.fields["ref.input_ids"].contents())
        row = {
            "case": i,
            "tokens": len(ids),
            "native_vs_original": {"hidden": metric(raw, t["final_norm"])},
        }
        hidden = {"native": raw}
        if args.same_weights_refs:
            qref, q = tensors(args.same_weights_refs / f"{i:03}.gguf")
            np.testing.assert_array_equal(ids, qref.fields["ref.input_ids"].contents())
            row["native_vs_same_weights"] = {"hidden": metric(raw, q["final_norm"])}
            row["python_same_weights_vs_original"] = {
                "hidden": metric(q["final_norm"], t["final_norm"])
            }
        if args.upstream_prefix:
            up = np.fromfile(
                str(args.upstream_prefix) + f"-{i:03}.f32", dtype=np.float32
            ).reshape(raw.shape)
            hidden["upstream"] = up
            row["upstream_vs_original"] = {"hidden": metric(up, t["final_norm"])}
            row["native_vs_upstream"] = {"hidden": metric(raw, up)}
        if "masked_logits" in t:
            pos, logits = model.masked_logits(text)
            np.testing.assert_array_equal(
                pos, ref.fields["ref.mask_positions"].contents()
            )
            row["native_vs_original"]["logits"] = logits_metrics(
                logits, t["masked_logits"], decoded
            )
            if args.same_weights_refs:
                row["native_vs_same_weights"]["logits"] = logits_metrics(
                    logits, q["masked_logits"], decoded
                )
                row["python_same_weights_vs_original"]["logits"] = logits_metrics(
                    q["masked_logits"], t["masked_logits"], decoded
                )
            for name, h in hidden.items():
                # Match the official helper's F32 projection, with proper Q dequantization.
                projected = h[pos] @ weight.T
                row[name + "_f32_head_vs_original"] = {
                    "logits": logits_metrics(projected, t["masked_logits"], decoded)
                }
        rows.append(row)
        print("Measured case", i, flush=True)
    result = {"model": Path(args.model).name, "cases": rows, "summary": {}}
    for key in rows[0]:
        if key in ("case", "tokens"):
            continue
        data = [r[key] for r in rows if key in r]
        s = {}
        if "hidden" in data[0]:
            s.update(
                hidden_min_cos=min(r["hidden"]["cos_min"] for r in data),
                hidden_min_global=min(r["hidden"]["cos_global"] for r in data),
                hidden_max_relative=max(r["hidden"]["relative_error"] for r in data),
            )
        if any("logits" in r for r in data):
            masks = [r["logits"] for r in data if "logits" in r]
            s.update(
                mask_matches=sum(r["top1_matches"] for r in masks),
                masks=sum(r["masks"] for r in masks),
            )
        result["summary"][key] = s
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
