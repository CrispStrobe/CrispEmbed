#!/usr/bin/env python
"""Dump actual Python encoder intermediates and typed decisions at FP32."""

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import AutoModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    torch.set_num_threads(2)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    ).eval()
    model.encoder.language_model.config._attn_implementation = "eager"
    model.encoder.config._attn_implementation = "eager"
    args.output.mkdir(parents=True, exist_ok=True)
    stages = {}
    hooks = []

    def attach(module, name):
        def hook(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            stages[name] = output.detach().float().cpu().numpy()[0].copy()

        hooks.append(module.register_forward_hook(hook))

    body = model.encoder.language_model
    attach(body.embed_tokens, "post_embed")
    attach(body.ple, "ple")
    for i, layer in enumerate(body.layers):
        attach(layer, f"layer_{i}")
    attach(body.norm, "final_norm")
    attach(body.embedding_projection, "tokens")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "日本語の判断。¿Qué debería hacer el agente?",
        '  tabs\t and\nnewlines\n\n, escaped "quotes", emoji 🧭. ',
        "Literal <bos> <mask> <|audio|> <|video|> tokens.",
        "Decision workflow. " * 180,
    ]
    for case, text in enumerate(texts):
        stages.clear()
        features = model.tokenize([text])
        with torch.no_grad():
            embedding = model.embed(**features).cpu().numpy()
        writer = gguf.GGUFWriter(
            str(args.output / f"{case:03d}.gguf"), "gemma_decision_reference"
        )
        writer.add_string("ref.text", text)
        writer.add_tensor("input_ids", features["input_ids"].float().numpy())
        writer.add_tensor("embedding", embedding)
        for name, value in stages.items():
            assert np.isfinite(value).all(), name
            writer.add_tensor(name, value)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        print(
            f"reference {case}: {features['input_ids'].shape[1]} tokens / {len(stages)} stages",
            flush=True,
        )
    for hook in hooks:
        hook.remove()
    scenarios = [
        {
            "workflow": "agent_trace_observability",
            "state": {
                "agent": {"autonomy": "unsupervised"},
                "task": "Delete old branches",
                "constraints": ["Preserve release branches"],
                "trace_summary": {
                    "constraint_violations": 1,
                    "irreversible_actions": 1,
                },
            },
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "What should we do?",
                    "criteria": {
                        "continue": "Proceed",
                        "review": "Ask a human",
                        "stop": "Halt the agent",
                    },
                },
                "risk": {
                    "type": "score",
                    "instructions": "How risky is this?",
                    "criteria": ["none", "low", "high", "severe"],
                },
                "needs_review": {
                    "type": "noul",
                    "instructions": "Does a human need to review this?",
                },
            },
        },
        {
            "workflow": "soporte",
            "state": {
                "problema": "No puedo entrar en mi cuenta",
                "urgente": True,
                "precio": 12.5,
                "large_float": 1e7,
                "tiny_float": 1e-5,
                "negative_zero": -0.0,
                "list": ["a'b", "c\\d", True, None],
                "detalle": None,
            },
            "questions": {
                "accion": {
                    "criteria": ["ayudar", "escalar"],
                    "instructions": "¿Cómo responder?",
                },
                "nivel": {
                    "type": "score",
                    "criteria": {"10": "very high", "-2": "low", "1": "medium"},
                },
                "named": {
                    "type": "score",
                    "criteria": {"small": "low risk", "big": "high risk"},
                },
            },
        },
        {
            "workflow": "",
            "state": "plain text rather than an object",
            "questions": {
                "one": {"type": "choice", "criteria": {"only": "only option"}},
                "no": {"type": "noul"},
            },
        },
    ]
    decisions = []
    for case in scenarios:
        inputs, specs, bank = model._build_inputs(
            [(case["workflow"], case["state"], case["questions"], None)]
        )
        with torch.no_grad():
            result = model(**inputs)
        answers = model.decide(
            case["state"], case["questions"], workflow=case["workflow"]
        )
        decisions.append(
            {
                "request": case,
                "texts": bank,
                "answers": {k: v.to_dict() for k, v in answers.items()},
                "logits": result.logits.tolist(),
                "probabilities": result.probabilities.tolist(),
                "text_embeds": result.text_embeds.tolist(),
                "indices": {
                    k: inputs[k].tolist()
                    for k in (
                        "state_index",
                        "question_index",
                        "option_index",
                        "question_type",
                    )
                },
            }
        )
        print(
            f"decisions: {len(specs)} questions / {len(bank)} unique texts", flush=True
        )
    (args.output / "decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2)
    )
    (args.output / "versions.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "dtype": "float32",
                "attention": "eager",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
