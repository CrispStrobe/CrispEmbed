# fullstop-punc blueprint reference (XLM-R large)

Reference for **`oliverguhr/fullstop-punctuation-multilang-large`** (XLM-R
large, 24L/1024, 6 labels) — the model behind the `fullstop-punc` /
`fullstop-punc-q8` registry entries, i.e. what `--punct-model fullstop-punc`
actually downloads. Same format and comparator as
`tests/regression/xlmr_punc/`, which covers the 12L/768 `punctuate-all`.

```bash
python tests/xlmr_punc_parity.py build /path/to/fullstop-punc-q8_0.gguf \
    tests/regression/fullstop_punc/blueprint_ref.txt
```

## Measured — the registry artifacts are CORRECT

Both were downloaded from `cstr/fullstop-punc-multilang-GGUF` and checked:

| artifact | token ids | cos_min | max_abs | preds | decoded |
|---|--:|--:|--:|--:|--:|
| `fullstop-punc-q8_0.gguf` | 6/6 | 0.998680 | 0.4013 | **67/67** | 6/6 |
| `fullstop-punc-q4_k.gguf` | 6/6 | 0.922908 | 5.0325 | **67/67** | 6/6 |

Read those two rows together, in that order. q4_k's 0.923 looks alarming on its
own — more deviation than a quant is usually allowed to explain — and the way to
tell a lossy quant from a broken graph is to run the higher-precision arm of the
SAME graph. q8_0 comes back at 0.9987, so the graph is right and the q4_k dip is
the quantiser on a 24-layer model. Both arms agree with the blueprint on
**every** argmax, so the user-visible output is identical either way.

They also carry `tokenizer.ggml.scores`, which is what the local
`punctuate-all-f16.gguf` was missing (see `../xlmr_punc/README.md`) — without
them the runtime falls back to greedy longest-match and mis-segments Unigram
pieces. Having scores is necessary but not sufficient, which is why this
reference exists rather than just a metadata check.
