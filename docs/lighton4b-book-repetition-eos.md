# Retained book repetition and EOS diagnostics

Artifact-only Kaggle execution144605485/version1 completed without loading a
model or repeating inference. The [selected numerical receipt](../tests/results/qwen35-ocr/lightonocr3-4b-book-repetition-eos-review.json)
records the complete1,722,279-byte report SHA256, exact three source receipts and
archive hashes. Every complete score row was independently hashed and checked
against its retained greedy argmax. Token sequences and text remain private.
The [protocol and implementation](../tools/book-repetition-eos/README.md) declare
fixed token-window screens and stable float64 EOS softmax/rank calculations;
three model-free controls pass.

| Arm | Generated tokens | Duplicate4/8/16-token windows | First repeated8-token window | EOS selected |
|---|---:|---|---:|---|
| Source |654|1 /0 /0|none|yes, row653|
| Planner OFF |660|1 /0 /0|none|yes, row659|
| Planner ON |1536|938 /907 /869|row527|no; cap reached|

The ON first repeated16-token window starts607. A stricter contiguous-cycle
screen (period1–64, at least3 cycles and12 tokens) finds no match in any arm.
The degradation therefore has substantial repeated longer substrings without
establishing a short periodic loop under that declared screen. These token
structure counts describe this retained output; they are not a quality scorer,
repetition penalty or evidence supporting a stopping heuristic.

Source and ON share405 output tokens. At score row405 their consumed prefixes
are identical: summed EOS probabilities are4.11496e-7 and3.27019e-7, respectively;
the two EOS ranks are9728/1264 and9084/1437. Their top-two score margins are
0.374729 and0.0114956. This disagreement occurs while both trajectories assign
little probability to EOS. Source/OFF share only92 output tokens, so its later
rows cannot enter the same-prefix comparison.

Across the entire independent ON trajectory, neither EOS is ever top-ranked.
Its maximum summed EOS probability is0.0287315 at row903. At the capped final
row1535, summed EOS probability is7.03971e-8 and top-two margin15.3603. The
source ends with EOS probability0.921799 at653; OFF ends with0.953628 at659.
These ending rows consume different prefixes. They describe each trajectory's
termination behavior and do **not** establish causal EOS suppression, a broken
stop-token mapping or a numerical parity failure at matching context.

No prompt, recipe, runtime, cap, reference, scorer or default was changed.
The candidate remains unqualified. Neutral source/native tracing at shared-prefix
row405 is the next attribution step; these diagnostics do not justify a runtime
fix or output truncation rule.
