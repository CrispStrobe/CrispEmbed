# LightOn4B original BF16 single-prefill attribution

Actual CPU-only source8d3b2331d1b57e285f7ba348fc1eacd689b79aa8 completes
one original prefill and streams71 stage records/130163104 bytes. The unchanged
07543dcc BF16 setup authenticates the original checkpoint, processor/tokenizer,
image/prompt and exact runtime packages. All five prepared inputs reproduce the
retained original bytes. The full248320 first-score row is bit-exact at SHA
`a4bece2310050081b502a3910a19469e47016829d74abebb29861739f3205ffd`, with ID27,
matching the observer-off original receipt. There is exactly one model forward;
no F32 model, GPU, legacy cache dump or177-token generation repeat occurs.

The [unaltered streamed-stage manifest](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-prefill.json)
retains original dtype, exact shape/layout, magnitude and payload hash for every
stage. Its SHA is036a936cdbd360208b5cf54c2604d671b9a919f1c8777bc7cf97b2585b010a78.
The [independent review](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-prefill-review.json)
binds sources, runtime, first score and measured resources. Large arrays stay
remote; only small reports were downloaded for review.

The original decoder input is[1,154,2560], vision tokens are[560,1024], and merged
image tokens are[140,2560]. Original position IDs are[4,1,154]: the chronological
text plane must be verified and removed explicitly before comparing native THW
axes. Do not silently reshape or transpose incompatible arrays.

[The exact upstream model source](https://github.com/huggingface/transformers/blob/75d3bdcd4b3ba70cee21287218d9764f33da41f0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
casts processor pixels to BF16 before the patch convolution; position weights
and interpolation products/sums also use BF16. Native's current vision executes
F32 arithmetic over the protected weights. A shared precision difference is not
proof that the Q4-specific decoded regression is caused there. Compare the first
differing stage and absolute errors/magnitudes, then use identical-input/weight
operator controls before changing arithmetic. Numerical/decode/EOS and Q8/Q4
qualification remain open; one-token observer neutrality proves no document
completion or speed improvement.
