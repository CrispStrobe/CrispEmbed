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

## Actual retained Q8/Q4 stage comparison

[Hosted42002886215](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42002886215)
authenticates all71 original payloads and both existing native smoke archives
before comparison. No original/native forward is repeated. Source image pixels
and THW positions are exact. Both native recipes have byte-identical vision,
raw text embedding and decoder input; their first recipe-specific difference is
`prefill.layer.0`.

The first shared difference is`vision.patch`: max absolute0.0126757622,
RMS0.000805617467, cosine0.999998609; original/native L2 are364.6366634/364.5889670
and absolute maxima4.3125/4.29982424. Position max absolute0.0263156891;
embedded0.0322451591. Vision layer23 reaches max1963.4453125/RMS8.26713570/
cosine0.994490075 against original absolute magnitude5728. Merged image
features have max1.67908716/RMS0.0112389284/cosine0.994596081. This shared
precision discrepancy precedes either quantized decoder.

First logits remain negative: Q8 max0.568259001/cosine0.998469735; Q4
max2.49422884/cosine0.976987556. Q8 smoke begins`<figure`, Q4`<table>`;
these two-token caps establish no EOS completion. Preserve all raw strict flags.
The [full comparison](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-native-stages.json)
and [review](../tests/results/qwen35-ocr/lightonocr3-4b-bf16-native-stages-review.json)
retain every stage's magnitudes/errors/source/payload hashes. The separate
BF16-vision semantics candidate requires actual stage and decoded replay before
any correction or quality claim.
