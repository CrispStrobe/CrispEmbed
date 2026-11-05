# LightOn4B linear decoder block component

Draft PR148 stacks PR146 (`8157a3137`) and its reviewed text/grouped dependencies.
The separate `lighton4b_linear_block_cpu` entry executes one full-width linear
block at hidden2560, FFN9216 and16-key/32-value heads with width128. It follows
the real original decoder blueprint: zero-centered RMS, grouped GDN, attention
residual, zero-centered post-RMS, SwiGLU FFN and final residual. It returns
intermediate stages and the compact convolution/recurrent cache.

`CRISPEMBED_LIGHTON4B_LINEAR_BLOCK`, `CRISPEMBED_QWEN35_OCR` and
`CRISPEMBED_QWEN35_GDN_GROUPED_HEADS` must all be enabled. The new entry is
default off; existing decoder/cache code stays byte-identical. Shapes, layout,
nonaliasing tensor identities, allocated data and F32 protected parameters are
checked explicitly. Matrix storage admits F32/F16 headers; actual numerical
evidence below is F32 only. The API bounds batch1 and1–16 new tokens, tracks at
most4096 cumulative tokens and validates exact32768-value convolution and524288-
value recurrent caches. Previous caches are immutable inputs. This is a standalone
component, with no32-layer decoder, full-attention block or vocabulary projection.

Hosted [41976928016](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41976928016)
at `5b1c813dcaf54b7a6409e49b8a3dd1df9603c334` passes all33 original/native stage
pairs and52 native controls. The original Transformers5.5.4 module/file/function
pins and original CPU deterministic settings are checked before constructing
`Qwen3_5DecoderLayer`. All fourteen full-width parameter arrays are synthetic,
seed195032, with exact shapes/hashes recorded. No checkpoint is acquired.
The original runs first, capturing two-token prefill, one cached token and the
three-token full prefix. Native cache inputs come only from native prefill.
Nine original hidden/cache comparisons remain byte-exact after removing hooks.
The native controls cover disabled gates, token bounds, malformed shape/full-
attention rejection, cache lengths/token count, immutable input cache and stage
output writes. They do not cover every possible malformed weight or cache value.

Independent retained-artifact review rehashes all six directly recorded source
files and every original stage array, then recomputes all33 absolute errors and
minimum row cosines from the original/native raw vectors. Earliest nonzero error
is `prefill.input_norm`, max3.5762786865e-7 against magnitude1.429962039. The
largest stage error is2.8610229492e-6 at FFN activation. Final hidden max errors
are1.9073486328e-6 for prefill/full-prefix and1.3113021851e-6 for the cached token;
reference magnitudes are1.827731371,1.933295608 and1.933295250 respectively.
The [review receipt](../tests/results/qwen35-ocr/lightonocr3-4b-linear-block.json)
retains every measured stage and magnitude, original runtime/source pins,
parameter hashes, executable hash, observer neutrality and the exact source commit.
Raw vectors remain in the hosted artifact; generated synthetic weight GGUF stays
in runner scratch and is not a production artifact. Public receipt text replaces
provider scratch paths; this changes no array or metric.

Evidence is Linux CPU/F32, batch1,2+1/3 synthetic tokens at one linear layer.
F16, full-attention/32-layer composition, real checkpoint stages, final norm,
streamed full-vocabulary logits, tokenizer/preprocessing, actual decoded text,
EOS and quality remain open. Component parity does not qualify any model,
format5 recognition, public alias, release or default flip. Streamed projection
remains a separate default-off component to compose after decoder stage proof.
