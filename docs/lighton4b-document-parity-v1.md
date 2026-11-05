# LightOn4B bounded document parity preparation

This protocol freezes one existing public-domain form before any original/native
model output. It supplies a real `crispembed_diff::Ref` archive comparer and
model-free controls. No original forward, native document inference, downloaded
checkpoint or document-quality result follows from preparation.

The exact contract is [the JSON protocol](lighton4b-document-parity-v1.json).
The original image is `tests/regression/images/cc0/simple_form.png`, 452×317,
SHA256 `3d986ec0d131e4e429127627929e12fd4845313f43849c26bb3d1231c3b038dc`.
The source manifest links its public-domain Commons parent. Existing medium
confidence human text explicitly records an occluded checkbox and an ambiguous
separator: this is a development diagnostic, not held-out calibration or a
benchmark. The frozen case uses image-only chat, no added text, thinking disabled,
greedy CPU/F32 eager generation, two threads, seed0, cap256, EOS248046 and
padding248044. Input RGB bytes and the exact template rendering are hashed.
The source-derived expected processor resize is448×320, grid1×20×28,560patches
and140merged image tokens. These geometry values are expectations; the actual
processor arrays have not been observed. A future adapter must check them and
freeze full tokenizer IDs, modality labels, pixel values and positional arrays
before native execution, without replacing the pinned request to fit outputs.

The original source is `lightonai/LightOnOCR-3-4B` revision
`b71010f095dc3735de7ee6b9969bff8f8f50b39a`; full checkpoint and tokenizer hashes,
package versions, metadata and helper hashes are fixed in JSON. Existing
`dump_qwen35_ocr_reference.py` observes the actual modules, all configured
vision/text blocks, recurrent/full-attention caches and generated scores. It
does not currently expose every final-normalization boundary required here.
The page dumper captures prefill final norm and all vision blocks only when its
explicit `vision_probes` option is used; its older comments mention twelve
vision blocks. Neither entry point is admitted as a complete4B document adapter.
The protocol admits no original or native producer source yet. A separately
reviewed version must bind the actual adapters and complete native format5
context/tokenizer/cache/vocabulary projection before model execution.

The comparer is executable now:

```sh
python tools/lighton4b_document_parity.py plan
cmake --build build --target crispembed-diff-lighton4b-document --parallel 2
python tools/lighton4b_document_parity.py compare \
  --original original-bundle --candidate native-bundle \
  --binary build/crispembed-diff-lighton4b-document --output comparison
```

Build and comparison belong on hosted workers. The preparation version fails
closed on unadmitted producer sources; there is deliberately no model-download
or forward command. Each future bundle contains `bundle.json` and a diagnostic
`stages.gguf`, with all tensors stored F32, never installed as model weights.
The manifest binds the protocol, immutable source/run, original framework,
request, archive SHA, ordered names and shapes, independent free IDs/text/stop,
and observer-neutral controls. Native manifests additionally bind the frozen
original receipt hash. The CLI requires all24 vision and32 decoder layers,
prefill/cached/full-prefix cache geometry, final norms, and all248320 logits for
every original-generated prefix. Independent native generation remains separate
from that teacher-forced trace. No reference cache enters native continuation.
Archives are bounded to2GiB each and stages to64Mi elements; capped original
source output is preserved as negative evidence and blocks native qualification.

The C++ comparer uses the unchanged `crispembed_diff::Ref`, fixed max error0.002
and minimum row cosine0.99999, without environment threshold overrides. It
reports every stage's max/mean/RMS error and reference/native magnitude, retaining
the earliest failure in frozen order. Input IDs, positions and RGB must be exact.
Missing/extra stages, shape drift and nonfinite arrays reject. Independent native
IDs/text/EOS/cap must also match; numerical success alone cannot pass the paired
document result. Quantized arms are blocked until a distinct F32/F16 closure and
separate licensed calibration/held-out protocol exist. Every model/public alias
and quality flag remains false.

Pending actual work: implement and source-admit the original adapter; verify
actual original processor geometry and complete EOS output; freeze raw vectors
and all-position logits before native dispatch; complete native format5
composition and run F32 then F16; compare their real staged archives and free
outputs. GPU availability does not remove these missing implementation gates.
