# MiniCPM-OCR stored-type decoder experiment v1

This is a default-off **source candidate**, with no build, numerical or speed
result yet. `CRISPEMBED_MINICPM_OCR_GRAPH=1` selects it only in MiniCPM-OCR.
Unset, empty and literal `0` retain `decoder_cpu`. Existing model, source,
preprocessing, prompts, image layout, five-case gold, stop IDs and token cap
remain unchanged. Quantized tensors additionally require the existing
`CRISPEMBED_QWEN35_QUANT=1` gate. Calibration mode
`CRISPEMBED_QWEN35_IMATRIX=1` is explicitly unsupported and rejected rather
than silently skipping its observer. No default or publication follows from this file.

The original remains Transformers 5.21.0 at the checkpoint and exact pins in
[minicpm-ocr-source.json](minicpm-ocr-source.json). The original inference
blueprint and file/line decisions are in
[minicpm-ocr-inference-spec.md](minicpm-ocr-inference-spec.md). In particular,
the source defines zero-centered RMSNorm, gated full attention, partial NeoX
RoPE on equal sequential axes, causal chronology independent of coordinates,
and a hybrid linear/full-attention schedule. None is inferred from speed.

The source inventory before implementation identified these reusable pieces:

| Existing piece | Reuse in this candidate |
| --- | --- |
| `src/qwen35_decoder.cpp` | Layer schedule, residual/FFN order, zero-centered RMS arithmetic, final last-token projection |
| `src/qwen35_attention.cpp` | Gated-Q split, head norms, equal-axis RoPE, causal softmax and F32 KV layout/arithmetic |
| `src/qwen35_linear.cpp` | Matrix validation, small F32-vector loading and linear projection layout |
| `src/qwen35_delta.cpp` | **Called directly:** convolution, recurrent state update, ggml gated-delta operator and gated normalization |
| `src/minicpm_ocr_vision.cpp` | The existing backend graph allocator/borrowed weight lifetime pattern; vision arithmetic unchanged |
| `src/core/gguf_loader.h` | Existing loaded tensor ownership, original stored tensor types and backend buffers |

The new `src/qwen35_graph_decoder.cpp` is composed from those repository
routines. It sends each original stored F32/F16/Q8_0/Q4_K/Q6_K matrix directly
to `ggml_mul_mat`, explicitly selecting F32 accumulation precision. Its
per-context CPU backend is reused; graph/input/output allocations end after
synchronous readback. Loaded weight tensors are borrowed without mutation or
conversion and must have host backend buffers. Small norms, convolution
weights and vectors still use the existing F32 conversion. The norm, SiLU,
full-attention and residual arithmetic is retained from the old path. This is
a **projection-graph decoder and prefill**, not a claim that the whole network
or full-attention softmax is fused. It does not implement a new attention
kernel or copy an upstream engine implementation.

Request state remains explicitly owned by `decoder_outputs::cache`: six
full-attention F32 KV caches and eighteen linear conv/recurrent states. Cached
single-token calls receive the previous state; independent requests start with
null previous state. This preserves the qualified cache layout and chronology.
The candidate still copies growing full-attention KV vectors; eliminating those
copies and fusing more operators are separate steps after the first measured
A/B. No persistent process-global cache, weight expansion or hidden fallback is
introduced. A selected candidate failure clears output and rejects the request.

The initial profile is intentionally strict: batch1, width1024, vocabulary
248100, tied embedding/head,24 layers, alternating three linear/one full,
4096 total tokens, exact original head/RoPE/epsilon configuration, and position
axes equal to `past + arange(tokens)`. Other Qwen/Ovis/Josev engines continue
to use their existing route.

`tests/minicpm-graph` is a standalone hosted build. Its actual C++ controls
cover stored F32/F16/Q8_0/Q4_K projection against the retained dequantized
control, original bytes/types, same-context repeat, independent contexts,
unset/empty/0 gates, invalid inputs, quant gate, failure clearing, clear and
recovery. Release checks use explicit requirements, not disabled assertions.
These small fixtures do not qualify model arithmetic. The root integrator must
add `src/qwen35_graph_decoder.cpp` to MiniCPM's `CRISPEMBED_SOURCES` before
building the real CLI/static/shared library; no root CMake change is included
in this owned source patch.

Before adoption, hosted execution must retain:

1. The existing benchmark's exact uploaded source, original Transformers and
   llama.cpp external-speed baselines. No original gold recapture.
2. F32/F16 graph off/on for **all five** immutable cases (`line`, `para-en`,
   `table`, `tall`, `tiny`), all input/generated IDs, text, EOS/cap and confidence.
   Compare prefill layers/norm/logits through actual `crispembed_diff::Ref`
   lossless F32 adapters plus independent metrics at fixed0.002/0.99999;
   preserve first divergence, mean/RMS, L2 and magnitudes, even on failure.
3. A test-only, production-excluded same-input projected-stage comparison,
   including actual stored tensor types. A projection component result cannot
   substitute for the composed decoded panel. Ordinary/observed/repeat outputs
   must be neutral before comparing performance.
4. Paired per-stage and whole-case timings on the same host/thread count,
   with actual source/build and model hashes. A speed target is not evidence.

Q8_0 currently awaits the running benchmark's all-five verdict. The publisher
recipe now agrees with that benchmark: preserve the4-D
`v.emb.patch_embedding.weight` at F16. This fixes format/load compatibility;
it does not prove Q8 numerical or decoded quality.

A distinct protected Q4_K recipe must be declared before outputs. Preserve
vision, merger, patch embedding, embeddings/tied vocabulary head, norms,
convolution and all small vectors. Only decoder2-D projection matrices may
be selected. Validate every actual stored type and unchanged protected bytes,
then run the identical five-case decoded and metric gates before publication.
No Q4 model has been produced or qualified by this preparation.

The follow-up hosted recipe is `tools/kaggle/minicpm-ocr-graph`. It requires
an explicitly rendered immutable source commit, the existing mounted original
capture,32GiB scratch and10GiB free RAM. It builds root CLI/quantizer and a
separate observed probe, with no upstream framework installation or recapture.
The exact original report is220355B, SHA256
`aeb1756dcbcfb72d72c13cf5a8d8131bb90e8d8884cf07c70723cc1fe4a64c32`.
The source contract/driver in that completed original report match the frozen
repository files; input lengths are256/1290/1290/2372/274 and generated counts
19/63/56/279/3. Full original row/array SHA verification is hosted only.

`CRISPEMBED_MINICPM_GRAPH_AUDIT` compiles an observer only into
`minicpm-graph-probe`. The recipe checks its symbol absent from the real CLI
and present in the separate observed executable. The probe requires exactly187
projection calls. It records **only first16 plus last16 token rows** at each
projection boundary, compares those actual graph outputs against the retained
same-input F32-materialized control, and publishes the original stored type,
selected row indices, full token count and weight name. This is a bounded
operator attribution, not full projected-tensor coverage. The full26
prefill outputs are separately required byte-identical to the real root CLI.
Ordinary/observed/repeat prefill outputs and all cache values must be exact.

The new MiniCPM-specific test adapter preserves raw F32 bits, singleton batch1
alignment is explicitly declared, and the actual Ref comparator exits0/1/2
for passed/numerically rejected/structurally rejected. Row-sensitive positive
and negative controls run before model conversion. Every full prefill stage
uses fixed0.002 maximum absolute and0.99999 minimum row cosine, independent
NumPy metrics, RMS/mean error, both L2 norms and min/max magnitudes. Adapter
shape bounds are versioned separately; historical adapters/guards are unchanged.

The frozen panel runs all five cases under F32 off/on, F16 off/on, Q8_0 on,
and protected Q4_K on, retaining all30 case-arm outcomes even if numerics fail.
First-stage divergence and detailed per-case Ref records are retained. Existing
native dump mode also records all generated confidences; they must be finite,
bounded and complete, and paired F32/F16 off/on confidences are checked at0.002.
The original capture has no full generated confidence vector: no original
full-confidence identity is claimed.

The protected Q4_K input is the exact same-run F16 export. Only the186 declared
decoder2-D projection matrices may become actual Q4_K. The other593 tensors
must retain exact type, shape and payload SHA256, including the tied head,
vision, patch embedding, projector, norms, convolution and scalar/vector state
parameters. `graph_panel.py` authenticates this **before any forward**. Its
candidate quality decisions do not reuse a quantized-only permissive cosine.
A rejected quantized arm stays rejected even if it is faster or decodes the
small panel correctly. No automatic publication is implemented by this kernel.


Both builds explicitly use the root's fixed CPU profile: GGML_LLAMAFILE=ON,
GGML_OPENMP=OFF, GGML_NATIVE=OFF and CRISPEMBED_NATIVE=OFF. The standalone
copies the exact root CPU ISA selection helper, including its platform guards;
it does not invent a second ISA policy. Both actual configured caches and
actual compiler argument rows must agree before model conversion. The guard
checks all six shared Qwen implementation TUs for the same fixed ISA and
contraction-off, plus the actual ggml CPU C/C++ backend definitions and ISA.
The small `build-profile.json` retains these compiler/cache observations.
This repairs a preparation mismatch: the submodule's LLAMAFILE and OpenMP
defaults differ from the root's. No inference result is attributed to it.


The separate Q8 publication barrier also rejects partial, wrong-schema or
wrong-original-revision benchmark producers. Both selected legacy Q8 arms
(`q8_0-t4` and `q8_0-t2`) must finish all five cases with exit0, exact-ID/text
predicates, zero original-agreement CER and correct generated/reference counts.
That legacy producer intentionally retained predicates/counts after discarding
the actual vectors. The publication receipt states this evidence limit instead
of claiming an independent ID-vector rederivation. The full-ID graph panel is
separate evidence; neither a partial initial benchmark nor a queued job can
satisfy publication.


The corrected shared-source worker also compiles and executes root
`test-ultrabert-profile` and standalone `josev-quant-controls` before any
checkpoint conversion. They are model-free precision-marker/negative-loader
controls, not inference or qualification of those families. Their exact source
and command logs are retained in the same run; no extra CPU session is created.


Before the first actual inference, execution order is explicitly frozen as
**case-major**: line, paragraph, table, tall, tiny; within each, F32 off/on,
F16 off/on, Q8 on, protected Q4_K on. This exposes the first quantized-case
diagnostics early while retaining exactly the same30 case-arms, inputs,
metrics and final all-case qualification. Numerical rejection never drops
later rows. The initial v1 scratch failure produced no model outputs.


Actual version2 compiled the root and observed targets and passed stored-type,
Ref, Ultra profile and Josev quant model-free controls. Before any inference,
it rejected Q8's patch artifact: the generic quantizer had flattened the F32
patch to `[588,1152]` before its protected-patch copy branch, ignoring the
literal F16 override. Q4's186/593 stored-type/protected-byte checks passed
separately; zero model cases ran. The original version2 producer is retained
unchanged. Neither artifact typing nor component controls qualify the model.

The bounded quantizer correction applies only to an **exact-name** `=f16`
request on a safe F32 four-dimensional patch weight, before the old flattening
step. It preserves all axes and streams finite FP16 rounding in fixed chunks.
Norm/bias/LoRA and host-only kernel names, PP-OCRv6 and Tesseract keep existing
critical rules; wildcard and block-quant overrides keep the old behavior.
No-override behavior is unchanged. The hosted actual C++ quantizer fixture
checks dtype, all axes, half-precision tie-even values and unchanged default,
F32 exclusions and protected rules before bulk conversion. The model preflight
now requires Q8's patch dtype, shape and payload SHA to equal the same-run
F16 source exactly. No original inputs or numeric budgets change.
