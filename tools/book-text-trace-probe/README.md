# Book same-prefix trace preparation

This is a test-only source observer and Kaggle CPU worker. The
[source capture receipt](../../tests/results/qwen35-ocr/lightonocr3-4b-book-source405/v1/receipt.json)
records an observer-neutral run; comparison with native states remains pending.
The existing form live-evaluation callback is reused with
separate environment variables. Capture sample count0 (all actual prefill
graphs) and405 (the first differing source/native plannerON greedy prediction).
The sampling bound is1536 so the unchanged654-token source can reach EOS.
The16MiB per-array/256MiB total bounds admit242-row FFN matrices while retaining
bounded output. The index still records rejected noncontiguous/unsupported arrays.

The historical ON/OFF row92 difference is not an upstream-parity regression:
plannerON follows the source through405. The existing form source-fed proof
already establishes exact scheduled QKV for all154 rows; ON/OFF arithmetic
sensitivity alone does not support changing quantization or recurrence.

Before submission, independently review `worker.py` and provide an immutable
source-pinned Kaggle bootstrap mounting the original book input/response artifacts. Apply the immutable feature/sampler
observer, dispatch observer, then this patch. Run source OFF/ON independently to
full original EOS and require every raw score unchanged. No source BF16/F32
model is instantiated. Keep source graphs, raw scores and native arrays remotely.

For native capture, use the restored `lighton4b-native-document` harness with
existing `CRISPEMBED_LIGHTON4B_GREEDY_TRACE=1` and
`CRISPEMBED_LIGHTON4B_GREEDY_TRACE_POSITION=405`. Preserve greedy generation and
its existing256-row teacher limit; no driver changes are needed. Capture prefill layer
outputs and step405 layer outputs/norm/logits. First replay the historical plannerON
prefix/raw score rows to prove the new driver is neutral. Existing source capture
includes layer0 convolution and recurrent intermediates; full persistent-cache
capture needs verified source tensor names/layouts and explicit identity checks.
Do not label absent source state as observed or compare differently ordered state.

Only after locating the earliest source/native layer/stage on identical context
should a same-input operator replay be selected. Scalar native FFN SiLU versus
source ggml SwiGLU is a concrete remaining implementation difference, but its
historical first4 maximum error5.96e-8 does not establish the final decoded cause.
