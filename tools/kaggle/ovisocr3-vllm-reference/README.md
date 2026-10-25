# Pinned OvisOCR3 original vLLM T4 recipe

This is a **prepared, undispatched** source recipe for the original
[OvisOCR3](https://huggingface.co/ATH-MaaS/OvisOCR3/tree/5f6ce6932efee9ec10c30746db55bd780214b7fd)
plain OCR prompt. It requires a successfully completed and independently
reviewed [OvisOCR3 CPU metadata run](../../ci-heavy/ovisocr3_metadata.py).
The launcher rejects the unresolved review, artifact and worker pins at its
first gate, before service queries, package installation or checkpoint access.
Fill them only from the committed review receipt and immutable worker commit.

The pinned source revision is `5f6ce6932efee9ec10c30746db55bd780214b7fd`.
The checkpoint declaration is 1,706,030,528 bytes with SHA-256
`aa2311c6b22b07cf3afe9aac6e8bcb02a2b19dd8671b2a91d0b144071577e600`.
The separate source-only audit is
[`ovisocr3-source-contract.json`](../../../tests/results/qwen35-ocr/ovisocr3-source-contract.json).
The source task prompt is `prompts/plain_ocr.txt` (SHA-256
`e037cf0013a757e19eae6d4d701f3b90848449fbebe44d1f8fa7d8ca4a5af9a7`).
The rendered prompt, installed processor arrays and effective vLLM stop set
still require the reviewed CPU metadata and actual GPU execution; source
declarations alone do not qualify them.

The GPU script checks the actual T4 capability, driver and CUDA driver API
before installing the pinned CUDA 12.9 vLLM 0.22.1 wheel or accessing weights.
It uses Torch 2.11.0, Transformers 5.16.1 and tokenizers 0.23.3. The CPU
metadata uses Torch 2.14.0; all three processor cases must match field
names, shapes, dtypes, hashes, IDs, grid and mask exactly across these
stacks. A mismatch is retained and stops before checkpoint access. The
worker then checks the full checkpoint SHA before model construction. The
source example's Triton GDN is retained. Decoder attention is explicitly
`TRITON_ATTN`, a controlled T4 departure based on the retained OvisOCR2
auto-FlashInfer failure; auto-backend equivalence is unproved. The source
example leaves dtype on auto, which must resolve to FP16 on T4.

Two independent one-image prompts (source synthetic square and deterministic
readable text) run with greedy generation. The 256-token cap is an explicit
departure from the source's 16,384 tokens. The report retains raw generated
IDs and text, every selected-token logprob plus the requested top 20 scores,
finish/stop reasons and cap/EOS status. This is a smoke reference, not a task
quality or native parity result. The source's two-image request path is not
qualified by this run.

Kaggle uploads only `code_file`, so the launcher fetches `worker.py` from an
immutable CrispEmbed Git commit and checks its SHA. It writes small reports,
command log hashes, processor replay arrays and the owned readable PNG to
Kaggle output. The checkpoint and wheel stay in task-owned temporary storage.
Do not push this Kaggle kernel until the metadata review pins and worker
commit are filled, reviewed, and the genuine GPU slot is available. No
credentials are stored in this directory.
