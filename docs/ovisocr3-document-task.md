# OvisOCR3 document Markdown: separate source contract

**Status: exact-source task audit and weight-free controls. Original Markdown
execution, its input/gold manifest and native support remain unqualified.**
The [machine-readable protocol](ovisocr3-document-protocol.json) uses the
checkpoint already audited for OvisOCR3, with its separate upstream Markdown
task. It does not replace the [plain OCR decision](ovisocr3.md).

## Exact source and task

The source is [ATH-MaaS/OvisOCR3 at
5f6ce6932efee9ec10c30746db55bd780214b7fd](https://huggingface.co/ATH-MaaS/OvisOCR3/tree/5f6ce6932efee9ec10c30746db55bd780214b7fd).
Its Apache-2.0 license and six unmodified small source files are retained in
[the document source fixture](../tests/fixtures/ovisocr3-document-source/NOTICE.txt).
The source validator checks their exact lengths and SHA-256 values against
[the existing immutable source audit](../tests/results/qwen35-ocr/ovisocr3-source-contract.json).
No checkpoint payload was downloaded for this task audit.

| Contract | Pinned source requirement |
| --- | --- |
| Checkpoint | 1,706,030,528 bytes; SHA-256 `aa2311c6b22b07cf3afe9aac6e8bcb02a2b19dd8671b2a91d0b144071577e600`; 473 tensors, MTP=0 |
| Task | `document_markdown`; one RGB page per request for this protocol |
| Prompt | [`prompts/document_markdown.txt`](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/5f6ce6932efee9ec10c30746db55bd780214b7fd/prompts/document_markdown.txt), 512 bytes; SHA-256 `dbf31d5899a86047ce0cd4d0d12cdc5616df1bea5d9a600b823098442c32997c` |
| Chat | One user message: image item then exact task text; `add_generation_prompt=True`, `enable_thinking=False` |
| Pixels | Explicit nested `images_kwargs`: minimum 448², maximum 2880²; RGB conversion |
| Generation | Upstream example's 16,384-token maximum, temperature 0, sampling disabled; declared EOS IDs 248044 and 248046 |
| Original implementation | vLLM 0.22.1 at `0decac0d96c42b49572498019f0a0e3600f50398`; Triton GDN prefill, tensor parallelism 1, GPU memory utilization 0.8 |

These settings come from the pinned [README inference and task
examples](https://huggingface.co/ATH-MaaS/OvisOCR3/blob/5f6ce6932efee9ec10c30746db55bd780214b7fd/README.md).
They are usage settings, not an OmniDocBench or OmniParseBench reproduction.
The two-image upstream allowance supports cross-page tasks; this protocol
selects one page and makes no cross-page claim.

The Markdown prompt requests natural reading order, HTML tables, LaTeX formulas
and image references whose box coordinates are scaled to `[0,1000)`. It requests
transcription without translation or paraphrase. Unlike the separate structured
task, it supplies no explicit boxes or categories for every text block. A later
evaluation adapter must not invent those text-region boxes.

The source's saved Fast processor name does not establish the installed class.
The old plain OCR metadata run resolved a slow Qwen2VL image processor under
Transformers 5.16.1. The new task still needs an actual rendered-prompt hash,
input IDs, packed pixels, grids, masks and effective stop evidence. Existing
native evidence remains bounded to 1024 image patches; the larger upstream
pixel maximum has not been qualified by this source audit.

## Cleanup preserves generation failures

The README applies these steps in order: trim outer whitespace; optionally
remove blocks starting with its image-reference tag; run its repeated-tail
cleanup. Image filtering defaults to enabled. Cleanup runs regardless of the
finish reason, with the upstream thresholds unchanged: 8,000 characters,
periods 1–200, at least 100 repeat characters and five repetitions.

[The source tool](../tools/ovisocr3_document_contract.py) extracts only that
hash-verified pure string helper from the retained README. It records raw
Markdown, raw IDs and finish reason alongside both the upstream default and
image-preserving presentation variants. A length stop stays incomplete even
when cleanup makes the displayed output short and plausible. EOS completion
requires explicit evidence that the EOS was not forced, and traces containing
tokens after an EOS are rejected. EOS completion also does not establish
document quality.

## Preserve the old plain OCR result

The existing 419-byte `plain_ocr` prompt excludes markup. Its original
[eleven-case receipt](../tests/results/qwen35-ocr/ovisocr3-original-quality.json)
retains the two-column page's 1024-token length stop, no EOS, CER
0.523649781952365 and WER 0.8399280575539568. The paired native F16 result remains
a faithful reproduction of that failure. The new protocol pins the exact old
panel, original receipt and native receipt hashes and validates that the failed
case is still incomplete. Its old prompt, cap, input/gold and scorer stay intact.

The 16,384-token setting above belongs to the separately named Markdown task
and is fixed before its outputs are observed. It neither re-scores the old
plain OCR panel nor grants a model alias. The existing format-3 runtime profile
still rejects a custom task prompt; a new task needs its own original-first
profile and qualification.

## Next executable gates

1. Freeze an independent licensed or authored document input/gold manifest
   before model outputs, covering reading order, headings/lists, HTML table
   cells and spans, LaTeX, image references and multilingual text. Declare
   splits, image/RGB hashes, task settings and scorer version. The current
   protocol intentionally has no invented annotations or measured score.
2. Capture the actual original Markdown processor/request and generation on
   hosted resources. Retain raw output and completion, all required model
   stages/caches and complete 248,320-wide raw logit rows at every generated
   position. Bind artifacts to model, source, task and case. Allocate storage
   before dispatch: at the source cap, one F32 logit stream alone can reach
   16,273,899,520 bytes.
3. Add a separate native task profile only after those captures. Compare on
   the same original token prefixes under the portable CPU precision policy,
   preserving strict intermediate and decoded diagnostics. Report document
   quality and source completion separately from port fidelity. Publication
   requires all later gates; this audit sets every qualification flag false.

Run the source audit and 14 bounded controls without model dependencies:

```bash
python tools/ovisocr3_document_contract.py
python -m unittest discover -s tests -p 'test_ovisocr3_document_contract.py' -v
```
