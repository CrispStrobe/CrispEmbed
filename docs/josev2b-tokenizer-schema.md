# Josev text/schema frontend preparation

This lane prepares a dedicated, default-off text frontend for
[mayflowergmbh/Josev-2B at6f0a31c8](https://huggingface.co/mayflowergmbh/Josev-2B/tree/6f0a31c8ed683aba79097eb3f48df2c9ab560afd).
It has no model weights, backbone/head inference, public API, response serializer,
registry, binding or release integration. Source preparation and tokenizer parity
are separate outcomes. No hosted result is claimed yet.

The publisher's Apache-2.0 `joint_schema_model.py` is the rendering blueprint,
SHA256 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`.
The [publisher license text (LF-normalized)](josev2b-tokenizer-schema-LICENSE.txt) is retained
with this source; the hosted task also retains the original LICENSE. The C++ implementation independently
expresses these semantics; it does not copy Torch implementation code. It reuses
CrispEmbed's ordered JSON parser, Python-compatible shortest float formatter,
strict JSON quoting, Unicode/NFC and byte-level BPE primitives. It does not call
Gemma's state normalization, question renderer, or tokenizer loader.

`josev2b::text_frontend::load()` captures
`core_env::on("CRISPEMBED_JOSEV_TEXT_FRONTEND")`; unset/empty/0 reject. Existing
contexts keep their gate value. Failed load resets the context; failed encode
clears all caller output. The standalone metadata-only GGUF component contains
zero tensors, full vocabulary and merges, all33 original added-token IDs/flags,
source hashes, exact regex/NFC declarations and complete tokenizer descriptors
and config. Unsupported flags/descriptor changes reject before native encoding.
The hosted conversion receipt binds the actual component SHA; a future full-model
caller must preserve that verified component identity when merging metadata into
its model. Source labels alone are not an authenticity check on arbitrary files.

`encode_record(request_json, output, error, max_length)` accepts at most64KiB of
finite, Unicode-scalar JSON,16 fields and128 options, batch one, max_length1–512.
It returns IDs/all-ones attention mask, record ID, ordered fields with publisher
kinds `noul=0`, `choice=1`, `score=2`, question/option spans, option IDs and lexical
IDs. Lexical IDs are the exact complete option-semantic slices for the head's
output-embedding means. Field order follows first key insertion with duplicate
keys taking their last value. Choice head order is sorted by key; score order is
list position; noul is true then false. Strings stay literal; other values use
compact sorted UTF8 JSON. Only null/empty-string instructions fall back to field
ID. Manual publisher prefix/schema/suffix fragments are encoded separately with
`add_special_tokens=False`; the unrelated chat template is unused. Only state is
clipped, and fixed schema overflow rejects. Media, unsupported criteria/types,
nonfinite numbers and invalid Unicode reject explicitly. Response formatting,
original criteria-order ties and rounding remain with the publisher's unchanged
`systemone_answer`; this lane does not implement a new output serializer.

The registered metadata phase `josev-tokenizer-schema` invokes the new reusable
workflow. It first verifies exact source closure and the frozen original
[receipt](../tests/results/josev2b/original-reference.json)/
[input freeze](../tests/results/josev2b/original-input-freeze.json).
Only the exact original lock's tokenizers0.22.2 wheel is installed, without
transitive packages, in an isolated environment; the basic Rust backend is used
without Hub loading, Transformers or Torch. Only tokenizer JSON/config, the23KB
serializer and LICENSE are downloaded. The task explicitly preserves both the
17MB-class tokenizer metadata and the additional artifacts remotely; no model
checkpoint or original activation arrays are needed.

Before C++ is built, the SHA-guarded pure publisher declarations are AST-extracted
using the same audited declaration set as `inspect_serialization`. All four owned
requests must reproduce original51a2e16b IDs, masks and spans exactly. Additional
owned literal-JSON, duplicate/order, IEEE-float, nested instruction, Unicode,
added-token, clipping and overflow cases are frozen as actual Rust-tokenizer gold
before any native execution. Original rejection controls are recorded too. Only
then does the two-worker standalone build link GGUF parsing from ggml-base; no
compute backend is built or initialized. Every accepted native result must match
all encoded fields exactly. Native subset negatives, metadata negatives, gate
controls and valid-before/reject/valid-after recovery checks preserve failures.
All source, descriptor, case, binary and output hashes are retained. The reusable
job has a25-minute budget and preserves partial evidence on failure.

These authored cases do not establish arbitrary-Unicode tokenization, multimodal,
long-context, task-quality, composed-head/backbone or release qualification.
Existing model families and all original gold stay unchanged. Keep the gate off
until actual reviewed tokenizer evidence and later composed-model evidence justify
any integration. Builds, tokenizer payloads, packages and batches run only on
hosted resources.
