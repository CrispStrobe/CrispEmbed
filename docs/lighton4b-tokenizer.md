# Separate experimental LightOn4B frontend

`lighton4b::frontend_tokenizer` in `src/lighton4b_frontend.h` is a distinct
format5 adapter behind default-off `CRISPEMBED_LIGHTON4B_FRONTEND`. It validates
the pinned4B repository/revision, profile, checkpoint/tokenizer hashes, format,
NFC normalizer, exact Unicode-mark-aware regex and primary/generation EOS248046.
Its bounded byte-level BPE and UTF8 handling derive from the existing frontend;
`src/qwen35_tokenizer.cpp` remains byte-identical. No metadata provenance is
rewritten to impersonate another model. The adapter requires the frozen empty
image-only prompt and expands its one image token into an explicit bounded count;
ordinary text encode/decode remain separate methods. EOS248044 is not a stop
for this original protocol.

The metadata-only hosted controls build target
`crispembed-test-lighton4b-tokenizer` and verify the exact154 input IDs and
modality IDs via original BF16 capture NPY hashes, complete177-ID decoded text,
rendered prompt, seven official tokenizer Unicode/added-token cases and six
negative gates. Full tokenizer payload hash is authenticated before use; no
checkpoint is downloaded or model forward executed. The metadata GGUF and build
stay hosted. These are frontend controls, not complete model/precision parity.
