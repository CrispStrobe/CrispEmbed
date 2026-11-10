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

Actual [41999152372](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41999152372)
at `9ba48e388708f9013b0e648fb7acee4214256187` passes every control above.
Independent small-artifact review rehashes both emitted NPY files against the
frozen original and compares the emitted decoded UTF8 text directly. The
[receipt](../tests/results/qwen35-ocr/lightonocr3-4b-tokenizer-controls.json)
retains exact source/tokenizer hashes. Initial41997924479 failed probe linkage
because the metadata loader implementation was omitted;41998489041 then failed
safe because the adapter expected the audit profile rather than actual export
profile `lightonocr3-4b`. Both negative receipts remain preserved. The one-line
profile correction changes neither historical frontend nor model output gold.
