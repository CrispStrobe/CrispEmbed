# LightOn4B owning GGUF structural inspection

Draft PR147 stacks atomic-owner PR146 (`8157a3137`), text-owner PR145 and the
unchanged vision component from PR144. Production format5 recognition remains
closed. The separate `lighton4b_gguf_inspection` context owns actual GGUF metadata
and no-allocation ggml tensor headers, applying typed source/profile/geometry/EOS
checks before returning the atomic723-tensor structure. It owns no backend or
weight data and exposes no tokenizer, decoder or recognition method.

The new `CRISPEMBED_LIGHTON4B_GGUF_INSPECTION` gate is default off; all four
underlying owner/OCR gates must also be enabled. Metadata source/revision strings
are declarations, not cryptographic proof of checkpoint contents. Header-only
files are intentionally valid inspection inputs, which precludes treating this
API as payload-backed model admission.

Synthetic hosted [41975595631](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41975595631)
at `3061e16a4` passes86 controls. The fixture builds all723 tensor headers from
the frozen original schema and34 typed metadata fields from original config.
Every field is tested with wrong values and wrong types. Positive controls cover
two simultaneously owned independent contexts, remaining-context lifetime after
one teardown, repeated creation/destruction and five disabled gates. The36
existing GDN/frozen replay/larger-input controls also pass.

Actual-header hosted [41975926521](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41975926521)
at `f4cc98568` repeats the86 controls and inspects the uploaded
[experimental baseline](https://huggingface.co/cstr/LightOnOCR-3-4B-GGUF-experimental-6b9dfe91eaa2/tree/a7ec2e54103165a96b8bb6dc3478a2887edd7459)
`lightonocr3-4b-vision-f32.gguf`. Three bounded4MiB requests each require HTTP206
and the exact Content-Range, including declared total9,757,493,600 bytes. A
nonconforming response stops acquisition; the protocol has a64MiB prefix ceiling.

The requests fetched12,582,912 bytes. The parsed/padded header occupies10,010,976
bytes, SHA-256 `babd50697d19b06d8f7ea35a8b0a129074c92c100be354b68fab407b87f96ab6`.
The remaining2,571,936 fetched bytes are the beginning of weight payload and
were discarded. Only real header bytes are retained: no zero-filled or sparse
full-model stand-in is created. Native inspection dereferences no payload data.
The header is a roughly3MiB compressed artifact alongside the small logs/reports.

Independent retained-artifact review verifies all14 recorded source digests,
34 metadata fields and every723 tensor name/shape/type/offset interval against
the original schema:474F32/249F16. Aligned tensor spans plus the real metadata
offset equal the declared full file size. It rehashes the retained header and
recomputes the86-control census. The discarded weight-prefix bytes cannot be
independently rehashed from this retained artifact; their per-range hashes are
acquisition records. The [review receipt](../tests/results/qwen35-ocr/lightonocr3-4b-gguf-inspection.json)
retains exact source/Hub pins and the original reports. Reporting-only successor
`2c3628732` explicitly labels range overshoot and retained zero weight bytes;
the native inspector and parser are unchanged.

This prefix check does not verify the declared full-file SHA-256
`757ad5d0ea7699b3d0ce4cbb70978de442e5b95e660da0a13cf048980e255606`
or any tensor payload hash. Separate conversion/upload evidence remains separate.
Payload-backed owning admission, tokenizer/preprocessing, composed native
execution and same-input stage/logit/decoded/EOS parity remain required before
recognition admission or support claims. No runtime release, alias or default flip.
