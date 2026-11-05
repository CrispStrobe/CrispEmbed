# Document-v2 packaging and admission

[This separate v2 contract](lighton4b-document-v2.json) packages already-produced
diagnostic outputs and consumes authenticated stage ranges. Document-v1 bytes,
its2GiB archive limit and empty producer lists remain unchanged. Both new
producer lists are also empty: actual original/native output and execution are
not admitted. The original adapter's capture hard-default and40GiB RAM/48GiB
disk guards remain intact. No original model forward is part of this work.

`tools/lighton4b_document_v2.py` builds an ordered source/range/hash manifest:
970 input/vision/prefill/cached/full-prefix fields plus67 fields at each generated
position. The latter contain positions,64 cache fields across32 layers, final
norm and all248320 logits. At the frozen256-token cap this is at most18122
stages. The already-pinned image-only452×317 form, source model/revision, package
recipe, prompt and EOS248046 come from document-v1 without modification.

For base fields, the packager parses the existing F32 diagnostic GGUF and hashes
the selected byte spans. It reuses the unchanged8GiB legacy parser bound, rather
than silently expanding a historical parser. Generated fields reference the
observer's separately completed F32 files, with raw-logit/generation-score
equivalence already required by the observer. The source transport manifest and
every complete source-file SHA must match. Missing, extra, duplicate or reordered
package stages, changed ranges/shapes, invalid IDs or readiness claims reject.
Manifests are at most16MiB, decoded UTF8 at most1MiB and each stage at most64Mi
elements/256MiB. Only F32/F16 comparison labels are accepted; quantized arms still
need a separate closure and held-out protocol.

The consumer does not require two whole large captures to be restored together.
`RemoteReader` validates an immutable transport manifest and canonical byte
ranges, caches one verified part per side and reads the required stage slices.
The consumer verifies each slice SHA and feeds one bounded original/native GGUF
pair into the unchanged `crispembed_diff::Ref` executable. The diagnostic pair
contains one stage, so its256MiB maximum remains below the unchanged2GiB v1
reader limit. Raw strict max-error0.002/minimum-row-cosine0.99999 checks and
earliest-stage metrics are preserved. Independent free IDs/text/EOS also have
to match; complete numerical agreement cannot conceal a decoded regression.

Lazy consumption proves authenticated selected slices. It is not a second full
restoration of unselected probes or proof of large retention. Source/reader part
caches, temporary single-stage archives and provider internal caches still need
independent placement/headroom review and measured capture-scale evidence.
The previously proven13,239-byte real-provider fixture does not provide that.

```sh
python tools/lighton4b_document_v2.py package-synthetic \
  --source synthetic-original --transport original-transport.json \
  --side original --output original-v2.json
python tools/lighton4b_document_v2.py readiness --manifest original-v2.json
python tools/lighton4b_document_v2.py consume-local \
  --original original-v2.json --native native-v2.json \
  --original-root synthetic-original --native-root synthetic-native \
  --binary build/crispembed-diff-lighton4b-document --output comparison-v2
```

The real bounded remote reader is also executable through the module API with
an authorized backend and pinned source prefix/revision/transport-manifest SHA;
hosted controls exercise it through an independent immutable HTTP service fixture.
The CLI exposes synthetic packaging explicitly and does not offer a model-forward
or real-admission override. `readiness` reports synthetic packaging/coverage
separately from actual248320-logit geometry and always retains actual-output,
execution and large-retention admissions as false in this preparation version.

Before real output can be admitted: review and bind actual producer source,
checkpoint/packages/input and transport closure; verify observed original
geometry, complete EOS and observer neutrality; prove full-width/large-storage
readiness on suitable hardware; then review a separate explicit admission change.
Native format5 context/tokenizer/vocabulary/cache composition and actual decoded/
intermediate document parity remain separate. No32GB or GPU use is authorized.

Actual [initial41981531300](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41981531300)
and [bounded41981959660](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41981959660)
each pass all5 hosted methods, zero skips. Final tested source is
`bb9f367a0fc3438a4ca4ab83909b462e2d2d7480`. Independent review verifies all7
recorded source hashes and log digests, including actual1037-stage synthetic
remote Ref consumption, independent decoded mismatch and corruption/coverage/
false-readiness negatives. [The exact-source receipt](../tests/results/qwen35-ocr/lightonocr3-4b-document-v2-controls.json)
retains the no-real-model scope. Actual4B output readiness remains false.
