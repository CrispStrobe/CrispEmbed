# Original4B diagnostic shard transport v1

This separately versioned transport implements the mechanics proposed in
[the earlier draft](lighton4b-original-transport-proposal-v1.json). It does not
enable original capture. `ORIGINAL_TRANSPORT_ADMITTED=False`, the adapter's
capture hard-block, the original document-v1 empty producer lists and its2GiB
archive limit remain unchanged. The [new transport contract](lighton4b-original-sharded-transport-v1.json)
is distinct from an admitted original/native document bundle.

`tools/lighton4b_sharded_transport.py` checks every source field's shape, byte
count and SHA, exact67-field coverage at every generated position, complete EOS
and observer-neutrality markers. Synthetic controls must be explicitly labeled
and cannot relabel an original forward as a synthetic fixture. Original capture
publication fails before reading source files while its admission is disabled.
Only the declared generated shards, observer/adapter reports and two legacy
reference representations are eligible; credentials and arbitrary source files
are not transport inputs.

The source files are sorted deterministically and streamed as contiguous byte
ranges into parts of at most512MiB. A single source file may span parts, and a
part may contain several fields. Every part and original file has its own SHA.
Bounds are4096 parts,24000 segments,16MiB manifest,17156 files,24GiB generated
shards and48GiB total source payload. Legacy files are individually bounded to
9GiB, allowing container overhead beyond the adapter's8GiB retained-array cap.
The adapter's40GiB RAM/48GiB disk admission remains unchanged; none of these
mechanics authorizes a32GB worker or changes GPU quota.

Publication requires a fresh unique session prefix. Each part is uploaded and
read back at its immutable revision into an independent empty temporary cache
before local part buffers are discarded. A final manifest is committed only
after every part passes and all source-file hashes remain unchanged. Failures
retain a failed receipt and committed partial-object identities; partial uploads
have no completed manifest and cannot be treated as archived. There is no
automatic overwrite, deletion or force push.

Independent readback starts in a fresh directory with the pinned final revision
and manifest SHA. It rejects duplicate JSON keys, repeated or unsafe file names,
noncanonical/overlapping/incomplete ranges, wrong part inventory, corruption and
truncation. It reconstructs original files from verified parts with bounded
buffers, verifies every whole-file hash, then repeats field/EOS/neutrality
inventory checks. The remote service is the only source of readback bytes.

The48GiB capture admission is not a promise that a full source capture and its
complete restored copy fit together. Full cold reconstruction needs the declared
source byte count plus one part and1GiB headroom; use a separate appropriately
provisioned hosted reader or independently budget both copies. Provider-specific
download/upload caches also need measured placement and headroom before admission.

The optional Hugging Face backend uses an existing caller-authorized repo,
checked parent commits and40-hex immutable readback revisions. Runtime credentials
are never stored in manifests. The backend is implemented, but its actual service
acceptance, durable retention and full storage/cache budget are **unverified**.
The current controls instead use a separately implemented HTTP service fixture
with immutable version snapshots and independent byte-level readback. That proves
transport mechanics only, not a real provider's retention or original-model parity.

```sh
python tools/lighton4b_sharded_transport.py --repo-id "$AUTHORIZED_REPO" \
  --repo-type dataset publish-synthetic --source synthetic-fixture \
  --prefix synthetic-reference-session --receipt publication.json
python tools/lighton4b_sharded_transport.py --repo-id "$AUTHORIZED_REPO" \
  --repo-type dataset readback --prefix synthetic-reference-session \
  --revision "$IMMUTABLE_REVISION" --manifest-sha256 "$MANIFEST_SHA256" \
  --output independent-readback
```

These are interfaces for an authorized hosted service test, not evidence that
such a provider operation has run. Before capture can be admitted, independently
review the transport/source closure, exercise the actual authorized provider with
synthetic bytes and immutable cold readback, verify cache/scratch placement and
headroom, and retain provider run/version/object identities and failure evidence.
Then review a separate capture-admission change and use a suitable high-memory
host. No model forward, native qualification, alias or release follows here.

Actual [initial41977694403](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41977694403)
and [bounded41977934618](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41977934618)
each pass all6 hosted synthetic methods, zero skips. Final source is
`2a9e03b194ee636b3dac57338e81387fc573505e`; independent review verifies all5
recorded source hashes, the log digest and the68-file independent fixture restore.
[The receipt](../tests/results/qwen35-ocr/lightonocr3-4b-transport-controls.json)
retains actual source/artifact identities and every false provider/model admission
flag. No external provider operation or original model capture is established.
