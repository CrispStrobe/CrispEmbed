# Tiny real-provider diagnostic transport proof

Actual hosted [41979390868](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41979390868)
at `1447b6b0dd595471e239210f2bee9f33fc640020` exercises the unchanged sharded
transport against an existing authorized private Hub diagnostic dataset. The
runtime confirms private status and write authorization; it creates no repo and
adds only a fresh unique prefix. Private account/repo/commit access locators stay
outside Git. The [public receipt](../tests/results/qwen35-ocr/lightonocr3-4b-real-provider-transport.json)
contains fingerprints and content hashes, not credential values or private paths.

The actual fixture has67 authored F32 fields plus observer metadata:13,239 source
bytes, split into four parts of at most4096 bytes and one final manifest. After
per-part immutable readback and final-manifest commit, a new provider client and
empty download caches reconstruct all68 files and compare against the independent
authored bytes. The report is `passed`, with actual remote writes/readback; it is
not an auth-blocked result. All17 existing GitOID/size entries are identical at
the tested commit, with exactly five new paths and22 paths total. No existing
repo/file is removed or overwritten, and no dataset or secret binding is created.

A second independent read-only review locates the exact tested immutable commit
by fingerprint, cold-fetches its manifest and all four parts, verifies canonical
coverage and every part/whole-file SHA, independently regenerates all67 float
vectors and compares the real before/after tree metadata. The manifest hashes to
`f1bdcbeff38c136f8d20d6cde257672967db8ce0d69ad477e79ec6bd118ef652`.
Full private locator/tree evidence is retained privately for authorized review.

This proof covers tiny non-model data at that immutable service revision. It
does not qualify long-term retention,512MiB-sized parts,24GiB original caches,
provider cache/storage placement, throughput or32GB execution. The original
document-v1 and transport-v1 contracts remain unchanged, with empty admitted
original producers. Both original transport and capture stay hard-disabled, and
the40GiB RAM/48GiB disk gates remain intact. No original/native model forward,
model-weight bytes, GPU session, registry alias or runtime release follows.

The next admission decision must independently review the actual provider proof
and source closure, complete the capture-scale storage/resource contract and
review an explicit versioned capture-admission change. Actual original full-width
observer neutrality and native format5 document parity remain separate work.
