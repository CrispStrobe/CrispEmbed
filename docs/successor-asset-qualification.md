# Immutable successor asset qualification

This protocol qualifies actual unpublished archives at one exact Git commit.
It does not create a tag, publish a release, upgrade the application from0.17.12,
qualify experimental models or infer task quality from an archive hash.

Dispatch `.github/workflows/release.yml` with`cpu_only=true` on the final candidate
source. This builds the actual seven CPU/mobile archives, retaining the normal
portable baseline and packaging/dependency checks; optional CUDA/Vulkan variants
are explicitly outside this bounded candidate. Only artifact-producing desktop
targets are built in this dry run. Ordinary tag builds keep their existing path.
Dispatch `.github/workflows/release-wasm.yml` on the same source with an empty tag.
Publication steps remain guarded. Both runs must complete successfully.

`tools/successor_asset_manifest.py` binds exact source, ggml gitlink, version,
archive SHA256/size and every member hash/link. Duplicate normalized paths,
traversal, special files, dangling links and cycles reject. Native coverage
requires all seven CPU/mobile archives; WASM coverage requires both bundles.
Hashing alone leaves qualification false. Eight small archive/layout controls exercise
these rejection cases without builds or models.

The reusable `.github/workflows/successor-assets.yml` then downloads actual
producer artifacts into fresh hosted consumers. Its source must equal both
producer commits. The registered metadata phase`successor-consumer` forwards
`native_run_id` as the native producer and`original_run` as the WASM producer.
No latest-run lookup or source-tree substitution is accepted.

- Four desktop consumers verify complete archive/member bytes and Linux glibc2.28
  dependencies. They test empty-cache offline rejection, actual CLI alias download
  of the pinned published MiniLM IQ4_XS model, its weight hash, cached offline
  reuse, every packaged-header function export, compiled C linkage/layout, and
  actual C/Python/CLI vectors. The shipped header must match the exact source.
  Vector gates are maximum absolute0.002/minimum cosine0.99999; CLI formatting
  loss is measured rather than called bit-exact. The Python structure is a
  supported prefix of the appended C structure, not an equal-size claim.
- The browser consumes both unchanged WASM bundles. It loads the exact same
  published model bytes as the native consumer, checks actual embedding output
  against the native vector, exported wrapper functions, and a classical RGB
  no-op cleanup through the OCR module. This is not an OCR model-quality audit.
- Apple additionally downloads the actual released0.17.12 archive, verifies its
  frozen hash, and runs the existing fifty-case PosFormer public-ABI probe on
  released and candidate unset/literal1/legacy0 paths. Released versus unset/1
  requires exact IDs/text/EOS; confidence drift is measured without a new
  tolerance. Candidate unset/1 outputs and confidences must agree exactly.
  The prior recognition score remains7/50; no recognition improvement is claimed.
- Android and iOS consumers inspect and cross-link the actual packaged C ABI
  with their platform SDKs. Unresolved archive dependencies reject. These jobs
  do not execute on devices; device inference remains an explicit platform gap.

Keep initial failures, artifact service identities, raw output comparisons and
logs in their remote artifacts. Download small reports only for independent
review. A candidate may be rejected despite successful builds. Any packaging
fix requires a new producer and all affected fresh consumers; never replace a
failed receipt or silently waive a missing archive/device check. A later tag or
application-pin decision is separate from this qualification.


Source review before the first build found incomplete mobile dependency closure:
Android's default shared ggml dependencies were omitted from its single-library
archive, and iOS packaged a non-amalgamated static archive. The corrected
candidate links ggml and the Android C++ runtime statically, and amalgamates the
iOS CrispEmbed/core/ggml archives. The cross-link consumers must verify these
changes on actual artifacts; source review is not an executed failure receipt.
The packaging change advances development manifests to0.17.14. No tag is created
and the application remains on0.17.12. Initial preliminary producers40338612955
and40338616900 were cancelled while every build was queued, before compilation;
WASM extraction was also corrected to preserve the archive's existing directory
root, with a dependency-free round-trip regression.
