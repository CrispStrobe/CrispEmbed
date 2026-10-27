# Prepared Josev wheel metadata ownership audit

The fourth original GPU attempt installed all 57 locked wheels and passed
`pip check`, then rejected in installed-environment verification before framework,
CUDA-operator, tokenizer or model access. Preserve that failure. Its message
`installed wheel METADATA bytes differ: setuptools` does not establish a byte
mismatch: the [frozen verifier](https://github.com/CrispStrobe/CrispEmbed/blob/2133cfa12dbdd25653fd68b1a6054f69f80f7e9c/tools/josev2b_environment.py#L258)
uses the same rejection for multiple metadata candidates and for a hash mismatch.
Its candidate predicate also admits nested vendored `.dist-info/METADATA` files.
Whether the exact setuptools wheel contains those files remains a hosted audit
question; no wheel or installed framework was inspected locally.

The separate [protocol](josev2b-wheel-metadata-protocol-v1.json) freezes the existing
environment lock and requirements at
[c9aa40b7](https://github.com/CrispStrobe/CrispEmbed/commit/c9aa40b7), with all 57
original wheel URLs, byte lengths and SHA256s. Published payloads total
2,790,174,915 bytes. The CPU-only [reusable workflow](../.github/workflows/josev2b-wheel-metadata.yml)
authenticates each complete wheel one at a time in temporary scratch, reads only
metadata and RECORD from the ZIP, then deletes that wheel. It installs and
executes no wheel and imports no framework. A 30-minute audit bound, 40-minute
job bound and single-wheel disk check prevent an unbounded dependency exercise.
No GPU slot or model/tokenizer payload is involved.

The [driver](../tools/ci-heavy/josev2b_wheel_metadata.py) compares the one top-level
owning distribution's METADATA with the immutable PEP658 hash and the current
sidecar bytes. It records all top-level and vendored candidates, their identities,
hashes and lengths, and verifies that the ZIP candidate list matches RECORD.
The old selector's candidate count is derived from that RECORD. It separately
reports byte equality and exact package/version/Python/dependency/extra fields.
Any changed metadata bytes still reject even when dependency fields agree.
Different declarations are retained verbatim, without inferring semantic
equivalence or changing dependency resolution. The audit continues across all
57 packages after an individual mismatch and retains each exact rejection.

Only small reports and sidecar/owner metadata are uploaded for 90 days, including
failures. Package URLs are restricted to the frozen PyPI wheel and its sidecar;
unexpected redirects reject. All archive payload work belongs on hosted CI.
Local controls use tiny synthetic ZIPs, with vendored candidates, missing or
duplicate owners, malformed coverage, changed bytes/dependencies and rejected
URLs. They do not establish an actual wheel result.

This preparation does not change the original wheel lock, worker/bootstrap pins,
model contract, arithmetic, inputs or numerical gates, and authorizes no GPU
retry. An archive-only result cannot measure installation rewriting or establish
CUDA/runtime/model compatibility. Review exact archive evidence before framework or model access; the GPU-reference
setup described below enforces that barrier within its original task.

The corrected verifier selects exactly one top-level
`<own-name>-<own-version>.dist-info/METADATA` from the owning distribution's
RECORD. Nested vendored metadata is ineligible; missing/duplicate owners,
wrong origins and any locked byte-hash mismatch still reject. Synthetic controls
cover a real owner plus vendors and reject a corrupted owner even if a vendor
has the expected hash. These controls do not establish installed-wheel or GPU
compatibility.

The independent hosted CPU archive audit remains pending. The separately
[prepared GPU-reference setup](josev2b-gpu-audited-install.md) can produce the
same exact archive/sidecar evidence inside the genuine original inference task,
before installing each wheel. Its full57-package audit/installed-owner barrier
precedes all framework imports, CUDA smoke and model access. This is an explicit
source-versioned setup correction, with no lock, model, panel, arithmetic or
acceptance change; actual completion remains pending.
