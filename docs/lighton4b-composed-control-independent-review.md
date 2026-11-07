# Independent review: composed-vision control protocol

Reviewed 7 November 2026: tools-only source `b98567cfd`, based on component
source `b59e5d3a9d9e333f67f9646c1a43a1edc70fccfa`. This is a source and
model-free control review; it is not a completed model run or quality verdict.

The seven component Python controls passed at `b59e5d3a9`; the four composed
protocol controls passed at `b98567cfd`. The SHA-256 of the reference driver
matches the frozen protocol. `src/`, ggml, CMake and the native document driver
are byte-identical between these two sources. The separately reviewed adapter
hash matches its declared guard.

The candidate admits the exact prior component receipt and requires all 28
source-stage comparisons and the legacy anchor before acquiring model weights.
It rehashes receipt-owned artifacts and source stages, fixes the candidate
weight and input identities, requires the old composed output and stage anchors
before the candidate, and changes only the two declared vision gates. It records
independent greedy output, EOS/cap, reference distance and source-feature
comparison; a negative candidate must remain recorded. Resource and runtime
limits are explicit. The protocol contains no new production reference probes.

**Review limit:** the actual component receipt and its remote payloads were not
independently downloaded for this review. Its claimed stage identity and the
new composed execution must be verified from their own authenticated reports;
passing these controls alone does not authorize a quality claim, held-out claim,
default change or publication. The component and composed protocols retain
observed-form-only scope and the known negative baseline.
