# Conditional WASM rounding A/B v1

This is a new, versioned consumer for the unmerged
[default-off candidate](wasm-q8-rounding-candidate.md). It leaves the original890
trace driver, its production guard and the metadata dispatcher unchanged. The
zero-input reusable workflow is
[wasm-q8-rounding-ab-v1.yml](../.github/workflows/wasm-q8-rounding-ab-v1.yml).
It is prepared for review; no model A/B has run and no default or release is
promoted. The caller must use a reviewed commit. Builds run hosted with two
workers and a60-minute job ceiling.

The [protocol](wasm-q8-rounding-ab-v1.json) pins actual successful trace
[40379523009](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40379523009)
at f8bac049a7c143773b330cfc4c217b1f0a47ef56. Its reviewed small artifact is
12681776616,47,552 bytes, SHA256
`7e934a9f486c9d259b17783c9091e8f322f32a2e97e1d15bb1f98ee3379ca4a4`;
`trace.json` has SHA256
`381f2ad5687f76830b64c431a39b003c574ecdfd4ca25996dff102f973b0fb8b`.
Raw artifact12681437524 is962,060 bytes, SHA256
`70f4eb725d6d3083b23ad93ffc8821f69bee8ac0257f3aa403946cc1145cd5cb`.
The first trace's native/browser neutrality controls and all eight named stages
must remain complete. Missing review status or any missing artifact/report pin
rejects before model or artifact access; source controls exercise those negatives.

The reviewed trace found bit-exact`emb_ln_out`, then a first difference at
`attn_out_0` (maximum absolute0.0833039284), while the actual Q8_0 architecture
conversion differed by one signed quantized unit with unchanged block scales.
This justifies the conditional experiment. It does not yet establish that this
candidate restores the complete network.

The new source guard allows exactly the candidate encoder SHA256
`99bb8a2308021ce997e1570927322d749155c89155dd08af11fdcac6b99e7be1`
and ggml a06fc0bb46b73cc10011fc90ae4054aee8438aeb. It verifies every declared
changed dependency file, rejects dirty source and binds all other production,
build and binding Git objects to both973b200e and the immutable40a producer.
The declaration includes six dependency runtime files and two test files. No
old guard is relaxed or monkeypatched. Archive preparation independently binds
successful native40339796226/WASM40339800364 service identities, exact manifest
and payload ZIPs, the original generator, every selected archive and every member
against the first trace's package proofs. Initial fixture12671621333, model
77087a5dd455f5eeae6d6d098bac82793610438518a91071c72c253940db18a2 and native
report6cdd426e0b71fcbe80000494645deeb75fa75f9d5028e06d29abcdb9ff8cf435
stay unchanged.

The first trace's diagnostic modules are bound by the reviewed whole raw ZIP
and service identity. The consumer computes all member/module hashes hosted.
Those binary hashes were not previously present in the trace report's
GGUF/F32/log hash list; the consumer does not claim otherwise.

Before any candidate inference, the workflow freezes this supplementary owned
panel. Exact literal/recipe expansion is source-bound, with canonical panel SHA256
`d01c0fcc3046bf1c3a1024f890e1c721852052b2cfc84f37e7b7c4e4f2b03406`:

| Case | Fixed text or recipe |
| --- | --- |
| original | `A small reproducible embedding consumer.` |
| empty | Empty UTF-8 string |
| punctuation-currency | `Price: $12.50 / €9.99 — (test)!? #42` |
| diacritics-emoji | `Café naïve résumé — 😀🧮` |
| mixed-scripts | `Hello Ελληνικά 中文 العربية हिन्दी` |
| whitespace | The protocol's literal spaces, tabs, LF and CRLF around `Hello`/`world` |
| numbers | `0 1 -1 3.14159 1e-6 2026-10-26 000123` |
| one-character | `x` |
| near-limit |420 copies of`hello`, separated by one ASCII space; expected422 IDs |
| truncating |800 copies of`hello`, separated by one ASCII space; expected512 IDs |

Old packaged native1/2 processes capture every vector, actual public/graph token
ID pair and six-decimal raw-norm diagnostic. The unchanged packaged browser
provides actual vectors. Its retained first-trace diagnostic module provides
IDs, with exact packaged/diagnostic vector neutrality required for every case.
The first sentence additionally preserves initial native/browser vectors and
all original stage and actual pooling-buffer hashes. The supplementary panel
is written once, hashed and required before candidate build/execution. It never
replaces the first trace or its frozen original sentence.

The candidate uses a minimal test-only environment/pooling overlay, fresh modules
for unset, empty, literal`0` and literal`1`, and ordinary/dump execution for every
case. Legacy states must exactly preserve each old packaged-browser F32 vector.
All ordinary/dump pairs must be bit-exact with identical actual token IDs. The
original sentence additionally requires unchanged legacy full-stage hashes.
Literal`1` must meet maximum absolute0.002/cosine0.99999 against both frozen native
thread variants for every final vector, and all eight original-sentence full
stages must meet maximum absolute0.002/minimum token-row cosine0.99999.
`crispembed_diff::Ref` and independent stdlib metrics both inspect those arrays.

Dump mode captures actual WASM pre-L2 and final pooling buffers for every case.
Complete GGUF stages are retained only for the original sentence, bounding the
payload. Original-sentence native pre-L2 means are explicitly recomputed with
sequential F32 arithmetic from frozen`encoder_out` and its source-derived
all-one mask; they are not captured native buffers. Other native panel cases
have actual six-decimal raw-norm diagnostics only. A separate one-module sequence
initializes0/1/0 contexts, leaves the environment at1, and runs A0/B1/C0/A0 again
against the isolated-module outputs to check backend/context isolation.

The global Q8_0 type trait remains legacy. This lane exercises the candidate
through real encoder backend graphs and does not label a raw trait-converter
probe as the selected backend converter. Every vector, token sequence, norm,
absolute/RMS/cosine comparison and numerical failure is retained. Comparison
continues across the full matrix before rejecting numerical failures. Small
reports/logs and remote arrays/modules use separate artifacts; model weights
are excluded from both uploads.

Seven stdlib protocol/source controls, Python/JavaScript syntax checks and
workflow lint pass. No local build, inference, framework, model or raw-array
batch ran. Actual hosted weight-free operator controls and the complete model
A/B remain dependencies. Even an entirely passing numerical panel would not
establish representative task accuracy, qualify release assets or promote a
default. A production asset change requires new immutable producers and fresh
consumers after review.
