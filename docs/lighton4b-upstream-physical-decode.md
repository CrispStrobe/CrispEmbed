# Same physical upstream text weights: bounded decoded comparison

The [predeclared protocol](lighton4b-upstream-physical-decode-v1.json) compares the
losslessly transported publisher Q4_K_M text weights with pinned upstream
llama.cpp on the already observed form, table and Arabic diagnostics. It is
pending execution. Neither engine output is human OCR ground truth; frozen
original BF16 outputs and the independently retained upstream Q4_K_M outputs
remain separate references.

The importer preserves every physical text quantization block and authoritative
small scalar, with whole-row inverse reordering and an activation permutation
before the 24 unchanged GDN output projections. The model-free controls at
`0610af26d9cb8972ecdfe55204e2306e9e740d5e` passed; the decoded worker must also
verify the complete persisted imported file and all 723 tensor spans. The
format-6 path remains experimental and default-off. Public model admission,
release aliases and format-5 behavior remain unchanged.

Two arms answer different questions; this first worker executes only the
source-feature arm, and defers composed vision until its measured result:

- `upstream_features` feeds actual captured upstream merged F32 image features
  into the test executable, isolating the physical-weight text decoder. These
  activation captures are reference probes, never production model weights.
- `composed_original_vision` executes our protected original vision weights and
  verified vision corrections. Its text weights are the same physical publisher
  weights; its vision weights/arithmetic differ from the upstream F16 projector.

The upstream observer runs after MTMD encoding and after decoder computation,
without graph callbacks or graph splitting. Fresh observer-OFF and observer-ON
processes must exactly reproduce the first two IDs and probability records from
the retained full upstream baseline. The original chat wrapper, prompt token
bytes and actual image/prompt geometry are checked before either native arm.
The independently decoded upstream Q6 embedding rows, actual native prefill
splice and source image positions must agree byte for byte; the source-feature
arm must consume the exact captured feature bytes.

The native source-feature arm independently generates greedily up to 256 IDs and preserve raw
text, both admitted EOG IDs, cap status, full first scores and stage arrays.
The report compares first scores with actual source F32 logits and full output
with both references, including prefix, character/word edits, process time and
observed peak resident memory. No original BF16 forward, full F32 model,
requantizer or image-specific fitted constants are used. First-score drift after
exact input does not by itself localize a specific decoder operator.

Earlier audited Kaggle CPU capacity was 31.51 GB effective available RAM with no
swap; this is an observation, not a guarantee for every worker.

One sequential Kaggle CPU worker builds with four threads and requires 10 GiB
effective available RAM, 24 GiB scratch and 8 GiB output before any checkpoint
transfer. Its outer deadline is six hours; each native case is limited to 90
minutes, based on prior observed throughput of about 225 rows in 45 minutes.
All three source observer controls finish before native full generation starts;
per-token partial IDs/text and progress survive a timeout. Three authenticated source files total approximately 7.4 GB; the
imported quantized candidate is temporary and its full hash plus all tensor
readbacks are retained. Large stage arrays stay in remote output. Submission
must follow the separately running candidate-conversion worker; there is no new
GitHub Actions dispatch or PR.
