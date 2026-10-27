# UltraBERT frozen SciFact native v3

This is source preparation for a distinct CPU/F32 experiment. The initial hosted
run rejected library loading before embeddings; no
representative retrieval, default, release, public pair API or app qualification is
claimed. Both Unicode pretokenization and right truncation remain default off.

The exact contract is
[`scifact-native-protocol-v3.json`](../tests/regression/ultrabert/scifact-native-protocol-v3.json).
The caller binds runtime/build/binding Git objects to `02402a0d`, composed from
integration `f00b2221` and the three byte-identical tokenizer files from the reviewed
[40434717493 tokenizer proof](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40434717493).
The existing truncation implementation remains unchanged. This new authorization
permits native embeddings only in this caller; the token-only producer's
`embeddings_authorized:false` remains intact.

The [40375626336 original panel](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40375626336)
has eight successful original shards and a successful freeze, two failed native
shards and six cancellations before steps. The overall cancelled conclusion stays
negative. Their exact original F32 arrays, 5483 stable IDs, full frozen token rows,
canonical inputs, qrels and licenses are reused; no original forward is rerun.
The earlier v1 author-column rejections and distinct canonical BEIR v2 are retained.
All source/data/artifact and completed producer-job checks precede weight access.

The hosted workflow builds once with ggml `a06fc0bb`, converts the pinned original
checkpoint to the already qualified F32 GGUF hash, exercises truncation/recovery,
and bundles the actual shared library and its dependencies. Eight native shards
reuse this exact artifact, with at most two running concurrently. Every assigned
row passes actual public CAPI token IDs and lengths before vector acceptance.
The API exposes no mask getter: all-attended padding semantics are **DERIVED** from
the pinned UltraBERT forward, which attends BOS ID0 and applies only local/global
window masks. Reports never label that mask as API-observed. Duplicate text rows
retain distinct IDs. Unaccepted partial outputs are retained on failure.

Raw CLS and actual normalized scalar CAPI vectors compare against all original
rows, using maximum absolute error ≤0.002 and cosine ≥0.99999. The aggregate retains
all 10966 vector checks, magnitudes, 300 original/native rankings, misses, margins
and relevant ranks. NDCG@10, MRR@10 and Recall@100 degradation must each be ≤0.001,
and top1 agreement ≥99%; ties use ascending lexical document IDs. Numerical failures
retain the whole completed panel and return nonzero. No tolerance or gold changes.

Historical 183 F32/1294 F16 stage failures, M1 strict replay drift and owned32-row
smoke retain their existing scopes. This experiment does not address those broader
numerical debts. Raw datasets, model/vector files and attributed notices stay in
90-day remote artifacts; frozen original artifacts expire January24,2027.

After source review, dispatch the registered `ovisocr2-metadata.yml` phase
`ultrabert-scifact-native-v3` at an immutable reviewed branch. No extra inputs or
original jobs are needed. All builds, downloads and inference run on hosted CPU CI;
no Torch, Transformers or SentenceTransformers import is needed by the native lane.


The initial [40444873884](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40444873884)
at `ee51e10e` built successfully; all eight native shards rejected at ctypes load
with `libggml.so.0` unavailable. Their exact small producer receipts, archive
identities and member hashes are retained in
[`initial-scifact-native-v3-loader-failure.json`](../tests/results/ultrabert/initial-scifact-native-v3-loader-failure.json).
No native embeddings or numerical judgments were produced. `ldd` resolved the
bundled GGML libraries; changing a running Python process's loader environment
was insufficient. Exact glibc cache behavior was not independently measured.

The corrected caller authenticates/materializes references and the bundle first,
then launches a fresh Python child with `LD_LIBRARY_PATH` set to the exact existing
`bundle/lib` directory, with no appended paths or inherited preloads. Before
ctypes or embedding access, the child rechecks its committed source, SHA-bound
stage manifest, every staged file, immutable receipt hashes, dataset/gold/token
identities, model/library hashes, retained ZIP hashes, ZIP-bound build receipt,
service artifact identities and completed producer jobs. Cached archives are
reused; the child downloads no payload. Both parent and child phase receipts and
the child log are retained. Runtime base02402a0d, ggmla06, original frozen arrays,
inputs, source policies, masks and numerical/ranking budgets remain unchanged.
