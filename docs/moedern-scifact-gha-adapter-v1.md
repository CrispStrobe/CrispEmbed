# MoEdern full SciFact GHA provider adapter v1

This source preparation adds a separate GitHub-hosted provider for the corrected per-variant worker at `1543475b80622daf057ecf198136579cc1f7c051`. It does not dispatch a job, rerun an original, qualify native/Q4 retrieval, register another metadata phase or change inference. The first source checkpoint at`ef207422f3e6c211f72f548122a418c1f7d9ac23` prepared1.5M without dispatch. This extension enables only27M on GHA and does not activate another variant or assert the state of a separate Kaggle submission. The existing Kaggle worker, source guard, original reports and numerical/ranking/client gates remain byte-identical.

The repository is public. GitHub documents standard `ubuntu-24.04` runners with4 CPUs/16GB RAM but14GB SSD; neither14GiB available RAM nor24GiB free disk is guaranteed. The [standalone workflow](../.github/workflows/moedern-scifact-gha-adapter-v1.yml) records actual admission and rejects below the unchanged14GiB/24GiB/two-CPU floors before tool installation or payload fetch. No paid larger runner is assumed, no system-image cleanup is performed, and budgets are not lowered to fit. If the runner fails admission, the negative report is retained and this provider remains unavailable. [GitHub runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#standard-github-hosted-runners-for-public-repositories).

The [new declaration](moedern-scifact-gha-adapter-v1.json) authenticates every original corrected worker/helper/runtime dependency against154 and separately pins the provider files. The new numerical entry point copies the old worker: only its source-closure import and provider admission in `resources()` differ. Structural controls compare every other function and the entry-point AST exactly. No historical source guard is patched, monkeypatched or silently bypassed.

Admission requires a genuine GitHub-hosted Linux x64 environment, explicit checkout SHA and live Actions run/attempt/repository/workflow-source metadata. It records checkout source separately from the workflow's service source. It requires the existing `KAGGLE_TOKEN`/`KAGGLE_ACCOUNT` secrets and the job's read-only artifact token; absent secrets stop before fetch. Tokens, account identities and signed URLs never enter receipts or command arguments. Kaggle SDK2.2.2 tooling uses an isolated environment separate from the hash-locked five-package numerical worker.

The provider authenticates the exact originalv1 submitted script through `GetKernel`, requires the exact version complete, and uses `version_label` on status and every paginated `List` request. It fetches only the pinned original report and its six declared raw members: canonical data,5483-row token/mask/offset journal, all/selected F32 vectors, complete scores and rankings. All whole-file sizes/SHA256s are checked before native models; the unchanged numerical worker then independently checks per-row spans, canonical gold, every ranking and metric before any forward. It also authenticates the older authored27Mv3 submitted script/report and all24 required raw arrays (20 ordinary all/selected vectors plus4 query-expansion arrays) through explicit-version Get/Status/List requests before models. The older1.5M GitHub artifact helper remains byte-identical for its existing Kaggle caller. No original framework or model recapture is installed or executed.

The published F32 model, all5483 raw text rows, explicit masks/role IDs, eight actual Ref shards, independent metrics, lexical ID tie order, strict F32 control, padded-Q4 conversion and C/Python/CLI/HTTP/Rust/Dart consumers are unchanged. Numerical budgets remain0.002 maximum absolute/0.99999 minimum row cosine, retrieval degradation at most0.001 and top-one agreement at least0.99. Q4 strict numerical and retrieval judgments remain separate; completion does not imply qualification or publication.

The GHA wrapper gives the existing child270 minutes inside a360-minute job, retaining partial numerical results and measured child/process logs on timeout. It uploads review/logs and native raw outputs separately for90 days with `if: always()`. It does not upload duplicate original multi-GB vectors: those remain at the authenticated original service, with exact member identities in the retained provider receipt. No model, corpus, vector or compiler runs locally during source preparation. Actual first-run admission, source/tooling imports, builds, model/token/vector/ranking/client outcomes and artifact upload remain pending.

## Corrected native baseline and preserved queued cancellation

The adapter now binds exactly to native source
`1543475b80622daf057ecf198136579cc1f7c051`. That bounded source correction
finds the already built CPU archive in `ggml/src`, where the pinned GGML build
actually emits it. Only the native test CMake path and its source declarations
changed; worker functions, original files, inference, numerical gates and
retrieval/client checks remain unchanged. The provider's protected closure is
recomputed against this exact source with no exception, and the AST comparison
reads the original worker directly from that Git commit.

[Initial queued run40774208339](../tests/results/moedern-colbert/scifact-gha-initial-queued-cancellation.json)
was cancelled with one job and zero steps. Its service source and intended
numerical source are recorded separately; no checkout, package install, payload
access, build or model execution occurred. It is a pre-execution cancellation,
not a numerical or admission result. The corrected source requires a separately
reviewed future dispatch; none is performed by this update.
