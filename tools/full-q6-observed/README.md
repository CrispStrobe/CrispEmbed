# Observed physical full-Q6 comparison

Build the supported CPU runtime, run functional type/gate/cache/operator controls,
import the unchanged publisher full-Q6 decoder, and decode the three existing
observed table/form/Arabic images. Compare complete IDs/text/EOS at the source's
cap256 against its existing Q6 outputs, and score both against the separately
retained original BF16 outputs. No original or upstream model forward is needed.

The importer retains 249 quantized Q6 matrices and 177 direct physical small
parameters, preserves all 297 protected vision tensors, and checks the actual
small parameters against the prior publisher Q4 source. Its coordinate mapping
and tied head remain explicit. No F32 model or F32 text-weight cache is created.

The worker requires Kaggle CPU, 10 GiB available memory, 30 GiB scratch and
12 GiB output; each model arm allows 1200 seconds, the worker 9000 seconds.
Set `HEAVY_OUT` and `HEAVY_SCRATCH` to new task-owned output/scratch directories:

```sh
python tools/full-q6-observed/worker.py
```

The worker runs its small Python controls and native functional controls before
models. Ordinary transfer/setup fixes and reruns are allowed. No baseline mount,
provenance seal or separate approval acknowledgement is needed. It records source
versions, models, outputs and failures as evidence. Large models, actual binaries,
all vocabulary rows and hidden arrays remain on the worker; read only needed
small reports locally.

Publisher Q6 currently matches BF16 table222 and Arabic6 IDs/text/EOS. Its
form182 response still has 154 character and 31 word edits against BF16 form177.
These are model agreement measurements, not human OCR accuracy. Existing Q6
source captures lack full-vocabulary scores and hidden arrays, so complete decoded
agreement does not establish numerical/operator parity. Existing Q4 failures,
experimental gates, aliases, releases and app pins remain unchanged. Complete
all three observed cases after decoded negatives; use actual mismatches to choose
further attribution work.
