# Bounded sequential mixed-model calibration

The exact predeclared recipe is `lighton4b-q4k-mixers-q8-v1`: attention Q8,
FFN Q4, the unchanged Q8 tied head and protected F32 tensors. The worker
`tools/ci-heavy/lighton4b_calibration_native.py` uses one authenticated mixed
artifact download and one portable build, then runs only `simple_table` and
`arabic_printed_line` sequentially. No heldout case runs in this job.

The reviewed original BF16 source has222 and6 EOS-completed positions for these
cases. Every original capture, prepared input and full-vocabulary score is
rehash-checked from exact retained version1. Native pixels/tokens/modalities/grid
must match before model loading. Full original-prefix teacher scores and
independent greedy generation use the frozen cap256; actual Ref compares every
score with unchanged0.002/0.99999 gates. A negative first case does not suppress
the second case. Numerical rejection is recorded separately from execution or
retrieval failure.

The scoped job in `.github/workflows/heavy-cpu.yml` selects only this exact
script, requires read-only artifact access and has no build matrix. It installs
pinned transport/Pillow/NumPy dependencies without Torch, stages the mixed model
once, builds once, and authenticates binaries before reuse. RAM/free-disk checks
precede acquisition. Each native process group has a90-minute bound; the overall
script has225 minutes and the job240 minutes, with partial output upload.
Only a verified task-owned transport ZIP is removed after extraction. Models
remain in worker scratch and large evidence stays remote.

The heldout original receipt case reached cap256 without EOS. That negative
remains a model-qualification blocker even if both calibration cases improve.
This job cannot qualify a model, change a default, publish an alias or update a
runtime release. Direct quantized dots remain disabled.
