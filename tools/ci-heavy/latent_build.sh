#!/usr/bin/env bash
# heavy-cpu.yml wrapper: latent_inpaint.py build-only phase (no model access).
set -euo pipefail
python tools/ci-heavy/latent_inpaint.py --phase build-only \
  --output "$HEAVY_OUT/latent-build" --scratch "$HEAVY_SCRATCH/latent-build" --deadline-minutes 120
