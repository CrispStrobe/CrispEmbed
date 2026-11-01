#!/usr/bin/env bash
# heavy-cpu.yml wrapper: delivery-4 graph build and model-free controls on CPU.
set -euo pipefail
python tools/ci-heavy/inpaint_graphs.py --backend cpu --output "$HEAVY_OUT/graphs" \
  --scratch "$HEAVY_SCRATCH/graphs" --deadline-minutes 120 --threads 4 --parallel 4
