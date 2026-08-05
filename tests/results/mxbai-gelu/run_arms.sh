#!/bin/bash
# Serial A/B driver: one heavy process at a time, --gpu-backend cpu everywhere.
set -u
SC=${MXGELU_WORK:-.}
BIN=${CRISPEMBED_BIN:?path to the built crispembed CLI}
OUT=$SC/arms.txt
: > "$OUT"

run_one() {
  # $1 = arm label, $2 = gguf, $3 = query index, $4 = env spelling (unset|0|1)
  local label="$1" gguf="$2" qi="$3" spell="$4"
  local q; q=$(cat "$SC/query_q$qi.txt")
  echo "== $label | q$qi | env=$spell ==" >> "$OUT"
  if [ "$spell" = "unset" ]; then
    unset CRISPEMBED_RERANK_POOLER_GELU_ERF
    "$BIN" -m "$gguf" --rerank "$q" -f "$SC/docs_q$qi.txt" --gpu-backend cpu 2>/dev/null >> "$OUT"
  else
    CRISPEMBED_RERANK_POOLER_GELU_ERF="$spell" "$BIN" -m "$gguf" --rerank "$q" -f "$SC/docs_q$qi.txt" --gpu-backend cpu 2>/dev/null >> "$OUT"
  fi
}

for m in xsmall base; do
  for qi in 0 1; do
    for spell in unset 0 1; do
      run_one "$m-f16-new" "$SC/gguf/mxbai-$m-f16-new.gguf" "$qi" "$spell"
    done
    # shipped q8_0 (no pooler tensors -> gate inert, but record both spellings)
    for spell in unset 1; do
      run_one "$m-shipped-q8_0" "$SC/gguf/mxbai-rerank-$m-v1-q8_0.gguf" "$qi" "$spell"
    done
    # re-quantized q8_0 from the pooler-bearing f16
    if [ -f "$SC/gguf/mxbai-$m-q8_0-new.gguf" ]; then
      for spell in unset 0 1; do
        run_one "$m-q8_0-new" "$SC/gguf/mxbai-$m-q8_0-new.gguf" "$qi" "$spell"
      done
    fi
  done
done
echo "done" >> "$OUT"
