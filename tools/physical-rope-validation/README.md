# Source-bound physical RoPE controls

The [actual component result](../../tests/results/qwen35-ocr/lighton4b-physical-rope-source-controls.json)
binds candidate `a8a22749ab0b757ae1f317b6fa5b84ffd148b939`, its actual CPU
profile and the retained executable/library bundle. Eight supported or guard
units execute, including source-node/chunk/tail/position/alias/failure/concurrency
coverage, existing fused SwiGLU and physical operators, plus independent
unsupported and simulated WASM compilation. CLI512/1536 gates are checked in
both modes. The whole-decoder fixture target compiles but is not executed.

On the retained 266-row component, OFF preserves all 22 old stage/cache identities.
ON matches all 76 actual-source layer3 stage and canonical-cache comparisons
byte for byte, including the complete image chunk. Actual per-attention cache
mode toggles reject in both directions without changing prior K/V and recover
when the original mode is restored. No model/source or decoded forward occurs.
The [independent small-report audit](../../tests/results/qwen35-ocr/lighton4b-physical-rope-independent-small-audit.json)
checks identity and metric invariants; it does not independently download raw
vectors or recompute their reductions.

The [initial setup attempt](../../tests/results/qwen35-ocr/lighton4b-physical-rope-control-setup-negative.json)
stopped before compilation because a bounded candidate bundle needed additional
public prerequisites. The corrected launcher fetches all declared prerequisites
and checks out the exact candidate commit. Two pre-execution harness findings
were also fixed: explicit16/4 head configuration and SHA authentication of every
compared source stage.

Observed A/B must reuse this exact tested bundle, keep fused SwiGLU enabled and
change only the separate RoPE gate. All old fusedON baseline rows/stages and
stop states remain acceptance gates. Component equality does not qualify the
full decoded pipeline, the fixed panel, a platform, default or release.
