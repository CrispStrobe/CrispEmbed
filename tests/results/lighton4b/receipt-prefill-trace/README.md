# Receipt source-aligned prefill trace

The corrected source observer completed with source OFF/ON first-score,
first-sample/probability, vision/position and dispatch neutrality against the
original reference. All 256 retained native/source input embedding and position
rows match exactly. Source uses actual prompt graph sizes 4/242/6/4; the native
plan uses fourteen image tiles of 16 followed by 10 and 8.

The first **compared hidden boundary** difference is image graph2, GDN layer0,
starting at image row232 (prompt row236). Error RMS is
`9.724584391572298e-7`, maximum `7.82012939453125e-5`, against native/source
RMS magnitudes `0.19578077166613095` / `0.19578077416119666`. All31 compared
prefix hidden boundaries are exact. The difference precedes full-attention
layer3 and grows through later layers. Its first row is offset8 within the
native ten-row tile; that association does not establish a batching or operator
cause. The next experiment is a same-input layer0 component/projection replay
with the actual source recurrent state and authenticated quantized weights.

The final-score RMS `0.22138784887772267` and maximum `1.3122177124023438`
reproduce the original fixed RoPEON metrics exactly. The receipt floor still
fails; the original observed BF16 table/form regressions remain unchanged.
No native model inference, full source decode, full F32 text model, runtime
change, default or release promotion occurred.

Four raw first-prefix projection comparisons in the producer report did not
apply the physical head permutation. Their flat deltas are excluded from
semantic conclusions; the other eight comparable prefix operators and all
prefix hidden boundaries are exact. The source inner stages are retained for
same-input attribution; they are not yet paired native image operator captures.
The source's last layer is cropped, so only its final prompt token is compared.

[Successful safe evidence](successful-prefill-trace.json) records sources,
terminal report hashes, magnitudes, geometry and interpretation limits. The
[initial capture-limit negative](initial-capture-budget-negative.json) remains
separate: the first observer exceeded its cumulative 512MiB capture cap during
the trailer; a bounded 768MiB retry passed both neutrality controls. Only small
reports were inspected locally; vectors remain remote. Wrapper-owned mutable
report/log snapshots are not represented as independently rehashed immutable
raw artifacts.
