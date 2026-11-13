# Fresh authored document screen: Q6 ties Q4

The native Q6 and Q4_K_M profiles produce identical token lists and decoded text
on all three new authored pages. Every arm reaches EOS, and the invoice has all
16 cells exact. This panel demonstrates no Q6 quality gain over Q4.

| Page | Tokens per profile | Frozen v1 content character / word edits | Invoice cells |
|---|---:|---:|---:|
| Field report |182 / EOS|12 /4|not applicable|
| Supply invoice |296 / EOS|0 /0|16 /16 exact|
| Accented dispatch |199 / EOS|56 /15|not applicable|

The content denominators are respectively 532 characters/95 words, 288/47 and
429/69. Both profiles have the same scores. Actual decoded text was checked
against the visually inspected PNGs and authored gold: no transcription error
was found. The prose/dispatch residuals arise from Markdown `**` emphasis and
standalone `---` rules that the existing versioned content parser retains.
Removing only those presentation markers as a **post-hoc formatting diagnostic**
yields exact whitespace-normalized text and tables. The frozen v1 scores, raw
outputs, gold and scoring policy remain unchanged; this diagnostic is not a new
acceptance gate. The invoice's raw 302 character/32 word edits are HTML markup;
its parsed content and grid are exact. [The small result](../tests/results/lighton4b/authored-q6-q4-quality.json)
retains complete outputs/IDs, both score views and the formatting attribution.

Execution used [source 90914b674](https://github.com/CrispStrobe/CrispEmbed/commit/90914b6746155f0f9e4eac46adf3fd0c80598a4b)
on Kaggle CPU: four threads, a fixed 512-token cap, independent composed native
vision and greedy decoding, Q4 then Q6 for each page. It reused the exact
[observed full-Q6 runtime](lighton4b-full-q6-observed.md) at `0b576d0e3` and the
retained immutable 4,041,905,760-byte Q4 and 4,797,156,960-byte Q6 models. No
runtime build, import, BF16/original forward or previous-case replay was needed.
All 1,354 actual vocabulary rows pass finite/greedy checks; every arm confirms
zero F32 text-weight cache entries/bytes. The initial 1000×1000 preprocessing
rejection is [retained separately](../tests/results/lighton4b/authored-q6-q4-initial-setup.json).
Before any model output, the unchanged text was directly rerendered at 512×512
with 14-pixel body text/19-pixel headings and all three final PNGs inspected again.

All saved vision stages and `prefill.input` are bit exact between profiles.
The earliest saved difference is decoder layer 0: RMS 0.003982–0.004267,
maximum absolute error 0.201206–0.224107, with profile RMS magnitudes
0.160861–0.172217. Final-normalization RMS errors are 0.838211–0.939912;
first-logit RMS errors 0.357148–0.484672, maximum 1.826576–2.412513.
Complete decoded agreement does not establish numerical or operator parity.
These are cross-precision comparisons at saved boundaries, with no new
original-source hidden-state capture.

The worker took 3,822.76 seconds; per-arm times are diagnostic single executions
in fixed order, not a speed benchmark. Large raw arrays remain remote. These
three clean synthetic Latin-script pages do not establish representative scan,
Arabic, broad multilingual, other-platform or release qualification. Earlier
quality negatives remain binding. Defaults, aliases, releases and app pins stay
unchanged. Representative licensed scans and separately scoped source/operator
or public-consumer/platform validation remain open.
