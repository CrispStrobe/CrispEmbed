# Physical fused SwiGLU observed A/B

The initial combined job passes all five [actual native control targets](../tests/results/qwen35-ocr/lightonocr3-4b-swiglu-observed/v1-initial/controls.json) on immutable candidate `519c3d9765c234cd93e9069d5efdc239e3cad3f2`: source-prefill4165, GDN/cache26, physical-native142, F16 attention255297 and six GGML-exact SwiGLU arms. CPU baseline, supported physical ABI and actual cap512 CLI also pass. The exact tested executable/library bundle is retained remotely. No checkpoint is loaded during controls.

The [initial model evidence](../tests/results/qwen35-ocr/lightonocr3-4b-swiglu-observed/v1-initial/partial.json) is **incomplete** because diagnostic postprocessing fails after three native arms. Table OFF reproduces all309 historical plannerON rows, IDs/text, EOS and stages. Fused ON completes218 tokens with EOS and reduces original-reference character edits321→20/607 and word edits92→6/106. First-score RMS versus pinned llama changes0.179280136→0.165976534, while maximum error worsens0.910927415→1.08675575. This is observed calibration, not numerical or shipping qualification.

Form OFF completes168 tokens and its historical rows/stages close. A source-probe loop then shadows the command label with `down`, producing `KeyError` during resources lookup. Runtime, weights, gates, prompt and scoring are unchanged. Form ON and both Arabic arms have not run; no complete panel result follows. Preserve the initial negative report SHA `b85f3df503be35c4427cd160adf9fd78eeb0733e1cdd98c6c5767e7ab696ae52` and actual controls SHA `97721fed18de359d08d633e017c51b261950e7fb788952d67607e48a3ea7f0f4`.

A bounded resume must authenticate retained completed arrays and metadata, reuse the exact tested bundle, and execute only missing formON/ArabicOFF/ArabicON arms. It must not rebuild or repeat successful arms. The failed provider output is unavailable as a direct kernel input; task-owned artifact transport is being prepared without exposing access capabilities. All experimental gates remain default-off; no heldout/book data was mounted and no source-model or full F32/BF16 text-model forward occurred.


## Completed receipt-bound continuation

The [complete result](../tests/results/qwen35-ocr/lightonocr3-4b-swiglu-observed/v1-resume/result.json) authenticates all 996 retained owned files/959,232,361 bytes, then reuses table OFF/ON and form OFF without a forward. It executes only form ON and Arabic OFF/ON with the original tested candidate/bundle, unchanged physical weights, planner ON, cap512 and threads4. All three OFF outputs close every historical score row, text/IDs/EOS and retained stage. The resumed receipt is `67ff19db0230ddc329300cc3f6a0256eb85b3cdbbcf06a578b23eb93c2688ba1`; the initial postprocessor negative remains preserved above.

| Observed case | OFF → ON tokens (both EOS) | Character edits OFF → ON | Word edits OFF → ON | Upstream floor character/word edits |
| --- | --- | --- | --- | --- |
| Table | 309 → 218 | 321 → 20 /607 | 92 → 6 /106 | 348 /97 |
| Form | 168 → 167 | 165 → 175 /656 | 43 → 45 /90 | 189 /46 |
| Arabic line | 6 → 6 | 0 → 0 /12 | 0 → 0 /2 | 0 /0 |

All three unchanged provisional upstream floors pass in both arms. Form regresses by ten character edits and two word edits; no aggregate waiver follows from the table gain. Arabic decoded text remains `أبجدية عربية`. Complete decoded outputs and token IDs for these synthetic observed cases are retained in the public result.

The independent small-artifact audit recomputes every decoded metric from immutable original/upstream JSON and rehashes/recomputes all eight form first-four gate/up/activation/down arrays. Gate/up match actual source bytes in both arms. OFF activation RMS/max are `1.39921456e-9`/`5.96046448e-8` at source RMS magnitude `0.0381852837`; down RMS/max are `6.37168242e-9`/`3.72529030e-8` at magnitude `0.0465778177`. ON activation and down are both bit exact to that actual source probe. This local closure does not establish complete source-graph equivalence or decoded quality.

First-score RMS versus upstream changes table `0.179280136→0.165976534`, form `0.129571035→0.123510493`, and Arabic `0.118377677→0.129915419`. Table maximum error worsens `0.910927415→1.08675575`; Arabic maximum improves `0.804398298→0.677476168`. Retain both error measures and all per-layer magnitudes. Initial/resumed timings are from different hosted sessions and cannot establish a speed win.

The model worker authenticates old artifacts and validates generated raw rows; the independent review covers small reports/labels and eight selected arrays. It does not claim a separate independent full-raw inventory audit. The gate remains default-off and unqualified. No runtime rebuild, source forward, full F32/BF16 text model, heldout access, default, release, alias or application change follows. A separate fixed regression replay requires its own source/control/input contracts before dispatch.
