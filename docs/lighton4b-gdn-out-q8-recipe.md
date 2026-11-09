# Exploratory Q4_K plus24 GDN output Q8 recipe

`lighton4b-q4k-gdn-out-q8-v1` copies exactly24 whole
`l.layers.{i}.linear_attn.out_proj.weight` tensors from the authenticated Q8 source,
for every recurrent layer `i%4!=3`. All other699 tensors are unchanged Q4-source
payloads,including protected F32 vision/norms and tiedQ8 embedding/head. The
payload increase is125,829,120 bytes. This is an exploratory calibration-selected
policy,notQ4_K_M or a qualified model alias.

The[protocol](lighton4b-q4k-gdn-out-q8-v1.json) fixes names,all24 donor/base
hashes and selection rationale before complete output. Selection comes from the
projection screen42191354208,where output-only24 has the best table RMS among
single GDN classes; Arabic favors QKV,and every strict first-score gate still
fails. Those limits remain. Heldout receipt/letter are untouched.

`tools/lighton4b_gdn_out_pack.py` authenticates both full immutable source files,
copies payloads in bounded chunks without dequantization/requantization,and
rechecks all723 persisted tensor hashes/types/shapes. Original source/tokenizer/
geometry metadata bytes remain unchanged. Six added exact metadata fields bind
the recipe tag,source full hashes,canonical selected-manifest SHA256,count24 and
payload delta. Output census is474 F32,224Q4_K and25Q8_0 tensors. The resulting
full GGUF hash/file size are recorded only after packing and audit,then checked
again before native execution. Candidate/model arrays remain remotely retained.

New gate`CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8=1` is default-off. Admission requires
this exact tag and typed metadata plus the exact tensor type policy. Gate unset,
empty,0,false,no,off rejects the candidate; the old mixer gate simultaneously
on rejects ambiguity. UniformQ4/Q8 and the previous mixers recipe retain their
rules. Wrong source/manifest metadata or a single wrong protected/restored type
fails closed. Existing source/model/geometry/EOS checks remain mandatory. No
image-specific condition or reference activation enters model execution.

One hosted job first runs model-free pack/owner/recipe/parallel-conversion controls,
then real closed-context/metadata checks and four actual serialized-file smoke
controls. BothQ4 baseline and candidate must reproduce all32 block hashes,
first-score hashes and two-token prefixes from projection screen42191354208.
Both use verified vision and headBF16 corrections,parallel dequantization1,
BF16_QUANT_WEIGHTS0,BF16_LINEAR0 and rejected alternatives off. These smoke
controls establish bit equivalence against the original serial overlay screen
before any complete output is admitted.

Only after every smoke control passes,the job runs independent greedycap256 for
Q4 baseline and candidate on table/Arabic calibration,against retained genuine
original BF16 outputs. It retains one original-prefix first-score row,all greedy
score hashes/arrays,complete IDs/text/EOS/cap,CER/WER agreement and resource timing.
The old full-teacher mode remains; new`greedy-full` skips redundant full teacher
replay because first-score strict failure is already measured. It does not claim
full-position teacher parity or human OCR accuracy. Original receipt capped/noEOS
negative remains untouched.

Preflight requires8GiB available RAM including cgroup limits,20GiB scratch and
8GiB output disk. Each smoke process is bounded15min,each full table arm90min,
each Arabic arm20min; overall script270min. Process groups are task-owned and
terminated on timeout/cancellation,with incremental reports and partial artifacts.
The candidate GGUF itself is kept with90-day remote evidence. No runtime default,
release,app pin or public model qualification changes follow from preparation.
