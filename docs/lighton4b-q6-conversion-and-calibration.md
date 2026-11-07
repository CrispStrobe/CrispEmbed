# Reproduce experimental LightOn4B BF16→Q6 conversion

PR155 includes the native loader, bounded source converter, independent controls
and closed hosted experiment harnesses. Conversion starts with the authenticated
original BF16 checkpoint; it never requantizes Q8/Q4 matrices into Q6 or builds a
full F32 model. Named recipes remain experimental and default-off, with no model
alias, upload or release promotion.

| Fixed recipe | Fresh original-BF16 Q6 matrices | Unchanged payload copies | Tied embedding/head |
|---|---:|---:|---|
| `lighton4b-q4km-q6-q8head-v1` | 32 value-family/down matrices | 691 | Exact original Q8 |
| `lighton4b-q4k-gdn-out-q6-v1` | 24 recurrent output matrices | 699 | Exact original Q8 |

The32-matrix rule is Q4_K_M-style with the explicit Q8 tied-head exception; the
24-output policy is a separate custom recipe. Existing plainQ4/Q8, attention-mixed
and32-Q6 profiles remain intact. Exact selected names,723-tensor inventory,
metadata and sources are declared in the
[32-Q6 protocol](lighton4b-q6-conversion-protocol-v1.json) and
[24-Q6 protocol](lighton4b-gdn-out-q6-conversion-protocol-v1.json).

`tools/lighton4b_q6_rows.cpp` is built only by explicitly requesting
`lighton4b-q6-rows` (EXCLUDE_FROM_ALL); it is absent from production artifacts.
It reads whole BF16 rows in chunks of at most65,536 elements, expands only that
chunk exactly to F32 and invokes pinned ggml Q6 quantization. It authenticates
source bytes and records raw-Q6/effective-BF16 hashes. The Python writer clones
all protected GGUF metadata and payload bytes, writes strict M15/custom metadata,
authenticates all723 inputs/output tensors and independently checks each fresh
Q6 dequantized-to-BF16 hash with gguf/NumPy. It publishes the output filename only
after successful final readback. Buffers remain bounded; no neural reference
forward runs here.

For an adequately sized hosted runner with the pinned source files and built row
helper, the explicit API entry point is:

```python
import sys
from pathlib import Path
sys.path.insert(0, 'tools')
from lighton4b_q6_convert import convert
from lighton4b_gdn_out_q6_recipe import plan

receipt = convert(
    Path('plain-q4.gguf'), Path('original.safetensors'),
    Path('original-weight-oracle.json'), Path('build/lighton4b-q6-rows'),
    Path('lighton4b-q4k-gdn-out-q6-v1.gguf'), Path('conversion.json'),
    recipe_plan=plan(),
)
```

The input filenames do not waive identity checks: original checkpoint6b0dab74…
is pinned at BF16 revisionb71010f0…, baseQ4f873eb3d… at revisiona7ec2e54… and
[oracle03119ea4…](https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/6525e045e16906b2025cad4fa10696a2e918abac/tests/results/qwen35-ocr/lightonocr3-4b-bf16-quant-first-score-v1.json)
is immutable. To choose the preserved32-Q6 recipe, import `plan` from
`lighton4b_q6_recipe` instead. The first actual32-Q6 candidate is76911f05…;
24-output candidate83e065c4… is4,095,839,136 bytes. No quantized donor supplies
any freshly generated Q6 matrix; unselected Q4/Q8 payloads are intentionally copied.

The closed hosted routes build the helper and stage authenticated inputs. These
are manual, bounded selectors in `heavy-cpu.yml`; model selectors require empty
args/pip, portable mode and ubuntu-latest. They have read-only artifact/reference
credentials, fixed cases and90-day output retention. For example:

```sh
# Model-free streaming/chunk/source/independent-decoder controls.
gh workflow run heavy-cpu.yml --ref feat/lighton4b-native-document-20261105 \
  -f script=tools/ci-heavy/lighton4b_q6_conversion_controls.py \
  -f pip='numpy==2.3.5 gguf==0.19.0' -f build_mode=portable -f timeout_minutes=25

# Actual fixed24 conversion, then Q4/Q8-output/Q6-output first-score calibration.
gh workflow run heavy-cpu.yml --ref feat/lighton4b-native-document-20261105 \
  -f script=tools/ci-heavy/lighton4b_gdn_out_q6_candidate.py -f build_mode=portable
```

Other exact selectors are `lighton4b_q6_candidate.py` for the preserved32 policy,
`lighton4b_gdn_out_q6_full.py` for two-case full calibration using the retained
24-output candidate, and `lighton4b_gdn_out_q6_logo_diagnostic.py` for the explicitly
post-selection, already observed logo case. Native owner/operator controls are
`lighton4b_q6_controls.py` and `lighton4b_gdn_out_q6_controls.py`. No command here
runs automatically or expands into a matrix. Source protocols contain exact model,
image/input, score/capture and runtime-file hashes. They reject changed references;
expired comparison artifacts require separate review, never silent repinning.
The source-only converter remains reproducible from authenticated original/base
snapshots even when comparison archives expire.

Conversion buffers are below2MiB plus bounded metadata (conservative512MiB Python
allowance). Actual24 conversion/calibration guards require32GiB shared free disk
or28GiB scratch plus8GiB output,8GiB effective RAM and2GiB cgroup margin. Full
replays use16GiB shared or12GiB scratch plus4GiB output. Builds use two threads,
native runs four. Global/per-child deadlines and partial negatives are retained;
large files stay remotely, with no model-hub upload.

Verification includes the65,792-value hosted stream/neighbor/tail/source-hash and
independent-decode controls in
[42210897150](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42210897150),
700 new24-output,713 old32 and112 Q8-output controls in
[42218056668](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42218056668),
actual723-tensor conversion and six immutable first-score arms in
[42219710049](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42219710049),
and full independent calibration in
[42223984643](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42223984643).
The24-output Q6 candidate exactly matches Q8-output full IDs/text/EOS on table
and Arabic, while the table still differs from genuine BF16 (CER.369028/WER.575472).
The [post-selection logo diagnostic](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42230729575)
remains `A`, matching Q4/Q8 but differing from the65-ID BF16 caption. All strict
first-score negatives and the calibration limits remain in PR157 evidence. This
end-to-end support does not qualify general OCR quality, heldout performance,
numerical parity, default gates or a release.
