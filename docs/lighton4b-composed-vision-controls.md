# Composed-vision executor controls

Kaggle CPU service `lighton4b-composed-vision-model-free-controls-v1`, version1, completed model-free controls at source `91025dd4e588d5f550f6051454c584587dc15fb3`. [Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-composed-controls/composed-vision.json) SHA256 `52fd2ccd26c9387b8ed459c0c9be9e7d813e65bc9a3b9c37274fd002ce022b4d`; [independent review/inventory](../tests/results/qwen35-ocr/lightonocr3-4b-composed-controls/independent-review.json).

The actual native driver built unchanged production/CMake sources. Supported CPU/private-ABI controls passed255,297 checks. Nine actual CLI checks admitted old256, new composed512, new source-fed512 and existing table-only512 through metadata opening (missing fixture return4), while rejecting unflagged512,513, wrong threads/mode and unknown flags (return2). This validates argument handling and host support, not inference. Model acquisition and forwards were disabled.

Elapsed116.32seconds, build70.40seconds, sampled process peak144.3MB; effective available RAM31.48GB. Three local protocol/arity tests and runtime group-deadline tests passed before the worker. Only267KB of reports/logs were downloaded; model files remain absent locally.

The paired executor still requires observed model execution and independent artifact review. It compares deployed projectors with unproven canonical vision-weight equivalence, keeps actual upstream internal stages unsupported and labels BF16 captures as reference outputs, not human ground truth. Capped outputs remain incomplete. This is no shipping or held-out qualification; experimental gates remain off. The prospective model source changes only a report wording clarifying that historical table256 was capped while the required512 reference completed.
