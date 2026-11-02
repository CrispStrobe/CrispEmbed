# Local development diagnostics (macOS arm64, not qualification)

Produced while developing the port on an overloaded shared workstation
(load average ~85), before the hosted CI lane existed. Timings here are not
speed evidence. OpenCV on this host has no IPP, so the reference used OpenCV's
generic Lanczos4 path; native and reference preprocessed inputs are bit-identical
(`input` max_abs 0 on all five pages).

These diffs were produced by an earlier build of tests/oxr_dfine_diff.cpp:
post.* were then still budgeted positionally, and `stage_budget_pass`/
`selection_pass` were not separate fields. They are kept unchanged as the
initial results. The 5e-3 stage budget fails in decoder layers 2-3 on four of
five pages; the float64 yardstick (receipt) shows native is closer to exact
arithmetic than PyTorch FP32 on 56/56 stages. F16 is rejected here (receipt:
one selected box differs by 147 px).
