#!/usr/bin/env python3
"""Build-only same-weight PCS matmul alternative; never rewrites production."""
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
anchor = '#include "ggml-cpu.h"'
helper = r'''
// Test-only attribution: retain identical stored weights, avoid an additional
// activation quantization by materializing quantized matrices as F32.
static ggml_tensor * pcs_probe_mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    if (core_env::on("PCS_GATE_F32_MATMUL") && ggml_is_quantized(a->type)) {
        a = ggml_cast(ctx, a, GGML_TYPE_F32);
    }
    return ggml_mul_mat(ctx, a, b);
}
#define ggml_mul_mat pcs_probe_mul_mat
'''
if source.count(anchor) != 1:
    raise ValueError('Unique PCS include anchor required')
Path(sys.argv[2]).write_text(source.replace(anchor, anchor + '\n#include "core/env_gate.h"\n' + helper))
