"""Pure decision logic for the layer0 attention op-by-op replay (stdlib only)."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('ffn_analysis', ROOT / 'tools/wasm-layer3-ffn-matmul/analysis.py')
F = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(F)
require, metrics, admit = F.require, F.metrics, F.admit
H = 384
ORDER = ('ln_mul', 'ln_add', 'q_raw', 'k_raw', 'v_raw', 'q', 'k', 'v', 'flash', 'o_raw', 'o', 'residual')
KIND = {'ln_mul': 'elementwise mul', 'ln_add': 'elementwise add', 'q': 'elementwise add', 'k': 'elementwise add', 'v': 'elementwise add',
        'o': 'elementwise add', 'residual': 'elementwise add', 'q_raw': 'mul_mat', 'k_raw': 'mul_mat', 'v_raw': 'mul_mat',
        'o_raw': 'mul_mat', 'flash': 'flash_attn_ext'}


def m(a, b):
    return metrics(a, b, H)


def selftest(meta, runs):
    """runs[(side, run_name)][op] -> bytes. run_name in chain1, chain2, ops_own."""
    checks = []
    for side in ['native', 'wasm']:
        checks.append(dict(name=side + '/plain-cpu-buffer', passed=all(meta[(side, r)]['weight_buffer'] == 'CPU' for r in ['chain1', 'chain2', 'ops_own'])))
        checks.append(dict(name=side + '/ops-order', passed=all(tuple(meta[(side, r)]['ops']) == ORDER for r in ['chain1', 'chain2', 'ops_own'])))
        for op in ORDER:
            c1, c2, own = runs[(side, 'chain1')][op], runs[(side, 'chain2')][op], runs[(side, 'ops_own')][op]
            checks.append(dict(name=f'{side}/{op}/deterministic', passed=c1 == c2))
            checks.append(dict(name=f'{side}/{op}/isolated-op==chain', passed=own == c1))
    return dict(passed=all(c['passed'] for c in checks), checks=checks)


def replay(meta, runs, captured):
    """runs: (native|wasm, chain|ops_own|ops_cross)[op]; ops_cross = this build's op on the OTHER build's chain inputs."""
    controls = []
    types = meta[('native', 'chain')]['weights']
    controls.append(dict(name='weights-identical-metadata', passed=types == meta[('wasm', 'chain')]['weights']))
    for side in ['native', 'wasm']:
        for r in ['chain', 'ops_own', 'ops_cross']:
            controls.append(dict(name=f'{side}/{r}/plain-cpu-buffer', passed=meta[(side, r)]['weight_buffer'] == 'CPU'))
        for op in ORDER:
            controls.append(dict(name=f'{side}/{op}/isolated-op==chain', passed=runs[(side, 'ops_own')][op] == runs[(side, 'chain')][op]))
    controls.append(dict(name='shared-input/norm00_output-native==wasm', passed=captured[('native', 'norm00_output')] == captured[('wasm0', 'norm00_output')]))
    controls.append(dict(name='own-reproduction/native/norm01_input', passed=runs[('native', 'chain')]['residual'] == captured[('native', 'norm01_input')]))
    controls.append(dict(name='own-reproduction/wasm/norm01_input', passed=runs[('wasm', 'chain')]['residual'] == captured[('wasm0', 'norm01_input')]))
    table = []
    for op in ORDER:
        N = runs[('native', 'chain')][op]
        table.append(dict(op=op, kind=KIND[op],
                          same_input_wasm_vs_native=m(N, runs[('wasm', 'ops_cross')][op]),
                          same_input_on_wasm_inputs=m(runs[('native', 'ops_cross')][op], runs[('wasm', 'chain')][op]),
                          propagated_chain=m(N, runs[('wasm', 'chain')][op])))
    differing = [r['op'] for r in table if not (r['same_input_wasm_vs_native']['bit_exact'] and r['same_input_on_wasm_inputs']['bit_exact'])]
    first_prop = next((r['op'] for r in table if not r['propagated_chain']['bit_exact']), None)
    kinds = sorted({KIND[o] for o in differing})
    return dict(controls_passed=all(c['passed'] for c in controls), controls=controls, weights=types, ops=table,
                first_same_input_differing_op=differing[0] if differing else None, same_input_differing_ops=differing,
                differing_op_kinds=kinds, single_deterministic_op_kind=len(kinds) == 1,
                first_propagated_differing_op=first_prop,
                norm01_input=m(captured[('native', 'norm01_input')], captured[('wasm0', 'norm01_input')]))
