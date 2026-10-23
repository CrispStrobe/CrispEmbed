#!/usr/bin/env python3
"""Hosted original-module/backend diagnostics; never replace frozen references."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ultrabert_contract import MODEL, REVISION, profile, sha256, validate_sources, validate_weight_file
from ultrabert_native import read_frozen_manifest


def compare(actual, expected):
    import numpy as np
    a, b = actual.detach().float().cpu().numpy(), expected.detach().float().cpu().numpy()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid same-input comparison')
    delta = a.astype(np.float64) - b.astype(np.float64)
    return dict(bit_exact=a.tobytes() == b.tobytes(), max_abs=float(abs(delta).max()),
                rms=float(np.sqrt(np.mean(delta * delta))), actual_norm=float(np.linalg.norm(a.astype(np.float64))),
                reference_norm=float(np.linalg.norm(b.astype(np.float64))),
                actual_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                reference_sha256=hashlib.sha256(b.tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frozen', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    import gguf
    import numpy as np
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoModel
    from dump_ultrabert_reference import capture
    if torch.__version__.split('+')[0] != '2.14.0' or transformers.__version__ != '5.16.1':
        raise ValueError('Original pinned environment required')
    baseline = json.loads((ROOT / 'tests/results/ultrabert/reference-audit.json').read_text())
    frozen = read_frozen_manifest(baseline, args.frozen)
    directory = Path(snapshot_download(MODEL, revision=REVISION,
                         allow_patterns=list(profile()['inspected_file_sha256']) + ['model.safetensors']))
    sources = validate_sources(directory)
    weights = validate_weight_file(directory / 'model.safetensors')
    torch.set_num_threads(2)
    model, loading = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=True,
                        attn_implementation='eager', dtype=torch.float32, output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Original model loading mismatch')
    model.eval()
    modules = {'layer_0.q': model.layers[0].self_attn.q_proj,
               'layer_14.gate': model.layers[14].mlp.gate_proj,
               'layer_14.up': model.layers[14].mlp.up_proj,
               'layer_14.ffn_out': model.layers[14].mlp.down_proj}
    names = {'layer_0.q': 'layers.0.self_attn.q_proj', 'layer_14.gate': 'layers.14.mlp.gate_proj',
             'layer_14.up': 'layers.14.mlp.up_proj', 'layer_14.ffn_out': 'layers.14.mlp.down_proj'}
    loaded = {}
    with safe_open(str(directory / 'model.safetensors'), framework='np') as stored:
        for key, module in modules.items():
            for suffix in ('weight', 'bias'):
                value = getattr(module, suffix)
                if value is None:
                    continue
                original = stored.get_tensor(names[key] + '.' + suffix)
                if value.detach().numpy().tobytes() != original.tobytes():
                    raise ValueError('Loaded original parameter bits differ')
                loaded[key + '.' + suffix] = hashlib.sha256(original.tobytes()).hexdigest()
    report = dict(source_commit=os.environ['GITHUB_SHA'], original_source=frozen['source_commit'],
                  scope='Original module/backend reproducibility only; no native correction or replacement targets',
                  source_files=sources, source_weights=weights['sha256'], loaded_parameter_sha256=loaded,
                  environment=dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                                   torch=torch.__version__, transformers=transformers.__version__,
                                   torch_config=torch.__config__.show(), cpu_capability=torch.backends.cpu.get_cpu_capability(),
                                   cpuinfo=Path('/proc/cpuinfo').read_text(),
                                   relevant_env={k:v for k,v in os.environ.items() if k in
                                       ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','ATEN_CPU_CAPABILITY','MKL_CBWR')}),
                  arms=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    for threads, mkldnn in ((2, True), (1, True), (2, False), (1, False)):
        torch.set_num_threads(threads)
        torch.backends.mkldnn.enabled = mkldnn
        arm = dict(threads=threads, mkldnn=mkldnn, cases=[])
        report['arms'].append(arm)
        for case in frozen['reference']['cases']:
            path = args.frozen / 'reference' / case['archive']
            if sha256(path) != case['archive_sha256']:
                raise ValueError('Original archive hash changed')
            reader = gguf.GGUFReader(str(path))
            tensors = {t.name: t for t in reader.tensors}
            def old(name):
                r = case['stages'][name]
                t = tensors[name]
                if hashlib.sha256(t.data.tobytes()).hexdigest() != r['payload_sha256']:
                    raise ValueError('Original payload changed')
                return torch.from_numpy(t.data.copy().reshape(r['shape']))
            encoded = {k:old(k).long() for k in ('input_ids','attention_mask')}
            output, captures = capture(model, encoded)
            if len(captures) != 386:
                raise ValueError('Incomplete original forward')
            stages = {k:compare(v, old(k)) for k,v in captures.items()}
            first = next((k for k,v in stages.items() if not v['bit_exact']), None)
            entry = dict(case=case['case']['id'], original_archive_sha256=case['archive_sha256'],
                         first_differing_stage=first, differing_stages=sum(not v['bit_exact'] for v in stages.values()),
                         stages=stages, projections={})
            gate, up = old('layer_14.gate'), old('layer_14.up')
            with torch.inference_mode():
                product = model.layers[14].mlp.act_fn(gate) * up
                inputs = {'layer_0.q': old('layer_0.attn_norm'),
                          'layer_14.gate':old('layer_14.ffn_norm'),
                          'layer_14.up':old('layer_14.ffn_norm'), 'layer_14.ffn_out':product}
                for key, module in modules.items():
                    x = inputs[key]
                    original = module(x)
                    repeated = module(x)
                    if not torch.equal(original.view(torch.int32), repeated.view(torch.int32)):
                        raise ValueError('Repeated same-input original projection changed')
                    bias = module.bias
                    f64 = torch.nn.functional.linear(x.double(), module.weight.double(),
                                                   None if bias is None else bias.double()).float()
                    rounded = torch.nn.functional.linear(x, module.weight.half().float(), bias)
                    entry['projections'][key] = dict(input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                         original_vs_frozen=compare(original, old(key)),
                         f64_vs_original=compare(f64, original), f64_vs_frozen=compare(f64, old(key)),
                         f16_weight_vs_original=compare(rounded, original), repeated_original_exact=True)
            arm['cases'].append(entry)
            save()
            del tensors, reader, captures, output
    report['status'] = 'completed four backend/thread arms; all frozen-anchor discrepancies retained'
    save()


if __name__ == '__main__':
    main()
