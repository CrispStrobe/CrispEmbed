#!/usr/bin/env python3
"""Hosted native tokenizer/template audit against frozen original IDs; no weights."""
from pathlib import Path
import json
import os
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from ultrabert_contract import MODEL, REVISION, QUERY, profile, sha256, tokenizer_metadata, validate_sources


def parse(log, rows):
    actual, masks, verdicts, total = {}, {}, {}, None
    for line in log.splitlines():
        fields = line.split('\t')
        if fields[0] in ('IDS', 'MASK', 'ROW'):
            index = int(fields[1])
            dest = actual if fields[0] == 'IDS' else masks if fields[0] == 'MASK' else verdicts
            if index in dest or index < 0 or index >= len(rows):
                raise ValueError('Unexpected/duplicate native tokenizer row')
            dest[index] = [int(v) for v in fields[2:]]
        elif fields[0] == 'TOTAL':
            if total is not None:
                raise ValueError('Duplicate tokenizer total')
            total = [int(v) for v in fields[1:]]
    expected_set = set(range(len(rows)))
    if set(actual) != expected_set or set(masks) != expected_set or set(verdicts) != expected_set:
        raise ValueError('Incomplete native tokenizer records')
    result = []
    for i, row in enumerate(rows):
        passed = actual[i] == row['ids'] and masks[i] == row['mask']
        if verdicts[i] != [int(passed)]:
            raise ValueError('Forged tokenizer verdict')
        result.append(dict(case=row['case'], batch_row=row['batch_row'], ids=actual[i],
                           attention_mask=masks[i], original_ids=row['ids'], original_mask=row['mask'],
                           exact=passed))
    failures = sum(not r['exact'] for r in result)
    if total != [len(rows), failures]:
        raise ValueError('Incomplete tokenizer total')
    return dict(rows=result, failures=failures)


def main():
    subprocess.run([sys.executable, str(ROOT / 'tests/test_ultrabert_tokenizer_log.py'), '-v'], check=True)
    import gguf
    from huggingface_hub import snapshot_download
    output = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'ultrabert-tokenizer'
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    baseline = ROOT / 'tests/results/ultrabert/reference-audit.json'
    if sha256(baseline) != 'a3c16863b1a78ebaebcc2d50e24555b696e691d3309ee17b75075c0051032a36':
        raise ValueError('Reviewed original reference report changed')
    frozen = json.loads(baseline.read_text())
    if frozen['source_commit'] != '0573f1847db68914052a98341a22fceeb1358518':
        raise ValueError('Wrong original producer')
    directory = Path(snapshot_download(MODEL, revision=REVISION,
                         allow_patterns=list(profile()['inspected_file_sha256'])))
    sources = validate_sources(directory)
    vocab, types, merges = tokenizer_metadata(json.loads((directory / 'tokenizer.json').read_text()))
    rows = []
    for record in frozen['reference']['cases']:
        case = record['case']
        texts = case['text'] if isinstance(case['text'], list) else [case['text']]
        for i, text in enumerate(texts):
            if case.get('prompt') == 'query':
                text = QUERY + text
            rows.append(dict(case=case['id'], batch_row=i, text=text, pair=case.get('text_pair', ''),
                             has_pair='text_pair' in case, ids=record['ids'][i], mask=record['attention_mask'][i]))
    if len(rows) != 8:
        raise ValueError('Frozen case row coverage changed')
    fixture = scratch / 'tokenizer.gguf'
    writer = gguf.GGUFWriter(str(fixture), arch='ultrabert_tokenizer_ref', use_temp_file=True)
    writer.add_string('ref.revision', REVISION)
    writer.add_uint32('ref.rows', len(rows))
    writer.add_array('tokenizer.ggml.tokens', vocab)
    writer.add_array('tokenizer.ggml.merges', merges)
    for i, row in enumerate(rows):
        key = f'row.{i}.'
        for name in ('text', 'pair'):
            writer.add_string(key + name, row[name])
        writer.add_bool(key + 'has_pair', row['has_pair'])
        for name in ('ids', 'mask'):
            writer.add_key_value(key + name, row[name], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    build = scratch / 'build'
    subprocess.run(['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                    '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DGGML_METAL=OFF'], check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'test-ultrabert-tokenizer', '-j2'], check=True)
    binary = next(p for p in (build/'test-ultrabert-tokenizer', build/'bin/test-ultrabert-tokenizer') if p.is_file())
    report = dict(source_commit=os.environ['GITHUB_SHA'], source_files=sources,
                  frozen_report_sha256=sha256(baseline), native_sha256=sha256(ROOT/'tests/test_ultrabert_tokenizer.cpp'),
                  driver_sha256=sha256(__file__), fixture_sha256=sha256(fixture), arms=[],
                  scope='native BPE and test-only UltraBERT templates over frozen texts; not public loading/inference',
                  model_weights_downloaded=False)
    target = output / 'ultrabert-tokenizer.json'
    for value in (None, '0'):
        env = dict(os.environ)
        env.pop('CRISPEMBED_ULTRABERT', None)
        if value is not None:
            env['CRISPEMBED_ULTRABERT'] = value
        control = subprocess.run([str(binary), 'absent.gguf'], env=env, capture_output=True)
        if control.returncode != 2:
            raise ValueError('Default/zero tokenizer gate failed')
    report['default_and_zero_gate_off'] = True
    env = dict(os.environ, CRISPEMBED_ULTRABERT='1')
    for name, args in (('native', []), ('terminal-sep', ['--terminal-sep']), ('mask-zero', ['--mask-zero'])):
        proc = subprocess.run([str(binary), str(fixture), *args], env=env, capture_output=True, text=True)
        log = output / (name + '.log')
        log.write_text(proc.stdout + proc.stderr)
        if proc.returncode not in (0, 1):
            raise ValueError('Native tokenizer structural failure: ' + name)
        arm = parse(proc.stdout, rows)
        if proc.returncode != int(arm['failures'] > 0):
            raise ValueError('Tokenizer exit disagrees with failures')
        arm.update(name=name, log=log.name, log_sha256=sha256(log))
        report['arms'].append(arm)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    report['negative_controls_rejected'] = all(a['failures'] > 0 for a in report['arms'][1:])
    report['status'] = 'completed; all exact ID/mask differences retained'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    if report['arms'][0]['failures'] or not report['negative_controls_rejected']:
        raise AssertionError('Native tokenizer parity/negative controls failed; see report')


if __name__ == '__main__':
    main()
