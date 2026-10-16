"""One fixed receipt: original image-only vs explicit literal text, full outputs."""
import gc
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess

from qwen35_decoded_panel import RECIPES, validate
from qwen35_page_parity import parse
from qwen35_quant_calibration import errors, file_sha
from qwen35_vision_operators import prefill_records

LITERAL_PROMPT = ('Transcribe all visible text exactly as plain text. Preserve reading order and line breaks. '
                  'Do not add Markdown, HTML, descriptions, or commentary.')


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def add_confidences(log, measured, reference):
    matches = re.findall(r'^FREE confidences:(.*)$', log, re.M)
    if len(matches) != 1:
        raise ValueError('Missing/repeated native confidence vector')
    values = list(map(float, matches[0].split()))
    if (len(values) != len(measured['native_ids'])
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in values)
            or f32(sum(map(f32, values)) / len(values)) != f32(measured['native_mean_confidence'])):
        raise ValueError('Invalid native confidence count/value/mean')
    original = reference['selected_token_confidences']
    if len(original) != len(reference['generated_ids']) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in original):
        raise ValueError('Invalid original confidence vector')
    # Always retain both arrays; align positions only while generated IDs agree.
    prefix = next((i for i, (a, b) in enumerate(zip(measured['native_ids'], reference['generated_ids'])) if a != b),
        min(len(values), len(original)))
    difference = [f32(values[i]) - original[i] for i in range(prefix)]
    measured['native_confidences'] = values
    measured['confidence_comparison'] = dict(aligned_prefix_tokens=prefix,
        max_abs=max(map(abs, difference), default=0.),
        mean_abs=sum(map(abs, difference))/max(1, prefix),
        native_magnitude=max(map(abs, values), default=0.), reference_magnitude=max(map(abs, original), default=0.),
        scope='Actual generated probabilities; aligned by identical prefix IDs only. Diagnostic, not OCR accuracy/calibration.')
    return measured


def teacher_quality(log, count):
    stages = re.findall(r'^(PASS|FAIL) (generated\.scores\.\d+):', log, re.M)
    budgets = re.findall(r'^BUDGET (generated\.scores\.\d+): strict_absolute_pass=(\d+) calibrated=(\d+) self_max_abs=(\S+) max_ratio=(\S+)$', log, re.M)
    summary = re.findall(r'^TEACHER_QUALITY stages=(\d+) strict_failures=(\d+)$', log, re.M)
    expected = ['generated.scores.'+str(i) for i in range(count)]
    if (count<1 or len(summary)!=1 or int(summary[0][0])!=count or [s for _,s in stages]!=expected
            or [s for s,*_ in budgets]!=expected or sum(p=='FAIL' for p,_ in stages)!=int(summary[0][1])):
        raise ValueError('Incomplete/duplicate/contradictory teacher score quality verdicts')
    rows=[]
    for (verdict,stage), (_,strict,calibrated,self_error,ratio) in zip(stages,budgets):
        if (strict not in ('0','1') or calibrated!='0' or float(self_error)!=0 or not math.isfinite(float(ratio)) or float(ratio)<0 or (verdict=='PASS' and (strict!='1' or float(ratio)>1))):
            raise ValueError('Invalid or relaxed generated-score budget')
        rows.append(dict(stage=stage,qualified_pass=verdict=='PASS',strict_absolute_pass=bool(int(strict)),
            calibrated=False,self_max_abs=float(self_error),max_ratio=float(ratio)))
    return dict(stages=rows,qualified_failures=int(summary[0][1]),
        scope='Unchanged qwen35_compare: min row cosine0.99999 and absolute0.002; generated scores receive no calibrated allowance.')


def frozen_prefix(reference, measured, old_case, old_decode, prior_prefix, decoded_prefix):
    count = len(old_case['reference']['generated_ids'])
    old = old_case['reference']
    fields = ('transformers', 'tokenizers', 'torch', 'dtype', 'attention', 'image_sha256', 'image_size', 'grid',
              'patches', 'prompt', 'rendered_prompt', 'eos_ids', 'cpu_controls')
    if (any(reference[k] != old[k] for k in fields)
            or reference['generated_ids'][:count] != old['generated_ids']
            or reference['selected_token_confidences'][:count] != old['selected_token_confidences']
            or decoded_prefix != old['decoded']
            or measured['native_ids'][:count] != old_decode['native_ids']
            or measured['metrics'][:77+count] != old_decode['metrics']
            or measured['teacher_forced'][:count] != old_decode['teacher_forced']
            or len(measured['native_confidences']) < count
            or f32(sum(map(f32, measured['native_confidences'][:count])) / count) != f32(old_decode['native_mean_confidence'])
            or measured['native_confidences'][:len(prior_prefix)] != prior_prefix):
        raise ValueError('Changed frozen receipt prefix, source/backend controls, scores or confidence')
    return dict(tokens=count, original_ids_text_confidence_exact=True, original_cpu_controls_exact=True,
        native_ids_scores_teacher_mean_exact=True, native_first32_probabilities_exact=True)


def run(repo, build, baseline, local, out, module, anchor_image, anchor_npz):
    from PIL import Image
    from transformers import AutoProcessor
    capture = module(repo / 'tools/dump_qwen35_ocr_page_reference.py', 'qwen35_receipt_reference')
    full = module(repo / 'tools/dump_qwen35_ocr_reference.py', 'qwen35_receipt_full_reference')
    results = repo / 'tests/results/qwen35-ocr'
    prior = json.loads((results / 'decoded-panel.json').read_text())
    material = json.loads((results / 'f16-materialization.json').read_text())
    old_case = next(c for c in prior['cases'] if c['name'] == 'receipt')
    old_decode = next(c['decode'] for c in prior['arms'][0]['cases'] if c['name'] == 'receipt')
    prefix = next(c['runs'][0]['confidences'] for c in material['benchmarks'] if c['name'] == 'receipt')
    root = out / 'receipt'; root.mkdir(exist_ok=True)
    target = out / 'receipt.json'
    report = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        model=prior['model'], revision=prior['revision'],
        scope='One frozen receipt derivative; image-only and explicit literal plain-text prompts, cap1024. No input resize/default or precision promotion.',
        production_changes='none', publication='none', source=old_case['source'], crop=old_case['crop'],
        explicit_resize=old_case['explicit_resize'], input_sha256=old_case['input_sha256'], gold=old_case['gold'],
        annotation_policy=old_case['annotation_policy'], cases=[],
        prior_evidence=dict(decoded_panel_sha256=file_sha(results/'decoded-panel.json'),
            materialization_sha256=file_sha(results/'f16-materialization.json'),
            archived_prefill_records=924, archived_qualified_failures=56,
            rerun_scope='Only four-arm receipt prefill subset (308 prior records); other616 prior records retained, not rerun here. Literal prompt adds77 new baseline records.'))
    env = dict(os.environ, CRISPEMBED_QWEN35_PROFILE='0', CRISPEMBED_QWEN35_PARALLEL_F16='1',
        CRISPEMBED_QWEN35_PRECISE_RMS='0', CRISPEMBED_QWEN35_IMATRIX='0', CRISPEMBED_QWEN35_VISION_PROBES='0',
        CRISPEMBED_QWEN35_VISION_PRECISE_NORM='0', CRISPEMBED_QWEN35_VISION_PRECISE_ATTN='0',
        CRISPEMBED_QWEN35_VISION_CASCADE_NORM='0')
    try:
        report['runtime_sources'] = {name: file_sha(repo/name) for name in material['sources']}
        if report['runtime_sources'] != material['sources']:
            raise ValueError('Changed frozen runtime arithmetic')
        report['weights'] = dict(bytes=baseline.stat().st_size, sha256=file_sha(baseline),
            tensor_audit=json.loads((out/'export-manifest.json').read_text())['records'])
        if report['weights'] != prior['weights']: raise ValueError('Changed frozen weights')
        source = repo / old_case['source']['local_file']
        if old_case['source']['license'] != 'CC0-1.0' or file_sha(source) != old_case['source']['sha256']:
            raise ValueError('Changed receipt source/license')
        image = Image.open(source).convert('RGB')
        if old_case['crop']: image = image.crop(old_case['crop'])
        if old_case['explicit_resize']: image = image.resize(old_case['explicit_resize'], resample=Image.Resampling.BICUBIC)
        image.save(root/'input.png')
        if file_sha(root/'input.png') != old_case['input_sha256']: raise ValueError('Changed frozen derivative')
        report['compact_anchor'] = capture.dump_reference(local, anchor_image, root/'anchor', max_new_tokens=12, anchor=anchor_npz)
        full.dump_reference(local, anchor_image, root/'prompted-anchor.npz', prompt=LITERAL_PROMPT)
        report['prompted_compact_anchor'] = capture.dump_reference(local, anchor_image, root/'prompted-anchor',
            max_new_tokens=12, prompt=LITERAL_PROMPT, anchor=root/'prompted-anchor.npz')
        gc.collect()
        # Freeze BOTH independent original outputs before any receipt native run.
        for name, prompt in [('upstream', ''), ('literal', LITERAL_PROMPT)]:
            folder = root/name; folder.mkdir(exist_ok=True)
            case = dict(name=name, prompt=prompt, max_new_tokens=1024, prefill=[])
            report['cases'].append(case)
            case['reference'] = capture.dump_reference(local, root/'input.png', folder, max_new_tokens=1024,
                prompt=prompt, long_document=True, prefill_controls=True, vision_probes=True)
            case['reference_vs_gold'] = errors(old_case['gold']['text'], case['reference']['decoded'])
            case['reference_counts'] = dict(characters=len(case['reference']['decoded']),
                lines=len(case['reference']['decoded'].splitlines()), tokens=len(case['reference']['generated_ids']))
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
            gc.collect()
        processor = AutoProcessor.from_pretrained(local, local_files_only=True, trust_remote_code=False)
        for case in report['cases']:
            folder = root/case['name']
            command = [str(build/'test-qwen35-quant'), str(baseline), str(root/'input.png'),
                str(folder/'official-delta.gguf'), case['prompt'], str(case['max_new_tokens'])]
            for name, norm, attention, cascade in (RECIPES if case['name']=='upstream' else RECIPES[:1]):
                arm_env = dict(env, CRISPEMBED_QWEN35_VISION_PRECISE_NORM=str(int(norm)),
                    CRISPEMBED_QWEN35_VISION_PRECISE_ATTN=str(int(attention)),
                    CRISPEMBED_QWEN35_VISION_CASCADE_NORM=str(int(cascade)))
                p = subprocess.run(command+['--prefill-only'], capture_output=True, text=True, env=arm_env)
                log = p.stdout+p.stderr; (folder/(name+'-prefill.log')).write_text(log); p.check_returncode()
                entry = dict(name=name, measurements=prefill_records(log))
                case['prefill'].append(entry)
                if case['name']=='upstream':
                    old_arm = next(a for a in prior['arms'] if a['name']==name)
                    old = next(c['prefill'] for c in old_arm['cases'] if c['name']=='receipt')
                    entry['prior_exact'] = entry['measurements']==old
                    if not entry['prior_exact']: raise ValueError('Changed frozen receipt prefill: '+name)
                target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
            p = subprocess.run(command+['--full-quality'], capture_output=True, text=True, env=env)
            log = p.stdout+p.stderr; (folder/'baseline-decode.log').write_text(log); p.check_returncode()
            case['decode'] = add_confidences(log, validate(parse(log, case['reference']), case['reference'], 1024), case['reference'])
            case['teacher_quality'] = teacher_quality(log, len(case['reference']['generated_ids']))
            case['decode']['native_vs_gold'] = errors(old_case['gold']['text'], case['decode']['native_text'])
            if case['name']=='upstream':
                text = processor.batch_decode([case['reference']['generated_ids'][:384]], skip_special_tokens=True)[0]
                case['frozen_prefix'] = frozen_prefix(case['reference'], case['decode'], old_case, old_decode, prefix, text)
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
            print(case['name'], 'tokens',len(case['decode']['native_ids']),'complete',case['decode']['complete_document'],
                'IDs/text',case['decode']['ids_exact'],case['decode']['text_exact'],'gold',case['decode']['native_vs_gold'],flush=True)
        report['incomplete_cases'] = sum(not c['decode']['complete_document'] for c in report['cases'])
        report['output_failures'] = sum(not c['decode']['ids_exact'] or not c['decode']['text_exact'] or not c['decode']['same_stop'] or
            c['decode']['first_teacher_argmax_difference'] is not None for c in report['cases'])
        report['qualified_prefill_failures'] = sum(len(a['measurements']['qualified_failed_stages']) for c in report['cases'] for a in c['prefill'])
        report['teacher_score_failures'] = sum(c['teacher_quality']['qualified_failures'] for c in report['cases'])
        report['result'] = 'completed receipt cap/prompt audit; numerical failures remain separate, no default/quality promotion'
        if report['incomplete_cases'] or report['output_failures'] or report['qualified_prefill_failures'] or report['teacher_score_failures']:
            raise AssertionError('Receipt audit retains incomplete/output/prefill counts: '+str((report['incomplete_cases'],report['output_failures'],report['qualified_prefill_failures'],report['teacher_score_failures'])))
    finally:
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
