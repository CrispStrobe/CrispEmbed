"""Hosted-only preparation of the frozen OCR quality panel; downloads no weights."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_quality import POLICY, words

PANEL = 'tests/regression/ocr-quality/panel.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(repo, name):
    path = (repo / name).resolve()
    if not path.is_relative_to(repo.resolve()) or not path.is_file():
        raise ValueError('Missing/non-repository panel asset')
    return path


def validate_panel(repo, panel=None):
    panel = json.loads(local(repo, PANEL).read_text()) if panel is None else panel
    if panel['version'] != 1 or panel['policy'] != POLICY or len(panel['cases']) != 3:
        raise ValueError('Unsupported panel version/policy/count')
    previous = [json.loads(local(repo, name).read_text()) for name in panel['development_reports']]
    used_hashes = {c['source']['sha256'] for r in previous for c in r.get('cases', []) if 'source' in c}
    used_hashes |= {r['source']['sha256'] for r in previous if 'source' in r}
    used_text = {words(c['gold']['text']) for r in previous for c in r.get('cases', []) if c.get('gold')}
    used_text |= {words(r['gold']['text']) for r in previous if r.get('gold')}
    calibration = json.loads(local(repo, panel['calibration_report']).read_text())['native_importance']['examples']
    used_hashes |= {c['image_sha256'] for c in calibration}
    used_text |= {words(c['gold']) for c in calibration}
    identifiers, sources, texts = set(), set(), set()
    for c in panel['cases']:
        source, gold_path = local(repo, c['source']), local(repo, c['gold'])
        gold = json.loads(gold_path.read_text())
        if (c['id'] in identifiers or c['split'] != 'evaluation' or c['prompt'] != ''
                or c['max_new_tokens'] != 1024 or c['crop'] is not None
                or c['license'] not in ('MIT', 'CC0-1.0')
                or sha(source) != c['source_sha256'] or sha(gold_path) != c['gold_sha256']
                or c['source_sha256'] in used_hashes | sources
                or not gold['text'] or words(gold['text']) in used_text | texts):
            raise ValueError('Changed, duplicate, overlapping or invalid evaluation case')
        if any(not t or not t[0] or any(len(row) != len(t[0]) or any(not isinstance(cell, str) or words(cell) != cell for cell in row) for row in t) for t in (gold['tables'] or [])):
            raise ValueError('Invalid rectangular gold table')
        if isinstance(gold['provenance'], dict):
            provenance = gold['provenance']
            annotations = local(repo, provenance['annotations'])
            record = next(r for r in json.loads(annotations.read_text())['records'] if r['file'] == provenance['record'])
            if sha(annotations) != provenance['sha256'] or gold['text'] != record['text']:
                raise ValueError('Changed independent manual annotation')
        if c['resize'] is not None:
            size = c['resize']
            if (len(size) != 2 or any(type(v) is not int or v < 32 or v % 32 for v in size)
                    or size[0] * size[1] // 256 > 1024):
                raise ValueError('Invalid explicit derivative geometry')
        identifiers.add(c['id']); sources.add(c['source_sha256']); texts.add(words(gold['text']))
    return panel



def verify_prepared(previous, current):
    """Reject changed pixels/annotations before inference, including renderer drift."""
    keys = ('id', 'source_sha256', 'gold_sha256', 'original_size', 'size',
            'png_sha256', 'rgb_sha256', 'file', 'nominal_patches')
    for report in (previous, current):
        if (report['policy'] != POLICY or len(report['cases']) != 3
                or len({c['id'] for c in report['cases']}) != 3):
            raise ValueError('Incomplete/repeated prepared cases')
    if (previous['panel_sha256'] != current['panel_sha256']
            or previous['font'] != current['font']
            or previous['split_evidence_sha256'] != current['split_evidence_sha256']
            or [[c[k] for k in keys] for c in previous['cases']]
               != [[c[k] for k in keys] for c in current['cases']]):
        raise ValueError('Frozen evaluation pixels, annotations, font or split evidence changed')
    return True


def prepare(repo, out):
    import cairosvg
    import PIL
    from PIL import Image
    panel = validate_panel(repo)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        policy=POLICY, panel_sha256=sha(repo / PANEL), scorer_sha256=sha(repo / 'tools/ocr_quality.py'),
        pillow=PIL.__version__, cairosvg=cairosvg.__version__, cases=[],
        inference='Not run; no weights, native outputs, references or quality scores. Freeze before inference.',
        geometry='Derivative dimensions only; actual pinned AutoProcessor grid must be checked in model audit.',
        split='Source byte hashes and normalized transcripts disjoint from frozen Qwen development/calibration reports; not proof of absence from checkpoint pretraining or other model evaluations.')
    report['split_evidence_sha256'] = {name: sha(local(repo, name))
        for name in [*panel['development_reports'], panel['calibration_report']]}
    font = Path(subprocess.check_output(['fc-match', '-f', '%{file}', 'DejaVu Sans'], text=True).strip())
    if font.name != 'DejaVuSans.ttf':
        raise ValueError('Expected DejaVu Sans renderer font')
    report['font'] = dict(name=font.name, sha256=sha(font))
    import cairocffi
    report['cairo'] = cairocffi.cairo_version_string()
    for c in panel['cases']:
        source = local(repo, c['source'])
        path = out / (c['id'] + '.png')
        original_size = None
        if source.suffix == '.svg':
            cairosvg.svg2png(url=str(source), write_to=str(path))
            image = Image.open(path).convert('RGB')
        else:
            image = Image.open(source).convert('RGB')
            original_size = list(image.size)
            if c['id'] == 'historical-receipt':
                # Annotation refers to the existing lossless PNG of this original TIFF.
                png = Image.open(repo / 'tests/regression/images/cc0/receipt_historical.png').convert('RGB')
                if image.size != png.size or image.tobytes() != png.tobytes():
                    raise ValueError('Historical TIFF/annotated PNG pixels differ')
            image = image.resize(c['resize'], resample=Image.Resampling.BICUBIC)
        image.save(path)
        if any(v % 32 for v in image.size) or image.width * image.height // 256 > 1024:
            raise ValueError('Prepared geometry exceeds planned patch bound')
        report['cases'].append(dict(id=c['id'], source_sha256=c['source_sha256'], gold_sha256=c['gold_sha256'],
            original_size=original_size, size=list(image.size), png_sha256=sha(path),
            rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(), file=path.name,
            nominal_patches=image.width * image.height // 256))
    frozen = repo / 'tests/results/ocr-quality-prepared.json'
    report['frozen_derivatives_exact'] = (verify_prepared(json.loads(frozen.read_text()), report)
        if frozen.exists() else None)
    (out / 'prepared.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    prepare(Path(__file__).resolve().parents[2], Path(os.environ['HEAVY_OUT']))
