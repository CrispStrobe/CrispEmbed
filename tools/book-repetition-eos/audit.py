#!/usr/bin/env python3
"""Artifact-only book repetition/EOS diagnostics. Never emits text or token IDs."""
import hashlib
import json
from pathlib import Path
import zipfile

VOCAB = 248320
EOS = (248044, 248046)
REFERENCES = {
    'source': ('heldout-result.json', '9b88195d5a8058dea4fab87472009d3826bc1fa3b5d0575aa2e33698316d5722', 'upstream-on', 'source'),
    'planner_on': ('planner-book-extended.json', 'd2ea232376c2fa4697613d615d33b8b10a4dc0d5ab987fdac7d1ed3d99561760', 'planner-on', 'greedy'),
    'planner_off': ('planner-heldout.json', 'f231ab48c9bff411a1f9c96cfb79eda084f2f0b7cee46f9ea747ccc734d5f8e6', 'baseline-old', 'greedy'),
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def repetition(ids):
    """Fixed token-structure screens; matches are diagnostic, not quality labels."""
    report = {'tokens': len(ids), 'duplicate_ngrams': {}}
    for width in (4, 8, 16):
        seen, duplicates, first = set(), 0, None
        for offset in range(len(ids) - width + 1):
            gram = tuple(ids[offset:offset + width])
            if gram in seen:
                duplicates += 1
                if first is None:
                    first = offset
            seen.add(gram)
        report['duplicate_ngrams'][str(width)] = dict(repeated_windows=duplicates, first_repeated_start=first)
    longest, first = None, None
    # Require >=3 cycles and >=12 tokens, periods1..64; no phrase/case-specific rules.
    for start in range(len(ids)):
        for period in range(1, min(64, (len(ids) - start) // 3) + 1):
            end = start + period
            while end < len(ids) and ids[end] == ids[start + (end - start) % period]:
                end += 1
            span = end - start
            if span < max(12, 3 * period):
                continue
            row = dict(start=start, end_exclusive=end, period=period, span=span, whole_cycles=span // period)
            if first is None or (start, period) < (first['start'], first['period']):
                first = row
            if longest is None or (span, -start, -period) > (longest['span'], -longest['start'], -longest['period']):
                longest = row
    report.update(first_periodic_run=first, longest_periodic_run=longest)
    return report


def common_prefix(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


def score_stats(values):
    import numpy as np
    if values.shape != (VOCAB,) or not np.isfinite(values).all():
        raise ValueError('finite complete vocabulary required')
    x = values.astype('float64')
    x -= x.max()
    weights = np.exp(x)
    denominator = weights.sum()
    probabilities = weights / denominator
    order = np.partition(x, -2)[-2:]
    result = dict(top_two_gap=float(order.max() - order.min()), entropy=float(-np.sum(probabilities * x) + np.log(denominator)))
    # Report EOS slots by semantic position only; no vocabulary IDs are exposed.
    result['eos'] = [dict(rank_strict=int(np.count_nonzero(values > values[eos])) + 1,
                          ties=int(np.count_nonzero(values == values[eos])),
                          probability=float(probabilities[eos]), gap_from_top=float(-x[eos])) for eos in EOS]
    result['either_eos_probability'] = sum(row['probability'] for row in result['eos'])
    return result


def main():
    import numpy as np
    root = Path('/kaggle/input')
    if not root.is_dir():
        raise RuntimeError('artifact-only worker requires mounted Kaggle artifacts')
    output = Path('/kaggle/working/book-repetition-eos.json')
    report = dict(schema='lighton4b-book-repetition-eos/v1', numpy_version=np.__version__, model_forward_executed=False,
                  text_or_token_ids_emitted=False, qualified=False, status='failed', arms={})
    ids_by_role, rows_by_role = {}, {}
    try:
        for role, (name, digest, folder, stem) in REFERENCES.items():
            matches = [path for path in root.rglob(name) if path.is_file() and path.stat().st_size < 4 << 20 and sha(path) == digest]
            if len(matches) != 1:
                raise ValueError('one exact retained report required: ' + role)
            path = matches[0]
            receipt = json.loads(path.read_text())
            arm = receipt['arm'] if role == 'planner_on' else receipt['arms']['upstream-on' if role == 'source' else 'baseline-old']
            ids = arm['ids']
            raw = receipt['source_raw_rows'] if role == 'source' else arm['raw_rows']
            if len(ids) != len(raw['rows']) or not ids:
                raise ValueError('complete metadata rows required')
            archive = path.parent / folder / raw['archive']['file']
            if archive.stat().st_size != raw['archive']['bytes'] or sha(archive) != raw['archive']['sha256']:
                raise ValueError('authenticated archive required')
            numerical = []
            with zipfile.ZipFile(archive) as sealed:
                for i, (token, row) in enumerate(zip(ids, raw['rows'])):
                    member = ('source.logits.%d.f32' if stem == 'source' else 'greedy.%d.logits.f32') % i
                    if row['position'] != i or row['argmax'] != token:
                        raise ValueError('sealed row/ID alignment')
                    value = sealed.read(member)
                    if len(value) != VOCAB * 4 or hashlib.sha256(value).hexdigest() != row['sha256']:
                        raise ValueError('complete row hash mismatch')
                    array = np.frombuffer(value, dtype='<f4')
                    if int(array.argmax()) != token:
                        raise ValueError('actual greedy row mismatch')
                    numerical.append(dict(position=i, **score_stats(array)))
            ids_by_role[role] = ids
            rows_by_role[role] = numerical
            report['arms'][role] = dict(receipt_sha256=digest, archive_sha256=raw['archive']['sha256'],
                                       stop_is_eos=ids[-1] in EOS, repetition=repetition(ids), rows=numerical)
        report['prefix_comparison_scope'] = {}
        for role in ['planner_on', 'planner_off']:
            count = common_prefix(ids_by_role['source'], ids_by_role[role])
            report['prefix_comparison_scope'][role] = dict(shared_output_prefix_tokens=count,
                last_same_input_score_row=min(count, len(ids_by_role['source']) - 1, len(ids_by_role[role]) - 1),
                later_rows='independent trajectory diagnostics only; differing consumed prefixes exclude parity')
        report['status'] = 'complete-artifact-only'
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
