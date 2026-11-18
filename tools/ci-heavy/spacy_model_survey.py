#!/usr/bin/env python3
"""Which spaCy pipeline sizes/languages are worth porting? (hosted, CPU only)

For German, English, French (official spaCy 3.8 pipelines) and Latin
(LatinCy, MIT), runs the small CNN pipeline and the transformer pipeline over
the human-annotated UD r2.14 test set with *gold tokenization* (a Doc built
from the gold words), so every model's output aligns token by token with the
gold annotation and with the other model:

- UPOS, UFeats and lemma accuracy against gold, and head/label attachment
  against gold (only meaningful where the model's dependency scheme is UD:
  French and Latin; German uses TIGER-derived labels, English ClearNLP);
- sm-vs-trf agreement per field (how much a trf port would change output);
- CPU throughput (4 hosted vCPUs; PyTorch default threads for trf).

Every wheel and treebank is hash-checked or recorded. No native code runs.
The numbers decide porting order; they are not a quality claim for any port.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

OUT = Path(os.environ['HEAVY_OUT'])
SCRATCH = Path(os.environ['HEAVY_SCRATCH']) / 'spacy-survey'
SCRATCH.mkdir(parents=True, exist_ok=True)
UD = {
    'de': ('UD_German-GSD', 'de_gsd-ud-test.conllu', 'f64867239949b842aab306d508b0f0e0c81c9edebfbb19629390adafc6d1c629'),
    'en': ('UD_English-EWT', 'en_ewt-ud-test.conllu', '5e0044aa9b786bfb81dbd5687400583a61635a65839e747b59b2ec907c8ab662'),
    'fr': ('UD_French-GSD', 'fr_gsd-ud-test.conllu', '158386a42ece1fca4fbcfa905b1a584d3e375415f4e0ffa7b3de20b1ea2d768a'),
    'la': ('UD_Latin-PROIEL', 'la_proiel-ud-test.conllu', '9d028d664480263b5de5c123f4a6f35600ca1b42baa3ecab25e8ba187ef9ee1d'),
}
OFFICIAL = 'https://github.com/explosion/spacy-models/releases/download/{0}-3.8.0/{0}-3.8.0-py3-none-any.whl'
LATINCY = 'https://huggingface.co/latincy/{0}/resolve/main/{0}-3.9.8-py3-none-any.whl'
MODELS = {
    'de': [('de_core_news_sm', OFFICIAL), ('de_core_news_md', OFFICIAL), ('de_dep_news_trf', OFFICIAL)],
    'en': [('en_core_web_sm', OFFICIAL), ('en_core_web_trf', OFFICIAL)],
    'fr': [('fr_core_news_sm', OFFICIAL), ('fr_dep_news_trf', OFFICIAL)],
    'la': [('la_core_web_sm', LATINCY), ('la_core_web_trf', LATINCY)],
}
UD_SCHEME = {'fr', 'la'}


def fetch(url, path, sha=None):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=300) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha is None or digest == sha, f'{path.name}: SHA-256 {digest} != {sha}'
    return {'url': url, 'sha256': digest, 'bytes': path.stat().st_size}


def read_conllu(path):
    sentences, words = [], []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            if words:
                sentences.append(words)
            words = []
            continue
        if line.startswith('#'):
            continue
        cols = line.split('\t')
        if '-' in cols[0] or '.' in cols[0]:
            continue  # multiword ranges / empty nodes: gold syntactic words only
        words.append({'form': cols[1], 'lemma': cols[2], 'upos': cols[3],
                      'feats': '' if cols[5] == '_' else cols[5], 'head': int(cols[6]), 'deprel': cols[7].split(':')[0],
                      'space': 'SpaceAfter=No' not in cols[9]})
    if words:
        sentences.append(words)
    return sentences


def main():
    import spacy
    from spacy.tokens import Doc
    report = {'schema': 'crispembed.spacy.model_survey.v1', 'spacy': spacy.__version__,
              'source': os.environ.get('GITHUB_SHA'), 'run': os.environ.get('GITHUB_RUN_ID'),
              'cpu_count': os.cpu_count(), 'languages': {}, 'wheels': {}, 'treebanks': {}}

    def save():
        (OUT / 'model-survey.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    for lang, (repo, name, sha) in UD.items():
        treebank = SCRATCH / name
        report['treebanks'][lang] = fetch(f'https://raw.githubusercontent.com/UniversalDependencies/{repo}/r2.14/{name}',
                                          treebank, sha)
        gold = read_conllu(treebank)
        result = {'treebank': repo, 'sentences': len(gold), 'tokens': sum(map(len, gold)), 'models': {}}
        report['languages'][lang] = result
        outputs = {}
        for model, pattern in MODELS[lang]:
            entry = {}
            result['models'][model] = entry
            try:
                wheel = SCRATCH / f'{model}.whl'
                report['wheels'][model] = fetch(pattern.format(model), wheel)
                # Install the wheel under its real file name (pip requires it).
                named = SCRATCH / pattern.format(model).rsplit('/', 1)[1]
                if not named.exists():
                    named.write_bytes(wheel.read_bytes())
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', str(named)], check=True)
                nlp = spacy.load(model)
                entry['pipeline'] = nlp.pipe_names
                docs = [Doc(nlp.vocab, words=[w['form'] for w in s], spaces=[w['space'] for w in s]) for s in gold]
                # Warm-up (lazy initialisation) on fresh docs, outside the timing.
                list(nlp.pipe([Doc(nlp.vocab, words=[w['form'] for w in s], spaces=[w['space'] for w in s])
                               for s in gold[:8]]))
                start = time.perf_counter()
                predicted = list(nlp.pipe(docs, batch_size=64))
                elapsed = time.perf_counter() - start
                entry['tokens_per_second'] = round(result['tokens'] / elapsed, 1)
                aligned = [d for d, s in zip(predicted, gold) if len(d) == len(s)]
                entry['misaligned_sentences'] = len(gold) - len(aligned)
                counts = dict.fromkeys(('upos', 'feats', 'lemma', 'uas', 'las', 'n'), 0)
                rows = []
                for doc, sent in zip(predicted, gold):
                    if len(doc) != len(sent):
                        rows.append(None)
                        continue
                    row = []
                    for token, g in zip(doc, sent):
                        head = 0 if token.head.i == token.i else token.head.i + 1
                        row.append((token.tag_, token.pos_, str(token.morph), token.lemma_, token.dep_, head))
                        counts['n'] += 1
                        counts['upos'] += token.pos_ == g['upos']
                        counts['feats'] += str(token.morph) == g['feats']
                        counts['lemma'] += token.lemma_ == g['lemma']
                        counts['uas'] += head == g['head']
                        counts['las'] += head == g['head'] and token.dep_.split(':')[0].lower() == g['deprel'].lower()
                    rows.append(row)
                outputs[model] = rows
                n = counts['n'] or 1
                entry['vs_gold'] = {k: round(counts[k] / n, 4) for k in ('upos', 'feats', 'lemma', 'uas', 'las')}
                entry['vs_gold']['scheme_note'] = ('UD dependency scheme: attachment comparable' if lang in UD_SCHEME
                                                   else 'non-UD dependency scheme: attachment vs UD gold not comparable')
                del nlp
            except Exception as exc:  # record and continue with the next model
                entry['failure'] = f'{type(exc).__name__}: {str(exc)[:600]}'
            save()
        models = [m for m, _ in MODELS[lang] if m in outputs]
        small = models[0] if models else None
        for other in models[1:]:
            agree = dict.fromkeys(('tag', 'pos', 'morph', 'lemma', 'dep', 'head', 'n'), 0)
            for a, b in zip(outputs[small], outputs[other]):
                if a is None or b is None:
                    continue
                for x, y in zip(a, b):
                    agree['n'] += 1
                    for i, key in enumerate(('tag', 'pos', 'morph', 'lemma', 'dep', 'head')):
                        agree[key] += x[i] == y[i]
            n = agree['n'] or 1
            result[f'agreement_{small}_vs_{other}'] = {k: round(agree[k] / n, 4)
                                                       for k in ('tag', 'pos', 'morph', 'lemma', 'dep', 'head')}
        save()
    lines = ['# spaCy model survey', '']
    for lang, result in report['languages'].items():
        lines.append(f"## {lang} ({result['treebank']}, {result['tokens']} gold tokens)")
        for model, entry in result['models'].items():
            text = entry.get('failure') or '{} | {} tok/s'.format(entry.get('vs_gold'), entry.get('tokens_per_second'))
            lines.append(f'- {model}: {text}')
        lines += [f'- {k}: {v}' for k, v in result.items() if k.startswith('agreement_')] + ['']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save()


if __name__ == '__main__':
    main()
