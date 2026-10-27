"""Versioned tokenizer-only A/B contracts; stdlib only."""
import json
from pathlib import Path
from scifact_tokenizer_diagnostic import digest,canonical,expose_pretokens
ROOT=Path(__file__).resolve().parents[1]
BASELINE_PATH=ROOT/'docs/ultrabert-unicode-pretoken-baseline.json'
CASES_PATH=ROOT/'docs/ultrabert-unicode-pretoken-cases.json'
CONTRACT_PATH=ROOT/'docs/ultrabert-unicode-pretoken-contract.json'
RUNTIME=('src/tokenizer.h','src/tokenizer_bpe.cpp','src/ultrabert_profile.h','src/ultrabert_tokenizer_mode.h')
SOURCES=(*RUNTIME,'tests/test_ultrabert_unicode_pretoken.cpp','tools/ultrabert_unicode_pretoken.py',
    'tools/ci-heavy/ultrabert_unicode_pretoken.py','tests/test_ultrabert_unicode_pretoken.py',
    '.github/workflows/ultrabert-unicode-pretoken.yml','docs/ultrabert-unicode-pretoken-baseline.json',
    'docs/ultrabert-unicode-pretoken-cases.json')
RAW={'id':12701842149,'bytes':25792830,'sha256':'d17d581dde9edfdac2e5e6b74d01019d669626e0defc2e50e671755fc728ec34'}


def observed_header(text):
    # Accessors exist only in a test-owned copied header, never production code.
    text=expose_pretokens(text)
    anchor='    // Test executable only: expose the unchanged const split for observation.\n'
    return text.replace(anchor,'    std::vector<std::string> diagnostic_clip_pretokens(const std::string & text) const { return clip_pretokenize(text); }\n'+anchor)


def legacy_projection(row):
    keys=('row','id','observer_bit_exact','ids','untruncated_ids','actual_native_pretokens')
    return {k:row[k] for k in keys if k in row}


class ParityError(ValueError):
    def __init__(self,message,details):
        super().__init__(message);self.details=details


def compare_rows(actual,frozen,legacy,baseline,mode):
    if len(actual)!=5483 or len(frozen)!=5483 or len(legacy)!=5483:raise ValueError('Complete5483 rows required')
    clipped=[];full=[]
    for index,(row,token,old) in enumerate(zip(actual,frozen,legacy)):
        if (row['row']!=index or row['id']!=token['id'] or old['id']!=token['id'] or
                row['observer_bit_exact'] is not True or row['ids']!=row['untruncated_ids'][:512] or
                row['attention_mask']!=[1]*len(row['ids'])):raise ValueError('Actual ordered ID/mask/observer contract differs: '+token['id'])
        if mode!='1' and legacy_projection(row)!=old:raise ValueError('Gate0 differs from actual frozen native output: '+token['id'])
        exact=row['ids']==token['ids'] and row['attention_mask']==token['attention_mask']
        complete=(digest(canonical(row['untruncated_ids']))==token['untruncated_ids_sha256'] and len(row['untruncated_ids'])==token['untruncated_tokens'])
        if not exact:clipped.append(row['id'])
        if not complete:full.append(row['id'])
    details=dict(rows=5483,clipped_mismatches=clipped,full_mismatches=full,
        original_clipped_masks_exact=not clipped,original_full_lengths_hashes_exact=not full,
        historical_native_exact=mode!='1',observer_bit_exact=True)
    if mode=='1':
        if clipped or full:raise ParityError('Gate1 complete original token parity rejected',details)
    elif clipped!=baseline['clipped_ids'] or full!=baseline['full_ids']:
        raise ParityError('Gate0 actual24/25 mismatch sets differ',details)
    return details
