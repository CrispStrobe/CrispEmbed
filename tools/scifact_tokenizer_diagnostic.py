#!/usr/bin/env python3
"""Strict source-only controls for the retained SciFact tokenizer diagnosis."""
import hashlib
import json
from pathlib import Path
import struct
import unicodedata

BASE='e35cd1a0ba3c2dcac1fdfbded739f5663d0b8902'
RUN=40375626336
PREP_RUN=40372423343
PREP={'id':12673961367,'bytes':17382270,'sha256':'09cdc1503b8f0fa6cba194d6941c7b65c69151a4183eabe88747596a21d19ce7'}
PREP_RECEIPT='b2a10ed5dd9f164534129a15a507784679d655c3f05f1a891c0fbddd441a7ce9'
FREEZE={'id':12693593726,'bytes':172964,'sha256':'4db657c428b61e205b3752ab5cfaafd5efaeb652acda76a96f89f627fe9ec323'}
FREEZE_REPORT='04a881169987f937f44e29ca6c604dfd324b72bd4bfd6b4ea51b21887f79d696'
DATA_FREEZE='8775b0a4caea0b1677e0357cffc139cf9d5ae0c85aa711168e5c59b4a3121f1f'
TOKENIZER_SHA='cec6e1c25c319d4de7d15e94cc69e95c68d75fee2e20c86c1bdac23b2db4d624'
REVISION='367d5343b5a0164d57b810cedf5de4b8983d5984'
QUERY='Instruct: Given a question, retrieve passages that answer the question\nQuery: '
NEGATIVES={
 0:{'id':12697173830,'bytes':14871,'sha256':'39f9b45541fecc0e1dbba10334e79f544546d8c022a3b23d18d648bb8574569f','row':'d:21676556','job':121891185945},
 1:{'id':12696503771,'bytes':14846,'sha256':'4e7b1d43cf25a0a2e5d09ca4f19b27c2827b70e8ca57ab75a98f0e005e86d89c','row':'d:104143831','job':121891185906}}
RUNTIME_PATHS=frozenset(('LICENSE','docs/ultrabert-scifact-notices.md','src/tokenizer.h','src/tokenizer_bpe.cpp','src/core/bpe.h','src/core/unicode_categ.h','src/core/unicode_class.h','src/core/json.h','src/core/clean_exit.h','src/core/env_gate.h','src/ultrabert_profile.h'))
DIAGNOSTIC_PATHS=frozenset(('tools/scifact_tokenizer_diagnostic.py','tools/ci-heavy/scifact_tokenizer_diagnostic.py','tools/ultrabert_contract.py','tests/test_scifact_tokenizer_diagnostic.cpp','tests/test_scifact_tokenizer_diagnostic.py','.github/workflows/scifact-tokenizer-diagnostic.yml'))
ORIGINAL_REPORT={'id':12679472596,'bytes':33253,'sha256':'8a63808e6009a6bd450fc756b32fe55d80aa89b1df0614b969966999b3a21a08'}
ORIGINAL_ENV_SHA='1df5bb659dfae5c09848e788d68cf089b2f0ccbc764d14095edd3eacb143831e'
TOKENIZER_WHEEL='https://files.pythonhosted.org/packages/3a/40/6706b82693715581457c6d5423eaa7faae576bb0526c5738a57085eb4449/tokenizers-0.23.3-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
TOKENIZER_WHEEL_SHA='376851d22bcf9d650a5c3090bb83e6cf9e895fbf0595369fa4cd43c1f69b5f87'


def digest(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def check_jobs(run,jobs):
    if (run['id']!=RUN or run['head_sha']!=BASE or run['path']!='.github/workflows/ovisocr2-metadata.yml' or
            run['status']!='completed' or run['conclusion']!='cancelled' or run['event']!='workflow_dispatch' or jobs['total_count']>100):
        raise ValueError('Retained cancelled parent identity differs')
    def one(name,conclusion,zero=False):
        rows=[r for r in jobs['jobs'] if r['name'].split(' / ')[-1]==name]
        if len(rows)!=1 or rows[0]['status']!='completed' or rows[0]['conclusion']!=conclusion or zero and rows[0]['steps']:
            raise ValueError('Retained job state differs: '+name)
        r=rows[0]
        return {'id':r['id'],'name':r['name'],'conclusion':r['conclusion'],'steps':len(r['steps'])}
    result={'original':[one('original-shard-'+str(i),'success') for i in range(8)],
            'freeze':one('freeze-original','success'),
            'native_failures':[one('native-shard-'+str(i),'failure') for i in (0,1)],
            'native_cancelled_before_steps':[one('native-shard-'+str(i),'cancelled',True) for i in range(2,8)]}
    if [r['id'] for r in result['native_failures']]!=[NEGATIVES[i]['job'] for i in (0,1)]:
        raise ValueError('Executed negative job identities differ')
    return result


def byte_decoder():
    printable=list(range(33,127))+list(range(161,173))+list(range(174,256))
    encoded={};extra=256
    for value in range(256):
        encoded[chr(value if value in printable else extra)]=value
        if value not in printable:extra+=1
    return encoded


def piece_bytes(piece):
    decoder=byte_decoder()
    try:return bytes(decoder[c] for c in piece)
    except KeyError as error:raise ValueError('Vocabulary piece outside ByteLevel alphabet') from error


def describe_ids(ids,vocab,text):
    result=[];offset=0;data=text.encode()
    for index,token in enumerate(ids):
        if type(token) is not int or not 0<=token<len(vocab):raise ValueError('Native token range differs')
        piece=vocab[token]
        if index==0 and token==0:
            result.append({'id':token,'piece':piece,'special':True,'derived_byte_span':[0,0]});continue
        raw=piece_bytes(piece)
        if data[offset:offset+len(raw)]!=raw:raise ValueError('Actual ID pieces do not reconstruct literal input prefix')
        result.append({'id':token,'piece':piece,'special':False,'raw_hex':raw.hex(),
            'decoded_utf8_backslashreplace':raw.decode('utf-8',errors='backslashreplace'),
            'derived_byte_span':[offset,offset+len(raw)]})
        offset+=len(raw)
    if offset!=len(data):raise ValueError('Two-row full token pieces do not cover literal input')
    return result


def codepoints(text):
    result=[];byte=0
    for index,c in enumerate(text):
        width=len(c.encode());result.append({'character_index':index,'byte_span':[byte,byte+width],
            'codepoint':f'U+{ord(c):04X}','character':c,'unicode_name':unicodedata.name(c,'UNASSIGNED'),
            'python_category':unicodedata.category(c)});byte+=width
    return result


def packet(vocab,merges,rows):
    result=bytearray(b'SCITOK1\0')
    def number(value):result.extend(struct.pack('<I',value))
    def string(value):
        data=value.encode();number(len(data));result.extend(data)
    for values in (vocab,merges):
        number(len(values))
        for value in values:string(value)
    number(len(rows))
    for row in rows:
        string(row['id']);string(row['input']);number(int(row['id'] in {n['row'] for n in NEGATIVES.values()}))
    return bytes(result)


def expose_pretokens(text):
    anchor='class BPETokenizer {\npublic:\n'
    if text.count(anchor)!=1:raise ValueError('Frozen tokenizer header anchor differs')
    return text.replace(anchor,anchor+'    // Test executable only: expose the unchanged const split for observation.\n'
        '    std::vector<std::string> diagnostic_gpt2_pretokens(const std::string & text) const { return gpt2_pretokenize(text); }\n')


def validate_manifest(spec):
    if (spec.get('schema')!='crispembed-scifact-tokenizer-diagnostic-source-v1' or
            spec.get('runtime_source')!=BASE or set(spec.get('runtime_sources',{}))!=RUNTIME_PATHS or
            set(spec.get('diagnostic_sources',{}))!=DIAGNOSTIC_PATHS):
        raise ValueError('Exact tokenizer source closure differs')
    for mapping in ('runtime_sources','diagnostic_sources'):
        if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
               for v in spec[mapping].values()):raise ValueError('Invalid source SHA')


def pretoken_spans(raw_parts,text):
    data=text.encode();offset=0;result=[]
    for raw in raw_parts:
        if not raw or data[offset:offset+len(raw)]!=raw:raise ValueError('Observed pretoken bytes differ from literal input')
        result.append({'derived_byte_span':[offset,offset+len(raw)],'raw_hex':raw.hex()});offset+=len(raw)
    if offset!=len(data):raise ValueError('Observed pretokens do not cover literal input')
    return result
