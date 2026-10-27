"""Hosted Rust tokenizer oracle + exact pinned ST prefix registration, no Torch."""
import ast
import hashlib
import json
import math
import types
from contract import ROOT,REV,REPORT,TOKENIZER,ST_SOURCE,require,sha,dump


def added_token_state(token):
 state=token.__getstate__()
 require(type(state)is dict and set(state)=={'content','single_word','lstrip','rstrip','normalized','special'},'Rust0.22.2 AddedToken state schema differs')
 require(type(state['content'])is str and state['content'] and all(type(state[k])is bool for k in ('single_word','lstrip','rstrip','normalized','special')),'Rust0.22.2 AddedToken state types differ')
 return dict(state)


def register(backend,source,config):
 require(sha(source.read_bytes())==ST_SOURCE,'ST source changed')
 tree=ast.parse(source.read_text())
 # The method lives on MultiVectorEncoder, not Transformer. Source selected by caller.
 method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_register_prefix_tokens')
 require(not method.decorator_list,'prefix registration decorator changed')
 import tokenizers
 namespace={'AddedToken':tokenizers.AddedToken}
 exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+[method],type_ignores=[]),'pinned-st-prefix-registration','exec'),namespace)
 actions=[]
 class Adapter:
  @property
  def added_tokens_encoder(self):return {v.content:k for k,v in backend.get_added_tokens_decoder().items()}
  def get_vocab(self):return backend.get_vocab()
  def tokenize(self,text):return backend.encode(text,add_special_tokens=False).tokens
  def add_special_tokens(self,value):
   actions.extend([added_token_state(x) for x in value['additional_special_tokens']]);backend.add_special_tokens(value['additional_special_tokens'])
 owner=types.SimpleNamespace(tokenizer=Adapter())
 namespace['_register_prefix_tokens'](owner,{'query':config['query_prefix'],'document':config['document_prefix']})
 return actions


def tokens(backend,text,role,skip):
 require(role in ('query','document'),'explicit role required')
 maximum=32 if role=='query' else 512
 backend.enable_truncation(maximum,strategy='longest_first',direction='right')
 if role=='query':backend.enable_padding(direction='right',pad_id=4,pad_type_id=0,pad_token='[MASK]',length=32)
 else:backend.no_padding()
 encoded=backend.encode(('[Q] ' if role=='query' else '[D] ')+text,add_special_tokens=True)
 ids,mask=encoded.ids,encoded.attention_mask
 expansion=[int(not m) for m in mask] if role=='query' else []
 return {'ids':ids,'supplied_mask':mask,'effective_mask':[1]*len(ids) if role=='query' else mask,
         'scoring_mask':[1]*len(ids) if role=='query' else [int(bool(m) and i not in skip) for i,m in zip(ids,mask)],
         'expansion_positions':expansion}, {'tokens':encoded.tokens,'offset_unit':'original Unicode character offsets of prefixed text','offsets':encoded.offsets}


def mapped(text,mapping):
 value=text.replace('``','"').replace("''",'"')
 out=''.join(mapping.get(ord(c),c) for c in value)
 # Scalar normalization already contains the second NFKD/strip/lower pass.
 import re
 return re.sub(' {2,}',' ',out)


def component(raw,effective,config,mapping,skip):
 require(raw['model']['type']=='Unigram' and raw['model']['unk_id']==1 and raw['model']['byte_fallback']is False,'Unigram flags differ')
 require(len(raw['model']['vocab'])==30000,'base vocabulary count differs')
 require(effective['normalizer']==raw['normalizer'] and effective['pre_tokenizer']==raw['pre_tokenizer'] and effective['post_processor']==raw['post_processor'],'effective normalizer/template changed')
 vocab=[x[0] for x in raw['model']['vocab']]+['[Q] ','[D] '];scores=[x[1] for x in effective['model']['vocab']]
 require([x[0] for x in effective['model']['vocab']]==vocab[:30000],'Rust vocabulary roundtrip changed pieces')
 require(len(set(vocab))==30002 and all(isinstance(s,(float,int)) and math.isfinite(s) for s in scores),'invalid exact F64 scores/vocab')
 added=effective['added_tokens'];require([x['id'] for x in added]==[0,1,2,3,4,30000,30001],'effective added ID order differs')
 flags=[]
 for x in added:
  require(not x['single_word'] and not x['rstrip'],'unsupported added-token flag')
  flags.append(int(x['single_word'])+2*int(x['lstrip'])+4*int(x['rstrip'])+8*int(x['normalized'])+16*int(x['special']))
 keys=sorted(mapping)
 return {'general.architecture':'moedern_unigram_tokenizer','general.license':'apache-2.0',
  'moedern.tokenizer.format':'crispembed-moedern-unigram-v1','moedern.tokenizer.revision':REV,
  'moedern.tokenizer.source_sha256':TOKENIZER,'moedern.tokenizer.original_report_sha256':REPORT,
  'moedern.tokenizer.normalizer':'rust0.22.2-two-nfkd-stripaccents-lowercase-passes',
  'moedern.tokenizer.pretokenizer':'WhitespaceSplit+Metaspace-always-split',
  'moedern.tokenizer.model':'Unigram-f64-unk1-fuse-unknown-no-byte-fallback',
  'tokenizer.ggml.tokens':vocab,'moedern.tokenizer.scores_f64':('f64',scores),
  'moedern.tokenizer.added_ids':[x['id'] for x in added],'moedern.tokenizer.added_flags':flags,
  'moedern.tokenizer.added_content':[x['content'] for x in added],
  'moedern.tokenizer.normalizer_codepoints':keys,'moedern.tokenizer.normalizer_values':[mapping[k] for k in keys],
  'moedern.tokenizer.skip_ids':skip,'moedern.tokenizer.raw_descriptor_json':dump({k:v for k,v in raw.items() if k!='model'}),
  'moedern.tokenizer.effective_descriptor_json':dump({k:v for k,v in effective.items() if k!='model'}),
  'moedern.tokenizer.role_config_json':dump(config)}
