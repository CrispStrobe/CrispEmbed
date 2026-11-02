#!/usr/bin/env python3
"""Layer0 attention op-by-op replay: decisions, admission reuse, source anchors. No build/model."""
import copy,hashlib,importlib.util,json,pathlib,struct,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
def load(n,p):
 s=importlib.util.spec_from_file_location(n,ROOT/p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
A=load('l0a','tools/wasm-layer0-attention/analysis.py')
T=load('ffn_tests','tests/test_wasm_layer3_ffn_matmul_v1.py')
GG=ROOT/'ggml/src'

def f32(v):return struct.pack('<%df'%len(v),*v)
def run(shift=None,op_shift=()):
 base=[float(i%13)/9+.5 for i in range(384*12)]
 return {op:f32([x+(2**-18 if (op in op_shift and i==0)else 0)for i,x in enumerate(base)])for op in A.ORDER}
def meta(buf='CPU'):return dict(weight_buffer=buf,ops=list(A.ORDER),weights={'q_w':dict(type='iq4_nl')})

class Selftest(unittest.TestCase):
 def runs(self):return {(s,r):run()for s in['native','wasm']for r in['chain1','chain2','ops_own']}
 def test_pass_and_rejections(self):
  m={(s,r):meta()for s in['native','wasm']for r in['chain1','chain2','ops_own']}
  self.assertTrue(A.selftest(m,self.runs())['passed'])
  r=self.runs();r[('wasm','chain2')]=run(op_shift={'flash'});self.assertFalse(A.selftest(m,r)['passed'])   # nondeterminism
  r=self.runs();r[('native','ops_own')]=run(op_shift={'q_raw'});self.assertFalse(A.selftest(m,r)['passed'])  # isolation broken
  m2=copy.deepcopy(m);m2[('native','chain1')]['weight_buffer']='CPU_REPACK';self.assertFalse(A.selftest(m2,self.runs())['passed'])

class Replay(unittest.TestCase):
 def setup(self,diff_cross=(),diff_chain=()):
  runs={('native','chain'):run(),('native','ops_own'):run(),('native','ops_cross'):run(op_shift=set(diff_chain)-set(diff_cross)),
        ('wasm','chain'):run(op_shift=set(diff_chain)),('wasm','ops_own'):run(op_shift=set(diff_chain)),('wasm','ops_cross'):run(op_shift=set(diff_cross))}
  m={(s,r):meta()for s in['native','wasm']for r in['chain','ops_own','ops_cross']}
  cap={('native','norm00_output'):b'x',('wasm0','norm00_output'):b'x',('native','norm01_input'):runs[('native','chain')]['residual'],('wasm0','norm01_input'):runs[('wasm','chain')]['residual']}
  return m,runs,cap
 def test_identifies_first_and_kinds(self):
  m,r,c=self.setup(diff_cross=('q_raw','k_raw','flash'),diff_chain=('q_raw','k_raw','v_raw','q','k','v','flash','o_raw','o','residual'))
  out=A.replay(m,r,c);self.assertTrue(out['controls_passed'])
  self.assertEqual(out['first_same_input_differing_op'],'q_raw');self.assertEqual(out['first_propagated_differing_op'],'q_raw')
  self.assertEqual(out['differing_op_kinds'],['flash_attn_ext','mul_mat']);self.assertFalse(out['single_deterministic_op_kind'])
  m,r,c=self.setup(diff_cross=('o_raw',),diff_chain=('o_raw','o','residual'));out=A.replay(m,r,c)
  self.assertTrue(out['single_deterministic_op_kind']);self.assertEqual(out['first_same_input_differing_op'],'o_raw')
 def test_controls(self):
  m,r,c=self.setup();c[('native','norm01_input')]=f32([0.0]*(384*12));self.assertFalse(A.replay(m,r,c)['controls_passed'])
  m,r,c=self.setup();c[('wasm0','norm00_output')]=b'y';self.assertFalse(A.replay(m,r,c)['controls_passed'])
  m,r,c=self.setup();r[('wasm','ops_own')]=run(op_shift={'flash'});self.assertFalse(A.replay(m,r,c)['controls_passed'])

class Admission(unittest.TestCase):
 def test_reuses_production_admission(self):
  for kind in['native','wasm']:
   rows=T.static_from(kind)
   for r in rows:
    if r['file'].endswith('probe.cpp'):r['file']=r['file'].replace('wasm-layer3-ffn-matmul','wasm-layer0-attention')
   A.admit(kind,rows,T.production(kind),'tests/wasm-layer0-attention/probe.cpp')

class Anchors(unittest.TestCase):
 def test_production_layer0_and_flash_path(self):
  c=(ROOT/'src/crispembed.cpp').read_text()
  for s in['Q = ggml_mul_mat(gctx, L.q_w, cur);','if (L.q_b) Q = ggml_add(gctx, Q, L.q_b);','Q = ggml_reshape_4d(gctx, Q, head_dim, n_heads, T, B);',
           'Q = ggml_permute(gctx, Q, 0, 2, 1, 3);','float scale = 1.0f / sqrtf((float)head_dim);','attn = ggml_flash_attn_ext(gctx, Q, K, V, attn_mask, scale, 0.0f, 0.0f);',
           'attn = ggml_mul_mat(gctx, L.o_w, attn);','cur = ggml_add(gctx, inp, attn);','m.embd_ln_w = get_any({ "embd_ln.weight", "token_embd_norm.weight" });']:
   self.assertIn(s,c)
  o=(GG/'ggml-cpu/ops.cpp').read_text()
  self.assertIn('neq1 >= Q_TILE_SZ);',o);self.assertIn('#define GGML_FA_TILE_Q  64',(GG/'ggml-cpu/common.h').read_text())
  self.assertIn('kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);',o);self.assertIn('vs = expf(s - M);',o)
  p=(ROOT/'tests/wasm-layer0-attention/probe.cpp').read_text()
  self.assertIn('ggml_flash_attn_ext(g, heads(g, i[0]), heads(g, i[1]), heads(g, i[2]), nullptr, scale, 0.0f, 0.0f)',p)
  self.assertIn('ggml_permute(g, ggml_reshape_4d(g, x, HD, NH, T, 1), 0, 2, 1, 3)',p)

class Protocol(unittest.TestCase):
 def test_profile(self):
  D=load('l0c','tools/ci-heavy/wasm_layer0_attention_v1.py');p=json.loads(D.PROFILE.read_text());D.validate(p)
  for path,d in p['owned_sha256'].items():self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),d,path)
  for k,v in[('candidate_enabled',True),('ops',list(A.ORDER)[:-1]),('runtime_change',True)]:
   q=copy.deepcopy(p);q[k]=v
   with self.assertRaises(ValueError):D.validate(q)

if __name__=='__main__':unittest.main()
