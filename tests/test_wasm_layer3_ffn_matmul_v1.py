#!/usr/bin/env python3
"""Layer3 FFN mul_mat replay: admission, metrics, decisions and source anchors. No build/model."""
import copy,hashlib,importlib.util,json,pathlib,re,shlex,struct,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools/wasm-layer3-ffn-matmul')]
import analysis as A
SRC=ROOT/'tests/wasm-layer3-ffn-matmul'
GG=ROOT/'ggml/src'

def production(kind):return json.loads((SRC/f'frozen-40852835039-full-{kind}-compile.json').read_text())

def static_from(kind):
 """A probe-like static build of the production profile: linkage-only defines and -fPIC dropped, probe entry added."""
 rows=[]
 for r in production(kind):
  r=copy.deepcopy(r);t=[x for x in shlex.split(r['command'])if x not in A.LINKAGE_ONLY and x!='-fPIC'and not x.endswith('_EXPORTS')]
  r['command']=shlex.join(t);rows.append(r)
 cxx='em++'if kind=='wasm'else'/usr/bin/c++'
 rows.append(dict(directory='/b',file='/src/tests/wasm-layer3-ffn-matmul/probe.cpp',output='p.o',
  command=cxx+' -O3 -DNDEBUG -std=gnu++17 -ffp-contract=off -o p.o -c /src/tests/wasm-layer3-ffn-matmul/probe.cpp'))
 return rows

def edit(rows,suffix,fn):
 r=next(r for r in rows if r['file'].endswith(suffix));r['command']=fn(r['command']);return rows

class Admission(unittest.TestCase):
 def test_production_profile_admitted(self):
  for kind in ['native','wasm']:
   out=A.admit(kind,static_from(kind),production(kind),'tests/wasm-layer3-ffn-matmul/probe.cpp')
   self.assertEqual(len(out['kernel_files']),len(A.KERNEL_FILES[kind]))
  prof={r['file']:r['profile']for r in A.admit('native',static_from('native'),production('native'),'tests/wasm-layer3-ffn-matmul/probe.cpp')['kernel_files']}
  self.assertTrue({'-mavx2','-mfma','-DGGML_USE_LLAMAFILE'}<=set(prof['ggml-cpu/arch/x86/quants.c']))
  prof={r['file']:r['profile']for r in A.admit('wasm',static_from('wasm'),production('wasm'),'tests/wasm-layer3-ffn-matmul/probe.cpp')['kernel_files']}
  self.assertTrue({'-msimd128','-pthread'}<=set(prof['ggml-cpu/arch/wasm/quants.c']));self.assertNotIn('-DGGML_USE_LLAMAFILE',prof['ggml-cpu/ggml-cpu.c'])
 def reject(self,kind,rows,msg):
  with self.assertRaisesRegex(ValueError,msg):A.admit(kind,rows,production(kind),'tests/wasm-layer3-ffn-matmul/probe.cpp')
 def test_profile_negatives(self):
  self.reject('native',edit(static_from('native'),'arch/x86/quants.c',lambda c:c.replace(' -mavx2','')),'profile differs')
  self.reject('native',edit(static_from('native'),'llamafile/sgemm.cpp',lambda c:c.replace(' -DGGML_USE_LLAMAFILE','')),'profile differs')
  self.reject('native',edit(static_from('native'),'ggml-cpu/ggml-cpu.c',lambda c:c.replace(' -O3',' -O3 -ffast-math')),'profile differs')
  self.reject('wasm',edit(static_from('wasm'),'arch/wasm/quants.c',lambda c:c.replace(' -msimd128','')),'profile differs')
  self.reject('wasm',edit(static_from('wasm'),'arch/wasm/quants.c',lambda c:c.replace(' -pthread','')),'profile differs')
  rows=static_from('native');rows=[r for r in rows if not r['file'].endswith('llamafile/sgemm.cpp')];self.reject('native',rows,'absent')
  rows=static_from('wasm');extra=copy.deepcopy(next(r for r in rows if r['file'].endswith('ggml-cpu/ggml-cpu.c')));extra['file']=extra['file'].replace('ggml-cpu.c','llamafile/sgemm.cpp');rows.append(extra)
  self.reject('wasm',rows,'must be absent')
 def test_probe_flag_negatives(self):
  for fn in [lambda c:c.replace(' -ffp-contract=off',''),lambda c:c.replace('-ffp-contract=off','-ffp-contract=fast'),lambda c:c.replace(' -O3',' -O3 -ffast-math')]:
   self.reject('native',edit(static_from('native'),'probe.cpp',fn),'Probe rounding flags')
  rows=static_from('native');rows.append(copy.deepcopy(rows[-1]));self.reject('native',rows,'Exactly one probe')

def f32(v):return struct.pack('<%df'%len(v),*v)

class Metrics(unittest.TestCase):
 def test_full_row_metrics_and_bit_exact(self):
  a=[float(i%7-3)+.25 for i in range(12*384)];m=A.metrics(f32(a),f32(a),384)
  self.assertTrue(m['bit_exact']and m['passed']);self.assertEqual(m['max_absolute'],0)
  b=list(a);b[5]+=.01;m=A.metrics(f32(a),f32(b),384);self.assertFalse(m['bit_exact']);self.assertEqual(m['differing_elements'],1);self.assertFalse(m['passed'])
  with self.assertRaises(ValueError):A.metrics(f32(a),f32(a),1536)
 def test_block_diff(self):
  self.assertEqual(A.block_diff(b'a'*68,b'a'*34+b'b'*34,34),dict(bytes_equal=False,blocks=2,differing_blocks=1))

def synthetic(selftest=True):
 meta,files={},{}
 for side in ['native','wasm']:
  cases=[]
  labels=[('authored_iq4_nl','iq4_nl','q8_0',1536),('authored_iq4_xs','iq4_xs','q8_K',384)]if selftest else \
   [('fc1_native','iq4_nl','q8_0',1536),('fc1_wasm0','iq4_nl','q8_0',1536),('fc1_wasm1','iq4_nl','q8_0',1536),('fc2_native','iq4_xs','q8_K',384),('fc2_wasm0','iq4_xs','q8_K',384)]
  for label,wt,at,width in labels:
   cases.append(dict(label=label,weight_type=wt,activation_type=at,weight_buffer='CPU'))
   base=[float(i%11)/8+1 for i in range(12*width)]
   nat=f32(base);gen=f32([x+(2**-20 if i==0 else 0)for i,x in enumerate(base)])
   files[(side,label,'native_order')]=nat;files[(side,label,'generic_order')]=gen
   files[(side,label,'production')]=nat if side=='native'else gen
   files[(side,label,'activation')]=bytes(34*12 if at=='q8_0'else 292*12)
  meta[side]=dict(cases=cases,fc1=dict(type='iq4_nl',ne=[384,1536]),fc2=dict(type='iq4_xs',ne=[1536,384]))
 return meta,files

class Decisions(unittest.TestCase):
 def test_selftest_pass_and_rejections(self):
  meta,files=synthetic();self.assertTrue(A.selftest(meta,files)['passed'])
  bad=dict(files);bad[('native','authored_iq4_nl','production')]=files[('native','authored_iq4_nl','generic_order')]
  self.assertFalse(A.selftest(meta,bad)['passed'])  # native order model wrong
  bad=dict(files);bad[('wasm','authored_iq4_xs','native_order')]=files[('wasm','authored_iq4_xs','generic_order')]
  self.assertFalse(A.selftest(meta,bad)['passed'])  # insensitive emulation
  m2=copy.deepcopy(meta);m2['native']['cases'][0]['weight_buffer']='CPU_REPACK';self.assertFalse(A.selftest(m2,files)['passed'])
 def test_replay_controls_and_numeric_scope(self):
  meta,files=synthetic(False)
  captured={(arm,op):files[('native'if arm=='native'else'wasm',op.split('_')[0]+'_'+arm,'production')]for arm in['native','wasm0','wasm1']for op in['fc1_raw','fc2_raw']if not(op=='fc2_raw'and arm=='wasm1')}
  r=A.replay(meta,files,captured);self.assertTrue(r['controls_passed'])
  self.assertEqual(set(r['ops']['fc1']['four_way']),{'native_kernel(native_input)','wasm_kernel(wasm_input)','native_kernel(wasm_input)','wasm_kernel(native_input)'})
  self.assertTrue(r['ops']['fc1']['candidate_expected_effect']['candidate(wasm_input) vs native_kernel(wasm_input)']['bit_exact'])
  bad=dict(captured);bad[('native','fc1_raw')]=files[('wasm','fc1_wasm0','production')];self.assertFalse(A.replay(meta,files,bad)['controls_passed'])
  m2=copy.deepcopy(meta);m2['native']['fc1']['type']='q8_0';self.assertFalse(A.replay(m2,files,captured)['controls_passed'])
  far=dict(files);far[('wasm','fc2_wasm0','production')]=f32([x+.01 for x in struct.unpack('<%df'%(12*384),files[('wasm','fc2_wasm0','production')])])
  self.assertFalse(A.replay(meta,far,captured)['four_way_within_budget'])

class SourceAnchors(unittest.TestCase):
 """The probe emulates exactly these a06 statements; fail if the frozen sources move."""
 def test_ggml_kernel_anchors(self):
  q=(GG/'ggml-cpu/quants.c').read_text();x=(GG/'ggml-cpu/arch/x86/quants.c').read_text();s=(GG/'ggml-cpu/llamafile/sgemm.cpp').read_text()
  f=(GG/'ggml-cpu/arch-fallback.h').read_text();w=f[f.index('#elif defined(__wasm__)'):]
  self.assertIn('#define ggml_vec_dot_iq4_nl_q8_0_generic ggml_vec_dot_iq4_nl_q8_0',w);self.assertIn('#define ggml_vec_dot_iq4_xs_q8_K_generic ggml_vec_dot_iq4_xs_q8_K',w)
  self.assertIn('        sumf += d * (sumi1 + sumi2);',q);self.assertIn('            sumf += d1 * (sumi1 + sumi2);',q)
  self.assertIn('accum = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ibl].d)*y[ibl].d),',x)
  self.assertIn('    res = _mm_add_ps(res, _mm_movehl_ps(res, res));',x)
  self.assertIn('case GGML_TYPE_IQ4_NL: {',s);self.assertIn('tinyBLAS_Q0_AVX<block_iq4_nl, block_q8_0, float> tb{',s)
  self.assertIn('return MM256_SET_M128I(load1(b), load0(b));',s);self.assertIn('return _mm256_fmadd_ps(a, b, c);',s)
  self.assertIn('void quantize_row_q8_0_wasm_nearest(',(GG/'ggml-cpu/quants.h').read_text())
  self.assertIn('if (params->wasm_q8_0_round && type == GGML_TYPE_Q8_0) {',(GG/'ggml-cpu/ggml-cpu.c').read_text())
 def test_probe_mirrors_generic_statements(self):
  p=(SRC/'probe.cpp').read_text()
  for line in ['sumf += d * (sumi1 + sumi2);','sumf += d1 * (sumi1 + sumi2);','sumf += d2 * (sumi1 + sumi2);','acc[L] = std::fmaf(d, (float) s, acc[L]);',
               'for (int L = 0; L < 8; ++L) acc[L] = std::fmaf(dd, (float) lane[L], acc[L]);','const float r0 = v[4] + v[0], r1 = v[5] + v[1], r2 = v[6] + v[2], r3 = v[7] + v[3];',
               'ggml_backend_cpu_set_wasm_q8_0_rounding(b, true);','quantize_row_q8_0_wasm_nearest(x + (size_t) r * k, q.data() + r * row, k);']:
   self.assertIn(line,p)

class Protocol(unittest.TestCase):
 def test_profile(self):
  spec=importlib.util.spec_from_file_location('ffn',ROOT/'tools/ci-heavy/wasm_layer3_ffn_matmul_v1.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
  p=json.loads(D.PROFILE.read_text());D.validate(p)
  for path,d in p['owned_sha256'].items():self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),d,path)
  for kind in['native','wasm']:self.assertEqual(hashlib.sha256((SRC/f'frozen-40852835039-full-{kind}-compile.json').read_bytes()).hexdigest(),p['production_inventories'][kind])
  for key,value in[('candidate_enabled',True),('runtime_change',True),('budgets',dict(max_absolute=.003,minimum_row_cosine=.99999))]:
   q=copy.deepcopy(p);q[key]=value
   with self.assertRaises(ValueError):D.validate(q)
  self.assertEqual(p['captures']['artifact']['run'],41432135298)

if __name__=='__main__':unittest.main()
