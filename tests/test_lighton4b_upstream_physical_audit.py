"""Weight-free complete mapping/permutation preflight for the fixed upstream file."""
import hashlib,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audit',ROOT/'tools/ci-heavy/lighton4b_upstream_physical_audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
class PhysicalMap(unittest.TestCase):
 def test_actual426_name_shape_type_contract(self):
  physical=json.loads((ROOT/'tests/fixtures/lighton4b-upstream-q4km-header-v1.json').read_text());source=json.loads((ROOT/'tests/fixtures/lightonocr3-4b-source/model.safetensors.header.json').read_text());cfg=json.loads((ROOT/'docs/lighton4b-upstream-physical-audit-v1.json').read_text());mapping={}
  for name,row in source.items():
   if name.startswith('model.language_model.'):
    native='l.'+name.removeprefix('model.language_model.');up=audit.physical_name(native);p=physical['tensors'][up];self.assertEqual(__import__('math').prod(row['shape']),p['elements']);mapping[up]={'native_name':native,'upstream_shape':p['shape'],'type':p['type']}
  self.assertEqual(set(mapping),set(physical['tensors']));self.assertEqual(len(mapping),426);self.assertEqual(hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest(),cfg['predeclared_mapping_sha256']);self.assertEqual(physical['tensors']['token_embd.weight']['type'],14);self.assertNotIn('output.weight',physical['tensors'])
 def test_all_rows_bijective_and256_block_negative(self):
  p=audit.head_perm();self.assertEqual(sorted(p),list(range(4096)));self.assertEqual(p[:128],list(range(128)));self.assertEqual(p[128:256],list(range(2048,2176)))
  self.assertEqual({p[i]//256 for i in range(256)},{0,8})
  for suffix,n in [('in_proj_qkv.weight',8192),('in_proj_z.weight',4096),('in_proj_a.weight',32),('in_proj_b.weight',32)]:
   rows=audit.rows_for('l.layers.0.linear_attn.'+suffix,n);self.assertEqual(sorted(rows),list(range(n)));self.assertNotEqual(rows,list(range(n)))
if __name__=='__main__':unittest.main()
