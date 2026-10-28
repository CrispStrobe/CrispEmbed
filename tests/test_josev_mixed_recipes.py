#!/usr/bin/env python3
"""Tiny storage/rounding/closure fixtures; no framework, model or compiler."""
import ast
import copy
import importlib.util
import json
import io
import math
from pathlib import Path
import random
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import quantize_josev2b_mixed as q
import josev2b_mixed_contract as contract
VECTOR_CONTROLS='--vector-controls'in sys.argv
if VECTOR_CONTROLS:sys.argv.remove('--vector-controls')


def fixture(path,rows,identity=None):
    text=lambda s:struct.pack('<Q',len(s.encode()))+s.encode()
    metadata=[('general.architecture',8,'josev2b_systemone'),('general.license',8,'apache-2.0'),
              ('josev2b.packed.format',8,q.OLD_FORMAT),('general.file_type',4,12 if identity else 0),
              ('general.quantization_version',4,2),('tokenizer.fixture',8,'Ä [MASK]')]
    header=b'GGUF'+struct.pack('<IQQ',3,len(rows),len(metadata))
    for name,kind,value in metadata:header+=text(name)+struct.pack('<I',kind)+(text(value)if kind==8 else struct.pack('<I',value))
    wanted=q.recipe(rows,identity)if identity else {n:0 for n in rows}
    payload=bytearray();positions={}
    for name,shape in rows.items():
        offset=(len(payload)+31)//32*32;payload.extend(bytes(offset-len(payload)))
        t=wanted[name];n=math.prod(shape)
        blob=struct.pack('<f',1.)*n if t==0 else struct.pack('<e',1.)*n if t==1 else (struct.pack('<ee',1.,0.)+bytes(140))*(n//256)
        header+=text(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*shape)
        positions[name]=len(header);header+=struct.pack('<IQ',t,offset);payload.extend(blob)
    path.write_bytes(header+bytes((-len(header))%32)+payload)
    return positions


class Controls(unittest.TestCase):
    def authored(self,identity='A'):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);root=Path(td.name)
        rows={n:([256,1]if q.old.projection(n)else[1])for n in q.schema()}
        source=root/'source.gguf';out=root/'candidate.gguf'
        fixture(source,rows);positions=fixture(out,rows,identity)
        return rows,source,out,positions

    def verify(self,rows,source,out,identity='A',seal=False):
        with patch.object(q,'PARAMETERS',sum(math.prod(s)for s in rows.values())):
            return q.verify_conversion(source,out,rows,identity,seal=seal)

    def test_exact_real_roles_and_monotonic_down_protection(self):
        rows=q.schema();plans={x:q.recipe(rows,x)for x in 'ABC'}
        for x,plan in plans.items():
            self.assertEqual({t:sum(v==t for v in plan.values())for t in q.TYPES},q.COUNTS[x])
            self.assertTrue(all(plan[n]==0 for n in rows if not q.old.projection(n)))
            self.assertTrue(all(plan[n]==1 for n in rows if q.old.projection(n)and'.mlp.'not in n))
        for n in rows:
            if '.down_proj.'in n:self.assertEqual((plans['B'][n],plans['C'][n]),(0,0))
            if plans['C'][n]==12:self.assertGreaterEqual(int(n.split('.')[2]),20)
        with self.assertRaises(ValueError):q.recipe(rows,'D')

    def test_f16_independent_ties_subnormals_signedzero_and_overflow(self):
        samples=[0,0x80000000,0x33800000,0x33000000,0x33000001,0x387fe000,
                 0x38800000,0x3f801000,0x3f803000,0x477fe000,0x00000001]
        samples += [random.Random(n).getrandbits(32)for n in range(4096)]
        for bits in samples:
            value=struct.unpack('<f',struct.pack('<I',bits))[0]
            try:expected=struct.unpack('<H',struct.pack('<e',value))[0]
            except OverflowError:
                with self.assertRaises(ValueError):q.f16_bits(bits)
                continue
            if not math.isfinite(value):
                with self.assertRaises(ValueError):q.f16_bits(bits)
            else:self.assertEqual(q.f16_bits(bits),expected,hex(bits))
        for bits in (0x7f800000,0x7fc00000,0xff800000,0x47800000):
            with self.assertRaises(ValueError):q.f16_bits(bits)

    def test_all_three_storage_fixtures_verify_and_seal(self):
        for identity in 'ABC':
            rows,source,out,_=self.authored(identity);before=q.parse(out,rows)
            result=self.verify(rows,source,out,identity,True);after=q.parse(out,rows)
            self.assertEqual(after['tensors'],before['tensors'])
            self.assertEqual(after['metadata']['josev2b.packed.format']['value'],q.NEW_FORMAT)
            self.assertEqual(result['counts'],{q.TYPES[t]:n for t,n in q.COUNTS[identity].items()})
            self.assertTrue(result['critical292_f32_byte_exact']and result['independent_f16_rounding_exact'])

    def test_mutated_half_or_critical_never_seals(self):
        for kind in (0,1):
            rows,source,out,_=self.authored();parsed=q.parse(out,rows)
            n=next(n for n,r in parsed['tensors'].items()if r['type']==kind)
            r=parsed['tensors'][n]
            with out.open('r+b')as f:f.seek(r['data_offset']);f.write(struct.pack('<f'if kind==0 else'<e',2.))
            with self.assertRaisesRegex(ValueError,'F32 bytes|F16 rounding'):self.verify(rows,source,out)
            self.assertEqual(q.parse(out,rows)['metadata']['josev2b.packed.format']['value'],q.OLD_FORMAT)

    def test_nonfinite_source_and_q4_scales_reject(self):
        for which in ('source','q4'):
            rows,source,out,_=self.authored()
            parsed=q.parse(out if which=='q4'else source,rows)
            n=next(n for n,r in parsed['tensors'].items()if r['type']==12)if which=='q4'else next(iter(rows))
            r=parsed['tensors'][n]
            with (out if which=='q4'else source).open('r+b')as f:
                f.seek(r['data_offset']);f.write(struct.pack('<e'if which=='q4'else'<f',float('inf')))
            with self.assertRaisesRegex(ValueError,'nonfinite'):self.verify(rows,source,out)

    def test_wrong_recipe_storage_unknown_type_shape_and_extent(self):
        rows,source,out,positions=self.authored()
        with self.assertRaisesRegex(ValueError,'recipe type'):self.verify(rows,source,out,'B')
        data=out.read_bytes();n=next(iter(rows))
        with out.open('r+b')as f:f.seek(positions[n]);f.write(struct.pack('<I',2))
        with self.assertRaisesRegex(ValueError,'only F32/F16'):q.parse(out,rows)
        out.write_bytes(data[:-1])
        with self.assertRaises(ValueError):q.parse(out,rows)
        out.write_bytes(data);bad=copy.deepcopy(rows);bad[n]=[512,1]
        with self.assertRaisesRegex(ValueError,'semantic shape'):q.parse(out,bad)

    def test_predictions_and_exact_c_quantizer_only(self):
        rows=q.schema()
        for name in 'ABC':
            wanted=q.recipe(rows,name)
            size=9989328+sum(math.prod(rows[n])*(4 if t==0 else 2 if t==1 else 144/256)for n,t in wanted.items())
            self.assertEqual(size,q.EXPECTED_BYTES[name])
        with patch.object(q,'hosted',side_effect=ValueError('hosted')):
            with self.assertRaisesRegex(ValueError,'hosted'):q.quantize('/missing','/missing','/missing','A')
        self.assertNotIn('numpy',sys.modules);self.assertNotIn('torch',sys.modules)

    def test_contract_closes_legacy_and_runtime_before_payload(self):
        cfg=contract.validate(json.loads(contract.PATH.read_text()))
        for name in ('src/qwen35_linear.cpp','tools/kaggle/josev-q4k/worker.py','tests/josev-q4k/CMakeLists.txt'):
            bad=copy.deepcopy(cfg);bad['protected_sha256'].pop(name)
            with self.assertRaisesRegex(ValueError,'closure'):contract.validate(bad)
        for field in ('holdout_access','publication_qualified','recipe_selected'):
            bad=copy.deepcopy(cfg);bad[field]=True
            with self.assertRaises(ValueError):contract.validate(bad)
        bad=copy.deepcopy(cfg);bad['recipes']['C']['counts']['12']=12
        with self.assertRaises(ValueError):contract.validate(bad)

    def test_worker_strict_control_before_all_recipes_and_no_holdout(self):
        source=(ROOT/'tools/ci-heavy/josev2b_mixed_recipes.py').read_text();tree=ast.parse(source)
        imports=[n for n in ast.walk(tree)if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('torch'in ast.unparse(n)or'transformers'in ast.unparse(n)for n in imports))
        self.assertLess(source.index("control=execute('f32'"),source.index("for identity in ('A','B','C')"))
        self.assertIn("require(control['numeric_and_complete_typed_parity']",source)
        self.assertIn("comparison['compared_stages']==227",source)
        self.assertIn("all_metrics_agree",source)
        self.assertIn("holdout_native_outputs=False,recipe_selected=False",source.replace('\n                      ',''))
        self.assertNotIn('task-quality',source)
        workflow=(ROOT/'.github/workflows/josev-mixed-recipes-v1.yml').read_text()
        self.assertLess(workflow.index('--fetch-reference'),workflow.index('--reviewed-source'))
        self.assertIn('retention-days: 90',workflow);self.assertIn('if: always()',workflow)


@unittest.skipUnless(VECTOR_CONTROLS,'run after locked NumPy install on hosted CI')
class VectorControls(unittest.TestCase):
    def test_actual_vector_matches_scalar_and_struct_gold(self):
        samples=[0,0x80000000,1,0x80000001,0x33000000,0x33000001,0x33800000,
                 0x387fe000,0x38800000,0x3f801000,0x3f803000,0x477fe000]
        samples += [random.Random(n).getrandbits(32)for n in range(4096)]
        good=[]
        for bits in samples:
            value=struct.unpack('<f',struct.pack('<I',bits))[0]
            try:expected=struct.pack('<e',value)
            except OverflowError:
                with self.assertRaisesRegex(ValueError,'overflows'):q.vector_f16(struct.pack('<I',bits))
                continue
            if not math.isfinite(value):
                with self.assertRaisesRegex(ValueError,'nonfinite'):q.vector_f16(struct.pack('<I',bits))
            else:
                self.assertEqual(q.vector_f16(struct.pack('<I',bits)),expected)
                self.assertEqual(q.vector_f16(struct.pack('<I',bits)),struct.pack('<H',q.f16_bits(bits)))
                good.append(bits)
        data=b''.join(struct.pack('<I',v)for v in good)
        expected=b''.join(struct.pack('<H',q.f16_bits(v))for v in good)
        self.assertEqual(q.vector_f16(data),expected)
        for bad in (b'',b'x',bytes(4*65536+4)):
            with self.assertRaises(ValueError):q.vector_f16(bad)
        for bits in (0x7f800000,0xff800000,0x7fc00001,0x47800000):
            with self.assertRaises(ValueError):q.vector_f16(struct.pack('<I',bits))

    def test_actual_chunked_hash_finite_and_rounding_path(self):
        bits=[0x80000000,0x33800000,0x3f801000,0x3f803000,0x387fe000]
        data=b''.join(struct.pack('<I',bits[i%len(bits)])for i in range(65539))
        half=b''.join(struct.pack('<H',q.f16_bits(bits[i%len(bits)]))for i in range(65539))
        scalar=q.source_hash_and_half(io.BytesIO(b'1234'+data),4,len(data),io.BytesIO(b'xx'+half),2)
        vector=q.source_hash_and_half(io.BytesIO(b'1234'+data),4,len(data),io.BytesIO(b'xx'+half),2,vector=True)
        self.assertEqual(vector,scalar)
        with self.assertRaisesRegex(ValueError,'rounding differs'):
            q.source_hash_and_half(io.BytesIO(data),0,len(data),io.BytesIO(bytes(len(half))),vector=True)
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            q.source_hash_and_half(io.BytesIO(struct.pack('<I',0x7fc00000)),0,4,vector=True)


if __name__=='__main__':unittest.main()
