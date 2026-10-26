"""Small original-first schema/parser controls; no model/framework loading."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from josev2b_head_schema import HEAD_CONFIG, checked_readback, checked_schema, inventory, reference, tensor_schema
from convert_josev2b_head_f32 import convert


class HeadControls(unittest.TestCase):
    def test_complete_source_derived_dimensions_and_scalar_adapters(self):
        import math
        rows=tensor_schema()
        self.assertEqual(len(rows),122)
        self.assertEqual(sum(math.prod(s) for s in rows.values()),30472708)
        self.assertEqual(rows['hidden_norm.weight'],[2048])
        self.assertEqual(rows['evidence_layers.1.attention.in_proj_weight'],[1536,512])
        self.assertEqual(rows['layers.3.norm3.bias'],[512])
        self.assertEqual(rows['residual_scorer.3.weight'],[1,512])
        self.assertEqual(rows['residual_gate'],[])
        self.assertEqual(tensor_schema(HEAD_CONFIG),rows)
        with self.assertRaisesRegex(ValueError,'configuration'):
            tensor_schema({**HEAD_CONFIG,'width':256})

    def test_mixed_bf16_f32_and_rank_zero_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'head'
            header={'vector':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]},
                    'scale':{'dtype':'F32','shape':[],'data_offsets':[4,8]}}
            raw=json.dumps(header).encode()
            path.write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*8)
            offset,rows=inventory(path)
            self.assertEqual(offset,8+len(raw))
            self.assertEqual(rows['scale']['shape'],[])
            self.assertEqual(rows['vector']['dtype'],'BF16')
            with self.assertRaisesRegex(ValueError,'122-tensor'):
                checked_schema(rows)

    def test_gaps_overlap_bad_dtype_and_trailing_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'head'
            for dtype,offsets,payload in [('F16',[0,4],b'\0'*4),('F32',[4,8],b'\0'*8),
                                          ('BF16',[0,4],b'\0'*3),('BF16',[0,4],b'\0'*5),
                                          ('F32',[0,8.0],b'\0'*8)]:
                raw=json.dumps({'v':dict(dtype=dtype,shape=[2],data_offsets=offsets)}).encode()
                path.write_bytes(struct.pack('<Q',len(raw))+raw+payload)
                with self.subTest(dtype=dtype,offsets=offsets), self.assertRaises(ValueError):
                    inventory(path)
            row='{"dtype":"F32","shape":[],"data_offsets":[0,4]}'
            raw=('{"v":'+row+',"v":'+row+'}').encode()
            path.write_bytes(struct.pack('<Q',len(raw))+raw+b'\0'*4)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                inventory(path)

    def test_missing_original_rejects_before_payload_or_optional_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'receipt.json';report.write_text('{}')
            with self.assertRaisesRegex(ValueError,'receipt SHA'):
                convert(Path(directory)/'missing-head',report,'0'*64,'0'*40,Path(directory)/'out.gguf')
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('numpy',sys.modules)
        self.assertNotIn('gguf',sys.modules)

    def test_a_non_completed_receipt_is_not_an_original(self):
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'receipt.json';report.write_text('{}')
            source='1e723a96f8fed0ea34b0e071e88f29561a315aaa'
            with self.assertRaisesRegex(ValueError,'completion/source'):
                reference(report,hashlib.sha256(report.read_bytes()).hexdigest(),source)

    def test_new_worker_receipt_coverage_and_bare_dtype_spellings(self):
        # Synthetic receipt tests source validation only, never model parity.
        source='2133cfa12dbdd25653fd68b1a6054f69f80f7e9c'
        def blob(name):
            return subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)
        raw=blob('docs/josev2b-source-contract.json'); cfg=json.loads(raw)
        panel=json.loads(blob(cfg['panel']['path']))
        names={'docs/josev2b-source-contract.json',cfg['panel']['path'],
               'tools/dump_josev2b_reference.py','tools/josev2b_contract.py',
               'tools/josev2b_environment.py',cfg['environment_lock']['path'],
               cfg['environment_lock']['requirements_path']}
        proof={name:dict(shape=shape,checkpoint_dtype='bfloat16',loaded_dtype='float32',
            checkpoint_promoted_f32_sha256='a'*64,loaded_f32_sha256='a'*64)
            for name,shape in tensor_schema().items()}
        report=dict(schema=cfg['schema'],status='completed',conclusive=True,native_outputs=False,
            git_commit=source,model_revision=cfg['revision'],
            contract_sha256=hashlib.sha256(raw).hexdigest(),panel_sha256=cfg['panel']['sha256'],
            cases=[dict(id=c['id'],observer_bit_exact=True,repeat_bit_exact=True) for c in panel['cases']],
            source={name:hashlib.sha256(blob(name)).hexdigest() for name in names},
            weights=dict(head_tensors=proof,head_identity_exact=True,head_tensor_count=122,
                head_state_elements=30472708,
                head_checkpoint_sha256=cfg['files']['joint_head.safetensors']['sha256']))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'synthetic-receipt.json'
            def validate():
                path.write_text(json.dumps(report))
                return reference(path,hashlib.sha256(path.read_bytes()).hexdigest(),source)
            self.assertEqual(len(validate()[2]),122)
            proof['residual_gate']['checkpoint_dtype']='float32'
            self.assertEqual(len(validate()[2]),122)
            del report['source'][cfg['environment_lock']['path']]
            with self.assertRaisesRegex(ValueError,'coverage'):
                validate()

    def test_readback_rejects_reshaped_payload_and_sign_zero_changes(self):
        payload=struct.pack('<4f',0.,1.,2.,3.)
        digest=hashlib.sha256(payload).hexdigest()
        checked_readback([2,2],payload,[2,2],digest)
        with self.assertRaisesRegex(ValueError,'dimensions'):
            checked_readback([4,1],payload,[2,2],digest)
        with self.assertRaisesRegex(ValueError,'bytes'):
            checked_readback([2,2],struct.pack('<4f',-0.,1.,2.,3.),[2,2],digest)

    def test_help_is_payload_and_framework_free(self):
        result=subprocess.run([sys.executable,ROOT/'tools/convert_josev2b_head_f32.py','--help'],
                              capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0)
        self.assertIn('--original-source',result.stdout)


if __name__=='__main__':unittest.main()
