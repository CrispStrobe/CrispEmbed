#!/usr/bin/env python3
"""Small authored storage fixtures and source checks, no real model/framework."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import quantize_josev2b_q4k as q
import josev2b_q4k_contract as contract


def fake_schema():
    return {**{f'l.layers.{i}.mlp.up_proj.weight':[256,1]for i in range(150)},
            **{f'head.authored_{i}':[1]for i in range(291)},'l.embed_tokens.weight':[1]}


def fixture(path,expected,quantized=False):
    """Unusable as a real model: tiny fake shapes, no original weight values."""
    text=lambda s:struct.pack('<Q',len(s.encode()))+s.encode()
    metadata=[('general.architecture',8,'josev2b_systemone'),('general.license',8,'apache-2.0'),
              ('josev2b.packed.format',8,q.OLD_FORMAT),('general.file_type',4,12 if quantized else 0),
              ('general.quantization_version',4,2),('tokenizer.authored',8,'Ä [Q] [MASK]')]
    data=b'GGUF'+struct.pack('<IQQ',3,len(expected),len(metadata))
    for key,kind,value in metadata:data+=text(key)+struct.pack('<I',kind)+(text(value)if kind==8 else struct.pack('<I',value))
    payload=bytearray();types=q.recipe(expected)
    for name,shape in expected.items():
        offset=(len(payload)+31)//32*32;payload.extend(b'\0'*(offset-len(payload)))
        kind=types[name]if quantized else 0
        blob=(struct.pack('<ee',1.,0.)+bytes(140))if kind==12 else struct.pack('<'+'f'*__import__('math').prod(shape),*([1.]*__import__('math').prod(shape)))
        # Unit axes may be stored with a shorter GGUF rank; logical ne is exact.
        ne=shape[:1]if quantized and shape==[256,1]else shape
        data+=text(name)+struct.pack('<I',len(ne))+struct.pack('<'+'Q'*len(ne),*ne)+struct.pack('<IQ',kind,offset)
        payload.extend(blob)
    path.write_bytes(data+b'\0'*((-len(data))%32)+payload)


class Controls(unittest.TestCase):
    def test_frozen_real_inventory_and_protection(self):
        rows=q.schema();plan=q.recipe(rows)
        self.assertEqual(len(rows),442);self.assertEqual(sum(k==12 for k in plan.values()),150)
        self.assertEqual(sum(k==0 for k in plan.values()),292)
        self.assertEqual(sum(__import__('math').prod(s)for s in rows.values()),1912297796)
        self.assertTrue(all(s[0]in(2048,6144)for n,s in rows.items()if plan[n]==12))
        for n in rows:
            if n.startswith('head.')or n=='l.embed_tokens.weight'or any(t in n for t in ('norm','conv1d','A_log','dt_bias','in_proj_a.','in_proj_b.')):
                self.assertEqual(plan[n],0,n)

    def test_runtime_and_converter_projection_whitelists_match(self):
        import re
        source=(ROOT/'src/josev2b_packed_profile.h').read_text()
        suffixes=set(re.findall(r'"(\.(?:mlp|self_attn|linear_attn)\.[^"]+\.weight)"',source))
        self.assertEqual(suffixes,set(q.PROJECTIONS))

    def test_host_barrier_precedes_source_file_access(self):
        with patch.dict(q.os.environ,{},clear=True),patch.object(q.Path,'is_dir',return_value=False):
            with self.assertRaisesRegex(ValueError,'hosted'):q.quantize('/missing','/missing','/missing')
        self.assertNotIn('numpy',sys.modules);self.assertNotIn('torch',sys.modules)

    def authored(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        root=Path(directory.name);rows=fake_schema();source=root/'f32.gguf';out=root/'unsealed.gguf'
        fixture(source,rows);fixture(out,rows,True)
        return rows,source,out

    def test_seal_after_all_critical_byte_and_semantic_checks(self):
        rows,source,out=self.authored();before=q.parse(out,rows)
        with patch.object(q,'PARAMETERS',38692):result=q.verify_conversion(source,out,rows,seal=True)
        after=q.parse(out,rows)
        self.assertTrue(result['sealed']and result['critical_bytes_exact'])
        self.assertEqual((result['q4k_tensors'],result['f32_tensors']),(150,292))
        self.assertEqual(before['tensors'],after['tensors']);self.assertEqual(before['bytes'],after['bytes'])
        self.assertEqual(after['metadata']['josev2b.packed.format']['value'],q.NEW_FORMAT)
        self.assertEqual(len(q.OLD_FORMAT),len(q.NEW_FORMAT))

    def test_bad_critical_bytes_reject_without_changing_format(self):
        rows,source,out=self.authored();info=q.parse(out,rows)
        with out.open('r+b')as f:f.seek(info['tensors']['l.embed_tokens.weight']['data_offset']);f.write(struct.pack('<f',2.))
        with patch.object(q,'PARAMETERS',38692),self.assertRaisesRegex(ValueError,'critical F32 tensor bytes'):q.verify_conversion(source,out,rows,seal=True)
        self.assertEqual(q.parse(out,rows)['metadata']['josev2b.packed.format']['value'],q.OLD_FORMAT)

    def test_nonfinite_q4_scale_reject_without_sealing(self):
        rows,source,out=self.authored();info=q.parse(out,rows);name=next(iter(rows))
        with out.open('r+b')as f:f.seek(info['tensors'][name]['data_offset']);f.write(struct.pack('<e',float('inf')))
        with patch.object(q,'PARAMETERS',38692),self.assertRaisesRegex(ValueError,'nonfinite Q4_K'):q.verify_conversion(source,out,rows,seal=True)
        self.assertEqual(q.parse(out,rows)['metadata']['josev2b.packed.format']['value'],q.OLD_FORMAT)

    def test_wrong_filetype_and_metadata_changes_reject_before_sealing(self):
        for key in ('general.file_type','tokenizer.authored'):
            rows,source,out=self.authored();info=q.parse(out,rows)
            with out.open('r+b')as f:
                field=info['metadata'][key];f.seek(field['value_offset']if key=='general.file_type'else field['string_offset'])
                f.write(struct.pack('<I',8)if key=='general.file_type'else b'X')
            with patch.object(q,'PARAMETERS',38692),self.assertRaises(ValueError):q.verify_conversion(source,out,rows,seal=True)

    def test_shape_truncation_and_bad_tail_reject(self):
        rows,source,out=self.authored();bad=copy.deepcopy(rows);bad[next(iter(bad))]=[512,1]
        with self.assertRaisesRegex(ValueError,'semantic shape'):q.parse(out,bad)
        data=out.read_bytes();out.write_bytes(data[:-1])
        with self.assertRaises(ValueError):q.parse(out,rows)
        out.write_bytes(data+b'X')
        with self.assertRaisesRegex(ValueError,'tail padding'):q.parse(out,rows)

    def test_closed_source_pending_quality_and_omission_rejections(self):
        cfg=json.loads(contract.PATH.read_text());contract.validate(cfg)
        for name in ['src/josev2b_head.cpp','src/qwen35_decoder.cpp',*contract.REQUIRED]:
            bad=copy.deepcopy(cfg);bad['protected_sha256'].pop(name)
            with self.assertRaisesRegex(ValueError,'closure'):contract.validate(bad)
        bad=copy.deepcopy(cfg);bad['original']['version_label']='latest'
        with self.assertRaisesRegex(ValueError,'original'):contract.validate(bad)

    def test_no_original_rerun_and_strict_control_before_candidate(self):
        source=(ROOT/'tools/kaggle/josev-q4k/worker.py').read_text()
        self.assertNotIn('import torch',source)
        self.assertLess(source.index("control=execute('f32'"),source.index('packer.quantize('))
        self.assertIn("comparison['compared_stages']==227",source)
        self.assertIn('all_metrics_agree',source)
        self.assertIn('publication_qualified=False',source)
        self.assertIn('strict_q4_f32_gates_passed',source)

    def test_isolated_cpu_setup_and_locked_numpy(self):
        lock=(ROOT/'tools/kaggle/josev-q4k/requirements.lock').read_text().splitlines()
        self.assertEqual(len(lock),2);self.assertTrue(all('--hash=sha256:'in line for line in lock))
        numpy=next(r for r in json.loads((ROOT/'docs/josev2b-environment-lock.json').read_text())['packages']if r['name']=='numpy')
        self.assertIn(numpy['url'],lock[1]);self.assertIn(numpy['sha256'],lock[1]);self.assertEqual(numpy['requires_dist'],[])
        bootstrap=(ROOT/'tools/kaggle/josev-q4k/kernel.py').read_text()
        self.assertIn('with_pip=False',bootstrap);self.assertIn('--require-hashes',bootstrap);self.assertIn('4*3600',bootstrap)
        metadata=json.loads((ROOT/'tools/kaggle/josev-q4k/kernel-metadata.json').read_text())
        self.assertEqual(metadata['enable_gpu'],'false');self.assertEqual(metadata['title'],metadata['id'].split('/')[-1])
        cmake=(ROOT/'tests/josev-q4k/CMakeLists.txt').read_text()
        self.assertIn('GGML_LLAMAFILE ON',cmake);self.assertIn('CRISPEMBED_JOSEV_HEAD_TESTING',cmake)


if __name__=='__main__':unittest.main()
