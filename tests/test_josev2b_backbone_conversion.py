"""No-payload conversion boundary controls; actual BF16/GGUF readback hosted."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import convert_josev2b_backbone_f32 as m


class Controls(unittest.TestCase):
    def test_source_derived_text_inventory_and_non_square_layouts(self):
        rows=m.schema();self.assertEqual(len(rows),320)
        self.assertEqual(sum(__import__('math').prod(v) for v in rows.values()),1881825088)
        self.assertEqual(rows[m.PREFIX+'layers.3.self_attn.q_proj.weight'],[4096,2048])
        self.assertEqual(rows[m.PREFIX+'layers.0.linear_attn.conv1d.weight'],[6144,1,4])
        self.assertTrue(all(k.startswith(m.PREFIX) for k in rows))
        self.assertNotIn('lm_head.weight',rows)

    def test_exact_source_namespace_shapes_and_index_required(self):
        name='model.safetensors-00001-of-00001.safetensors'
        rows={k:dict(shape=v,dtype='BF16') for k,v in m.schema().items()}
        rows['model.visual.original.weight']=dict(shape=[2,3],dtype='F32')
        index={'weight_map':dict.fromkeys(rows,name)}
        self.assertEqual(len(m.checked_text(rows,index,name)),320)
        bad=json.loads(json.dumps(rows));bad[m.PREFIX+'norm.weight']['shape']=[1024]
        with self.assertRaises(ValueError):m.checked_text(bad,index,name)
        bad=dict(index['weight_map']);bad[next(iter(bad))]='wrong-shard'
        with self.assertRaises(ValueError):m.checked_text(rows,{'weight_map':bad},name)
        for extra in ('lm_head.weight','unreviewed.namespace'):
            changed={**rows,extra:dict(shape=[1],dtype='F32')}
            with self.assertRaises(ValueError):m.checked_text(changed,{'weight_map':dict.fromkeys(changed,name)},name)

    def test_missing_original_rejects_before_optional_payload_readers(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'reference.json').write_text('{}')
            with self.assertRaises(ValueError):m.convert(p/'missing',p/'missing-index',p/'reference.json','wrong','bad',p/'output.gguf')
            self.assertFalse((p/'output.gguf').exists())
        for name in ('numpy','gguf','torch'):self.assertNotIn(name,sys.modules)

    def test_actual_backbone_inventory_accepts_sparse_extent_above_head_bound(self):
        import struct
        # A sparse synthetic extent exercises the real header parser without
        # allocating/readback of a model payload or importing an array package.
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'synthetic.safetensors';payload=130*1024*1024
            # Use bounded dimensions while retaining the same total extent.
            header=json.dumps({'synthetic':dict(shape=[130,262144],dtype='F32',data_offsets=[0,payload])}).encode()
            with path.open('wb') as stream:
                stream.write(struct.pack('<Q',len(header)));stream.write(header);stream.truncate(8+len(header)+payload)
            offset,rows=m.backbone_inventory(path,path.stat().st_size)
            self.assertEqual(offset,8+len(header));self.assertEqual(rows['synthetic']['offsets'],[0,payload])
            from josev2b_head_schema import inventory
            with self.assertRaises(ValueError):inventory(path)
            with self.assertRaises(ValueError):m.backbone_inventory(path,path.stat().st_size+4)

    def test_help_is_framework_and_payload_free(self):
        p=subprocess.run([sys.executable,ROOT/'tools/convert_josev2b_backbone_f32.py','--help'],capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,0);self.assertIn('--original-sha256',p.stdout)


if __name__=='__main__':unittest.main()
