"""Small stdlib contract/adapter controls; never compile or import arrays."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import josev2b_backbone_probe as c
import josev2b_backbone_diff as d
spec=importlib.util.spec_from_file_location('backbone_driver',ROOT/'tools/ci-heavy/josev2b_backbone.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class Controls(unittest.TestCase):
    def report(self):return json.loads((ROOT/'tests/results/josev2b/original-reference.json').read_text())
    def freeze(self):return json.loads((ROOT/'tests/results/josev2b/original-input-freeze.json').read_text())
    def test_actual_original104stages_exact_order_shapes_and_finite_policy(self):
        report=self.report();events=c.original_events(report)
        self.assertEqual(sum(map(len,events.values())),104)
        self.assertEqual([len(events[k]) for k,_ in c.CASES],[26]*4)
        for mutation in ('missing','duplicate','shape','nonfinite','order'):
            bad=copy.deepcopy(report)
            index=next(i for i,r in enumerate(bad['arrays']) if r['stage']=='backbone/layer/0')
            if mutation=='missing':bad['arrays'].pop(index)
            if mutation=='duplicate':bad['arrays'].insert(index,bad['arrays'][index])
            if mutation=='shape':bad['arrays'][index]['shape']=[319,2048]
            if mutation=='nonfinite':bad['arrays'][index]['negative_infinity']=1
            if mutation=='order':bad['arrays'][index],bad['arrays'][index+1]=bad['arrays'][index+1],bad['arrays'][index]
            with self.assertRaises(ValueError,msg=mutation):c.original_events(bad)

    def test_frozen_packet_is_exact_with_no_truncation_or_id_normalization(self):
        raw=(ROOT/'tests/results/josev2b/original-input-freeze.json').read_bytes();self.assertEqual(c.sha(raw),c.FREEZE)
        freeze=json.loads(raw);packet=c.packet(freeze)
        self.assertTrue(packet.startswith(b'JOSBKV1\0'+c.RECEIPT.encode()+c.SOURCE.encode()+c.FREEZE.encode()))
        at=8+64+40+64;self.assertEqual(struct.unpack_from('<I',packet,at)[0],4);at+=4
        for record in freeze['records']:
            length,tokens=struct.unpack_from('<II',packet,at);at+=8
            self.assertEqual(packet[at:at+length].decode(),record['id']);at+=length
            self.assertEqual(list(struct.unpack_from('<'+'I'*tokens,packet,at)),record['encoded']['input_ids']);at+=4*tokens
            self.assertEqual(packet[at:at+tokens],bytes(record['attention_mask']));at+=tokens
        self.assertEqual(at,len(packet))
        for key,value in [('attention_mask',[0]*319),('id','different')]:
            bad=copy.deepcopy(freeze);bad['records'][0][key]=value
            with self.assertRaises(ValueError):c.packet(bad)
        bad=copy.deepcopy(freeze);bad['records'][0]['encoded']['input_ids'][0]=248320
        with self.assertRaises(ValueError):c.packet(bad)

    def test_native_shape_occurrence_and_complete_panel_reject(self):
        # Source descriptors are validated independently of the emitted binary.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);record=dict(id='mixed-de',tokens=319,completed=True,ordinary_repeat_bit_exact=True,logits_empty=True,
                stages=[dict(e,file=f'00/{i:02d}.f32') for i,e in enumerate(c.stages(319))])
            for row in record['stages']:
                p=root/row['file'];p.parent.mkdir(exist_ok=True)
                # Sparse metadata-sized fixture; never instantiate a tensor.
                with p.open('wb') as stream:stream.truncate(4*math.prod(row['shape']))
            self.assertEqual(len(c.validate_native(0,record,root)),26)
            for field,value in [('shape',[319,2048]),('occurrence',1),('file','../outside.f32')]:
                bad=copy.deepcopy(record);bad['stages'][0][field]=value
                with self.assertRaises(ValueError):c.validate_native(0,bad,root)
            for field in ('ordinary_repeat_bit_exact','logits_empty','completed'):
                bad=copy.deepcopy(record);bad[field]=False
                with self.assertRaises(ValueError):c.validate_native(0,bad,root)

    def test_lossless_non_square_adapter_and_numeric_exit_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=struct.pack('<6f',1,0,0,1,1,0);p=root/'a.f32';p.write_bytes(data)
            pin=dict(path=p,shape=[1,3,2],sha256=c.sha(data))
            result=d.write_pair(root/'a.gguf',root/'b.gguf',[dict(name='stage.00',original=pin,native=pin)])
            self.assertEqual(result['original']['tensors']['stage.00']['shape'],[1,3,2])
            self.assertEqual(d.read_gguf(root/'a.gguf')['stage.00']['sha256'],c.sha(data))
            with self.assertRaises(ValueError):d.write_pair(root/'x.gguf',root/'y.gguf',[dict(name='stage.00',original=pin,native={**pin,'shape':[3,2]})])
            with self.assertRaises(ValueError):d.collect(2,root/'absent',{})
            p.write_bytes(data[:-1]+b'\xff')
            with self.assertRaises(ValueError):d.source_bytes(pin)

    def test_collected_metrics_keep_original_batch_axes_and_stage_identity(self):
        events=c.original_events(self.report())['mixed-de']
        native=[dict(e,file=f'00/{i:02d}.f32') for i,e in enumerate(c.stages(319))]
        metrics={f'stage.{i:02d}':dict(shape=e['shape'],passed=i!=3,max_absolute=.003 if i==3 else 0)
                 for i,e in enumerate(events)}
        rows=c.comparison_records(events,native,metrics)
        self.assertEqual(len(rows),26);self.assertEqual(rows[3]['stage'],'backbone/layer/2')
        self.assertEqual(rows[3]['shape'],[1,319,2048]);self.assertFalse(rows[3]['passed'])
        self.assertTrue(rows[4]['passed'])
        bad=copy.deepcopy(metrics);bad['stage.00']['shape']=[319,2048]
        with self.assertRaises(ValueError):c.comparison_records(events,native,bad)

    def test_npy_header_extent_validation_without_numpy(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'owned.npy';header=repr(dict(descr='<f4',fortran_order=False,shape=(1,2,2))).encode()+b'\n'
            data=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<4f',1,2,3,4);p.write_bytes(data)
            row=dict(bytes=len(data),dtype='float32',shape=[1,2,2])
            self.assertEqual(driver.original.array_header(p,row),16)
            with self.assertRaises(ValueError):driver.original.array_header(p,{**row,'shape':[2,2]})
            p.write_bytes(data+b'x')
            with self.assertRaises(ValueError):driver.original.array_header(p,row)
        self.assertTrue(all(n not in sys.modules for n in ('numpy','gguf','torch','transformers')))

    def test_historical_source_guard_and_dispatcher_union(self):
        current=subprocess.check_output(['git','diff','--name-only',c.BASE,'HEAD','--',*c.PROTECTED],cwd=ROOT,text=True)
        self.assertEqual(current,'')
        old=subprocess.check_output(['git','show',c.BASE+':.github/workflows/ovisocr2-metadata.yml'],cwd=ROOT,text=True)
        now=(ROOT/'.github/workflows/ovisocr2-metadata.yml').read_text()
        old_options=old.split('options: [',1)[1].split(']',1)[0].split(', ')
        now_options=now.split('options: [',1)[1].split(']',1)[0].split(', ')
        self.assertEqual(set(now_options)-set(old_options),{'josev-backbone'})
        self.assertTrue(set(old_options)<=set(now_options))
        self.assertEqual(len(now_options),len(set(now_options)))
        self.assertIn('uses: ./.github/workflows/josev2b-backbone.yml',now)

    def test_original_first_barrier_actual_ref_and_cpu_resource_bounds(self):
        source=(ROOT/'tools/ci-heavy/josev2b_backbone.py').read_text();workflow=(ROOT/'.github/workflows/josev2b-backbone.yml').read_text()
        self.assertLess(source.index('retained(scratch)',source.index('def run(')),source.index('import numpy as np',source.index('def run(')))
        self.assertIn('10*2**30',source);self.assertIn('25*2**30',source)
        self.assertIn("compared_stages']!=104",source)
        self.assertIn('timeout-minutes: 240',workflow);self.assertIn('--parallel',source)
        self.assertNotIn('KAGGLE',workflow)
        diff=(ROOT/'tests/josev-backbone/diff.cpp').read_text()
        self.assertIn('original.compare(name, b.first, b.second, 0)',diff)
        self.assertIn('double(r.max_abs) <= 0.002',diff);self.assertIn('double(r.cos_min) >= 0.99999',diff)
        self.assertIn('return failed ? 1 : 0',diff);self.assertIn('core_util::clean_exit',diff)

if __name__=='__main__':unittest.main()
