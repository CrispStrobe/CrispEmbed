#!/usr/bin/env python3
"""Synthetic BF16 capture plumbing; no checkpoint or original model forward."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bf16_capture',ROOT/'tools/lighton4b_bf16_freegen.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

class Tensor:
    def __init__(self,value):self.value=np.asarray(value,dtype=np.float32)
    def detach(self):return self
    def cpu(self):return self
    def float(self):return self
    def contiguous(self):return self
    def numpy(self):return self.value

class Controls(unittest.TestCase):
    def test_full_vocabulary_scores_ids_hashes_and_read_only_source(self):
        with tempfile.TemporaryDirectory()as folder:
            a=np.zeros((1,248320),dtype=np.float32);a[0,42]=2
            b=np.zeros_like(a);b[0,248046]=1;before=a.tobytes();events=[]
            rows=driver.score_files([Tensor(a),Tensor(b)],[42,248046],Path(folder),events.append)
            self.assertEqual(len(events),2);self.assertEqual(sum(r['bytes']for r in rows),2*248320*4)
            self.assertEqual(a.tobytes(),before)
            for row in rows:self.assertEqual(driver.digest(Path(folder)/row['file']),row['sha256'])
            self.assertTrue(all(0<r['selected_probability']<=1 for r in rows))
    def test_partial_score_receipt_survives_second_position_failure(self):
        with tempfile.TemporaryDirectory()as folder:
            good=np.zeros((1,248320),dtype=np.float32);good[0,1]=1
            bad=good.copy();bad[0,0]=np.nan;events=[]
            with self.assertRaises(ValueError):driver.score_files([Tensor(good),Tensor(bad)],[1,1],Path(folder),events.append)
            self.assertEqual(len(events),1);self.assertTrue((Path(folder)/events[0]['file']).is_file())
    def test_wrong_shape_id_or_position_count_reject(self):
        with tempfile.TemporaryDirectory()as folder:
            for scores,ids in [([Tensor([[1.]])],[0]),([Tensor(np.zeros((1,248320)))],[1]),([],[])]:
                with self.assertRaises(ValueError):driver.score_files(scores,ids,Path(folder))
    def test_cgroup_v2_and_v1_limits_cache_reclaim_and_peak(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);proc=root/'proc';cg=root/'cg';(proc/'self').mkdir(parents=True);cg.mkdir()
            (proc/'meminfo').write_text('MemAvailable: 33554432 kB\n');(proc/'self/status').write_text('VmRSS: 1000 kB\nVmHWM: 2000 kB\n')
            (cg/'memory.max').write_text(str(32*2**30));(cg/'memory.current').write_text(str(12*2**30))
            (cg/'memory.stat').write_text('inactive_file '+str(8*2**30)+'\nfile_dirty '+str(2**30)+'\n')
            (cg/'memory.peak').write_text(str(15*2**30))
            r=driver.memory(proc,cg);self.assertEqual(r['available_bytes'],27*2**30);self.assertEqual(r['worker_cgroup_peak_bytes'],15*2**30)
            (cg/'memory.max').unlink();(cg/'memory').mkdir();(cg/'memory/memory.limit_in_bytes').write_text(str(16*2**30))
            (cg/'memory/memory.usage_in_bytes').write_text(str(4*2**30));r=driver.memory(proc,cg);self.assertEqual(r['available_bytes'],12*2**30)
    def test_distinct_protocol_and_all_f32_guards_unchanged(self):
        p=json.loads(driver.PROTOCOL.read_text());self.assertEqual(p['precision'],'bfloat16');self.assertEqual(p['case']['max_new_tokens'],256)
        self.assertEqual(p['guards']['max_score_bytes'],254279680);self.assertEqual(p['guards']['minimum_available_ram_bytes'],22*2**30)
        self.assertNotIn('torch',driver.__dict__)
        source=(ROOT/'tools/lighton4b_original_document.py').read_text();self.assertIn('CAPTURE_TRANSPORT_ADMITTED=False',source)
        self.assertIn('available>=40*2**30',source)

if __name__=='__main__':unittest.main()
