import ast,copy,json,tempfile,unittest,zipfile
from pathlib import Path
import contracts as c
class Controls(unittest.TestCase):
 def test_frozen_selection_and_negative_mutations(self):
  cfg=json.loads((c.ROOT/'docs/lighton4b-q6-heldout-v1.json').read_text());c.admit(cfg)
  for key,value in [('cases',list(reversed(cfg['cases']))),('cap',512),('context',4096),('threads',2),('worker_seconds',0),('qualified',True),('reference_forward_allowed',True),('model_sha256','0'*64),('changed_gates',[])]:
   bad=copy.deepcopy(cfg);bad[key]=value
   with self.assertRaises(ValueError):c.admit(bad)
 def test_exact_three_gate_comparison(self):
  old={k:'0'for k in c.CHANGED};old['CRISPEMBED_LIGHTON4B_HEAD_BF16']='0';new=dict(old);new.update({k:'1'for k in c.CHANGED});a,b=c.gates(old,new);self.assertEqual({k for k in a if a[k]!=b[k]},c.CHANGED)
  for key,value in [('CRISPEMBED_LIGHTON4B_HEAD_BF16','1'),('CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED','0')]:
   bad=dict(new);bad[key]=value
   with self.assertRaises(ValueError):c.gates(old,bad)
 def test_whole_historical_archive_identity(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'greedy-raw-rows.zip'
   with zipfile.ZipFile(p,'w')as z:z.writestr('greedy.0.logits.f32',b'\0'*(248320*4))
   arm={'raw_rows':{'archive':{'file':p.name,'bytes':p.stat().st_size,'sha256':c.sha(p)},'rows':[{'bytes':248320*4,'sha256':'0'*64}]}}
   self.assertEqual(c.archive_closed(Path(d),arm)['rows'],1)
   with p.open('ab')as f:f.write(b'changed')
   with self.assertRaises(ValueError):c.archive_closed(Path(d),arm)
 def test_real_worker_syntax_and_deadline_catches(self):
  tree=ast.parse(Path(__file__).with_name('worker.py').read_text());self.assertTrue(tree.body)
  # Worker delegates transfers to the earlier independently tested finite retry
  # helper; no broad inner handler may convert cancellation into success.
  text=Path(__file__).with_name('worker.py').read_text();self.assertNotIn('except Exception',text);self.assertIn("if opt.controls_only:r['status']='passed_model_free_q6_heldout_controls';return",text)
 def test_native_complete_rejects_partial_and_f32(self):
  s=dict(generated_tokens=1,eos=True,provisional_heldout_cap=1536,source_fed_vision=False,teacher_positions=1,converted_f32_weight_entries=0,converted_f32_weight_bytes=0);c.native_complete(s,[248046],False)
  for key,value in [('generated_tokens',2),('eos',False),('provisional_heldout_cap',512),('converted_f32_weight_bytes',4),('source_fed_vision',True)]:
   bad=dict(s);bad[key]=value
   with self.assertRaises(ValueError):c.native_complete(bad,[248046],False)
 def test_actual_shared_position_exporter(self):
  text=(c.ROOT/'tests/lighton4b_native_case_preflight.cpp').read_text()
  self.assertIn('export_positions(mods, image.grid, out)',text)
  self.assertIn('export_positions({0,1,1,1,1,1,1,0}, {1,4,6}, argv[2])',text)
  self.assertIn('std::memcmp(bytes.data(), expected, sizeof(expected))',text)
  worker=Path(__file__).with_name('worker.py').read_text()
  self.assertLess(worker.index("run('position-export-control'"),worker.index('if opt.controls_only:'))
  self.assertIn("position_env['CRISPEMBED_QWEN35_OCR']='1'",worker)
 def test_reference_cli_is_explicit_and_bounded(self):
  args=c.ex.native_args('binary','model','rgb',32,32,'teacher','out','native-composed');self.assertEqual(args[7:],[1536,4,'greedy-full','--provisional-heldout-cap1536'])
  source=c.ex.native_args('binary','model','rgb',32,32,'teacher','out','native-source-features','feature');self.assertEqual(len(source),13)
  text=(c.ROOT/'tests/lighton4b_native_document.cpp').read_text();self.assertIn('ids.size() > 512',text);self.assertIn('cap != 1536 || threads != 4 || mode != "greedy-full"',text);self.assertIn('ids.size() + size_t(cap) > 2048',text)
if __name__=='__main__':unittest.main()
