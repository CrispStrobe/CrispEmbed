import ast,copy,json,unittest
from pathlib import Path
import worker
class Controls(unittest.TestCase):
 def test_immutable_scope(self):
  cfg=json.loads((Path(__file__).parent/'protocol.json').read_text());worker.admit(cfg)
  for name,value in [('worker_seconds',0),('heldout_allowed',True),('model_forward_allowed',True),('prompt_tokens',155)]:
   bad=copy.deepcopy(cfg);bad[name]=value
   with self.assertRaises(RuntimeError):worker.admit(bad)
 def test_exact_startup_exclusion_and_actual_phases(self):
  probe={'graph':0,'tokens':[0,0],'positions_4axes':[[0,1]]*3+[[0,0]]};trace={'analysis':{'excluded_nonprompt_graphs':[probe],'tensor_metadata':[{'index':i,'sample_count':0,'tokens':n}for i,n in enumerate([2,4,140,6,4])]+[{'index':5,'sample_count':31,'tokens':1}]}}
  graphs=worker.source_graphs(trace);self.assertEqual([g['tokens']for g in graphs],[4,140,6,4]);self.assertEqual(sum(g['tokens']for g in graphs),154)
  bad=copy.deepcopy(trace);bad['analysis']['excluded_nonprompt_graphs'][0]['positions_4axes']=[[0,1]]*4
  with self.assertRaises(RuntimeError):worker.source_graphs(bad)
 def test_unchanged_math_and_local_paths(self):
  cfg=json.loads((Path(__file__).parent/'protocol.json').read_text())
  for name,digest in cfg['production_file_sha256'].items():self.assertEqual(worker.sha(worker.ROOT/name),digest,name)
  self.assertEqual(cfg['matrix']['type'],14);self.assertEqual(cfg['matrix']['bytes'],17203200)
 def test_actual_compiled_geometry_contract(self):
  source=(worker.ROOT/'src/lighton4b_source_prefill_chunks.h').read_text();self.assertIn('tokens<=512-past',source);self.assertIn('left-count<8',source);self.assertIn('end-start>4',source)
  native=(worker.ROOT/'src/lighton4b_native_context.h').read_text();self.assertIn('lighton4b_source_prefill_budget(tokens,previous?previous->tokens:0)',native);self.assertIn('lighton4b_prefill_chunk_positions',native)
  driver=(worker.ROOT/'tests/lighton4b_native_document.cpp').read_text();self.assertIn('initial, &mods',driver);self.assertIn('const int count = chunk_sizes.front()',driver)
 def test_worker_cancellation_and_admission(self):
  text=Path(__file__).with_name('worker.py').read_text();ast.parse(text);self.assertNotIn('except Exception',text);self.assertIn('signal.SIGTERM,signal.SIGINT',text);self.assertIn('physical.memory()',text);self.assertLess(text.index("run('build'"),text.index('base.download_range'))
if __name__=='__main__':unittest.main()
