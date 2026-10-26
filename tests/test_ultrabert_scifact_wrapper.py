"""Small source/config and all-row wrapper-preflight guards; no frameworks."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('scifact_wrapper_driver', ROOT/'tools/ci-heavy/ultrabert_scifact.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
CONFIG = b'{\n    "transformer_task": "feature-extraction",\n    "modality_config": {\n        "text": {\n            "method": "forward",\n            "method_output_name": "last_hidden_state"\n        }\n    },\n    "module_output_name": "token_embeddings"\n}'

class Tensor:
 def __init__(self, values): self.values=values
 def tolist(self): return self.values

class Wrapper:
 def __init__(self,bad=False): self.calls=[];self.bad=bad
 def result(self,text):
  return {'input_ids':Tensor([[0,43 if self.bad and text==' document-last ' else 42]]),
          'attention_mask':Tensor([[1,1]])}
 def preprocess(self,texts,prompt): self.calls.append(('prompt',texts[0],prompt));return self.result(texts[0])
 def tokenize(self,texts): self.calls.append(('literal',texts[0],None));return self.result(texts[0])
 def forward(self,*args,**kwargs): raise AssertionError('A token preflight must not run a model')

class Controls(unittest.TestCase):
 def fixture(self):
  queries=[{'id':'q:'+str(i),'text':' query '+str(i)+' '} for i in range(300)]
  docs=[{'id':'d:'+str(i),'text':' document '+str(i)+' '} for i in range(5183)]
  docs[-1]['text']=' document-last '
  data={'queries':queries,'documents':docs}
  tokens=[dict(id=r['id'],ids=[0,42],attention_mask=[1,1]) for r in queries+docs]
  return data,tokens
 def test_published_module_bytes_required(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'sentence_bert_config.json'
   with self.assertRaises(ValueError):driver.module_config(d)
   p.write_bytes(CONFIG)
   self.assertEqual(driver.module_config(d)['module_output_name'],'token_embeddings')
   p.write_bytes(CONFIG.replace(b'feature-extraction',b'feature-extractioN'))
   with self.assertRaises(ValueError):driver.module_config(d)
 def test_all_rows_and_both_literal_prompt_paths_before_forward(self):
  data,tokens=self.fixture();model=Wrapper();report=driver.wrapper_token_preflight(model,data,tokens)
  self.assertEqual(report['rows'],5483);self.assertEqual(len(model.calls),10966)
  self.assertTrue(report['before_first_forward']);self.assertTrue(report['both_prompt_paths_checked'])
  self.assertEqual(model.calls[0],('prompt',' query 0 ',driver.QUERY))
  self.assertEqual(model.calls[1],('literal',driver.QUERY+' query 0 ',None))
  self.assertEqual(model.calls[-1],('literal',' document-last ',None))
 def test_last_row_mismatch_has_actual_ids_and_does_not_get_dropped(self):
  data,tokens=self.fixture()
  with self.assertRaises(driver.WrapperTokenMismatch) as caught:driver.wrapper_token_preflight(Wrapper(True),data,tokens)
  self.assertEqual(caught.exception.details['id'],'d:5182')
  self.assertEqual(caught.exception.details['actual_ids'],[[0,43]])
 def test_incomplete_rows_reject_before_wrapper_calls(self):
  data,tokens=self.fixture();tokens.pop();model=Wrapper()
  with self.assertRaises(ValueError):driver.wrapper_token_preflight(model,data,tokens)
  self.assertEqual(model.calls,[])
 def test_unreviewed_provenance_service_rejects_before_artifact_download(self):
  pin=driver.protocol()['data_provenance_review']
  good=dict(path=pin['workflow'],head_sha=pin['source_commit'],status='completed',conclusion='failure',event='workflow_dispatch')
  for key,value in [('path','wrong'),('head_sha','0'*40),('status','in_progress'),('conclusion','success'),('event','push')]:
   with self.subTest(field=key),tempfile.TemporaryDirectory() as d:
    root=Path(d)
    with patch.object(driver,'api',return_value=dict(good,**{key:value})),patch.object(driver.subprocess,'run') as download:
     with self.assertRaises(ValueError):driver.reviewed_data_provenance(root,root,root)
     download.assert_not_called()
if __name__=='__main__':unittest.main()
