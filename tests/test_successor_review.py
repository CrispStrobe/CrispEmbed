import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ci-heavy'))
import successor_review as review

class ReviewControls(unittest.TestCase):
 def fixture(self,root):
  run='123';audit='b'*40
  base={'source_commit':'a'*40,'consumer_source':audit,'runtime_source_closure':{'objects':['fixture source object']}}
  def save(artifact,name,data):
   path=root/(artifact+'-'+run)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
  metric={'passed':True,'max_abs':0.0,'cosine':1.0}
  for platform in review.DESKTOPS:
   data=dict(base,desktop_bounded_consumer_positive=True,empty_cache_download=True,empty_cache_offline_rejected=True,
    offline_reuse=True,comparisons={k:metric for k in ('cli_to_cabi','offline_cli','python_to_cabi','compiled_C_to_cabi')},
    vector=[1]+[0]*383,compiled_hparams_size=40,abi_exports=['crispembed_init','crispembed_encode','crispembed_free','crispembed_resolve_model','crispembed_set_offline'])
   data.update({k:'c'*64 for k in ('library_sha256','cli_binary_sha256','compiled_c_consumer_sha256','model_sha256')})
   save('successor-consumer-'+platform,'consumer.json',data)
  for platform in review.MOBILES:
   save('successor-mobile-'+platform,'consumer.json',dict(base,mobile_abi=platform,cross_link_passed=True,cross_link_exit_code=0,device_execution_performed=False))
  save('successor-wasm-consumer','browser.json',dict(base,page_errors=[],ocr={'authored_black_blob_cleared_rgb_exact':True},embed={'vector':[1]+[0]*383},max_abs=0,cosine=1))
  apple={'cases':50,'token_ids_exact':True,'text_exact':True,'stops_exact':True}
  save('successor-apple-released','comparison.json',dict(base,candidate_unset_one_exact=True,legacy_zero_preserved=True,comparisons={'candidate-unset':apple,'candidate-one':apple}))
  for name in ('released','candidate-unset','candidate-one','candidate-zero'):
   save('successor-apple-released',name+'.json',{'completed':True,'cases':[{}]*50})
  binding={'passed':True,'comparisons':[metric]*12}
  save('successor-prebuilt-bindings','successor-bindings.json',dict(audit_source=audit,passed=True,producer=base,native_build_fallback=False,offline_cache_unchanged=True,surfaces={'rust':binding,'dart':binding}))
  return run,audit
 def test_complete_scoped_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root);result=review.review(root,run,audit)
   self.assertTrue(result['desktop_browser_apple50_and_linux_bindings_positive'])
   self.assertFalse(result['mobile_device_execution_verified'])
 def test_missing_platform_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root)
   (root/('successor-mobile-ios-arm64-'+run)/'consumer.json').unlink()
   with self.assertRaises(FileNotFoundError):review.review(root,run,audit)
 def test_mixed_producer_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root);path=root/('successor-consumer-macos-arm64-'+run)/'consumer.json'
   data=json.loads(path.read_text());data['source_commit']='d'*40;path.write_text(json.dumps(data))
   with self.assertRaises(ValueError):review.review(root,run,audit)
if __name__=='__main__':unittest.main()
