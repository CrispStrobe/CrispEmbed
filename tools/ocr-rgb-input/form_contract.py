"""Separately versioned form-only source closure; historical813 guard unchanged."""
import hashlib,json,subprocess
from pathlib import Path
import contract as old
ROOT=old.ROOT;BASE='b3ed607d02bbd10ec042777870491bf9f7a334e8'
PATH=ROOT/'docs/ocr-rgb-form-v1.json'
PAGE_ID='en-form-economic-journal-67'
INITIAL_REPORT='76a168944cf3fe51ffef03c9851e42fc047863b786505380861c13b5dbe837d3'
OWNED={'PLAN.md','docs/ocr-rgb-form-v1.json','docs/ocr-rgb-form-v1.md','tools/ocr-rgb-input/form_contract.py',
 'tools/ocr-rgb-input/form_audit.py','tools/ocr-rgb-input/form_provider.py','tools/ocr-real-scans/retained_inputs.py',
 'tools/kaggle/ocr-real-scans/kernel.py','tests/test_ocr_rgb_form.py','.github/workflows/ocr-rgb-form-v1.yml',
 'tests/results/ocr-real-scans/rgb-input-initial.json'}
require=old.require;sha=old.sha;prefix=old.prefix;admit=old.admit;admitted_decode=old.admitted_decode;PROFILE=old.PROFILE

def declaration():
 cfg=json.loads(PATH.read_text())
 require(cfg['schema']=='crispembed-ocr-rgb-form-v1'and cfg['base']==BASE and cfg['page_id']==PAGE_ID and cfg['carried_pages']==19 and not cfg['model_execution']and not cfg['cause_established'],'Exact form-only scope required')
 return cfg

def source(header=False):
 cfg=declaration();baseline=old.validate(json.loads(old.PATH.read_text()));git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT)
 require(git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'Dirty form protocol')
 require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED,'Undeclared form source delta')
 required=(set(baseline['source_sha256'])-{'tools/kaggle/ocr-real-scans/kernel.py'})|{'docs/ocr-rgb-input-v1.json','tools/kaggle/ocr-real-scans/receipt.py'}
 require(set(cfg['protected_sha256'])==required and set(cfg['owned_sha256'])==OWNED-{'PLAN.md',str(PATH.relative_to(ROOT))},'Exact protected/owned closure required')
 for group in ['protected_sha256','owned_sha256']:
  for name,pin in cfg[group].items():
   p=ROOT/name;require(p.is_file()and not p.is_symlink()and sha(p)==pin and git('show','HEAD:'+name)==p.read_bytes(),'Form source differs: '+name)
   if group=='protected_sha256':require(git('show',BASE+':'+name)==p.read_bytes(),'Frozen813 dependency changed: '+name)
 require(git('rev-parse','HEAD:src')==git('rev-parse',BASE+':src')and git('rev-parse','HEAD:ggml').decode().strip()==old.GGML,'Production runtime/STB unchanged required')
 require(not git('status','--porcelain','--','src','ggml','CMakeLists.txt'),'Dirty runtime/decoder')
 require(sha(ROOT/'docs/ocr-real-scans-v1.json')==old.MANIFEST['sha256'],'Frozen manifest changed')
 if header:
  p=ROOT/'ggml/examples/stb_image.h';require(p.stat().st_size==baseline['stb_header']['bytes']and sha(p)==baseline['stb_header']['sha256'],'Exact production STB header required')
 return baseline

def verify_provider(directory,proof):
 cfg=declaration();pin=cfg['retained_original'];image=directory/(PAGE_ID+'.jpg')
 require(proof['schema']=='crispembed-ocr-rgb-form-provider-v1'and proof['completed']is True and proof['model_execution']is False and
  proof['original']['source']==pin['source']and proof['original']['kernel_slug']==pin['kernel_slug']and proof['original']['version_label']=='v1'and proof['original']['status']=='COMPLETE'and proof['original']['submitted_script_bytes']==pin['submitted_script_bytes']and proof['original']['submitted_script_sha256']==pin['submitted_script_sha256']and
  proof['original']['actual_sdk_serialized_version_on_get_status_and_all_pages']is True and proof['initial_report_sha256']==INITIAL_REPORT and
  proof['original_report_sha256']==pin['report_sha256'],'Complete retained producer admission required')
 require(image.is_file()and not image.is_symlink()and image.stat().st_size==pin['image']['bytes']and sha(image)==pin['image']['sha256'],'Retained form identity differs beforedecode')
