"""Mac-only retry source/guard controls: no build, corpus or model access."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
BASE='20e58588f2c357b8156f94f61b38d8158b906e10'
WORKFLOW=ROOT/'.github/workflows/successor-apple-only.yml'


def job(text,name):
 start=text.index('  '+name+':\n');match=re.search(r'^  [A-Za-z0-9_-]+:\n',text[start+len('  '+name+':\n'):],re.M)
 return text[start:] if match is None else text[start:start+len('  '+name+':\n')+match.start()]


class AppleOnlyControls(unittest.TestCase):
 def test_one_mac_job_and_unchanged_probe_body(self):
  original=job((ROOT/'.github/workflows/successor-assets.yml').read_text(),'apple-released')
  actual=job(WORKFLOW.read_text(),'apple-only')
  self.assertEqual(re.findall(r'^  ([\w-]+):$',WORKFLOW.read_text().split('jobs:\n',1)[1],re.M),['apple-only'])
  actual=actual.replace('  apple-only:\n','  apple-released:\n',1)
  actual=actual.replace('    runs-on: macos-15\n','    needs: producer-identity\n    runs-on: macos-15\n',1)
  start=actual.index('      - name: Lightweight Mac-only scope controls\n');end=actual.index('      - name: Actual candidate archive and complete member checks\n',start)
  actual=actual[:start]+actual[end:]
  actual=actual.replace('name: successor-apple-only-${{ github.run_id }}','name: successor-apple-released-${{ github.run_id }}',1)
  actual=actual.replace('            ${{ runner.temp }}/apple-only-scope.json\n','',1)
  self.assertEqual(actual,original)
 def test_metadata_only_adds_choice_and_scoped_forwarder_with_ten_inputs(self):
  path='.github/workflows/ovisocr2-metadata.yml'
  before=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT,text=True)
  after=(ROOT/path).read_text()
  dispatch=after.split('  workflow_dispatch:\n',1)[1].split('  push:\n',1)[0]
  self.assertEqual(len(re.findall(r'^      [\w-]+:$',dispatch,re.M)),10)
  scoped=job(after,'successor-apple-only')
  self.assertIn("inputs.phase == 'successor-apple-only'",scoped)
  self.assertIn('native_run: ${{ inputs.native_run_id }}',scoped)
  self.assertIn('corpus_run: ${{ inputs.candidate_run_id }}',scoped)
  self.assertEqual(after.replace(scoped,'',1).replace('successor-consumer, successor-apple-only,','successor-consumer,',1),before)
 def guard(self,native,corpus):
  text=WORKFLOW.read_text();start=text.index("          python - <<'PYCODE'\n")+len("          python - <<'PYCODE'\n");end=text.index('          PYCODE\n',start)
  code='\n'.join(line[10:] for line in text[start:end].splitlines())
  with tempfile.TemporaryDirectory() as directory:
   env=dict(RUNNER_TEMP=directory,GITHUB_SHA='a'*40,NATIVE_RUN=native,CORPUS_RUN=corpus)
   with mock.patch.dict(os.environ,env):
    error=None
    try:exec(compile(code,'workflow-input-guard','exec'),{})
    except ValueError as e:error=e
   receipt=json.loads((Path(directory)/'apple-only-scope.json').read_text())
   return error,receipt
 def test_exact_inputs_are_diagnostic_only(self):
  error,receipt=self.guard('40339796226','40386968045')
  self.assertIsNone(error);self.assertTrue(receipt['diagnostic_mac_only'])
  for key in ('aggregate_performed','release_qualified','final_integrated_source_validated'):self.assertFalse(receipt[key])
 def test_wrong_inputs_keep_rejection_receipt(self):
  for native,corpus in (('0','40386968045'),('40339796226','0'),('latest','40386968045'),('40339796226','40376735641')):
   with self.subTest(native=native,corpus=corpus):
    error,receipt=self.guard(native,corpus)
    self.assertIsInstance(error,ValueError);self.assertEqual(receipt['status'],'rejected')
    self.assertFalse(receipt['release_qualified'])
 def test_full_consumer_and_production_inputs_unchanged(self):
  paths=('src','ggml','cmake','CMakeLists.txt','VERSION','Cargo.lock','examples','python','flutter',
   'crispembed','crispembed-sys','wasm','tools/quantize.cpp','build-wasm.sh','build-embed-wasm.sh',
   'scripts/check-cpu-baseline.py','scripts/check-bundled-deps.py','tools/successor_asset_manifest.py',
   '.gitattributes','.github/workflows/release.yml','.github/workflows/release-wasm.yml','README.md','LICENSE',
   '.github/workflows/successor-assets.yml','tools/ci-heavy/successor_assets.py','tools/ci-heavy/successor_posformer.py',
   'tools/ci-heavy/successor_posformer_corpus.py','tools/probe_posformer_release.py')
  changed=subprocess.check_output(['git','diff',BASE,'--name-only','--',*paths],cwd=ROOT,text=True)
  self.assertEqual(changed,'')
 def test_failure_artifact_remains_always_and_contains_all_probe_reports(self):
  text=job(WORKFLOW.read_text(),'apple-only');upload=text.split('      - uses: actions/upload-artifact@v4\n',1)[1]
  self.assertIn('if: always()',upload)
  for name in ('apple-only-scope.json','apple-candidate/consumer.json','apple-corpus-receipt.json','apple-released/*.json','apple-released/*-logs/'):
   self.assertIn(name,upload)
if __name__=='__main__':unittest.main()
