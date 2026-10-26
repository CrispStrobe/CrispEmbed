#!/usr/bin/env python3
"""Compare actual released/candidate Apple archives on the frozen fifty cases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from successor_asset_manifest import sha, archive_members
from probe_posformer_release import compare_release_records
BASE='78493a31fc3043f3af4ab6fd2f197a32da6f07f0'

def run(candidate,app,out):
 if os.environ.get('GITHUB_ACTIONS')!='true':raise ValueError('Hosted only')
 out.mkdir(parents=True,exist_ok=False)
 receipt=json.loads((ROOT/'tests/results/posformer/released-compatibility.json').read_text())
 old=next(row for row in receipt['released_asset_manifest']['files'] if row['path']=='macos-arm64/crispembed-macos-arm64.tar.gz')
 subprocess.run(['gh','release','download','v0.17.12','--repo','CrispStrobe/CrispEmbed','--pattern','crispembed-macos-arm64.tar.gz','--dir',str(out)],check=True,timeout=600)
 archive=out/'crispembed-macos-arm64.tar.gz'
 if archive.stat().st_size!=old['bytes'] or sha(archive)!=old['sha256']:raise ValueError('Released Apple archive differs from frozen manifest')
 archive_members(archive)  # Validate normalized paths/links before extraction.
 baseline=out/'released';baseline.mkdir()
 with tarfile.open(archive) as package:package.extractall(baseline,filter='data')
 candidate_source=json.loads((candidate.parent/'consumer.json').read_text())['source_commit']
 results={}
 env={k:v for k,v in os.environ.items() if k not in ('CRISPEMBED_POSFORMER_TRAINED_FORWARD','POSFORMER_SCALAR_ENCODER')}
 for name,library,source,gate in (
   ('released',baseline/'libcrispembed.dylib',BASE,None),
   ('candidate-unset',candidate/'libcrispembed.dylib',candidate_source,None),
   ('candidate-one',candidate/'libcrispembed.dylib',candidate_source,'1'),
   ('candidate-zero',candidate/'libcrispembed.dylib',candidate_source,'0')):
  actual_env=dict(env)
  if gate is not None:actual_env['CRISPEMBED_POSFORMER_TRAINED_FORWARD']=gate
  subprocess.run([sys.executable,ROOT/'tools/probe_posformer_release.py','--library',library,'--library-source',source,
   '--model',app/'posformer-q8.gguf','--corpus',app/'.dart_tool/inference/handwriting-corpus',
   '--frozen-report',ROOT/'tests/results/posformer/main-backport-reference.json','--output',out/(name+'.json'),
   '--logs',out/(name+'-logs')],env=actual_env,check=True,timeout=2700)
  results[name]=json.loads((out/(name+'.json')).read_text())
 summary={'schema':'crispembed-successor-apple-released-comparison-v1','source_commit':candidate_source,'consumer_source':os.environ['GITHUB_SHA'],
   'released_archive_sha256':old['sha256'],'comparisons':{},'legacy_zero_preserved':True,
   'recognition_quality_improvement_claimed':False,'limits':['Fifty frozen MathWriting cases; the prior recognition score is7/50.',
    'Confidence drift is measured without a new tolerance. Legacy zero is retained as a separate diagnostic.']}
 for name in ('candidate-unset','candidate-one'):summary['comparisons'][name]=compare_release_records(results['released'],results[name])
 keys=('id','image_sha256','text','token_ids','stop','confidence','mean_confidence')
 if any(any(a[k]!=b[k] for k in keys) for a,b in zip(results['candidate-unset']['cases'],results['candidate-one']['cases'])):
  raise ValueError('Candidate unset/literal1 differ')
 summary['candidate_unset_one_exact']=True
 (out/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--app',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.candidate,a.app,a.output)
