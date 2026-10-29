"""Honest Kaggle CPU entry point after a reviewed source checkout/Python3.12 venv.

The operator owns private kernel metadata and the credential-only mounted input.
No model/software dataset, SDK import, GPU lottery or GITHUB_ACTIONS spoofing.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,atomic,hosted,module,require,sha,verify


def main(a):
    cfg=verify(a.reviewed_source);provider=hosted(a.reviewed_source)
    require(provider.startswith('Kaggle CPU')and os.environ.get('GITHUB_ACTIONS')is None,'genuine Kaggle adapter only')
    require(sys.prefix!=sys.base_prefix and sys.version_info[:2]==(3,12),'isolated CPython3.12 venv required before setup')
    require(not any(k in sys.modules for k in ('torch','numpy','transformers')),'no framework import before retained-byte authentication')
    a.out.mkdir(parents=True,exist_ok=False);report=dict(source=a.reviewed_source,completed=False,provider=provider,
        external_exact_version_service_proof_required=True,original_full_corpus_recaptured=False,commands={})
    save=lambda:atomic(a.out/'kernel.json',report);save()
    sys.path.insert(0,str(ROOT/'tools'));from ci_process import run
    def command(name,argv,timeout):
        log=a.out/(name+'.log');status=run(argv,log,cwd=ROOT,env=os.environ.copy(),timeout=timeout)
        report['commands'][name]=dict(status,log=log.name,sha256=sha(log));save()
        require(not status['timed_out']and status['exit']==0,'bounded setup failed: '+name)
    try:
        prepare=module(ROOT/'tools/moedern-first-row/prepare.py','first_row_cpu_retained')
        prepare.run(a.reviewed_source,a.out/'input')
        command('torch',[sys.executable,'-m','pip','install','torch==2.8.0','--index-url','https://download.pytorch.org/whl/cpu'],600)
        command('runtime',[sys.executable,'-m','pip','install','numpy==2.2.6','transformers==5.3.0',
            'sentence-transformers @ git+https://github.com/huggingface/sentence-transformers.git@b0bf39968ba461c34b29372144794eff1be06f6d','gguf==0.17.1'],600)
        command('pip-check',[sys.executable,'-m','pip','check'],60)
        log=a.out/'capture.log';status=run([sys.executable,'-I',ROOT/'tools/ci-heavy/moedern_first_row_v1.py','--reviewed-source',a.reviewed_source,
            '--input',a.out/'input','--scratch',a.out/'build','--out',a.out/'output'],log,cwd=ROOT,env=os.environ.copy(),timeout=3600)
        report['commands']['capture']=dict(status,log=log.name,sha256=sha(log));save()
        require(not status['timed_out']and status['exit']in(0,1),'structural/timeout rejection')
        captured=__import__('json').loads((a.out/'output/review/hosted.json').read_text())
        require(captured['completed']is True,'bounded actual capture incomplete')
        report.update(completed=True,numeric_exit=status['exit'],publication_qualified=False);save();return status['exit']
    except Exception as error:report['error']=type(error).__name__+': '+str(error);save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True);p.add_argument('--out',type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
