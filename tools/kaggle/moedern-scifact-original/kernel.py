"""Hosted CPU original-only retrieval capture; render @COMMIT@ exactly.
One separately rendered variant per kernel; no native/Q4 execution.
"""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
COMMIT='@COMMIT@'
VARIANT='@VARIANT@'
OUT=Path('/kaggle/working/moedern-scifact-original');OUT.mkdir(exist_ok=False)
SCRATCH=Path('/kaggle/temp/moedern-scifact-original');SCRATCH.mkdir(parents=True,exist_ok=False)
report=dict(source=COMMIT,completed=False,commands=[],original_only=True)
def save():(OUT/'setup.json').write_text(json.dumps(report,indent=2)+'\n')
def command(args,timeout=1800):
    log=OUT/('command-'+str(len(report['commands']))+'.log')
    with log.open('wb')as f:
        p=subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    report['commands'].append(dict(argv=list(map(str,args)),exit=p.returncode,log=log.name,sha256=hashlib.sha256(log.read_bytes()).hexdigest()));save()
    if p.returncode:raise RuntimeError('original capture command failed: '+log.name)
try:
    assert VARIANT in ['1.5M','27M','115M']
    assert len(COMMIT)==40 and all(c in '0123456789abcdef'for c in COMMIT)
    available=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
    report['resources']=dict(mem_available=available,disk_free=shutil.disk_usage(SCRATCH).free,cpu_count=os.cpu_count());save()
    assert available>=10*1024**3 and report['resources']['disk_free']>=18*1024**3,'remote resource guard'
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
    os.environ.update(TOKENIZERS_PARALLELISM='false',CRISPEMBED_CAPTURE_SOURCE=COMMIT)
    checkout=SCRATCH/'source';command(['git','clone','-q','--filter=blob:none','https://github.com/CrispStrobe/CrispEmbed.git',checkout]);os.chdir(checkout);command(['git','checkout','-q',COMMIT])
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==COMMIT
    command([sys.executable,'-m','pip','install','-q','uv==0.8.17'])
    command([sys.executable,'-m','uv','venv','-q','--python','3.12','--seed',SCRATCH/'venv']);py=SCRATCH/'venv/bin/python'
    command([py,'tests/test_moedern_scifact_parity.py','-v'])
    command([py,'-m','pip','install','-q','torch==2.8.0','--index-url','https://download.pytorch.org/whl/cpu'])
    command([py,'-m','pip','install','-q','numpy==2.2.6','pyarrow==21.0.0','transformers==5.3.0','sentence-transformers @ git+https://github.com/huggingface/sentence-transformers.git@b0bf39968ba461c34b29372144794eff1be06f6d'])
    command([py,'-m','pip','freeze'])
    command([py,'tools/moedern-colbert/scifact_original.py','--variant',VARIANT,'--output',OUT/VARIANT,'--scratch',SCRATCH/VARIANT],timeout=9*3600)
    report['completed']=True;save()
except Exception as e:report['error']=str(e);save();raise
