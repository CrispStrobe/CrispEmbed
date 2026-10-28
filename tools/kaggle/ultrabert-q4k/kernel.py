"""CPU qualification against frozen SciFact, no original model execution.
Render @COMMIT@ exactly. Hosted credentials are input-only and never printed.
"""
import glob,hashlib,json,os,shutil,signal,subprocess,sys
from pathlib import Path
COMMIT='@COMMIT@'
OUT=Path('/kaggle/working/ultrabert-q4k');OUT.mkdir(exist_ok=False)
SCRATCH=Path('/kaggle/temp/ultrabert-q4k');SCRATCH.mkdir(parents=True,exist_ok=False)
report=dict(source=COMMIT,completed=False,provider='Kaggle CPU',commands=[])
def save():(OUT/'setup.json').write_text(json.dumps(report,indent=2)+'\n')
def command(args,timeout=1800):
    log=OUT/('command-'+str(len(report['commands']))+'.log')
    with log.open('wb')as stream:
        p=subprocess.Popen(list(map(str,args)),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            report['timed_out_command']=len(report['commands'])
    report['commands'].append(dict(argv=list(map(str,args)),exit=p.returncode,log=log.name,sha256=hashlib.sha256(log.read_bytes()).hexdigest()));save()
    if p.returncode:raise RuntimeError('remote command failed: '+log.name)
try:
    assert len(COMMIT)==40 and all(c in '0123456789abcdef'for c in COMMIT)
    mem=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
    report['resources']=dict(mem_available=mem,disk_free=shutil.disk_usage(SCRATCH).free,cpu_count=os.cpu_count());save()
    assert mem>=10*1024**3 and report['resources']['disk_free']>=12*1024**3,'remote resources below bound'
    tokens=glob.glob('/kaggle/input/**/gh_token.txt',recursive=True);assert len(tokens)==1,'one authorized original-artifact credential required'
    os.environ['GH_TOKEN']=Path(tokens[0]).read_text().strip()
    os.environ['CRISPEMBED_CAPTURE_SOURCE']=COMMIT
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
    repo=SCRATCH/'source';command(['git','clone','-q','--filter=blob:none','https://github.com/CrispStrobe/CrispEmbed.git',repo]);os.chdir(repo);command(['git','checkout','-q',COMMIT]);command(['git','submodule','update','-q','--init','--depth','1','ggml'])
    command([sys.executable,'-m','pip','install','-q','uv==0.8.17']);command([sys.executable,'-m','uv','venv','-q','--python','3.12','--seed',SCRATCH/'venv']);py=SCRATCH/'venv/bin/python'
    command([py,'tests/test_portfolio_ultrabert_q4k.py','-v']);command([py,'tests/test_ultrabert_quant_profile.py','-v'])
    command([py,'-m','pip','install','-q','numpy==2.2.6','gguf==0.17.1','huggingface-hub==0.36.0','safetensors==0.6.2'])
    build=SCRATCH/'build';command(['cmake','-S','.','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCRISPEMBED_BUILD_SHARED=ON','-DBUILD_SHARED_LIBS=OFF','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_LLAMAFILE=ON','-DGGML_METAL=OFF','-DGGML_CUDA=OFF','-DGGML_BLAS=OFF'])
    command(['cmake','--build',build,'--target','crispembed-shared','crispembed-quantize','test-ultrabert-profile','--parallel','2'],5400);command([build/'test-ultrabert-profile'])
    diff=SCRATCH/'vector-diff';command(['c++','-std=c++17','-O2','-ffp-contract=off','-I','src','tests/moedern-native-v2/diff.cpp','-o',diff])
    command([py,'tools/portfolio/ultrabert_q4k.py','--output',OUT/'panel','--scratch',SCRATCH/'panel','--library',build/'libcrispembed.so','--quantizer',build/'crispembed-quantize','--diff',diff],9*3600)
    report['completed']=True;save()
except Exception as e:report['error']=str(e);save();raise
