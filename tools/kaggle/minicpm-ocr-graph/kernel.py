"""CPU-only hosted MiniCPM stored-type graph experiment. Replace @COMMIT@.
No new original capture. Small panel/setup receipt plus remotely retained raw
native/projection traces. No publication or default switch.
"""
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

COMMIT='@COMMIT@'
OUT=Path('/kaggle/working/minicpm-graph');OUT.mkdir(parents=True,exist_ok=False)
START=time.monotonic()
report=dict(schema='crispembed-minicpm-graph-setup-v1',completed=False,source=COMMIT,commands=[])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8 << 20),b''):h.update(b)
    return h.hexdigest()


def save():
    (OUT/'setup.json').write_text(json.dumps(report,indent=1)+'\n')


def run(args,timeout=1800):
    number=len(report['commands']);log=OUT/f'command-{number}.log'
    with log.open('wb') as stream:
        p=subprocess.Popen(list(map(str,args)),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            report['timeout_command']=number
    report['commands'].append(dict(argv=list(map(str,args)),exit=p.returncode,log=log.name,sha256=sha(log)))
    save()
    if p.returncode:raise RuntimeError('hosted command failed: '+str(number))


def resources(scratch='/kaggle/temp', meminfo='/proc/meminfo'):
    # A fresh Kaggle script need not have this task scratch parent yet.
    # Creating it is reversible and must precede disk_usage/statvfs.
    Path(scratch).mkdir(parents=True,exist_ok=True)
    available=int(next(line.split()[1] for line in Path(meminfo).read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    return dict(mem_available=available,disk_free=shutil.disk_usage(scratch).free,cpu_count=os.cpu_count())


try:
    assert len(COMMIT)==40 and all(c in '0123456789abcdef' for c in COMMIT),'exact source required'
    report['resources']=resources();save()
    assert report['resources']['disk_free'] >= 32*1024**3,'32GiB scratch required'
    assert report['resources']['mem_available'] >= 10*1024**3,'10GiB RAM required'
    refs={str(Path(p).resolve()) for p in glob.glob('/kaggle/input/**/minicpm-ocr/reference.json',recursive=True)}
    assert len(refs)==1,'one mounted original report required'
    reference=Path(refs.pop());assert sha(reference)=='aeb1756dcbcfb72d72c13cf5a8d8131bb90e8d8884cf07c70723cc1fe4a64c32'
    run(['git','clone','-q','--filter=blob:none','https://github.com/CrispStrobe/CrispEmbed.git','/kaggle/temp/ce'])
    os.chdir('/kaggle/temp/ce');run(['git','checkout','-q',COMMIT]);run(['git','submodule','update','-q','--init','--depth','1','ggml'])
    run([sys.executable,'-m','pip','install','-q','uv==0.8.17'])
    run([sys.executable,'-m','uv','venv','-q','--python','3.12','--seed','/kaggle/temp/venv'])
    py='/kaggle/temp/venv/bin/python'
    run([py,'-m','pip','install','-q','numpy==2.2.6','gguf==0.17.1','huggingface-hub==1.33.0'])
    run([py,'-m','pip','freeze'])
    for script in ['tests/test_minicpm_ocr_graph_source.py','tests/test_minicpm_ocr_graph_adapter.py','tests/test_minicpm_ocr_publish.py','tests/test_quantize_patch_float.py']:
        run([py,script])
    run(['cmake','-S','tests/minicpm-graph','-B','/kaggle/temp/g','-DCMAKE_BUILD_TYPE=Release','-DGGML_METAL=OFF','-DGGML_VULKAN=OFF'])
    run(['cmake','--build','/kaggle/temp/g','--parallel','2'],3600)
    run(['/kaggle/temp/g/minicpm-graph-controls'])
    run([py,'tools/minicpm-ocr/graph_panel.py','--ref-controls','/kaggle/temp/g/minicpm-graph-diff',OUT/'ref-controls'])
    run(['cmake','-S','.','-B','/kaggle/temp/b','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCRISPEMBED_NATIVE=OFF','-DGGML_NATIVE=OFF','-DGGML_LLAMAFILE=ON','-DGGML_CUDA=OFF'])
    run(['cmake','--build','/kaggle/temp/b','--target','crispembed-cli','crispembed-quantize','test-ultrabert-profile','--parallel','2'],5400)
    run(['/kaggle/temp/b/test-ultrabert-profile'])
    run([py,'tests/test_quantize_patch_float.py','--quantizer','/kaggle/temp/b/crispembed-quantize','--output',OUT/'patch-quantizer-controls.json'])
    # Model-free precision marker/load guards: same combined source, no extra
    # kernel session or original/model payload, before any bulk conversion.
    run(['cmake','-S','tests/josev-quant','-B','/kaggle/temp/jq','-DCMAKE_BUILD_TYPE=Release'])
    run(['cmake','--build','/kaggle/temp/jq','--target','josev-quant-controls','--parallel','2'],1800)
    run(['/kaggle/temp/jq/josev-quant-controls'])
    cli='/kaggle/temp/b/crispembed'
    symbols=subprocess.check_output(['nm','-C',cli],text=True)
    assert 'qwen35_ocr::graph_decoder::observe(' not in symbols,'production observer symbol leaked'
    assert 'qwen35_ocr::graph_decoder::observe(' in subprocess.check_output(['nm','-C','/kaggle/temp/g/minicpm-graph-probe'],text=True),'positive observer archive control'
    report['production_observer_excluded']=True;save()
    sys.path.insert(0,str(Path.cwd()/'tools/minicpm-ocr'))
    from build_profile import verify
    profile=verify('/kaggle/temp/b','/kaggle/temp/g')
    (OUT/'build-profile.json').write_text(json.dumps(profile,indent=1)+'\n')
    report['build_profile_sha256']=sha(OUT/'build-profile.json');save()
    checkpoint=reference.parent.parent/'checkpoint'
    assert (checkpoint/'model.safetensors').is_file(),'reuse frozen mounted checkpoint, no fallback capture/download'
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    for dtype in ['f32','f16']:
        run([py,'models/convert-minicpm-ocr-to-gguf.py',checkpoint,f'/kaggle/temp/minicpm-{dtype}.gguf','--dtype',dtype],1800)
    quant='/kaggle/temp/b/crispembed-quantize'
    run([quant,'/kaggle/temp/minicpm-f32.gguf','/kaggle/temp/minicpm-q8_0.gguf','q8_0','--tensor-type','v.emb.patch_embedding.weight=f16'],1800)
    run([quant,'/kaggle/temp/minicpm-f16.gguf','/kaggle/temp/minicpm-q4_k.gguf','q4_k','--tensor-type','v.*=f16','--tensor-type','m.*=f16','--tensor-type','l.embed_tokens.weight=f16'],1800)
    # The panel authenticates every source/original/model/tensor before forward.
    # Its numeric rejection retains all30 case-arms and actual Ref failures.
    command=[py,'tools/minicpm-ocr/graph_panel.py','--source',COMMIT,'--reference',reference.parent,'--cli',cli,'--probe','/kaggle/temp/g/minicpm-graph-probe','--diff','/kaggle/temp/g/minicpm-graph-diff','--output',OUT/'panel','--threads','4']
    for dtype in ['f32','f16','q8_0','q4_k']:command += ['--model',f'{dtype}=/kaggle/temp/minicpm-{dtype}.gguf']
    run(command, max(60,int(11.2*3600-(time.monotonic()-START))))
    report['completed']=True;save()
except Exception as error:
    report['error']=str(error);save();raise
