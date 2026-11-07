"""Hosted native-main and real Emscripten shared-decoder linkage, no model."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command
EMSDK_SOURCE='1693ce7359fc9694f198694f902a2cc061239b86'
EMSDK_VERSION='6.0.2'

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def main():
    require(Path('/kaggle/working').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'Kaggle CPU only')
    out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'sdk-link';work.mkdir()
    started=time.monotonic();r={'schema':'lighton4b-physical-f16-sdk-link/v1','status':'failed','commands':{},'model_forward_executed':False,'quality_qualified':False,'emsdk_source_sha':EMSDK_SOURCE,'emsdk_version':EMSDK_VERSION}
    def save():(out/'sdk-link.json').write_text(json.dumps(r,indent=2)+'\n')
    def run(label,args,timeout=900):
        r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
        try:command(args,out/(label+'.log'),min(timeout,2400-(time.monotonic()-started)),r['commands'][label])
        finally:save()
    deadline=Deadline(2400);deadline.__enter__()
    try:
        run('source-pin',['git','-C',ROOT,'rev-parse','HEAD']);r['source_sha']=(out/'source-pin.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source')
        available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit!='max':available=min(available,int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text()))
        require(available>=4<<30 and shutil.disk_usage(work).free>=10<<30,'RAM/disk before compiler SDK')
        r['preflight']={'available_bytes':available,'disk_free_bytes':shutil.disk_usage(work).free,'load':os.getloadavg()}
        run('source-membership',[sys.executable,ROOT/'tools/physical-f16-attention-probe/test_link_membership.py'])
        run('submodule',['git','-C',ROOT,'submodule','update','--init','ggml'])
        build=work/'native'
        run('native-configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCRISPEMBED_NATIVE=OFF','-DGGML_NATIVE=OFF','-DGGML_AVX2=ON','-DGGML_FMA=ON','-DGGML_F16C=ON','-DGGML_LLAMAFILE=ON','-DGGML_AVX512=OFF','-DGGML_OPENMP=OFF','-DGGML_BACKEND_DL=OFF','-DGGML_BLAS=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
        commands=json.loads((build/'compile_commands.json').read_text());rows=[x for x in commands if x['file'].endswith('/src/lighton4b_physical_f16_attention.cpp')]
        mainrows=[x for x in rows if 'CMakeFiles/crispembed.dir/'in x['command']];deltarows=[x for x in rows if 'CMakeFiles/crispembed-qwen35-delta.dir/'in x['command']]
        require(len(mainrows)==len(deltarows)==1,'both actual SDK library helper objects')
        require('CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI'not in mainrows[0]['command']and'CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI=1'in deltarows[0]['command'],'ABI stays private to delta')
        r['compile_contract']={'main_helper_present':True,'main_private_abi':False,'delta_private_abi':True};shutil.copy(build/'CMakeCache.txt',out/'native-CMakeCache.txt')
        run('native-main-build',['cmake','--build',build,'--target','test-lighton4b-main-f16-link','--parallel','2'],1200)
        run('native-main-link',[build/'test-lighton4b-main-f16-link']);require('MAIN_SDK_F16_CLOSED_STUB_LINK PASS 5'in(out/'native-main-link.log').read_text(),'actual native main archive stub')
        sdk=work/'emsdk'
        for label,args in [('sdk-init',['git','init',sdk]),('sdk-remote',['git','-C',sdk,'remote','add','origin','https://github.com/emscripten-core/emsdk.git']),('sdk-fetch',['git','-C',sdk,'fetch','--depth','1','origin',EMSDK_SOURCE]),('sdk-checkout',['git','-C',sdk,'checkout','--detach','FETCH_HEAD']),('sdk-install',[sys.executable,sdk/'emsdk.py','install',EMSDK_VERSION]),('sdk-activate',[sys.executable,sdk/'emsdk.py','activate',EMSDK_VERSION,'--embedded'])]:run(label,args,900)
        def sdk_run(label,args,timeout=900):run(label,['bash','-c','source "$1/emsdk_env.sh" >/dev/null && shift && exec "$@"','--',sdk,*args],timeout)
        sdk_run('emcc-version',['emcc','--version']);require('6.0.2'in(out/'emcc-version.log').read_text(),'repo-pinned real Emscripten')
        wasm=work/'wasm'
        sdk_run('wasm-configure',['emcmake','cmake','-S',ROOT/'tools/physical-f16-link-probe','-B',wasm,'-DCMAKE_BUILD_TYPE=Release','-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm','-DCRISPEMBED_SOURCE='+str(ROOT),'-DGGML_SOURCE='+str(ROOT/'ggml')])
        shutil.copy(wasm/'CMakeCache.txt',out/'wasm-CMakeCache.txt')
        require('EMSCRIPTEN_SYSTEM_PROCESSOR:UNINITIALIZED=wasm' in (wasm/'CMakeCache.txt').read_text() or 'EMSCRIPTEN_SYSTEM_PROCESSOR:STRING=wasm' in (wasm/'CMakeCache.txt').read_text(),'production WASM processor selection')
        require(any('arch/wasm/quants.c' in p.read_text() for p in wasm.rglob('build.make')),'real WASM quantizer source membership')
        sdk_run('wasm-build',['cmake','--build',wasm,'--target','physical-f16-shared-link','--parallel','2'],900)
        sdk_run('wasm-run',['node',wasm/'physical-f16-shared-link.js']);require('MAIN_SDK_F16_CLOSED_STUB_LINK PASS 5'in(out/'wasm-run.log').read_text(),'actual shared attention+helper WASM link/run')
        retained=out/'wasm';retained.mkdir()
        for artifact in wasm.glob('physical-f16-shared-link.*'):
            if artifact.is_file():shutil.copy2(artifact,retained/artifact.name)
        r['wasm_artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in retained.iterdir()if p.is_file()}
        r['status']='native_main_and_real_wasm_link_pass';r['limitations']='Focused actual shared-decoder WASM closure + main source membership, not a complete OCR/WASM artifact size or model qualification.'
    except BaseException as error:r['error_type'],r['error']=type(error).__name__,str(error);raise
    finally:
        deadline.close();r['elapsed_seconds']=time.monotonic()-started;r['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.rglob('*')if p.is_file()and p.name!='sdk-link.json'};save()

if __name__=='__main__':main()
