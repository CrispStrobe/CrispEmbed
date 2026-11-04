MINERU_PLAN={'schema': 'crispembed.mineru2605.resources.v1', 'cases_per_dispatch': 1, 'tiers': ['f32', 'f16', 'q8_0', 'q4_k'], 'positions_upper': 2048, 'logit_bytes_upper': 6223298560, 'stage_and_extra_dump_bytes_upper': 3489660928, 'scratch_bytes_required': 13853769141, 'output_bytes_required': 11860443136, 'available_memory_bytes_required': 12884901888, 'policy': 'one case; F32 plus one temporary candidate GGUF; raw arrays retained directly in uploaded output', 'qualification': False, 'resource_bounds_are_not_runtime_measurements': True}
import hashlib,json,os,platform,shutil,sys,urllib.request
from pathlib import Path
assert Path('/kaggle/working').is_dir() and os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Batch','Interactive')
assert os.environ.get('GITHUB_ACTIONS') is None and os.environ.get('GITHUB_SHA') is None
memory=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
out=Path('/kaggle/working/ocr-cpu-capacity.json')
report={'schema':'crispembed.ocr-cpu-provider-capacity.v1','platform':platform.platform(),'machine':platform.machine(),'cpus':os.cpu_count(),'python':platform.python_version(),'available_memory_bytes':memory,'filesystems':{},'mineru_plan':MINERU_PLAN,'model_forward_calls':0,'build_calls':0,'completed':False}
for label,path in [('output',Path('/kaggle/working')),('scratch',Path('/tmp'))]:
 d=shutil.disk_usage(path);report['filesystems'][label]={'device':path.stat().st_dev,'free_bytes':d.free,'total_bytes':d.total}
f=report['filesystems'];shared=f['output']['device']==f['scratch']['device']
report['shared_filesystem']=shared
report['unchanged_mineru_contract_fits']=memory>=MINERU_PLAN['available_memory_bytes_required'] and (min(f['output']['free_bytes'],f['scratch']['free_bytes'])>=MINERU_PLAN['output_bytes_required']+MINERU_PLAN['scratch_bytes_required'] if shared else f['output']['free_bytes']>=MINERU_PLAN['output_bytes_required'] and f['scratch']['free_bytes']>=MINERU_PLAN['scratch_bytes_required'])
try:
 url='https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B/resolve/08aaea840498d49ce16247b6263196cf02814885/config.json'
 with urllib.request.urlopen(url,timeout=20) as stream: data=stream.read(65537)
 assert len(data)<=65536
 report['public_config_network_probe']={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'expected_sha256':'f91cad3c52a3d8f1d420b51459455b27d792db28bf5d1d08097ae96bf1c44730'}
 report['public_config_network_probe']['verified']=report['public_config_network_probe']['sha256']==report['public_config_network_probe']['expected_sha256']
except Exception as error: report['public_config_network_probe']={'verified':False,'error_type':type(error).__name__}
report['completed']=True
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
