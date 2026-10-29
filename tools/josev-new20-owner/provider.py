"""Authenticated retained-service staging; no original/model forward."""
import hashlib,json,math,os,re,urllib.error,urllib.request,zipfile
from pathlib import Path,PurePosixPath
from josev2b_task_quality_native_v1 import require,sha256

def indexed_outputs(client,Get,List,Status,owner,pin):
    def request(kind):
        obj=kind();obj.user_name=owner;obj.kernel_slug=pin['kernel_slug'];obj.version_label=pin['version_label']
        require(json.loads(str(obj)).get('versionLabel')==pin['version_label'],
                'Actual SDK request serialization lost explicit version; no CLI fallback')
        return obj
    try:response=client.get_kernel(request(Get))
    except Exception:raise ValueError('Immutable Kaggle original metadata transfer rejected')from None
    metadata=json.loads(str(response))
    submitted=metadata['blob']['source']
    require(len(submitted.encode())==pin['submitted_script_bytes'],'Exact submitted script extent differs')
    require(hashlib.sha256(submitted.encode()).hexdigest()==pin['submitted_script_sha256']and pin['source']in submitted,
            'Exact submitted original source differs')
    try:status=client.get_kernel_session_status(request(Status))
    except Exception:raise ValueError('Immutable Kaggle original status transfer rejected')from None
    require(str(status.status).split('.')[-1]=='COMPLETE','Exact-version original is not complete')
    files={};token=None;pages=0
    while True:
        q=request(List);q.page_size=100
        if token:q.page_token=token
        try:page=client.list_kernel_session_output(q)
        except Exception:raise ValueError('Immutable Kaggle original output index rejected')from None
        pages+=1
        require(pages<=100,'Bounded original output pagination exceeded')
        for item in page.files or []:
            name=item.file_name
            require(name not in files,'Duplicate original output member')
            files[name]=item.url
        token=page.next_page_token
        if not token:break
    return files,dict(kernel_slug=pin['kernel_slug'],version_label=pin['version_label'],source=pin['source'],
                      submitted_script_sha256=pin['submitted_script_sha256'],submitted_script_bytes=len(submitted.encode()),status='COMPLETE',pages=pages,
                      actual_sdk_serialized_version_on_get_status_and_all_pages=True)


def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def stream(url,target,pin,maximum):
 require(url.startswith('https://')and 0<pin['bytes']<=maximum and not target.exists(),'Fresh bounded HTTPS member required')
 target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_suffix(target.suffix+'.partial');require(not partial.exists(),'Earlier partial transfer retained')
 digest=hashlib.sha256();count=0
 try:
  with urllib.request.urlopen(url,timeout=180)as response,partial.open('xb')as out:
   length=response.headers.get('Content-Length');require(length is None or int(length)==pin['bytes'],'Retained member Content-Length differs')
   while block:=response.read(4*2**20):
    count+=len(block);require(count<=pin['bytes'],'Retained member exceeded extent');out.write(block);digest.update(block)
 except Exception:raise ValueError('Authenticated transfer rejected; task-owned partial retained')from None
 require(count==pin['bytes']and digest.hexdigest()==pin['sha256'],'Retained member final bytes/SHA differ')
 partial.replace(target);target.chmod(0o444)
 return dict(bytes=count,sha256=digest.hexdigest())

def sdk_client():
 import importlib,importlib.metadata,sys
 require(importlib.metadata.version('kaggle')=='2.2.2'and sys.prefix!=sys.base_prefix,'Exact isolated retained-read SDK required')
 module=importlib.import_module('kaggle.api.kaggle_api_extended')
 require(Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'Loaded retained SDK escaped venv')
 client=module.KaggleApi()
 try:client.authenticate()
 except Exception:raise ValueError('Existing retained-read credentials rejected')from None
 return client,module

def sdk_index(pin):
 client,module=sdk_client()
 with client.build_kaggle_client()as api:
  return indexed_outputs(api.kernels.kernels_api_client,module.ApiGetKernelRequest,module.ApiListKernelSessionOutputRequest,module.ApiGetKernelSessionStatusRequest,os.environ['KAGGLE_ACCOUNT'],pin)

def original(cfg,root,review,guard):
 pin=cfg['original'];files,proof=sdk_index(pin);prefix=pin['root_member']
 root.mkdir(parents=True,exist_ok=False)
 for name,row in pin['reports'].items():
  member=prefix+'/'+name;require(member in files,'Original small receipt absent: '+name)
  stream(files[member],root/name,row,2*2**20)
 raw=json.loads((root/'raw-manifest.json').read_text());report=json.loads((root/'reference.json').read_text())
 require(len(raw)==1480 and report['arrays']==raw and report['git_commit']==cfg['original_source'],'Exact original20/1480 journal/source required before array transfer')
 names=set();total=0;proof['arrays']=[]
 for row in raw:
  rel=PurePosixPath(row['file']);require(len(rel.parts)==2 and rel.parts[0]=='arrays'and rel.suffix=='.npy'and '..'not in rel.parts and '\\'not in row['file']and row['file']not in names,'Unsafe/duplicate NPY member')
  names.add(row['file']);member=prefix+'/'+row['file'];require(member in files,'Original NPY member absent')
  proof['arrays'].append(dict(file=row['file'],**stream(files[member],root/row['file'],row,64*2**20)));total+=row['bytes']
  if len(names)%100==0:write(review/'original-transfer.json',dict(source=cfg['original_source'],completed_arrays=len(names),bytes=total,original_rerun=False))
 payload=sum(math.prod(row['shape'])*({'float32':4,'int64':8}[row['dtype']])for row in raw)
 require(total==1750605760 and payload==1750416320 and total-payload==1480*128,'Exact1480 NPY file/tensor/header byte totals differ')
 # Stdlib-only original_guard checks every file/header/dtype/shape/finite-policy,
 # complete snapshot122headidentity, requests/IDs/no-clipping and74 stages/case.
 guard(root,__import__('josev2b_task_quality_native_v1').native_prerequisites())
 proof.update(completed=True,array_files=1480,array_npy_bytes=total,array_tensor_bytes=payload,npy_header_bytes=total-payload,original_rerun=False,all_npy_sha_header_source_identity_verified=True)
 write(review/'original-service.json',proof);return proof

def old_four(cfg,root,review):
 pin=cfg['old_four_negative'];files,proof=sdk_index(pin);member=pin['report_member'];require(member in files,'Frozen old-four report absent')
 target=root/'old-all150-negative.json';stream(files[member],target,pin['report'],2**20)
 data=json.loads(target.read_text());require(data['source']==cfg['baseline_runtime_source']and data['completed']is True and data['f32']['compared_stages']==227 and data['f32']['strict_original_f32_numeric_and_typed_passed']is True and data['quant_task_decisions_passed']is False and data['publication_qualified']is False and data['conversion']['output_sha256']=='717f0283d9839d4b277e0591f50c7c3b75bbecad3c1cb7d3ecaeda99213640ef'and data['conversion']['output_bytes']==2944520672,'Reviewed old-four F32positive/Q4negative identity differs')
 proof.update(report=pin['report'],old_all150_model_downloaded=False,old_all150_new20_forward=False,qualification=False)
 write(review/'old-four-service.json',proof);return proof

def api(path):
 request=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/'+path,headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(request,timeout=90)as response:data=response.read(2*2**20+1)
 require(len(data)<=2*2**20,'Bounded artifact metadata required');return json.loads(data)

def artifact_metadata(selected,pin,maximum):
 run=api('actions/runs/'+str(pin['run']))
 require(run['head_sha']==selected['service_source']and run['path']==selected['workflow']and run['status']=='completed'and run['conclusion']=='success'and run['run_attempt']==selected['run_attempt'],'Exact completed mixed development service differs')
 row=api('actions/artifacts/'+str(pin['id']))
 require(row['workflow_run']['id']==pin['run']and row['size_in_bytes']==pin['bytes']and row['digest']=='sha256:'+pin['sha256']and not row['expired'],'Exact artifact descriptor differs')
 require(type(pin['bytes'])is int and 0<pin['bytes']<=maximum,'Bounded immutable artifact required')
 return {k:row[k]for k in ['id','name','size_in_bytes','digest']}, {k:run[k]for k in ['id','head_sha','path','run_attempt','conclusion']}

def artifact(selected,pin,target,maximum):
 artifact_metadata(selected,pin,maximum)
 class Stop(urllib.request.HTTPRedirectHandler):
  def redirect_request(self,*args,**kwargs):return None
 request=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(pin['id'])+'/zip',headers={'Authorization':'Bearer '+os.environ['GH_TOKEN']})
 try:response=urllib.request.build_opener(Stop).open(request,timeout=90)
 except urllib.error.HTTPError as error:
  require(error.code==302 and error.headers['Location'].startswith('https://'),'Signed HTTPS artifact storage required')
  # Fresh request to signed storage has no GitHub/Kaggle Authorization header.
  return stream(error.headers['Location'],target,pin,maximum)
 else:response.close();raise ValueError('Expected signed storage redirect absent')

def member(archive,name,target,pin,maximum):
 rel=PurePosixPath(name);require(not rel.is_absolute()and '\\'not in name and all(p not in ('','.','..')for p in rel.parts),'Safe exact ZIP member required')
 with zipfile.ZipFile(archive)as z:
  infos=z.infolist();require(len(infos)<=10000 and len(set(i.filename for i in infos))==len(infos),'Bounded unique ZIP directory required')
  hits=[i for i in infos if i.filename==name];require(len(hits)==1,'Exact selected member absent')
  item=hits[0];require(not item.flag_bits&1 and ((item.external_attr>>16)&0o170000)!=0o120000 and item.file_size==pin['bytes']and 0<item.file_size<=maximum,'Selected member type/extent differs')
  target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=0
  with z.open(item)as source,target.open('xb')as out:
   while block:=source.read(4*2**20):count+=len(block);require(count<=pin['bytes'],'ZIP member exceeds bound');h.update(block);out.write(block)
 require(count==pin['bytes']and h.hexdigest()==pin['sha256'],'Selected member final SHA differs')
 target.chmod(0o444)

def eligible(report):
 """Pre-output development screen, NEVER reclassifies old strict numeric flags."""
 require(report['completed']is True and report['holdout_native_outputs']is False and report['recipe_selected']is False and
  report['f32']['strict_original_f32_numeric_and_typed_passed']is True,'Complete original-four development/F32control required')
 results={}
 for recipe in ('A','B','C'):
  item=report['recipes'][recipe];conv=item['conversion'];panel=item['panel'];typed=panel['quant_task_decisions']
  wanted={'A':({'F32':292,'F16':78,'Q4_K':72},3613774304),'B':({'F32':316,'F16':78,'Q4_K':48},4651864544),'C':({'F32':316,'F16':118,'Q4_K':8},5375381984)}
  require(conv['recipe']==recipe and conv['counts']==wanted[recipe][0]and conv['output_bytes']==wanted[recipe][1]and len(conv['tensors'])==442,'Exact recipe storage/inventory differs')
  okay=bool(conv['completed']and conv['critical292_f32_byte_exact']and conv['independent_f16_rounding_exact']and conv['semantic_shapes_unchanged']and panel['structural_passed']and panel['compared_stages']==227 and len(typed['cases'])==4)
  details=[]
  for case in typed['cases']:
   a=case['original_response'];b=json.loads(case['native_response_text'])
   okay &= a['model']==b['model']and a['usage']==b['usage']and list(a['answers'])==list(b['answers'])and len(case['fields'])==len(a['answers'])
   for field in case['fields']:
    name=field['id'];x,y=a['answers'][name],b['answers'][name]
    okay &= field['original_argmax']==field['native_argmax']and x['type']==y['type']
    if field['type']==0:
     okay &= all(math_finite(v)and 0<=v<=1 for v in [x['noul'],y['noul']]);delta=abs(x['noul']-y['noul'])
    else:
     okay &= list(x['probabilities'])==list(y['probabilities'])
     okay &= all(math_finite(v)and 0<=v<=1 for row in [x,y]for v in row['probabilities'].values())
     values=[abs(x['probabilities'][k]-y['probabilities'][k])for k in x['probabilities']]
     delta=max(values)
    okay &= math_finite(delta)and delta<=.01
    ordinal=field['ordinal_expectation'];okay &= (field['type']==2)==(ordinal is not None)
    if ordinal:okay &= ordinal['budget']==.001 and math_finite(ordinal['absolute_error'])and ordinal['absolute_error']<=.001
    details.append(dict(case=case['id'],field=name,probability_absolute=delta,ordinal_absolute=None if ordinal is None else ordinal['absolute_error']))
  results[recipe]=dict(eligible=bool(okay),bytes=conv['output_bytes'],fields=details,strict_f32_flags=panel['strict_original_numeric_passed'])
 candidates=[key for key,row in results.items()if row['eligible']]
 return dict(recipes=results,selected=min(candidates,key=lambda k:results[k]['bytes'])if candidates else None,
  definition='strictF32control+structural/criticalbytes+alloldargmax+eachpublicprobability.01+eacholdordinal.001; smallestbytes',holdout_used=False)

def math_finite(value):
 import math
 return type(value)in(int,float)and math.isfinite(value)

def candidate(cfg,root,review):
 s=__import__('josev2b_task_quality_native_v1').selected_candidate(cfg);proof={};root.mkdir(parents=True,exist_ok=False)
 small=root/'selected-review.zip';proof['review_artifact']=artifact(s,s['review_artifact'],small,16*2**20)
 report_path=root/'selected-report.json';member(small,s['review_artifact']['member'],report_path,dict(bytes=s['report_bytes'],sha256=s['report_sha256']),8*2**20)
 report=json.loads(report_path.read_text());require(report['source']==s['checkout_source']and report['protocol_sha256']==s['protocol_sha256'],'Selected development report source/protocol differs')
 choice=eligible(report);require(choice['selected']==s['recipe'],'Candidate not selected by frozen original-four development screen')
 conv=report['recipes'][s['recipe']]['conversion'];require(conv['output_sha256']==s['legacy_model_sha256']and conv['output_bytes']==s['legacy_model_bytes'],'Reviewed candidate conversion/model identity differs')
 write(review/'development-selection.json',choice)
 # Preserve/authenticate the legacy raw artifact descriptor, without copying
 # or forwarding its multi-GB model into this held-out task.
 proof['legacy_raw_artifact_descriptor'],proof['legacy_service']=artifact_metadata(s,s['raw_artifact'],20*2**30)
 import canonical
 migration=__import__('josev2b_task_quality_native_v1').canonical_prerequisites(cfg)
 bridge=migration['bridge_service'];small_bridge=root/'canonical-review.zip'
 proof['canonical_review_artifact']=artifact(bridge,migration['review_artifact'],small_bridge,16*2**20)
 documents={}
 for key in ['descriptor','proof','same_payload_control']:
  row=migration[key];path=root/('canonical-'+key+'.json')
  member(small_bridge,row['member'],path,dict(bytes=row['bytes'],sha256=row['sha256']),8*2**20)
  def unique(pairs):
   result={}
   for name,value in pairs:require(name not in result,'Duplicate canonical JSON key');result[name]=value
   return result
  documents[key]=json.loads(path.read_text(),object_pairs_hook=unique,parse_constant=lambda _:require(False,'Nonfinite canonical JSON'))
 proof['canonical_small_evidence']=canonical.reports(cfg,documents['descriptor'],documents['proof'],documents['same_payload_control'],report)
 write(review/'canonical-evidence.json',proof['canonical_small_evidence'])
 raw=root/'canonical-raw.zip';proof['canonical_raw_artifact']=artifact(bridge,migration['raw_artifact'],raw,20*2**30)
 model=root/'selected-canonical.gguf';member(raw,migration['raw_artifact']['member'],model,migration['canonical_model'],6*2**30)
 proof['canonical_model_verification']=canonical.verify_model(model,cfg,documents['proof'])
 raw.unlink() # Only this authenticated scratch ZIP; immutable remote source stays.
 proof.update(selected=s['recipe'],report_sha256=s['report_sha256'],legacy_model_sha256=s['legacy_model_sha256'],legacy_model_bytes=s['legacy_model_bytes'],canonical_model=migration['canonical_model'],complete_remote_archive_preserved=True,verified_local_canonical_archive_removed=True,legacy_model_downloaded=False,holdout_used_for_selection=False)
 write(review/'candidate-service.json',proof);return model,report_path,proof
