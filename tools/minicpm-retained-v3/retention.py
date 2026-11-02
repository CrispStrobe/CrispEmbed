"""Canonical retention and reconstructible, task-owned adapter cleanup."""
import json
from pathlib import Path
import shutil
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'minicpm-ocr'),str(Path(__file__).resolve().parents[1])]
from common import atomic,checked,sha,require,safe,canonical,duplicate,room,used
from graph_adapter import write_pair,collect
from josev2b_composed_panel import metrics

LABELS=[f'model.language_model.layers.{i}'for i in range(24)]+['model.language_model.norm','lm_head']


def npy(path,descriptor=None):
    import numpy as np
    value=np.load(path,mmap_mode='r',allow_pickle=False)
    require(value.dtype==np.dtype('<f4')and value.size and value.ndim in [1,2,3,4]and np.isfinite(value).all(),
            'Finite bounded F32 original/canonical NPY required')
    require(value.size<=16*1024*1024,'NPY element cap exceeded')
    if descriptor:require(list(value.shape)==descriptor['shape']and sha(path)==descriptor['sha256'],'Original NPY descriptor differs')
    return value


def raw(path,array):
    path=Path(path);require(not path.exists()and not path.is_symlink(),'Fresh task-owned reconstructed F32 required')
    array.tofile(path)
    return dict(path=path,shape=list(array.shape),sha256=sha(path))


def compare(pairs,binary,temp,out):
    """Retain source hashes, exact adapter reconstruction and Ref before cleanup."""
    import numpy as np
    import subprocess
    temp=Path(temp);out=Path(out);temp.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    reconstruction=[];independent={}
    for pair in pairs:
        require('reconstruction'in pair,'Explicit canonical source reconstruction required')
        for side in ['original','native']:
            spec=pair[side];require(sha(spec['path'])==spec['sha256'],'Raw adapter input changed')
            anchor=pair['reconstruction'][side]
            require(sha(anchor['source_path'])==anchor['source_sha256'],'Canonical source changed before Ref')
        a=np.fromfile(pair['original']['path'],dtype='<f4').reshape(pair['original']['shape'])
        b=np.fromfile(pair['native']['path'],dtype='<f4').reshape(pair['native']['shape'])
        independent[pair['name']]=metrics(a,b)
        reconstruction.append(dict(name=pair['name'],shape=pair['original']['shape'],
            original_raw_sha256=pair['original']['sha256'],native_raw_sha256=pair['native']['sha256'],
            sources={side:{k:str(v)for k,v in pair['reconstruction'][side].items()if k!='source_path'}for side in ['original','native']}))
    adapters=write_pair(temp/'original.gguf',temp/'native.gguf',pairs)
    with(out/'diff.log').open('w')as stream:
        process=subprocess.run([str(binary),str(temp/'original.gguf'),str(temp/'native.gguf'),str(out/'diff.json')],stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    proof=collect(process.returncode,out/'diff.json',independent)
    require(proof['all_metrics_agree'],'Actual Ref and independent metrics disagree')
    result=dict(actual_ref=proof,stages=independent,passed=all(r['passed']for r in independent.values()),diff_log_sha256=sha(out/'diff.log'))
    atomic(out/'comparison.json',result)
    declaration=dict(schema='crispembed-lossless-adapter-reconstruction-v1',gguf_version=3,alignment=32,
                     tensor_type='F32',shape_rule='C axes reversed into GGML ne; no arithmetic transform',
                     adapters=adapters,pairs=reconstruction,actual_ref_sha256=sha(out/'diff.json'),
                     comparison_sha256=sha(out/'comparison.json'),canonical_sources_rechecked=True)
    atomic(out/'reconstruction.json',declaration)
    # Canonical inputs and manifest are checked again before deleting ONLY the
    # two adapters created in this fresh scratch directory. Original/canonical
    # arrays, models and other task outputs are never candidates for cleanup.
    for pair in pairs:
        for anchor in pair['reconstruction'].values():require(sha(anchor['source_path'])==anchor['source_sha256'],'Canonical source changed before cleanup')
    for name in ['original.gguf','native.gguf']:
        path=temp/name;require(path.parent==temp and path.is_file()and not path.is_symlink(),'Unsafe adapter cleanup')
        path.unlink()
    atomic(out/'cleanup.json',dict(adapters_removed=True,reconstruction_sha256=sha(out/'reconstruction.json'),canonical_removed=False))
    return result


def anchor(path,label,transform):
    return dict(source_path=Path(path),source_label=label,source_sha256=sha(path),transform=transform)


def compact_old(rows,reference,reference_directory,out):
    """Copy canonical bytes/small proofs; leave immutable v3 service untouched."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False);plan=[];total=0
    original={c['id']:c for c in reference['cases']}
    for item in rows:
        folder=item['folder'];destination=out/Path(item['pin']['case_report']).parent
        reconstruction=verify_old_duplicates(item,original[item['pin']['id']],reference_directory)
        for name,pin in item['files'].items():
            if duplicate(name):
                source=reconstruction[name]
                plan.append(dict(case_report=item['pin']['case_report'],omitted=name,identity=pin,reconstruction=source))
                continue
            # Every other old retained member must be present; fetch only the
            # canonical arrays and scalar/log files, never unknown bulk files.
            require(canonical(name)or Path(name).suffix in ['.json','.jsonl','.log','.stdout','.stderr'],
                    'Undeclared old noncanonical bulk member: '+name)
            source=checked(folder,name,pin);target=safe(destination,name);target.parent.mkdir(parents=True,exist_ok=True)
            room(used(out),pin['bytes'],shutil.disk_usage(out).free);shutil.copyfile(source,target)
            require(sha(target)==pin['sha256'],'Canonical compaction copy differs');total+=pin['bytes']
        for name in ['case-report.json','files.json']:
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/name,target)
    atomic(out/'old-reconstruction-manifest.json',dict(old_service_mutated=False,duplicates_deleted_from_old=False,
        retained_bytes=total,omitted_duplicate_members=plan,canonical_sha_verified=True,all_omitted_adapter_bytes_reconstructed_and_sha_verified=True,
        limits='Omitted old adapters remain immutablev3; their complete GGUF and raw payload SHA256s were reconstructed from authenticated canonical sources. New scratch adapters require actualRef/reconstruction receipts before cleanup.'))
    return dict(retained_bytes=total,reconstruction_sha256=sha(out/'old-reconstruction-manifest.json'),old_service_mutated=False)


def payload(spec):
    """Yield exact F32 bytes; no full tensor tobytes/materialization."""
    import hashlib
    path=spec['path'];count=0;digest=hashlib.sha256()
    if spec['kind']=='raw':
        with Path(path).open('rb')as stream:
            while block:=stream.read(2**20):
                count+=len(block);digest.update(block);yield block
    else:
        import numpy as np
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        if spec['kind']=='ids':
            require(a.dtype.kind in 'iu'and a.size<=4096 and (a>=0).all()and(a<2**24).all(),'Exact bounded ID-to-F32 representation required')
            a=a.astype('<f4')
        require(a.dtype==np.dtype('<f4')and a.flags.c_contiguous and a.size==__import__('math').prod(spec['shape']),
                'Exact C-order canonical payload shape/dtype required')
        view=memoryview(a).cast('B')
        for offset in range(0,len(view),2**20):
            block=view[offset:offset+2**20];count+=len(block);digest.update(block);yield block
    require(count==4*__import__('math').prod(spec['shape'])and digest.hexdigest()==spec['raw_sha256'],
            'Reconstructed duplicate F32 payload SHA differs')


def gguf_digest(tensors):
    import struct
    import hashlib
    require(0<len(tensors)<=512,'Bounded reconstructed GGUF tensor count required')
    header=b'GGUF'+struct.pack('<IQQ',3,len(tensors),0);offset=0;rows=[]
    for name,spec in tensors:
        text=name.encode('ascii');shape=spec['shape']
        header+=struct.pack('<Q',len(text))+text+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*reversed(shape))+struct.pack('<IQ',0,offset)
        rows.append((offset,spec));offset=(offset+4*__import__('math').prod(shape)+31)//32*32
    begin=(len(header)+31)//32*32;digest=hashlib.sha256();digest.update(header);digest.update(b'\0'*(begin-len(header)));position=begin
    for offset,spec in rows:
        digest.update(b'\0'*(begin+offset-position));position=begin+offset
        for block in payload(spec):digest.update(block);position+=len(block)
    return dict(bytes=position,sha256=digest.hexdigest(),format='GGUF3 F32 metadata0/alignment32')


def verify_old_duplicates(item,reference,reference_directory):
    """Prove omitted whole adapter hashes, not merely an ability to recreate."""
    import numpy as np
    folder=item['folder'];files=item['files'];plans={};pair={'original':[],'native':[]}
    ids=np.load(folder/'dump/input_ids.npy',allow_pickle=False)
    generated=np.load(folder/'dump/ids.npy',allow_pickle=False)
    confidence=np.load(folder/'dump/confidences.npy',allow_pickle=False)
    require(ids.dtype==np.dtype('<i4')and ids.tolist()==reference['input_ids']and generated.dtype==np.dtype('<i4')and
            generated.tolist()==reference['ids']and confidence.dtype==np.dtype('<f4')and np.isfinite(confidence).all()and
            confidence.reshape(-1).tolist()==item['case']['confidences']and(confidence>=0).all()and(confidence<=1).all(),'Actual retained input/output IDs or confidences differ')
    for index,label in enumerate(LABELS):
        shape=item['case']['prefill']['stages']['stage_'+str(index)]['shape']
        for side in pair:
            name=side+'_'+str(index)+'.f32'
            path=Path(reference_directory)/reference['arrays'][label]['path']if side=='original'else folder/'dump'/(label.replace('.','_')+'.npy')
            spec=dict(path=path,shape=shape,raw_sha256=files[name]['sha256'],kind='npy')
            # Drain to verify the duplicate raw SHA independently of its GGUF.
            for _ in payload(spec):pass
            pair[side].append(('stage_'+str(index),spec))
            plans[name]=dict(source_npy_sha256=sha(path),shape=shape,raw_sha256=files[name]['sha256'],
                             transform='Known singleton batch removed; exact F32 C payload')
            probe='projected/'+('layer_'+str(index)if index<24 else'norm'if index==24 else'logits')+'.f32'
            if side=='native'and probe in files:
                require(files[probe]==files[name],'Root/probe duplicate payload SHA/extent differs')
                plans[probe]=dict(canonical_npy_sha256=sha(path),shape=shape,raw_sha256=files[name]['sha256'])
    for side,tensors in pair.items():
        name='prefill/'+side+'.gguf';identity=gguf_digest(tensors)
        require({k:identity[k]for k in ['bytes','sha256']}==files[name],'Old whole prefill adapter reconstruction differs')
        plans[name]=dict(**identity,tensors=[dict(name=n,shape=s['shape'],raw_sha256=s['raw_sha256'])for n,s in tensors])
    if 'projected'in item['case']:
        journal=json.loads((folder/'projected/projections.json').read_text())
        for side,suffix in [('original','control'),('native','native')]:
            tensors=[]
            for row in journal['stages']:
                name='projected/'+row['key']+'.'+suffix+'.f32'
                tensors.append((row['key'],dict(path=folder/name,kind='raw',shape=[len(row['selected_rows']),row['output_width']],raw_sha256=files[name]['sha256'])))
            name='projected/comparison/'+side+'.gguf';identity=gguf_digest(tensors)
            require({k:identity[k]for k in ['bytes','sha256']}==files[name],'Old whole projected adapter reconstruction differs')
            plans[name]=dict(**identity,tensors=[dict(name=n,shape=s['shape'],raw_sha256=s['raw_sha256'])for n,s in tensors])
        meta=[('input_ids',dict(path=folder/'dump/input_ids.npy',kind='ids',shape=[len(reference['input_ids'])],raw_sha256=files['ids-as-exact-f32.f32']['sha256'])),
              ('merged',dict(path=folder/'dump/model_merger.npy',kind='npy',shape=list(npy(folder/'dump/model_merger.npy').shape),raw_sha256=files['merged.f32']['sha256']))]
        identity=gguf_digest(meta);require({k:identity[k]for k in ['bytes','sha256']}==files['probe-input.gguf'],'Old probe-input adapter reconstruction differs')
        plans['probe-input.gguf']=identity
        for key,name in [('input_ids','ids-as-exact-f32.f32'),('merged','merged.f32')]:
            spec=dict(meta)[key];plans[name]=dict(source_npy_sha256=sha(spec['path']),shape=spec['shape'],raw_sha256=spec['raw_sha256'],transform=spec['kind'])
    require({name for name in files if duplicate(name)}<=set(plans),'Missing old duplicate reconstruction mapping')
    return plans
