"""Small authenticated retained-image helpers; no decoder/model dependency.

Version-index helper adapted from the frozen SciFact GHA provider at3f82073e.
Account/token values and signed URLs never enter returned receipts.
"""
import hashlib,json,os
from pathlib import Path

def require(ok,why):
    if not ok:raise ValueError(why)

def _mounted_image(page,root):
    """Return exact admitted bytes+public proof, or None when no candidate exists.

    Matching wrong/ambiguous/symlink candidates reject; they never trigger URL
    fallback. Search only input-images directories and fixed literal filenames.
    """
    root=Path(root)
    require(not root.is_symlink(),'Retained input root symlink rejected')
    if not root.exists():return None
    require(root.is_dir()and not root.is_symlink(),'Retained input root must be a real directory')
    ident=page['id'];require('/'not in ident and '\\'not in ident and ident not in('.','..'),'Literal page ID required')
    hits=[];visited=0;entries=0
    def walk_error(error):raise error
    for current,dirs,files in os.walk(root,followlinks=False,onerror=walk_error):
        visited+=1;entries+=len(dirs)+len(files)
        require(visited<=2048 and entries<=20000,'Retained image search budget exceeded')
        require(len(Path(current).relative_to(root).parts)<=10,'Retained image search depth exceeded')
        if 'input-images'in dirs and (Path(current)/'input-images').is_symlink():raise ValueError('Symlink input-images directory rejected')
        dirs[:]=sorted(d for d in dirs if not (Path(current)/d).is_symlink())
        if Path(current).name!='input-images':continue
        for suffix in('.jpg','.png','.encoded'):
            p=Path(current)/(ident+suffix)
            if p.exists()or p.is_symlink():hits.append(p)
    if not hits:return None
    require(len(hits)==1,'Ambiguous mounted encoded image candidates')
    p=hits[0];pin=page['image']
    require(p.is_file()and not p.is_symlink()and p.stat().st_size==pin['bytes']and 0<pin['bytes']<=16*2**20,'Mounted encoded image extent/type differs')
    data=p.read_bytes();require(len(data)==pin['bytes']and hashlib.sha256(data).hexdigest()==pin['sha256'],'Mounted encoded image SHA differs')
    return data,dict(provider='authenticated retained mounted image',bytes=len(data),sha256=pin['sha256'],member_suffix='input-images/'+p.name)


def mounted_image(page,root):
    try:return _mounted_image(page,root)
    except OSError:raise ValueError('Retained image filesystem admission rejected')from None


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

