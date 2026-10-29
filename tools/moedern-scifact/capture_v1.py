"""One actual C-ABI context, raw text in and complete typed F32 streams out."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(Path(__file__).parent)]
from contract_native_v1 import require,sha
from original_inputs_v1 import expected_tokens


class Result(C.Structure):
    _fields_=[('abi_version',C.c_uint32),('struct_bytes',C.c_uint32),('role',C.c_int32),
              ('n_tokens',C.c_int32),('n_selected',C.c_int32),('dimension',C.c_int32),('expansion_count',C.c_int32),
              ('ids',C.POINTER(C.c_int32)),('supplied_mask',C.POINTER(C.c_uint8)),
              ('effective_mask',C.POINTER(C.c_uint8)),('scoring_mask',C.POINTER(C.c_uint8)),
              ('expansion_positions',C.POINTER(C.c_uint8)),('all_vectors',C.POINTER(C.c_float)),
              ('selected_vectors',C.POINTER(C.c_float))]


class Native:
    def __init__(self,library,model):
        self.lib=C.CDLL(str(Path(library).resolve()))
        lib=self.lib
        lib.crispembed_moedern_init.argtypes=[C.c_char_p,C.c_int];lib.crispembed_moedern_init.restype=C.c_void_p
        lib.crispembed_moedern_free.argtypes=[C.c_void_p];lib.crispembed_moedern_free.restype=None
        lib.crispembed_moedern_encode.argtypes=[C.c_void_p,C.c_char_p,C.c_int];lib.crispembed_moedern_encode.restype=C.POINTER(Result)
        lib.crispembed_moedern_score.argtypes=[C.c_void_p,C.c_char_p,C.c_char_p,C.POINTER(C.c_float)];lib.crispembed_moedern_score.restype=C.c_int
        lib.crispembed_moedern_error.argtypes=[C.c_void_p];lib.crispembed_moedern_error.restype=C.c_char_p
        self.ctx=lib.crispembed_moedern_init(str(model).encode(),2)
        require(self.ctx,'actual C-ABI initialization rejected: '+self.error())
    def error(self):
        return (self.lib.crispembed_moedern_error(getattr(self,'ctx',None))or b'').decode('utf-8',errors='strict')
    def close(self):
        if self.ctx:self.lib.crispembed_moedern_free(self.ctx);self.ctx=None
    def encode(self,text,role):
        require(role in('query','document')and isinstance(text,str)and '\0'not in text,'explicit NUL-free raw text/role required')
        pointer=self.lib.crispembed_moedern_encode(self.ctx,text.encode('utf-8'),0 if role=='query'else 1)
        require(bool(pointer),'actual C-ABI encode rejected: '+self.error());r=pointer.contents
        require(r.abi_version==1 and r.struct_bytes==C.sizeof(Result)and r.dimension==128 and
                3<=r.n_tokens<=512 and 1<=r.n_selected<=r.n_tokens and r.expansion_count==(r.n_tokens if role=='query'else 0)and
                r.role==(0 if role=='query'else 1),'actual C-ABI result version/layout/shape differs')
        require(all(bool(p)for p in (r.ids,r.supplied_mask,r.effective_mask,r.scoring_mask,r.all_vectors,r.selected_vectors))and
                (r.expansion_count==0 or bool(r.expansion_positions)),'actual C-ABI pointer missing')
        typed=dict(ids=list(r.ids[:r.n_tokens]),supplied_mask=list(r.supplied_mask[:r.n_tokens]),
                   effective_mask=list(r.effective_mask[:r.n_tokens]),scoring_mask=list(r.scoring_mask[:r.n_tokens]),
                   expansion_positions=list(r.expansion_positions[:r.expansion_count])if r.expansion_count else [])
        # Copy while the borrowed buffers are valid, before ANY following call.
        all_bytes=C.string_at(r.all_vectors,r.n_tokens*512);selected=C.string_at(r.selected_vectors,r.n_selected*512)
        return typed,all_bytes,selected
    def negatives(self):
        require(not self.lib.crispembed_moedern_encode(self.ctx,None,0)and self.error(),'null text accepted')
        require(not self.lib.crispembed_moedern_encode(self.ctx,b'x',7)and self.error(),'invalid role accepted')
        require(not self.lib.crispembed_moedern_encode(self.ctx,b'\xff',0)and self.error(),'invalid UTF8 accepted')
        score=C.c_float(5)
        require(not self.lib.crispembed_moedern_score(self.ctx,b'q',None,C.byref(score))and score.value==0 and self.error(),'failed score not cleared')
        return dict(null_text=True,invalid_role=True,invalid_utf8=True,null_score_cleared=True)


def main(args):
    require(os.environ.get('GITHUB_ACTIONS')=='true'or(Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()and os.environ.get('KAGGLE_KERNEL_RUN_TYPE')in('Batch','Interactive')),'hosted C-ABI model execution only')
    import numpy as np
    inputs=json.loads(args.data.read_text());gold=json.loads(args.rows.read_text())
    rows=[dict(**r,key='q:'+r['id'],role='query')for r in inputs['queries']]+[dict(**r,key='d:'+r['id'],role='document')for r in inputs['documents']]
    require(len(rows)==len(gold)==5483,'complete5483 raw-text/role inputs required')
    args.out.mkdir(parents=True,exist_ok=False)
    status=dict(schema='crispembed-moedern-scifact-capi-capture-v1',completed=False,rows=0,failures=[],
                library_sha256=sha(args.library),model_sha256=sha(args.model),threads=2,raw_text_only=True,original_vectors_injected=False,failure_count=0)
    def save():(args.out/'capture.json').write_text(json.dumps(status,indent=2)+'\n')
    save();ctx=Native(args.library,args.model);offset={'all':0,'selected':0};started=time.monotonic()
    def failure(value):
        status['failure_count']+=1
        if len(status['failures'])<16:status['failures'].append(value)
    try:
        status['negative_controls']=ctx.negatives()
        with (args.out/'all.f32').open('xb')as all_file,(args.out/'selected.f32').open('xb')as selected_file,(args.out/'rows.jsonl').open('x')as journal:
            for i,(row,target)in enumerate(zip(rows,gold)):
                entry=dict(key=row['key'],role=row['role'],completed=False)
                try:
                    require(row['key']==target['key'],'original row order differs')
                    typed,all_bytes,selected=ctx.encode(row['text'],row['role'])
                    a=np.frombuffer(all_bytes,dtype='<f4').reshape(-1,128);b=np.frombuffer(selected,dtype='<f4').reshape(-1,128)
                    require(np.isfinite(a).all()and np.isfinite(b).all()and
                            selected==a[np.asarray(typed['scoring_mask'],dtype=bool)].tobytes(),'actual all/selected F32 buffers differ from scoring mask')
                    exact=typed==expected_tokens(target)
                    entry.update(typed,completed=True,typed_exact=exact)
                    if not exact:failure(dict(key=row['key'],stage='token_ids_or_masks'))
                    for kind,data,stream in [('all',all_bytes,all_file),('selected',selected,selected_file)]:
                        entry[kind+'_offset']=offset[kind];entry[kind+'_bytes']=len(data);entry[kind+'_sha256']=hashlib.sha256(data).hexdigest();stream.write(data);offset[kind]+=len(data)
                except Exception as error:
                    entry['error']=str(error);failure(dict(key=row['key'],stage='native_capture',error=str(error)))
                journal.write(json.dumps(entry,allow_nan=False)+'\n');status['rows']=i+1
                if (i+1)%100==0:journal.flush();status['elapsed_seconds']=time.monotonic()-started;save()
        # Actual public score API checks use raw text again, not supplied vectors.
        pairs=[]
        for qi,di in ((0,0),(0,5182),(1,1),(1,5181),(299,0),(299,5182)):
            value=C.c_float(float('nan'))
            ok=ctx.lib.crispembed_moedern_score(ctx.ctx,inputs['queries'][qi]['text'].encode(),inputs['documents'][di]['text'].encode(),C.byref(value))
            require(ok and np.isfinite(value.value),'actual raw-text score rejected: '+ctx.error())
            pairs.append(dict(query=qi,document=di,score=value.value))
        status.update(completed=True,elapsed_seconds=time.monotonic()-started,pairs=pairs,
                      files={name:dict(bytes=(args.out/name).stat().st_size,sha256=sha(args.out/name))for name in ('all.f32','selected.f32','rows.jsonl')})
        save();return 0 if not status['failure_count']else 1
    except BaseException as error:
        status.update(error_type=type(error).__name__,error=str(error),completed=False,elapsed_seconds=time.monotonic()-started)
        save();raise
    finally:ctx.close()


if __name__=='__main__':
    def terminate(_signal,_frame):raise RuntimeError('capture termination requested')
    signal.signal(signal.SIGTERM,terminate)
    p=argparse.ArgumentParser()
    for name in ('library','model','data','rows','out'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
