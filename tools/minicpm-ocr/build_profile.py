"""Authenticate actual configured root/probe backend and implementation flags."""
import json
from pathlib import Path
import shlex

CACHE_KEYS=('GGML_NATIVE','CRISPEMBED_NATIVE','GGML_LLAMAFILE','GGML_CUDA','GGML_BLAS',
            'GGML_OPENMP','GGML_BACKEND_DL','GGML_CPU_REPACK','GGML_CPU_ALL_VARIANTS','GGML_AVX','GGML_AVX2','GGML_AVX512','GGML_FMA','GGML_F16C')
QWEN=('qwen35_graph_decoder.cpp','qwen35_decoder.cpp','qwen35_attention.cpp',
      'qwen35_linear.cpp','qwen35_delta.cpp','qwen35_profile.cpp')


def require(ok, why):
    if not ok:raise ValueError(why)


def cache(path):
    rows={}
    for line in Path(path).read_text().splitlines():
        if line.startswith(('#','//')) or ':' not in line or '=' not in line:continue
        name,rest=line.split(':',1);_,value=rest.split('=',1);rows[name]=value
    return rows


def command(row):
    return row.get('arguments') or shlex.split(row['command'])


def isa(argv):
    return sorted(x for x in argv if x.startswith(('-march=','-mavx','-mfma','-mf16c')))


def one(rows, basename, target=None):
    selected=[r for r in rows if Path(r['file']).name==basename and
              (target is None or ('CMakeFiles/'+target+'.dir/') in ' '.join(command(r)))]
    require(len(selected)==1,'exact compile row required: '+str((basename,target,len(selected))))
    return selected[0]


def verify(root, standalone):
    root,standalone=Path(root),Path(standalone)
    a,b=cache(root/'CMakeCache.txt'),cache(standalone/'CMakeCache.txt')
    for key in CACHE_KEYS:
        require(key in a and key in b and a[key]==b[key],'root/standalone configured profile differs: '+key)
    require(a['GGML_NATIVE']=='OFF' and a['CRISPEMBED_NATIVE']=='OFF' and a['GGML_LLAMAFILE']=='ON','fixed portable/LLAMAFILE-ON profile required')
    require(a['GGML_CUDA']=='OFF' and a['GGML_BLAS']=='OFF' and a['GGML_OPENMP']=='OFF','CPU-only internal backend required')
    aa=json.loads((root/'compile_commands.json').read_text());bb=json.loads((standalone/'compile_commands.json').read_text())
    sources=[]
    for name in QWEN:
        x,y=one(aa,name,'crispembed'),one(bb,name,'minicpm-graph-probe')
        xx,yy=command(x),command(y)
        require(isa(xx)==isa(yy),'root/standalone actual ISA flags differ: '+name)
        require('-ffp-contract=off' in xx and '-ffp-contract=off' in yy,'both implementation TUs require contraction off: '+name)
        require(not any(flag.startswith(('-ffast-math','-Ofast','-march=native')) for flag in xx+yy),'unreviewed arithmetic/host ISA flag')
        sources.append(dict(file=name,root_argv=xx,standalone_argv=yy,isa=isa(xx),contraction='off'))
    backend=[]
    for name in ('ggml-cpu.c','ggml-cpu.cpp'):
        x,y=one(aa,name),one(bb,name)
        xx,yy=command(x),command(y)
        require(isa(xx)==isa(yy),'actual backend ISA differs: '+name)
        dx=sorted(v for v in xx if v.startswith('-DGGML'))
        dy=sorted(v for v in yy if v.startswith('-DGGML'))
        require(dx==dy,'actual backend compile definitions differ: '+name)
        backend.append(dict(file=name,root_argv=xx,standalone_argv=yy,isa=isa(xx),definitions=dx))
    return dict(passed=True,cache={k:a[k] for k in CACHE_KEYS},implementation=sources,backend=backend)
