"""Tiny stdlib GGUF fixtures; actual C++ quantizer runs only with --quantizer.

The explicit4D floating patch exception keeps legacy defaults, wildcard/critical
policies and ordinary matrix precision controls. No model or NumPy is needed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
PATCH='v.emb.patch_embedding.weight'
NE=(2,2,3,4)
VALUES=[-0.0,2**-24,1+2**-11,1+3*2**-11,65504.0]+[(i-24)/32 for i in range(43)]


def require(ok, why):
    if not ok:raise ValueError(why)


def sha(data):return hashlib.sha256(data).hexdigest()


def string(value):
    raw=value.encode();return struct.pack('<Q',len(raw))+raw


def fixture(path,kind=0,architecture='patch-float-fixture',values=None):
    values=VALUES if values is None else values
    require(len(values)==math.prod(NE),'fixture extent')
    patch=struct.pack('<'+('f' if kind==0 else 'e')*len(values),*values)
    vector=struct.pack('<32f',*[(i-16)/8 for i in range(32)])
    matrix=struct.pack('<128f',*[(i%19-9)/16 for i in range(128)])
    rows=[(PATCH,NE,kind,patch),('critical.norm.weight',(32,),0,vector),
          ('critical.bias',(32,),0,vector),('protected32.weight',(32,4),0,matrix),
          ('ordinary.weight',(32,4),0,matrix),('critical.norm.patch_embed.weight',NE,0,struct.pack('<48f',*VALUES)),
          ('lora.patch_embed.weight',NE,0,struct.pack('<48f',*VALUES))]
    header=b'GGUF'+struct.pack('<IQQ',3,len(rows),1)+string('general.architecture')+struct.pack('<I',8)+string(architecture)
    offset=0;parts=[]
    for name,ne,dtype,data in rows:
        header+=string(name)+struct.pack('<I',len(ne))+struct.pack('<'+'Q'*len(ne),*ne)+struct.pack('<IQ',dtype,offset)
        parts.append(data+b'\0'*((-len(data))%32));offset+=len(parts[-1])
    Path(path).write_bytes(header+b'\0'*((-len(header))%32)+b''.join(parts))
    return {name:dict(ne=ne,type=dtype,data=data) for name,ne,dtype,data in rows}


def read(path):
    data=Path(path).read_bytes();require(len(data)<1024**2 and data[:4]==b'GGUF','tiny GGUF header')
    version,count,keys=struct.unpack_from('<IQQ',data,4);require(version==3 and count<20 and keys<20,'fixture coverage')
    at=24
    def text():
        nonlocal at
        length=struct.unpack_from('<Q',data,at)[0];at+=8
        require(length<1024,'string bound');value=data[at:at+length].decode();at+=length;return value
    for _ in range(keys):
        text();kind=struct.unpack_from('<I',data,at)[0];at+=4
        if kind==8:text()
        elif kind in (4,5):at+=4
        else:raise ValueError('unexpected fixture metadata type')
    rows=[]
    for _ in range(count):
        name=text();rank=struct.unpack_from('<I',data,at)[0];at+=4;require(1<=rank<=4,'rank')
        ne=list(struct.unpack_from('<'+'Q'*rank,data,at));at+=8*rank
        kind,offset=struct.unpack_from('<IQ',data,at);at+=12;rows.append((name,ne,kind,offset))
    start=(at+31)//32*32;result={}
    for name,ne,kind,offset in rows:
        elements=math.prod(ne)
        if kind==0:size=elements*4
        elif kind==1:size=elements*2
        elif kind==8:require(ne[0]%32==0,'Q8 rows');size=elements//32*34
        else:raise ValueError('unexpected stored tensor type')
        payload=data[start+offset:start+offset+size];require(len(payload)==size,'payload extent')
        result[name]=dict(type=kind,ne=ne,bytes=size,sha256=sha(payload),data=payload)
    return result


def native_controls(binary,out):
    proofs=[]
    with tempfile.TemporaryDirectory(prefix='crispembed-patch-float-') as directory:
        folder=Path(directory);source=folder/'source.gguf';fixture(source)
        def run(name,*rules,input_path=source,expected=0):
            output=folder/(name+'.gguf')
            process=subprocess.run([str(binary),str(input_path),str(output),'q8_0',*rules],capture_output=True)
            require(process.returncode==expected,'actual C++ quantizer exit differs: '+name+' '+process.stderr.decode('utf8','replace'))
            rows=read(output) if process.returncode==0 else None
            proofs.append(dict(name=name,exit=process.returncode,expected_exit=expected,stdout_sha256=sha(process.stdout),stderr_sha256=sha(process.stderr),
                               tensors={k:{field:v for field,v in row.items() if field!='data'} for k,row in (rows or {}).items()}))
            return rows
        default=run('default')
        original=read(source)
        require(default[PATCH]['type']==0 and default[PATCH]['ne']==[12,4] and default[PATCH]['data']==original[PATCH]['data'],'legacy no-override patch policy changed')
        exact=run('literal-f16','--tensor-type',PATCH+'=f16','--tensor-type','protected32.weight=f32',
                  '--tensor-type','critical.norm.weight=f16','--tensor-type','critical.bias=f16',
                  '--tensor-type','critical.norm.patch_embed.weight=f16','--tensor-type','lora.patch_embed.weight=f16')
        expected=struct.pack('<48e',*VALUES)
        require(exact[PATCH]['type']==1 and exact[PATCH]['ne']==list(NE) and exact[PATCH]['data']==expected,'exact4D F16 type/axes/tie-even decoded values differ')
        require(struct.unpack('<48e',exact[PATCH]['data'])==struct.unpack('<48e',expected),'actual half values differ')
        require(exact['protected32.weight']['type']==0 and exact['protected32.weight']['data']==original['protected32.weight']['data'],'existing explicitF32 matrix exclusion changed')
        for name in ('critical.norm.weight','critical.bias','critical.norm.patch_embed.weight','lora.patch_embed.weight'):
            require(exact[name]==default[name],'critical floating guard changed: '+name)
        require(exact['ordinary.weight']==default['ordinary.weight'] and exact['ordinary.weight']['type']==8,'ordinary quantization changed')
        for name,rules in [
            ('wildcard-f16',['--tensor-type','v.*=f16']),
            ('literal-block-quant',['--tensor-type',PATCH+'=q8_0']),
            ('last-wildcard-cancels',['--tensor-type',PATCH+'=f16','--tensor-type','v.*=f32'])]:
            result=run(name,*rules);require(result[PATCH]==default[PATCH],'unrequested4D exception: '+name)
        for architecture in ('ppocrv6','tesseract_lstm'):
            protected=folder/(architecture+'.gguf');fixture(protected,architecture=architecture)
            baseline=run(architecture+'-default',input_path=protected)
            result=run(architecture+'-override','--tensor-type',PATCH+'=f16',input_path=protected)
            require(result[PATCH]==baseline[PATCH] and result[PATCH]['type']==0,'host-only architecture guard changed')
        half=folder/'half.gguf';fixture(half,kind=1)
        result=run('already-f16','--tensor-type',PATCH+'=f16',input_path=half)
        require(result[PATCH]['ne']==list(NE) and result[PATCH]['type']==1 and result[PATCH]['data']==expected,'existingF16 patch changed')
        for name,value in [('overflow',70000.),('nonfinite',float('inf'))]:
            invalid=folder/(name+'.gguf');fixture(invalid,values=[value]+VALUES[1:])
            run(name,'--tensor-type',PATCH+'=f16',input_path=invalid,expected=1)
    report=dict(schema='crispembed-quantize-explicit-patch-f16-controls-v1',passed=True,quantizer_sha256=sha(Path(binary).read_bytes()),cases=proofs)
    if out:Path(out).write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: actual C++4D floating patch type/axes/rounding, legacy/wildcard/critical/F32 guards, finite policy')
    return report


class FixtureControls(unittest.TestCase):
    def test_float_fixture_readback_and_tie_even_expectations(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.gguf';fixture(p);r=read(p)
            self.assertEqual(r[PATCH]['ne'],list(NE));self.assertEqual(r[PATCH]['bytes'],192)
            half=struct.unpack('<48e',struct.pack('<48e',*VALUES))
            self.assertEqual(half[2],1.);self.assertEqual(half[3],1.001953125)
    def test_truncation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.gguf';fixture(p);p.write_bytes(p.read_bytes()[:-31])
            with self.assertRaises(ValueError):read(p)
    def test_literal_patch_scope_and_preflatten_order(self):
        s=(ROOT/'tools/quantize.cpp').read_text()
        self.assertIn('rule.pattern == name && rule.type == GGML_TYPE_F16',s)
        self.assertIn('!is_ppocrv6 && !is_tesseract_lstm && exact_f16_patch_cast(t, name)',s)
        self.assertIn('t->type == GGML_TYPE_F32 && !patch_f16[i]',s)
        self.assertLess(s.index('if (patch_f16[i])'),s.index('// Guard 1:'))
        self.assertIn('std::array<float, 16384>',s)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--quantizer',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(FixtureControls)
    if not unittest.TextTestRunner().run(suite).wasSuccessful():raise SystemExit(1)
    if args.quantizer:native_controls(args.quantizer.resolve(),args.output)
