"""Bounded post-inference score I/O; no model or legacy-protocol dependency."""
from pathlib import Path,PurePosixPath
import math
import struct

def require(ok,why):
    if not ok:raise ValueError(why)
class LocalReader:
    def __init__(self,root):self.root=Path(root)
    def read(self,file,offset,count):
        name=PurePosixPath(file)
        require(isinstance(file,str)and not name.is_absolute()and len(name.parts)==1 and name.name not in('', '.', '..')and '\\'not in file,'single safe score filename required')
        path=self.root/file
        require(path.resolve().is_relative_to(self.root.resolve())and not path.is_symlink(),'contained regular score source required')
        require(type(offset)is int and offset>=0 and type(count)is int and 0<count<=4*2**20,'bounded score range required')
        with path.open('rb')as stream:stream.seek(offset);data=stream.read(count)
        require(len(data)==count,'score range truncated');return data

def tiny_archive(path,name,shape,data):
    # Encoding is byte-identical to the retained v2 diagnostic writer.
    require(isinstance(name,str)and 0<len(name.encode())<=256 and 1<=len(shape)<=4 and all(type(n)is int and n>0 for n in shape),'bounded diagnostic tensor required')
    require(math.prod(shape)*4==len(data)and len(data)<=4*2**20,'bounded F32 score geometry required')
    raw=name.encode();ne=list(reversed(shape));header=struct.pack('<4sIQQQ',b'GGUF',3,1,0,len(raw))+raw
    header+=struct.pack('<I',len(ne))+struct.pack('<'+'Q'*len(ne),*ne)+struct.pack('<IQ',0,0)
    with Path(path).open('wb')as stream:stream.write(header);stream.write(b'\0'*((-len(header))%32));stream.write(data);stream.write(b'\0'*((-len(data))%32))
