"""Synthetic byte-composition controls; no native model qualification."""
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import lighton4b_mixed_pack as pack
from gguf_file_type_canonicalize import parse

def fixture(path,scale,file_type):
    names=['l.embed_tokens.weight','l.layers.0.self_attn.q_proj.weight','l.layers.0.mlp.up_proj.weight'];data=[struct.pack('<4f',*(scale*x for x in(1,2,3,4)))for _ in names]
    metadata=pack.string('general.file_type')+struct.pack('<II',4,file_type);info=b'';records={}
    for i,(name,raw)in enumerate(zip(names,data)):
        info+=pack.string(name)+struct.pack('<IQQIQ',2,2,2,0,i*32);records[name]={'shape':[2,2],'bytes':16,'sha256':hashlib.sha256(raw).hexdigest()}
    header=struct.pack('<4sIQQ',b'GGUF',3,3,1)+metadata+info;header+=b'\0'*((-len(header))%32)
    with path.open('wb')as stream:
        stream.write(header)
        for i,raw in enumerate(data):stream.write(raw+(b'\0'*16 if i<2 else b''))
    return records
class Controls(unittest.TestCase):
    def test_exact_selected_source_payloads_and_tag(self):
        with tempfile.TemporaryDirectory()as d:
            d=Path(d);a=d/'q4.gguf';b=d/'q8.gguf';out=d/'mixed.gguf';expected={'q4_k':fixture(a,1,14),'q8_0':fixture(b,2,7)}
            receipt=pack.compose(a,b,out,expected);self.assertEqual(receipt['tensor_count'],3);self.assertEqual(receipt['metadata_file_type'],14)
            by={r['name']:r for r in receipt['records']};self.assertEqual(by['l.layers.0.self_attn.q_proj.weight']['source_recipe'],'q8_0');self.assertEqual(by['l.layers.0.mlp.up_proj.weight']['source_recipe'],'q4_k');self.assertEqual(by['l.embed_tokens.weight']['source_recipe'],'q4_k')
            self.assertEqual(parse(out)['metadata'][pack.KEY]['value'],pack.RECIPE)
            with self.assertRaises(ValueError):pack.compose(out,b,d/'bad.gguf',expected)
    def test_corrupt_source_payload_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            d=Path(d);a=d/'q4.gguf';b=d/'q8.gguf';expected={'q4_k':fixture(a,1,14),'q8_0':fixture(b,2,7)}
            payload=bytearray(a.read_bytes());payload[-1]^=1;a.write_bytes(payload)
            with self.assertRaisesRegex(ValueError,'digest'):pack.compose(a,b,d/'bad.gguf',expected)
if __name__=='__main__':unittest.main()
