"""No model files: raw quant-row transport and full descriptor/header controls."""
import hashlib,io,json,struct,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_upstream_physical_pack as pack
class PhysicalPack(unittest.TestCase):
 def test_whole_q4_and_q6_rows_copy_without_requantization(self):
  for kind,width in [(12,144),(14,210)]:
   rows=[bytes([i+1])*width for i in range(4)];raw=b''.join(rows);src=io.BytesIO(b'prefix'+raw);dst=io.BytesIO(b'header'+bytes(len(raw)))
   sha,changed=pack.transport_tensor(src,dst,6,6,len(raw),[0,2,1,3],kind)
   wanted=b''.join(rows[i]for i in [0,2,1,3]);self.assertTrue(changed);self.assertEqual(dst.getvalue(),b'header'+wanted);self.assertEqual(sha,hashlib.sha256(wanted).hexdigest());self.assertEqual(src.getvalue(),b'prefix'+raw)
 def test_duplicate_or_half_block_permutation_rejected(self):
  for order,count in [([0,0],288),([1,0],144)]:
   with self.assertRaises(ValueError):pack.transport_tensor(io.BytesIO(bytes(count)),io.BytesIO(),0,0,count,order,12)
 def test_identity_and_small_f32_head_parameter_transport(self):
  for order in [None,[0,1,2,3]]:
   data=bytes(range(16));out=io.BytesIO();sha,permuted=pack.transport_tensor(io.BytesIO(data),out,0,0,16,order,0);self.assertFalse(permuted);self.assertEqual(out.getvalue(),data)
  data=struct.pack('<ffff',1,2,3,4);out=io.BytesIO();pack.transport_tensor(io.BytesIO(data),out,0,0,16,[0,2,1,3],0);self.assertEqual(struct.unpack('<ffff',out.getvalue()),(1,3,2,4))
 def test_model6_header_retains_physical_types_and_declares_semantics(self):
  original=json.loads((ROOT/'tests/fixtures/lightonocr3-4b-source/model.safetensors.header.json').read_text());physical=json.loads((ROOT/'tests/fixtures/lighton4b-upstream-q4km-header-v1.json').read_text());base={'metadata':{},'tensors':{}};up={'tensors':{}}
  raw=b'GGUF'+struct.pack('<IQQ',3,723,4)
  for name,kind,value in [('general.file_type',4,14),('qwen35_ocr.format_version',4,5),('qwen35_ocr.zero_centered_text_norm',7,True),('qwen35_ocr.special_token_ids',9,[248046])]:
   start=len(raw);raw+=pack.text(name)+struct.pack('<I',kind);offset=len(raw)
   if kind==4:raw+=struct.pack('<I',value)
   elif kind==7:raw+=bytes([value])
   else:raw+=struct.pack('<IQi',5,1,value[0])
   base['metadata'][name]={'kind':kind,'value':value,'offset':offset,'raw_sha256':hashlib.sha256(raw[start:]).hexdigest()}
  base['data_offset']=len(raw)
  for source,row in original.items():
   if source=='__metadata__':continue
   name=('l.'+source.removeprefix('model.language_model.')if source.startswith('model.language_model.')else'v.'+source.removeprefix('model.visual.'));shape=row['shape'];stored=list(reversed([shape[0],__import__('math').prod(shape[1:])]if name=='v.patch_embed.proj.weight'else shape));kind=8 if name=='l.embed_tokens.weight' else 12 if name.startswith('l.')and len(shape)==2 else 0;block,width=pack.TYPES[kind];base['tensors'][name]={'stored_shape':stored,'storage_rank':len(stored),'type':kind,'bytes':__import__('math').prod(shape)//block*width,'data_offset':0}
   if name.startswith('l.'):
    n=pack.mapping.physical_name(name);p=physical['tensors'][n];block,width=pack.TYPES[p['type']];up['tensors'][n]={'stored_shape':p['shape'],'type':p['type'],'bytes':p['elements']//block*width,'data_offset':0}
  with tempfile.TemporaryDirectory()as folder:
   path=Path(folder)/'header-only';path.write_bytes(raw);header,layout,size,changed=pack.prepare_header(base,up,path,{'upstream_control_ids':[248044,248046]})
  self.assertEqual(len(layout),723);self.assertEqual(sum(x['type']==14 for x in layout),33);self.assertEqual(sum(x['type']==12 for x in layout),216);self.assertEqual(sum(x['type']==0 for x in layout),474);self.assertIn('qwen35_ocr.special_token_ids',changed);self.assertEqual(struct.unpack_from('<Q',header,8)[0],723);self.assertEqual(len(header)%32,0);self.assertGreater(size,len(header));self.assertIn(pack.text(pack.RECIPE),header);self.assertEqual(sum(x['name'].endswith('.linear_attn.out_proj.weight')and x['row_order']==list(range(2560))for x in layout),24)
class Q4ReportAliases(unittest.TestCase):
    def test_report_aliases_cannot_overwrite_models_or_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sources = [root / name for name in ('base', 'upstream')]
            for i, path in enumerate(sources):
                path.write_bytes(bytes([i]) * 32)
            output = root / 'candidate'
            partial = root / 'candidate.partial'
            # Missing candidate paths matter too: report==output would otherwise
            # overwrite the fully converted model with JSON on success.
            for report in [*sources, output, partial, root / '.' / 'candidate']:
                with self.subTest(report=report.name), patch.object(pack, 'digest') as digest:
                    with self.assertRaisesRegex(ValueError, 'must not alias'):
                        pack.convert(*sources, output, report)
                    digest.assert_not_called()
                self.assertFalse(output.exists())
                self.assertFalse(partial.exists())
                for i, path in enumerate(sources):
                    self.assertEqual(path.read_bytes(), bytes([i]) * 32)
            for link_type in ('symlink', 'hardlink'):
                for i, source in enumerate(sources):
                    alias = root / f'{link_type}-{i}'
                    if link_type == 'symlink':
                        alias.symlink_to(source)
                    else:
                        alias.hardlink_to(source)
                    with self.subTest(link_type=link_type, source=source.name):
                        with self.assertRaisesRegex(ValueError, 'must not alias'):
                            pack.convert(*sources, output, alias)
                        self.assertEqual(source.read_bytes(), bytes([i]) * 32)
            # Keep ordinary failed-conversion diagnostics when paths are safe.
            report = root / 'report.json'
            with self.assertRaisesRegex(ValueError, 'whole protected source identity'):
                pack.convert(*sources, output, report)
            self.assertEqual(pack.json.loads(report.read_text())['status'], 'failed')

if __name__=='__main__':unittest.main()
