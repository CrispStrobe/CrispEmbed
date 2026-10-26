import io
import importlib.util
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import successor_asset_manifest as m

class ArchiveTests(unittest.TestCase):
 def archive(self,root,entries):
  path=root/'test.tar.gz'
  with tarfile.open(path,'w:gz') as archive:
   for name,kind,value in entries:
    info=tarfile.TarInfo(name)
    if kind=='file':
     payload=value.encode();info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
    else:info.type=tarfile.SYMTYPE;info.linkname=value;archive.addfile(info)
  return path
 def test_exact_file_and_link(self):
  with tempfile.TemporaryDirectory() as d:
   rows=m.archive_members(self.archive(Path(d),[('./lib.so.1','file','abc'),('./lib.so','link','lib.so.1')]))
   self.assertEqual(rows[1]['sha256'],'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
   self.assertEqual(rows[0]['resolved_target'],'lib.so.1')
 def test_duplicate_normalized(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):m.archive_members(self.archive(Path(d),[('x','file','a'),('./x','file','b')]))
 def test_traversal(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):m.archive_members(self.archive(Path(d),[('../escape','file','a')]))
 def test_dangling(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):m.archive_members(self.archive(Path(d),[('x','link','absent')]))
 def test_cycle(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):m.archive_members(self.archive(Path(d),[('x','link','y'),('y','link','x')]))
 def test_incomplete_asset_set(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):m.build(Path(d),'native',Path(__file__).resolve().parents[1])
if __name__=='__main__':unittest.main()
