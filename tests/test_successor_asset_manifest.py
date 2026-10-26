import io
import importlib.util
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import json
import os
import subprocess
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ci-heavy'))
import successor_assets as consumer
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
class ConsumerLayoutTests(unittest.TestCase):
 def test_wasm_named_roots_roundtrip(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);payload=root/'payload';payload.mkdir()
   for kind in ('ocr','embed'):
    name='crispembed-'+kind+'-wasm'
    with tarfile.open(payload/(name+'.tar.gz'),'w:gz') as archive:
     item=tarfile.TarInfo(name+'/crispembed_'+kind+'.js');data=b'fixture module';item.size=len(data)
     archive.addfile(item,io.BytesIO(data))
   manifest=m.build(payload,'wasm',consumer.ROOT)
   def artifact(run,name,destination,limit):
    with zipfile.ZipFile(destination,'w') as archive:
     if name=='successor-wasm-manifest':archive.writestr('successor-wasm-manifest.json',json.dumps(manifest))
     else:
      for file in payload.iterdir():archive.write(file,file.name)
    return {'id':1,'name':name,'digest':'sha256:'+m.sha(destination),'size_in_bytes':destination.stat().st_size}
   service={'head_sha':manifest['source_commit'],'path':'.github/workflows/release-wasm.yml','status':'completed','conclusion':'success','event':'workflow_dispatch'}
   with mock.patch.dict(os.environ,{'GITHUB_ACTIONS':'true','GITHUB_SHA':manifest['source_commit']}),mock.patch.object(consumer,'api',return_value=service),mock.patch.object(consumer,'artifact',side_effect=artifact):
    consumer.prepare(1,'wasm','crispembed-wasm-release',root/'out')
   for kind in ('ocr','embed'):
    self.assertTrue((root/'out'/('crispembed-'+kind+'-wasm')/('crispembed_'+kind+'.js')).is_file())
 def test_windows_drive_path_rejected(self):
  with self.assertRaises(ValueError):m.safe_name('C:escape.dll')

class RuntimeClosureTests(unittest.TestCase):
 def fixture(self,root):
  def git(*args):
   return subprocess.check_output(['git','-C',str(root),'-c','user.name=Audit Test',
    '-c','user.email=audit@example.invalid','-c','core.hooksPath='+str(root/'no-hooks'),*args],text=True).strip()
  subprocess.run(['git','init','-q',str(root)],check=True)
  (root/'src').mkdir();(root/'docs').mkdir()
  (root/'src/code.c').write_text('int fixture = 1;\n');(root/'VERSION').write_text('1.0.0\n')
  (root/'docs/audit.md').write_text('initial audit\n');git('add','.');git('commit','-qm','fixture')
  return git,git('rev-parse','HEAD')
 def test_audit_commit_allowed_runtime_commit_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);git,producer=self.fixture(root)
   (root/'docs/audit.md').write_text('new audit only\n');git('add','.');git('commit','-qm','audit update')
   with mock.patch.object(consumer,'ROOT',root):
    proof=consumer.runtime_closure(producer)
    self.assertEqual(proof['consumer_source'],git('rev-parse','HEAD'))
    (root/'src/code.c').write_text('int fixture = 2;\n');git('add','.');git('commit','-qm','runtime update')
    with self.assertRaises(ValueError):consumer.runtime_closure(producer)
 def test_dirty_production_source_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);git,producer=self.fixture(root)
   (root/'VERSION').write_text('2.0.0\n')
   with mock.patch.object(consumer,'ROOT',root),self.assertRaises(ValueError):consumer.runtime_closure(producer)

if __name__=='__main__':unittest.main()
