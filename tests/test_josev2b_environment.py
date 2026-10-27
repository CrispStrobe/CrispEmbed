#!/usr/bin/env python3
"""Bounded stdlib controls. No dependency/framework/payload download."""
import importlib.util
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import site
import importlib.metadata
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import josev2b_environment as environment
from josev2b_environment import metadata_url,MetadataNetworkGuard,verify_lock,dependency_closure,active_requirements,sha,requirements_bytes

INDEX='https://pypi.org/simple/torch/'
META='https://files.pythonhosted.org/packages/ab/cd/'+'a'*64+'/torch-2.13.0-cp313-cp313-manylinux_2_28_x86_64.whl.metadata'

class EnvironmentTests(unittest.TestCase):
    def test_metadata_only_allowlist(self):
        self.assertEqual(metadata_url(INDEX),'simple-index')
        self.assertEqual(metadata_url(META),'pep658-core-metadata')
        for url in [META[:-9],META.replace('.whl.metadata','.tar.gz'),
                    META.replace('.whl.metadata','.zip'),META.replace('.whl.metadata','.whl?metadata=1'),
                    META.replace('files.pythonhosted.org','example.org'),INDEX.replace('https:','http:'),
                    INDEX+'?format=wheel',INDEX+'#wheel']:
            with self.assertRaises(ValueError):metadata_url(url)

    def test_payload_never_reaches_transport_even_redirect_target(self):
        with tempfile.TemporaryDirectory() as d:
            guard=MetadataNetworkGuard(Path(d)/'journal.json')
            transport=Mock(side_effect=AssertionError('transport must not see payload'))
            wrapped=guard.wrap(transport)
            for url in [META[:-9],META.replace('.whl.metadata','.tar.gz')]:
                for method in ['GET','HEAD']:
                    with self.assertRaises(ValueError):wrapped(None,types.SimpleNamespace(url=url,method=method))
            transport.assert_not_called()
            self.assertTrue(all(not e['allowed'] for e in guard.events))

    def test_metadata_transport_and_bounded_body(self):
        with tempfile.TemporaryDirectory() as d:
            guard=MetadataNetworkGuard(Path(d)/'journal.json',max_bytes=3)
            raw=types.SimpleNamespace(read=Mock(return_value=b'abcd'))
            response=types.SimpleNamespace(headers={},raw=raw,close=Mock())
            wrapped=guard.wrap(Mock(return_value=response))
            result=wrapped(None,types.SimpleNamespace(url=META,method='GET'))
            with self.assertRaisesRegex(ValueError,'budget'):result.raw.read()
            response.close.assert_called_once()

    def test_complete57_lock_and_published_root_pins(self):
        lock=json.loads((ROOT/'docs/josev2b-environment-lock.json').read_text())
        requirements=(ROOT/'tools/kaggle/josev2b-original/requirements.lock').read_bytes()
        self.assertEqual(len(lock['packages']),57)
        self.assertEqual(lock['roots']['tokenizers'],'0.22.2')
        self.assertTrue(verify_lock(lock,lock['roots'],requirements)['complete'])
        self.assertEqual(lock['dependency_closure'],dependency_closure(lock['packages'],lock['roots'],'3.13.11'))
        self.assertFalse(any(r['name']=='wrapt' for r in lock['packages']))
        rows={r['name']:r for r in lock['packages']}
        self.assertEqual(rows['torch']['sha256'],'a3a9a21312872af8a26950b2c15680335a386a1f56ed03e780653d78b9607e9e')
        self.assertEqual(rows['torchvision']['sha256'],'ad7b3a439265cc3739a4ab5b4c998c0e38ea99c0ee7ca4dea35c5d0b099ec237')
        self.assertEqual(rows['tokenizers']['sha256'],'369cc9fc8cc10cb24143873a0d95438bb8ee257bb80c71989e3ee290e8d72c67')

    def test_transitive_cuda_extras_missing_or_conflicting_reject(self):
        lock=json.loads((ROOT/'docs/josev2b-environment-lock.json').read_text())
        self.assertIn('cublas',lock['dependency_closure']['active_extras']['cuda-toolkit'])
        missing=[r for r in lock['packages'] if r['name']!='nvidia-cublas']
        with self.assertRaisesRegex(ValueError,'missing|unsatisfied'):
            dependency_closure(missing,lock['roots'])
        bad=copy.deepcopy(lock['packages']);next(r for r in bad if r['name']=='nvidia-cublas')['version']='0.0.0'
        with self.assertRaisesRegex(ValueError,'unsatisfied'):dependency_closure(bad,lock['roots'])
        bad=copy.deepcopy(lock['packages'])+[{'name':'wrapt','version':'1','requires_dist':[]}]
        with self.assertRaisesRegex(ValueError,'unneeded'):dependency_closure(bad,lock['roots'])

    def test_requirements_or_platform_tampering_rejects(self):
        lock=json.loads((ROOT/'docs/josev2b-environment-lock.json').read_text())
        requirements=(ROOT/'tools/kaggle/josev2b-original/requirements.lock').read_bytes()
        with self.assertRaisesRegex(ValueError,'requirements'):verify_lock(lock,lock['roots'],requirements+b'\n')
        bad=copy.deepcopy(lock);bad['packages'][0]['filename']='bad-1-py3-none-macosx_10_9_x86_64.whl'
        with self.assertRaises(ValueError):verify_lock(bad,bad['roots'],requirements)
        bad=copy.deepcopy(lock);bad['roots']['tokenizers']='0.23.0'
        with self.assertRaises(ValueError):verify_lock(bad,lock['roots'],requirements)

    def test_linux_active_markers_keep_windows_variants_distinct(self):
        a=['demo==1; sys_platform == "linux"','other==1; sys_platform == "win32" and platform_machine == "AMD64"']
        b=['demo==1; sys_platform == "linux"','other==1; sys_platform == "win32" and platform_machine == "x86_64"']
        self.assertEqual(active_requirements(a),active_requirements(b))
        self.assertNotEqual(active_requirements(a),active_requirements(['demo==2; sys_platform == "linux"']))

    def test_expected_prefix_rejects_user_or_external_site(self):
        with tempfile.TemporaryDirectory() as d:
            prefix=Path(d)/'venv';prefix.mkdir()
            with patch.object(sys,'prefix',str(prefix)),patch.object(sys,'base_prefix',str(Path(d)/'base')), \
                 patch.object(sys,'executable',str(prefix/'bin/python')),patch.object(sys,'path',[str(prefix/'lib/python3.13/site-packages')]), \
                 patch.object(site,'ENABLE_USER_SITE',False):
                self.assertTrue(environment.isolated_prefix(prefix)['prefix_matches_expected'])
                with patch.object(sys,'path',['/ambient/site-packages']):
                    with self.assertRaisesRegex(ValueError,'external'):environment.isolated_prefix(prefix)
                with patch.object(site,'ENABLE_USER_SITE',True):
                    with self.assertRaisesRegex(ValueError,'prefix/user'):environment.isolated_prefix(prefix)

    def test_owner_metadata_selection_rejects_missing_duplicate_or_vendor_fallback(self):
        own=Path('demo_pkg-1.dist-info/METADATA')
        vendor=Path('demo_pkg/_vendor/demo_pkg-1.dist-info/METADATA')
        dist=types.SimpleNamespace(files=[vendor,own,Path('demo_pkg/__init__.py')])
        self.assertEqual(environment.owner_metadata_file(dist,'demo-pkg','1'),own)
        for files in [[vendor], [own,own], [Path('different-1.dist-info/METADATA')],
                      [Path('demo_pkg-2.dist-info/METADATA')], [Path('/demo_pkg-1.dist-info/METADATA')]]:
            dist.files=files
            with self.assertRaisesRegex(ValueError,'owning METADATA file missing/ambiguous'):
                environment.owner_metadata_file(dist,'demo-pkg','1')

    def test_installed_metadata_unlocked_package_and_loaded_origin_controls(self):
        with tempfile.TemporaryDirectory() as d:
            prefix=Path(d);file=prefix/'demo-1.dist-info/METADATA';file.parent.mkdir();file.write_bytes(b'demo metadata')
            vendor=prefix/'demo/_vendor/vendor-1.dist-info/METADATA'
            vendor.parent.mkdir(parents=True);vendor.write_bytes(b'vendored metadata')
            class Dist:
                def __init__(self,name,version):self.metadata={'Name':name};self.version=version
                files=[Path('demo-1.dist-info/METADATA'),Path('demo/_vendor/vendor-1.dist-info/METADATA')]
                def locate_file(self,f):return prefix/f
            demo,pip=Dist('demo','1'),Dist('pip','26.2.2')
            row={'name':'demo','version':'1','core_metadata_sha256':sha(file.read_bytes())}
            cfg={'environment_lock':{'sha256':'a'*64},'runtime':{'python_minor':'3.13','torch':'2.13.0'}}
            with patch.object(environment,'isolated_prefix',return_value={}), \
                 patch.object(environment,'runtime_target',return_value={}), \
                 patch.object(environment,'locked_files',return_value={'packages':[row]}), \
                 patch.object(importlib.metadata,'distributions',return_value=[demo,pip]):
                self.assertTrue(environment.verify_installed(cfg,prefix)['all_wheel_metadata_hashes_exact'])
                with patch.object(demo,'locate_file',side_effect=lambda f: prefix if str(f)=='' else prefix.parent/'outside-METADATA'):
                    with self.assertRaisesRegex(ValueError,'owning METADATA origin outside'):
                        environment.verify_installed(cfg,prefix)
                with patch.object(importlib.metadata,'distributions',return_value=[demo,pip,Dist('wrapt','1')]):
                    with self.assertRaisesRegex(ValueError,'unlocked'):environment.verify_installed(cfg,prefix)
                file.write_bytes(b'altered')
                # A vendor with the expected bytes cannot substitute for a
                # corrupted owning distribution's metadata.
                vendor.write_bytes(b'demo metadata')
                with self.assertRaisesRegex(ValueError,'METADATA'):environment.verify_installed(cfg,prefix)
                file.write_bytes(b'demo metadata')
                with patch('importlib.import_module',return_value=types.SimpleNamespace(__version__='2.13.0',__file__='/ambient/torch.py')):
                    with self.assertRaisesRegex(ValueError,'loaded framework'):environment.verify_installed(cfg,prefix,True)

if __name__=='__main__':unittest.main()
