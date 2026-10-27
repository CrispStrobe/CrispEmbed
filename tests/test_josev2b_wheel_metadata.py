#!/usr/bin/env python3
"""Tiny synthetic ZIPs only; no network, installed distributions or frameworks."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wheel_metadata',ROOT/'tools/ci-heavy/josev2b_wheel_metadata.py')
audit = importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
LIMITS = {'max_metadata_bytes':1048576,'max_metadata_bytes_per_wheel':8388608}
DATA = b'Metadata-Version: 2.4\nName: demo\nVersion: 1\nRequires-Python: >=3.10\nRequires-Dist: dep>=2; extra == "test"\nProvides-Extra: test\n\nDescription.\n'


def row():
    return {'name':'demo','version':'1','filename':'demo-1-py3-none-any.whl',
            'url':'https://files.pythonhosted.org/packages/aa/bb/'+'1'*64+'/demo-1-py3-none-any.whl',
            'core_metadata_sha256':hashlib.sha256(DATA).hexdigest(),'core_metadata_bytes':len(DATA),
            'requires_python':'>=3.10','requires_dist':['dep>=2; extra == "test"']}


class Controls(unittest.TestCase):
    def inspect(self, entries, *, sidecar=DATA, declaration=None):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tiny.zip'
            with zipfile.ZipFile(path,'w') as z:
                for name,data in entries:z.writestr(name,data)
                record=io.StringIO();csv.writer(record).writerows((name,'','') for name,_ in entries)
                z.writestr('demo-1.dist-info/RECORD',record.getvalue())
            return audit.archive_metadata(path,declaration or row(),sidecar,LIMITS)

    def test_exact_owner_and_vendored_candidates_are_distinct(self):
        vendor=DATA.replace(b'Name: demo',b'Name: vendored')
        result,owner=self.inspect([('demo-1.dist-info/METADATA',DATA),
            ('demo/_vendor/vendored-1.dist-info/METADATA',vendor)])
        self.assertTrue(result['passed']);self.assertEqual(owner,DATA)
        self.assertEqual(result['legacy_selector_candidate_count'],2)
        self.assertFalse(result['legacy_selector_unique'])
        self.assertEqual([r['top_level'] for r in result['candidate_inventory']],[True,False])
        self.assertEqual(result['candidate_inventory'][1]['name'],'vendored')

    def test_no_owner_multiple_owner_or_wrong_identity_reject(self):
        for entries in [ [('demo/vendor-1.dist-info/METADATA',DATA)],
            [('demo-1.dist-info/METADATA',DATA),('other-1.dist-info/METADATA',DATA)],
            [('other-1.dist-info/METADATA',DATA)],
            [('demo-1.dist-info/METADATA',DATA.replace(b'Name: demo',b'Name: other'))] ]:
            with self.assertRaises(ValueError):self.inspect(entries)

    def test_byte_difference_cannot_be_waived_by_equal_dependency_fields(self):
        changed=DATA.replace(b'Description.',b'Another description.')
        result,_=self.inspect([('demo-1.dist-info/METADATA',changed)])
        self.assertFalse(result['passed']);self.assertFalse(result['owner_matches_lock'])
        self.assertTrue(all(result['dependency_fields_exact'].values()))

    def test_semantic_declaration_differences_are_retained_and_rejected(self):
        changed=DATA.replace(b'dep>=2;',b'dep>=3;')
        result,_=self.inspect([('demo-1.dist-info/METADATA',changed)])
        self.assertFalse(result['passed']);self.assertFalse(result['dependency_fields_exact']['requires_dist'])
        self.assertIn('dep>=3',result['owner_fields']['requires_dist'][0])
        result,_=self.inspect([('demo-1.dist-info/METADATA',DATA)],sidecar=changed)
        self.assertFalse(result['passed']);self.assertFalse(result['sidecar_matches_lock'])
        result,_=self.inspect([('demo-1.dist-info/METADATA',DATA)],sidecar=None)
        self.assertFalse(result['passed']);self.assertTrue(result['owner_matches_lock'])
        self.assertIsNone(result['sidecar_metadata_sha256'])

    def test_oversized_or_duplicate_zip_metadata_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tiny.zip'
            with zipfile.ZipFile(path,'w') as z:z.writestr('demo-1.dist-info/METADATA',DATA)
            with self.assertRaisesRegex(ValueError,'extent'):
                audit.archive_metadata(path,row(),DATA,dict(LIMITS,max_metadata_bytes=1))
        with self.assertWarns(UserWarning),self.assertRaisesRegex(ValueError,'Duplicate'):
            self.inspect([('demo-1.dist-info/METADATA',DATA),('demo-1.dist-info/METADATA',DATA)])

    def test_only_locked_wheel_and_sidecar_urls_can_reach_transport(self):
        r=row()
        audit.request_url(r['url'],r,False);audit.request_url(r['url']+'.metadata',r,True)
        for url in [r['url']+'?download=1',r['url'].replace('https:','http:'),
                    r['url'].replace('files.pythonhosted.org','example.com'),r['url']+'.metadata',
                    r['url'].replace('.whl','.tar.gz')]:
            with self.assertRaises(ValueError):audit.request_url(url,r,False)

    def test_original_lock_profile_and_resource_bounds_stay_fixed(self):
        protocol=json.loads(audit.PROTOCOL.read_text());lock=json.loads((ROOT/'docs/josev2b-environment-lock.json').read_text())
        self.assertEqual(len(lock['packages']),57)
        self.assertEqual(sum(r['bytes'] for r in lock['packages']),protocol['total_wheel_bytes'])
        self.assertFalse(any(protocol['qualification'].values()))
        self.assertLess(max(r['bytes'] for r in lock['packages']),1024**3)
        self.assertEqual(hashlib.sha256((ROOT/'docs/josev2b-environment-lock.json').read_bytes()).hexdigest(),
                         protocol['old_source_files']['docs/josev2b-environment-lock.json'])


if __name__=='__main__':unittest.main()
