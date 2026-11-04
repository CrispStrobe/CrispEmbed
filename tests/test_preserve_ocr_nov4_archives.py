#!/usr/bin/env python3
"""Synthetic provenance/stream/readback controls, without network or model work."""
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import time
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/ci-heavy'))
spec=importlib.util.spec_from_file_location('preserve',ROOT/'tools/ci-heavy/preserve_ocr_nov4_archives.py')
worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


def metadata(target):
    repository=dict(id=worker.REPO_ID,full_name=worker.archive.REPOSITORY)
    run=dict(id=target['run_id'],head_sha=target['source'],head_branch=target['branch'],
             event='workflow_dispatch',path='.github/workflows/heavy-cpu.yml',run_attempt=1,
             status='completed',conclusion=target['conclusion'],repository=repository,head_repository=repository)
    item=dict(id=target['artifact_id'],name='heavy-'+str(target['run_id']),size_in_bytes=target['bytes'],
              digest='sha256:'+target['sha256'],expired=False,expires_at='2099-01-01T00:00:00Z',
              url=worker.archive.API+'/actions/artifacts/'+str(target['artifact_id']),
              archive_download_url=worker.archive.API+'/actions/artifacts/'+str(target['artifact_id'])+'/zip',
              workflow_run=dict(id=target['run_id'],head_sha=target['source'],head_branch=target['branch'],
                                repository_id=worker.REPO_ID,head_repository_id=worker.REPO_ID))
    return run,item


class Controls(unittest.TestCase):
    def test_frozen_targets_include_failed_original(self):
        self.assertEqual(len(worker.TARGETS),10)
        self.assertEqual(sum('german' in t['label'] for t in worker.TARGETS),1)
        for target in worker.TARGETS:
            worker.validate_metadata(target,*metadata(target))
        self.assertEqual(next(t for t in worker.TARGETS if t['label']=='lighton08-inventory-table')['conclusion'],'failure')

    def test_exact_batches_and_arguments(self):
        self.assertEqual(worker.select_targets(False),worker.TARGETS)
        self.assertEqual(worker.select_targets(True),(worker.GERMAN,))
        self.assertEqual(sum(t['bytes'] for t in worker.select_targets(True)),2543809676)
        self.assertEqual(sum(t['bytes'] for t in worker.select_targets(False)),worker.TOTAL)
        self.assertEqual(worker.GERMAN['source'],'1bb0ddb1d9ac12ded51eb5bd0edd89be0a8259e6')
        worker.validate_metadata(worker.GERMAN,*metadata(worker.GERMAN))
        for flag,value in ((False,''),(True,'--german-only')):
            worker.validate_batch_args({'PRESERVE_SELECTED_ARGS':value},flag)
        for flag,value in ((False,'--german-only'),(True,''),(True,'--german-only extra'),(False,' ')):
            with self.subTest(flag=flag,value=value),self.assertRaises(ValueError):
                worker.validate_batch_args({'PRESERVE_SELECTED_ARGS':value},flag)
        with self.assertRaises(ValueError):worker.select_targets(1)

    def test_producer_mutations(self):
        target=next(t for t in worker.TARGETS if t['label']=='lighton08-inventory-table')
        for key,value in dict(head_sha='a'*40,head_branch='other',event='push',path='other',
                              run_attempt=2,status='in_progress',conclusion='success',id=1).items():
            run,item=metadata(target);run[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):worker.validate_metadata(target,run,item)
        for key in ('repository','head_repository'):
            run,item=metadata(target);run[key]=dict(id=1,full_name=worker.archive.REPOSITORY)
            with self.assertRaises(ValueError):worker.validate_metadata(target,run,item)

    def test_artifact_mutations(self):
        target=worker.TARGETS[0]
        for key,value in dict(id=1,name='other',size_in_bytes=1,digest='sha256:'+'a'*64,expired=True,
                              expires_at='2000-01-01T00:00:00Z',url='other',archive_download_url='other').items():
            run,item=metadata(target);item[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):worker.validate_metadata(target,run,item)
        for key in ('id','head_sha','head_branch','repository_id','head_repository_id'):
            run,item=metadata(target);item['workflow_run'][key]='other'
            with self.subTest(key=key),self.assertRaises(ValueError):worker.validate_metadata(target,run,item)
        altered=copy.deepcopy(target);altered['label']='arbitrary'
        with self.assertRaises(ValueError):worker.validate_metadata(altered,*metadata(target))

    def test_own_producer_can_be_running(self):
        target=worker.TARGETS[0];run,_=metadata(target);run.update(status='in_progress',conclusion=None)
        worker.producer(run,target['source'],target['branch'],target['run_id'],completed=False)
        with self.assertRaises(ValueError):worker.producer(run,target['source'],target['branch'],target['run_id'],conclusion='success')

    def test_bounded_stream_and_retained_partial(self):
        payload=b'x'*(worker.archive.CHUNK+3)
        class Stream(io.BytesIO):
            def read(self,size=-1):
                if not 0<size<=worker.archive.CHUNK: raise AssertionError('unbounded read')
                return super().read(size)
        output=io.BytesIO()
        result=worker.stream_hash(Stream(payload),len(payload),hashlib.sha256(payload).hexdigest(),time.monotonic()+10,output)
        self.assertEqual(result['bytes'],len(payload));self.assertEqual(output.getvalue(),payload)
        for size,digest,deadline in ((len(payload)-1,hashlib.sha256(payload).hexdigest(),time.monotonic()+10),
                                    (len(payload)+1,hashlib.sha256(payload).hexdigest(),time.monotonic()+10),
                                    (len(payload),'0'*64,time.monotonic()+10),
                                    (len(payload),hashlib.sha256(payload).hexdigest(),time.monotonic()-1)):
            with self.assertRaises(ValueError):worker.stream_hash(Stream(payload),size,digest,deadline)

    def test_inner_zip_readback(self):
        payload=b'unchanged original ZIP transport'
        targets=(dict(label='fixture',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()),)
        for names,data in ((['fixture.zip'],payload),(['other.zip'],payload),(['fixture.zip','fixture.zip'],payload),
                           (['../fixture.zip'],payload),(['fixture.zip'],b'bad'),(['fixture.zip'],b'x'*len(payload))):
            with self.subTest(names=names,data=data),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'outer.zip'
                with zipfile.ZipFile(path,'w') as z:
                    for name in names:z.writestr(name,data)
                if names==['fixture.zip'] and data==payload:
                    self.assertEqual(worker.readback(path,targets)[0]['bytes'],len(payload))
                else:
                    with self.assertRaises(ValueError):worker.readback(path,targets)


if __name__=='__main__':unittest.main()
