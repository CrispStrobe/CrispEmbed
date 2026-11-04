#!/usr/bin/env python3
"""Pure completion barriers, without importing model packages or accessing network."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('completion',ROOT/'tools/ci-heavy/lighton3_kv_complete_first.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


def models():return {p:dict(bytes=100,sha256=('a' if p=='f32' else 'b')*64) for p in ('f32','f16')}


def executions():
    result={}
    for precision,gate,arm in worker.EXPECTED:
        result[precision,gate,arm]=dict(git_commit=worker.NATIVE,precision=precision,arm=arm,gate=gate,build_mode='portable',
              schema_version=1,model_sha256=models()[precision]['sha256'],model_bytes=100,
              binary_sha256='c'*64,process=dict(exit=0,timed_out=False))
    return result


def summary():
    return dict(source=worker.NATIVE,protocol_sha256=worker.OLD_PROTOCOL_SHA,completed=False,**{'pass':False},phase='first',
                case=worker.CASE,native_request_processes=10,original_forward_calls=0,
                failure=dict(type='ValueError',message='KV changed prefill bytes: generated_ids.f32'),
                rows=[dict(precision=p,gate=g) for p,g in (('f32','0'),('f32','1'),('f16','0'))],
                commands=[dict(name=p+'-kv'+g+'-'+a,exit=0,timed_out=False) for p,g,a in worker.EXPECTED])


class Controls(unittest.TestCase):
    def test_completion_protocol_native_closure(self):
        protocol=json.loads((ROOT/worker.PROTOCOL_NAME).read_text())
        self.assertEqual(protocol['native_source'],worker.NATIVE)
        self.assertEqual(protocol['preserved'],worker.PRESERVED)
        expected={'src/lightonocr.cpp','src/lightonocr3_profile.h','tests/test_lightonocr3_contract.cpp'}
        self.assertEqual(set(protocol['owned_native_sha256']),expected)
        for name,digest in protocol['owned_native_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)
        self.assertEqual(hashlib.sha256((ROOT/'docs/lighton3-kv-f32-ab-v1.json').read_bytes()).hexdigest(),worker.OLD_PROTOCOL_SHA)

    def test_operator_native_sources_bound_separately(self):
        protocol=dict(schema='crispembed.lighton3.kv-complete-first.v1',native_source=worker.NATIVE,preserved=worker.PRESERVED)
        worker.validate_closure(protocol,worker.NATIVE,'d'*40,b'script',b'protocol',b'script',b'protocol')
        for native,operator,script,document in (('e'*40,'d'*40,b'script',b'protocol'),
                                               (worker.NATIVE,'bad',b'script',b'protocol'),
                                               (worker.NATIVE,'d'*40,b'changed',b'protocol'),
                                               (worker.NATIVE,'d'*40,b'script',b'changed')):
            with self.assertRaises(ValueError):worker.validate_closure(protocol,native,operator,script,document,b'script',b'protocol')

    def test_preserved_metadata_bound(self):
        api='https://api.github.com/repos/CrispStrobe/CrispEmbed'
        repository=dict(id=1210669686,full_name='CrispStrobe/CrispEmbed')
        run=dict(id=worker.PRESERVED['run_id'],head_sha=worker.NATIVE,head_branch=worker.PRESERVED['branch'],
                 event='workflow_dispatch',path='.github/workflows/heavy-cpu.yml',status='completed',conclusion='failure',
                 run_attempt=1,repository=repository,head_repository=repository)
        for kind in ('raw','review'):
            target=worker.PRESERVED[kind]
            item=dict(id=target['artifact_id'],name='lighton3-kv-'+kind+'-'+str(worker.PRESERVED['run_id']),
                      size_in_bytes=target['bytes'],digest='sha256:'+target['sha256'],expired=False,
                      url=api+'/actions/artifacts/'+str(target['artifact_id']),
                      archive_download_url=api+'/actions/artifacts/'+str(target['artifact_id'])+'/zip',
                      workflow_run=dict(id=worker.PRESERVED['run_id'],head_sha=worker.NATIVE,head_branch=worker.PRESERVED['branch'],
                                        repository_id=1210669686,head_repository_id=1210669686))
            worker.validate_metadata(run,item,kind,api)
            for key,value in (('head_sha','d'*40),('head_branch','other'),('status','in_progress'),('conclusion','success'),('run_attempt',2)):
                changed=copy.deepcopy(run);changed[key]=value
                with self.subTest(kind=kind,key=key),self.assertRaises(ValueError):worker.validate_metadata(changed,item,kind,api)
            for key,value in (('size_in_bytes',1),('digest','sha256:'+'d'*64),('expired',True),('name','other')):
                changed=copy.deepcopy(item);changed[key]=value
                with self.subTest(kind=kind,key=key),self.assertRaises(ValueError):worker.validate_metadata(run,changed,kind,api)

    def test_binary_and_execution_identity(self):
        worker.validate_executions(executions(),models(),'c'*64)
        with self.assertRaises(ValueError):worker.validate_executions(executions(),models(),'d'*64)
        for key,value in (('git_commit','d'*40),('binary_sha256','d'*64),('model_sha256','d'*64),
                          ('model_bytes',99),('arm','other'),('gate','other'),('build_mode','native')):
            found=executions();next(iter(found.values()))[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):worker.validate_executions(found,models())
        found=executions();found.pop(next(iter(found)))
        with self.assertRaises(ValueError):worker.validate_executions(found,models())

    def test_summary_failure_is_exact(self):
        worker.validate_summary(summary())
        for key,value in (('source','d'*40),('protocol_sha256','d'*64),('completed',True),('native_request_processes',12),
                          ('original_forward_calls',1),('failure',dict(type='ValueError',message='other'))):
            record=summary();record[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):worker.validate_summary(record)
        for mutation in ('duplicate','missing','failed','timeout'):
            record=summary()
            if mutation=='duplicate':record['commands'][0]=copy.deepcopy(record['commands'][1])
            elif mutation=='missing':record['rows'].pop()
            elif mutation=='failed':record['commands'][0]['exit']=1
            else:record['commands'][0]['timed_out']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):worker.validate_summary(record)

    def dirs(self,root):
        a,b=root/'a',root/'b';a.mkdir();b.mkdir()
        for directory in (a,b):
            for name in ('pixel_values.f32','vis_ln_pre.f32','logits_step0.f32','logits_step6.f32','generated_ids.f32'):
                (directory/name).write_bytes(b'1234')
        return a,b

    def test_postdecode_ids_and_later_logits_can_change(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=self.dirs(Path(directory))
            (b/'generated_ids.f32').write_bytes(b'changed length and values')
            (b/'logits_step6.f32').unlink();(b/'logits_step7.f32').write_bytes(b'5678')
            result=worker.prefill_control(a,b)
            self.assertTrue(result['all_bytes_exact']);self.assertEqual(result['tensor_files'],3)

    def test_real_prefill_changes_fail(self):
        for mutation in ('bytes','missing','extra','step0'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                a,b=self.dirs(Path(directory))
                if mutation=='bytes':(b/'vis_ln_pre.f32').write_bytes(b'5678')
                elif mutation=='missing':(b/'vis_ln_pre.f32').unlink()
                elif mutation=='extra':(b/'new_stage.f32').write_bytes(b'1234')
                else:
                    (a/'logits_step0.f32').unlink();(b/'logits_step0.f32').unlink()
                with self.assertRaises(ValueError):worker.prefill_control(a,b)

    def test_stored_kv0_proofs_rehashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);record=summary()
            for row in record['rows']:
                if row['gate']!='0':continue
                controls={};row['byte_controls']=controls
                for arm in worker.ARMS:
                    path=root/worker.CASE/row['precision']/'kv0'/arm;path.mkdir(parents=True)
                    tensors=[]
                    if arm!='plain':
                        (path/'stage.f32').write_bytes(b'1234')
                        tensors=[dict(name='stage.f32',bytes=4,sha256=hashlib.sha256(b'1234').hexdigest())]
                    controls[arm]=dict(all_bytes_exact=True,tensor_files=len(tensors),tensors=tensors)
            self.assertEqual(len(worker.stored_byte_controls(record,root)),6)
            path=root/worker.CASE/'f32'/'kv0'/'forced'/'stage.f32';path.write_bytes(b'5678')
            with self.assertRaises(ValueError):worker.stored_byte_controls(record,root)


if __name__=='__main__':unittest.main()
