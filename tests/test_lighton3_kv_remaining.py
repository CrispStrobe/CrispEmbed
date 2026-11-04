#!/usr/bin/env python3
"""Pure first-proof, source, artifact and prefix controls; no network or models."""
import copy
import importlib.util
import json
import hashlib
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('remaining',ROOT/'tools/ci-heavy/lighton3_kv_remaining.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


def first_summary():
    summary=dict(schema='crispembed.lighton3.kv-complete-first.v1',phase='first',operator_source=worker.FIRST['source'],
                 native_source=worker.NATIVE,protocol_sha256=worker.FIRST['protocol_sha256'],completed=True,**{'pass':True},
                 rebuilt_binary_byte_exact=True,binary_sha256=worker.BINARY_SHA,reused_native_request_processes=10,
                 new_native_request_processes=2,native_request_processes=12,original_forward_calls=0,rows=[])
    for precision in ('f32','f16'):
        for gate in ('0','1'):
            negative=precision=='f16' and gate=='0'
            control=dict(all_bytes_exact=True,tensor_files=1,tensors=[dict(name='logits_step0.f32',bytes=4,sha256='a'*64)])
            summary['rows'].append(dict(precision=precision,gate=gate,step6_max_abs_reduced=True,
                 prefill_byte_controls=dict(forced=copy.deepcopy(control),autonomous=copy.deepcopy(control)),
                 comparison=dict(owner_bar_pass=True,decoded=dict(ids_exact=not negative,text_exact=not negative,
                    reference_ids=[0]*6+[2918],native_ids=[0]*6+[3377 if negative else 2918]))))
    return summary


def metadata(target,name):
    api='https://api.github.com/repos/CrispStrobe/CrispEmbed'
    repo=dict(id=1210669686,full_name='CrispStrobe/CrispEmbed')
    run=dict(id=target['run_id'],head_sha=target['source'],head_branch=target['branch'],event='workflow_dispatch',
             path='.github/workflows/heavy-cpu.yml',status='completed',conclusion=target['conclusion'],run_attempt=1,
             repository=repo,head_repository=repo)
    item=dict(id=target['artifact_id'],name=name,size_in_bytes=target['bytes'],digest='sha256:'+target['sha256'],expired=False,
              url=api+'/actions/artifacts/'+str(target['artifact_id']),archive_download_url=api+'/actions/artifacts/'+str(target['artifact_id'])+'/zip',
              workflow_run=dict(id=target['run_id'],head_sha=target['source'],head_branch=target['branch'],repository_id=1210669686,head_repository_id=1210669686))
    return run,item,api


class Controls(unittest.TestCase):
    def test_protocol_native_closure(self):
        protocol=json.loads((ROOT/worker.PROTOCOL_NAME).read_text())
        self.assertEqual(protocol['native_source'],worker.NATIVE)
        self.assertEqual(protocol['first_proof'],worker.FIRST)
        self.assertEqual(protocol['baselines'],worker.BASELINES)
        self.assertEqual(protocol['cases'],worker.CASES)
        expected={'src/lightonocr.cpp','src/lightonocr3_profile.h','tests/test_lightonocr3_contract.cpp'}
        self.assertEqual(set(protocol['owned_native_sha256']),expected)
        for name,digest in protocol['owned_native_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)

    def test_positive_first_proof(self):self.assertEqual(len(worker.validate_first(first_summary())),4)

    def test_first_source_counters_and_typed_flags(self):
        mutations=dict(operator_source='a'*40,native_source='a'*40,protocol_sha256='a'*64,binary_sha256='a'*64,
                       completed=1,**{'pass':1},rebuilt_binary_byte_exact=1,reused_native_request_processes=9,
                       new_native_request_processes=3,native_request_processes=11,original_forward_calls=False)
        for key,value in mutations.items():
            summary=first_summary();summary[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):worker.validate_first(summary)

    def test_first_rows_and_controls(self):
        for mutation in ('duplicate','missing','bar','ids','text','step6','control','inventory','negative'):
            summary=first_summary();row=summary['rows'][1]
            if mutation=='duplicate':summary['rows'][0]=copy.deepcopy(row)
            elif mutation=='missing':summary['rows'].pop()
            elif mutation=='bar':row['comparison']['owner_bar_pass']=False
            elif mutation=='ids':row['comparison']['decoded']['ids_exact']=False
            elif mutation=='text':row['comparison']['decoded']['text_exact']=False
            elif mutation=='step6':row['step6_max_abs_reduced']=False
            elif mutation=='control':row['prefill_byte_controls']['forced']['all_bytes_exact']=1
            elif mutation=='inventory':row['prefill_byte_controls']['forced']['tensors']=[]
            else:summary['rows'][2]['comparison']['decoded']['ids_exact']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):worker.validate_first(summary)

    def test_closed_source_blobs(self):
        protocol=dict(schema='crispembed.lighton3.kv-remaining.v1',native_source=worker.NATIVE,first_proof=worker.FIRST,
                      baselines=worker.BASELINES,cases=worker.CASES)
        worker.validate_closure(protocol,worker.NATIVE,'a'*40,[b'a',b'b',b'c'],[b'a',b'b',b'c'])
        for files in ([b'x',b'b',b'c'],[b'a',b'x',b'c'],[b'a',b'b',b'x']):
            with self.assertRaises(ValueError):worker.validate_closure(protocol,worker.NATIVE,'a'*40,files,[b'a',b'b',b'c'])
        with self.assertRaises(ValueError):worker.validate_closure(protocol,'a'*40,'b'*40,[b'a']*3,[b'a']*3)

    def test_artifact_identity(self):
        for target in (worker.FIRST,*worker.BASELINES.values()):
            name='fixture-name';run,item,api=metadata(target,name)
            worker.validate_metadata(target,run,item,api,name)
            for key,value in (('head_sha','a'*40),('head_branch','other'),('conclusion','other'),('run_attempt',2),('status','in_progress')):
                changed=copy.deepcopy(run);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):worker.validate_metadata(target,changed,item,api,name)
            for key,value in (('name','other'),('size_in_bytes',1),('digest','sha256:'+'a'*64),('expired',True)):
                changed=copy.deepcopy(item);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):worker.validate_metadata(target,run,changed,api,name)

    def test_baseline_prefixes(self):
        for case in worker.CASES:
            self.assertTrue(worker.selected('references/'+case+'/reference.json','full_eps'))
            self.assertFalse(worker.selected('references/'+case+'/reference.json','first_eps'))
            path=case+'/f16/gate1/forced/stage.f32'
            self.assertEqual(worker.selected(path,'full_eps'),case!='form-grounding')
            self.assertEqual(worker.selected(path,'first_eps'),case=='form-grounding')
        for kind in worker.BASELINES:
            for path in ('references/form-transcription/reference.json','form-transcription/f16/gate1/plain/decoded.txt',
                         'fox-transcription/f16/gate0/forced/a.f32','fox-transcription/q8/gate1/forced/a.f32'):
                self.assertFalse(worker.selected(path,kind),path)


if __name__=='__main__':unittest.main()
