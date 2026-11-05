#!/usr/bin/env python3
"""Tiny real-provider transport proof, unique prefix in an existing private repo.

No repository is created, no existing path is overwritten/deleted, no model is
loaded, and no original capture or resource gate is changed. Private locators
stay in scratch; the public report exposes fingerprints and immutable hashes.
"""
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
DESTINATION_NAME_SHA='8588fda7984a253f233034a337d46bff4f84c4a85f3a4c530b897f856df586e3'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(value):return hashlib.sha256(value.encode()).hexdigest()


def main():
    if os.environ.get('GITHUB_ACTIONS')!='true':raise SystemExit('hosted provider test required')
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'lighton4b-real-provider'
    scratch.mkdir(parents=True,exist_ok=False)
    report={'schema':'lighton4b-real-provider-transport/v1','status':'failed',
        'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'original_forward_executed':False,'native_forward_executed':False,'model_payload_bytes':0,
        'original_transport_admitted':False,'capture_enabled':False,'quality_qualified':False,
        'repository_created':False,'existing_paths_overwritten':False,'paths_deleted':False}
    report['source_sha256']={p:sha(ROOT/p)for p in('tools/ci-heavy/lighton4b_hub_transport.py',
        'tools/lighton4b_sharded_transport.py','tests/test_lighton4b_sharded_transport.py',
        'tools/lighton4b_original_document.py','docs/lighton4b-document-parity-v1.json')}
    repo=None;tokens=[os.environ.get(k)for k in('HF_TOKEN_2','HF_TOKEN')if os.environ.get(k)]
    try:
        logging.getLogger('huggingface_hub').setLevel(logging.CRITICAL)
        from huggingface_hub import HfApi
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
        for token in tokens:
            try:
                api=HfApi(token=token);namespace=api.whoami()['name']
                matches=[item.id for item in api.list_datasets(author=namespace,full=False)
                    if item.private and fingerprint(item.id.split('/')[-1])==DESTINATION_NAME_SHA]
                if len(matches)!=1:continue
                candidate=matches[0];api.auth_check(candidate,repo_type='dataset',write=True)
                repo=candidate;os.environ['HF_TOKEN']=token;break
            except Exception:continue
        if repo is None:
            report.update(status='blocked',reason='No supplied CI credential has write access to the existing authorized private diagnostic destination')
            return 2
        sys.path.insert(0,str(ROOT/'tools'))
        import lighton4b_sharded_transport as transport
        spec=importlib.util.spec_from_file_location('synthetic_transport_fixture',ROOT/'tests/test_lighton4b_sharded_transport.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        def tree(revision):
            return {item.path:dict(blob_id=item.blob_id,bytes=item.size)for item in
                api.list_repo_tree(repo,repo_type='dataset',revision=revision,recursive=True)
                if getattr(item,'blob_id',None)is not None}
        info=api.repo_info(repo,repo_type='dataset');transport.require(info.private is True,'private existing repo required')
        before=tree(info.sha);prefix='synthetic-hub-v1-'+os.environ['GITHUB_RUN_ID']+'-'+uuid.uuid4().hex[:8]
        transport.require(not any(name.startswith(prefix+'/')for name in before),'fresh unique prefix required')
        source=scratch/'source';fixture.fixture(source)
        expected={str(p.relative_to(source)):p.read_bytes()for p in source.rglob('*')if p.is_file()}
        size=sum(len(v)for v in expected.values());transport.require(size<=64*2**10,'tiny non-model fixture must remain<=64KiB')
        backend=transport.HuggingFaceBackend(repo,'dataset')
        receipt=transport.publish(source,backend,prefix,scratch/'publication.json',part_bytes=4096)
        report.update(remote_write_executed=True,destination_private=True,destination_fingerprint=fingerprint(repo),
            preexisting_paths=len(before),before_tree_fingerprint=fingerprint(json.dumps(before,sort_keys=True)),
            prefix=prefix,fixture_bytes=size,parts=len(receipt['parts']),manifest_sha256=receipt['manifest_sha256'],
            immutable_revision_fingerprint=fingerprint(receipt['revision']))
        # New client plus a fresh restore directory; no publisher cache or source
        # files are inputs to reconstruction. Compare against independently saved
        # authored fixture bytes only after remote reconstruction has completed.
        reader=transport.HuggingFaceBackend(repo,'dataset')
        restored=scratch/'cold-reader'
        result=transport.restore(reader,prefix,receipt['revision'],receipt['manifest_sha256'],restored)
        transport.require(result['status']=='passed','provider cold readback failed')
        for name,data in expected.items():
            actual=(restored/'files'/name).read_bytes()
            transport.require(actual==data and hashlib.sha256(actual).digest()==hashlib.sha256(data).digest(),
                'independent authored fixture readback mismatch')
        after=tree(receipt['revision'])
        transport.require(all(after.get(name)==row for name,row in before.items()),'preexisting user data changed')
        allowed={prefix+'/'+part['file']for part in receipt['parts']}|{prefix+'/manifest.json'}
        transport.require(set(after)-set(before)==allowed,'unexpected remote path change')
        report.update(status='passed',remote_provider_cold_readback_verified=True,independent_whole_file_hashes_verified=True,
            restored_files=len(expected),preexisting_tree_unchanged=True,new_paths_only_under_prefix=True,
            after_paths=len(after),new_paths=len(allowed),provider_scope='tiny synthetic fixture at immutable tested revision only; long-term retention/large capture remain unverified')
        # Contains private service access references, so deliberately outside the
        # public artifact directory and never printed or committed.
        (scratch/'private-provider-locator.json').write_text(json.dumps(dict(repo_id=repo,prefix=prefix,
            revision=receipt['revision'],manifest_sha256=receipt['manifest_sha256']),indent=2)+'\n')
        return 0
    except Exception as error:
        details=str(error)
        for value in tokens+([repo]if repo else[]):details=details.replace(value,'[redacted]')
        details=re.sub(r'https?://\S+','[service-url]',details)
        report.update(status='failed',error_type=type(error).__name__,error=details[:1000])
        return 1
    finally:
        (out/'real-provider-transport.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        print(json.dumps({k:report[k]for k in('status','original_forward_executed','model_payload_bytes','capture_enabled')}))


if __name__=='__main__':raise SystemExit(main())
