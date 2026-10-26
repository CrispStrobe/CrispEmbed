#!/usr/bin/env python3
"""Audit pip's selected closure against small primary PyPI/PEP658 metadata only."""
import argparse
from email.parser import BytesParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from josev2b_environment import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--resolver-output',type=Path,required=True);a=p.parse_args()
    from pip._vendor import requests
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name,parse_wheel_filename
    from pip._vendor.packaging.version import Version
    from pip._vendor.packaging.specifiers import SpecifierSet
    folder=a.resolver_output;report=json.loads((folder/'pip-report.json').read_text())
    network=json.loads((folder/'network.json').read_text())
    if network['wheel_or_sdist_transfer_permitted'] is not False or any(not e['allowed'] for e in network['events']):
        raise ValueError('resolver network was not metadata-only and complete')
    runtime=json.loads((ROOT/'docs/josev2b-source-contract.json').read_text())['runtime']
    roots={canonicalize_name(k):v for k,v in runtime.items() if k!='python_minor'}
    guard=MetadataNetworkGuard(folder/'audit-network.json',seconds=80)
    adapter=requests.adapters.HTTPAdapter;original=adapter.send;adapter.send=guard.wrap(original)
    packages=[]
    try:
        session=requests.Session();session.trust_env=False
        for item in report['install']:
            meta=item['metadata'];name=canonicalize_name(meta['name']);version=meta['version'];url=item['download_info']['url']
            if item['is_yanked']:raise ValueError('yanked release selected')
            response=session.get(f'https://pypi.org/pypi/{name}/{version}/json',timeout=15);response.raise_for_status();release=response.json()
            release_bytes=response.content
            cache=folder/'metadata';cache.mkdir(exist_ok=True)
            (cache/(name+'-'+version+'.json')).write_bytes(release_bytes)
            files=[f for f in release['urls'] if f['url']==url]
            if len(files)!=1:raise ValueError('selected wheel missing from primary PyPI release: '+name)
            file=files[0];expected=item['download_info']['archive_info']['hashes']['sha256']
            if file['packagetype']!='bdist_wheel' or file['yanked'] or file['digests']['sha256']!=expected:
                raise ValueError('primary wheel identity differs')
            wheel_name,wheel_version,_,tags=parse_wheel_filename(file['filename'])
            if canonicalize_name(wheel_name)!=name or wheel_version!=Version(version):raise ValueError('wheel name/version differs')
            # URL guard permits only the PEP658 sidecar, never url itself.
            response=session.get(url+'.metadata',timeout=15);response.raise_for_status();metadata_bytes=response.content
            (cache/(name+'-'+version+'.metadata')).write_bytes(metadata_bytes)
            metadata=BytesParser().parsebytes(metadata_bytes)
            requires=metadata.get_all('Requires-Dist') or []
            norm=lambda values:sorted(str(Requirement(v)) for v in values)
            if (canonicalize_name(metadata['Name'])!=name or Version(metadata['Version'])!=Version(version) or
                    norm(requires)!=norm(meta.get('requires_dist') or []) or
                    Version(release['info']['version'])!=Version(version) or
                    SpecifierSet(metadata.get('Requires-Python') or '')!=SpecifierSet(release['info'].get('requires_python') or '')):
                raise ValueError('independent core metadata/JSON/pip requirements differ: '+name)
            extras=sorted(set((metadata.get_all('Provides-Extra') or [])+(release['info'].get('provides_extra') or [])))
            for extra in ['']+extras:
                if active_requirements(requires,extra)!=active_requirements(release['info'].get('requires_dist') or [],extra):
                    raise ValueError('Linuxcp313 active dependencies differ between wheel and primary release: '+name+'/'+extra)
            packages.append({'name':name,'version':version,'filename':file['filename'],'url':url,
                'sha256':expected,'bytes':file['size'],'wheel_tags':sorted(str(t) for t in tags),
                'requires_python':metadata.get('Requires-Python') or '', 'requires_dist':requires,
                'core_metadata_sha256':sha(metadata_bytes),'core_metadata_bytes':len(metadata_bytes),
                'pypi_aggregate_requires_dist_strings_exact':norm(requires)==norm(release['info'].get('requires_dist') or []),
                'pypi_aggregate_linux_cp313_all_extras_equivalent':True,
                'pypi_release_json_sha256':sha(release_bytes),'pypi_release_json_bytes':len(release_bytes)})
    finally:
        adapter.send=original;guard.save()
    packages.sort(key=lambda x:x['name'])
    closure=dependency_closure(packages,roots)
    # Markers and Requires-Python also yield exactly the same closure for the
    # resolver's3.13.11 and actual observed GPU3.13.15; no host identity is retained.
    if dependency_closure(packages,roots,'3.13.11')!=closure:
        raise ValueError('Python patch changes dependency closure')
    requirements=requirements_bytes(packages)
    lock={'schema':'crispembed-josev-cp313-linux-wheel-lock-v1',
        'target':{'python_minor':'3.13','implementation':'cp','architecture':'x86_64','system':'Linux','glibc_minimum':'2.28'},
        'roots':roots,'packages':packages,'dependency_closure':closure,
        'requirements_sha256':sha(requirements),'wheel_payloads_downloaded_locally':False,
        'resolver':{'pip':'25.3','mode':'dry-run ignore-installed binary-only metadata-network-guard',
            'python_patches_metadata_checked':['3.13.11','3.13.15'],
            'metadata_requests':len(network['events']),'metadata_response_bytes':network['response_bytes_read']},
        'audit':{'primary_release_and_pep658_metadata':True,'requests':len(guard.events),'response_bytes':guard.bytes}}
    verify_lock(lock,roots,requirements)
    (folder/'environment-lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    (folder/'requirements.lock').write_bytes(requirements)
    print(json.dumps({'packages':len(packages),'wheel_bytes_published_sum':sum(p['bytes'] for p in packages),
        'lock_sha256':sha((folder/'environment-lock.json').read_bytes()),'requirements_sha256':sha(requirements),
        'metadata_requests':len(guard.events),'metadata_bytes':guard.bytes}))

if __name__=='__main__':main()
