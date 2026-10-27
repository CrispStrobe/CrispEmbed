#!/usr/bin/env python3
"""Josev metadata-only dependency audit and immutable environment lock controls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def metadata_url(url, method='GET'):
    """Reject wheel/sdist URLs BEFORE any adapter/transport is called."""
    u = urlsplit(url)
    path = unquote(u.path)
    if method != 'GET' or u.scheme != 'https' or u.username or u.password or u.port not in (None, 443) or u.query or u.fragment:
        raise ValueError('metadata-only request rejected')
    if u.hostname == 'pypi.org' and re.fullmatch(r'/simple/[a-zA-Z0-9_.-]+/?', path):
        return 'simple-index'
    if u.hostname == 'pypi.org' and re.fullmatch(r'/pypi/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.!+-]+/json', path):
        return 'pypi-release-json'
    if u.hostname == 'files.pythonhosted.org' and re.fullmatch(r'/packages/[0-9a-f]{2}/[0-9a-f]{2}/[0-9a-f]+/[^/]+\.whl\.metadata', path):
        return 'pep658-core-metadata'
    raise ValueError('wheel/sdist/other payload URL rejected before transfer')


class MetadataNetworkGuard:
    """Bounded in-process guard for pip's sole requests HTTP transport.

    Both redirects and metadata fallback downloads pass HTTPAdapter.send. Never
    invoke the original transport for a wheel, sdist or unrecognized endpoint.
    """
    def __init__(self, journal, seconds=80, max_bytes=32*1024**2):
        self.journal = Path(journal)
        self.deadline = time.monotonic()+seconds
        self.max_bytes = max_bytes
        self.bytes = 0
        self.events = []

    def save(self):
        self.journal.write_text(json.dumps({'schema':'crispembed-josev-metadata-network-v1',
            'wheel_or_sdist_transfer_permitted':False,'response_bytes_read':self.bytes,
            'events':self.events},indent=2)+'\n')

    def before(self,url,method):
        try:
            if time.monotonic()>self.deadline or len(self.events)>=200:
                raise ValueError('metadata network time/request budget exceeded')
            kind=metadata_url(url,method)
        except Exception as error:
            self.events.append({'url':url,'method':method,'allowed':False,'error':str(error)})
            self.save()
            raise
        self.events.append({'url':url,'method':method,'allowed':True,'kind':kind})
        self.save()

    def wrap(self,send):
        guard=self
        def guarded(adapter,request,*args,**kwargs):
            guard.before(request.url,request.method)
            response=send(adapter,request,*args,**kwargs)
            length=response.headers.get('Content-Length')
            if length and int(length)>8*1024**2:
                response.close()
                raise ValueError('single metadata response exceeds bound')
            original=response.raw.read
            read_bytes=0
            def consume(data):
                nonlocal read_bytes
                read_bytes+=len(data);guard.bytes+=len(data)
                if read_bytes>8*1024**2 or guard.bytes>guard.max_bytes or time.monotonic()>guard.deadline:
                    response.close();raise ValueError('metadata response budget exceeded')
                return data
            def read(*a,**kw):
                return consume(original(*a,**kw))
            response.raw.read=read
            if hasattr(response.raw,'read_chunked'):
                chunks=response.raw.read_chunked
                def chunked(*a,**kw):
                    for data in chunks(*a,**kw):yield consume(data)
                response.raw.read_chunked=chunked
            return response
        return guarded


def target_environment(python_full_version='3.13.15'):
    return {'implementation_name':'cpython','implementation_version':python_full_version,
        'os_name':'posix','platform_machine':'x86_64','platform_release':'',
        'platform_system':'Linux','platform_version':'','python_full_version':python_full_version,
        'platform_python_implementation':'CPython','python_version':'3.13','sys_platform':'linux'}


def active_requirements(values,extra='',python_full_version='3.13.15'):
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    selected=[]
    for text in values:
        req=Requirement(text)
        if req.marker and not req.marker.evaluate(dict(target_environment(python_full_version),extra=extra)):
            continue
        req.marker=None;req.name=canonicalize_name(req.name)
        req.extras={canonicalize_name(x) for x in req.extras}
        selected.append(str(req))
    return sorted(set(selected))


def dependency_closure(packages,roots,python_full_version='3.13.15'):
    """Independently walk all active Requires-Dist markers, versions and extras."""
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version
    environment=target_environment(python_full_version)
    rows={canonicalize_name(x['name']):x for x in packages}
    if len(rows)!=len(packages):raise ValueError('duplicate locked distribution')
    active={canonicalize_name(name):set() for name in roots}
    for name,version in roots.items():
        row=rows.get(canonicalize_name(name))
        if not row or row['version']!=version:raise ValueError('locked root version differs: '+name)
    edges={};changed=True
    while changed:
        changed=False
        for name,extras in list(active.items()):
            row=rows.get(name)
            if row is None:raise ValueError('dependency missing from complete lock: '+name)
            if not SpecifierSet(row.get('requires_python') or '').contains(Version(python_full_version),prereleases=True):
                raise ValueError('locked Python requirement differs: '+name)
            needed=[]
            for text in row.get('requires_dist',[]):
                req=Requirement(text)
                if req.url:raise ValueError('unreviewed direct dependency URL')
                if req.marker and not any(req.marker.evaluate(dict(environment,extra=extra)) for extra in ['']+sorted(extras)):
                    continue
                dep=canonicalize_name(req.name)
                if dep not in rows or not req.specifier.contains(Version(rows[dep]['version']),prereleases=True):
                    raise ValueError('locked dependency version unsatisfied: '+name+' -> '+text)
                needed.append({'requirement':text,'name':dep,'version':rows[dep]['version'],'extras':sorted(req.extras)})
                if dep not in active:
                    active[dep]=set();changed=True
                old=set(active[dep]);active[dep].update(req.extras)
                if old!=active[dep]:changed=True
            edges[name]=needed
    if set(active)!=set(rows):raise ValueError('lock contains unneeded or unreachable packages')
    return {'active_extras':{name:sorted(extras) for name,extras in sorted(active.items())},
            'edges':{name:edges[name] for name in sorted(edges)},'complete':True}


def requirements_bytes(packages):
    header='# Josev CUDA F32 CPython3.13 Linux x86_64: complete metadata-resolved wheel lock.\n# Install only on reviewed GPU host, with --no-deps --require-hashes.\n'
    lines=[]
    for row in sorted(packages,key=lambda x:x['name']):
        lines.append(f"{row['name']} @ {row['url']} --hash=sha256:{row['sha256']}")
    return (header+'\n'.join(lines)+'\n').encode()


def verify_lock(lock,roots,requirements,python_full_version='3.13.15'):
    from pip._vendor.packaging.utils import canonicalize_name,parse_wheel_filename
    from pip._vendor.packaging.version import Version
    from pip._vendor.packaging.tags import cpython_tags,compatible_tags
    platforms=[f'manylinux_2_{n}_x86_64' for n in range(28,16,-1)]+['manylinux2014_x86_64','manylinux2010_x86_64','manylinux1_x86_64','linux_x86_64']
    target_tags=set(cpython_tags((3,13),abis=['cp313','abi3','none'],platforms=platforms))|set(compatible_tags((3,13),interpreter='cp313',platforms=platforms))
    if (lock.get('schema')!='crispembed-josev-cp313-linux-wheel-lock-v1' or
            lock.get('target')!={'python_minor':'3.13','implementation':'cp','architecture':'x86_64','system':'Linux','glibc_minimum':'2.28'} or
            lock.get('roots')!=roots or lock.get('wheel_payloads_downloaded_locally') is not False):
        raise ValueError('locked environment profile differs')
    packages=lock['packages']
    for row in packages:
        u=urlsplit(row['url']);filename=unquote(u.path.rsplit('/',1)[-1])
        if (u.scheme!='https' or u.hostname!='files.pythonhosted.org' or u.query or u.fragment or
                u.username or u.password or not filename.endswith('.whl') or filename!=row['filename'] or
                row['name']!=canonicalize_name(row['name']) or not re.fullmatch('[a-f0-9]{64}',row['sha256']) or
                type(row['bytes']) is not int or row['bytes']<=0 or
                not re.fullmatch('[a-f0-9]{64}',row['core_metadata_sha256'])):
            raise ValueError('invalid exact wheel identity')
        wheel_name,wheel_version,_,tags=parse_wheel_filename(filename)
        if canonicalize_name(wheel_name)!=row['name'] or wheel_version!=Version(row['version']) or not tags.intersection(target_tags) or sorted(str(t) for t in tags)!=row['wheel_tags']:
            raise ValueError('locked wheel incompatible with reviewed Linuxcp313 target')
    if requirements_bytes(packages)!=requirements or sha(requirements)!=lock['requirements_sha256']:
        raise ValueError('complete requirements bytes differ')
    closure=dependency_closure(packages,roots,python_full_version)
    if closure!=lock['dependency_closure']:
        raise ValueError('complete dependency closure differs for this Python profile')
    return closure


def runtime_target():
    import platform
    import sys
    libc,version=platform.libc_ver()
    if (sys.version_info[:2]!=(3,13) or sys.platform!='linux' or platform.machine()!='x86_64' or
            libc!='glibc' or tuple(int(x) for x in version.split('.')[:2])<(2,28)):
        raise ValueError('reviewed Linuxx86_64 CPython3.13 glibc>=2.28 target required')
    return {'python_minor':'3.13','system':'Linux','architecture':'x86_64','glibc_minimum_satisfied':True}


def locked_files(cfg):
    runtime_target()
    declaration=cfg['environment_lock']
    lock_bytes=(ROOT/declaration['path']).read_bytes()
    requirements=(ROOT/declaration['requirements_path']).read_bytes()
    if sha(lock_bytes)!=declaration['sha256'] or sha(requirements)!=declaration['requirements_sha256']:
        raise ValueError('reviewed full environment lock bytes differ')
    lock=json.loads(lock_bytes)
    from pip._vendor.packaging.utils import canonicalize_name
    roots={canonicalize_name(k):v for k,v in cfg['runtime'].items() if k!='python_minor'}
    import platform
    verify_lock(lock,roots,requirements,platform.python_version())
    return lock


def isolated_prefix(expected):
    import os
    import site
    import sys
    prefix=Path(sys.prefix).resolve();expected=Path(expected).resolve()
    if prefix!=expected or prefix==Path(sys.base_prefix).resolve() or site.ENABLE_USER_SITE is not False:
        raise ValueError('reviewed isolated venv prefix/user-site differs')
    if not Path(os.path.abspath(sys.executable)).is_relative_to(expected):
        raise ValueError('executed interpreter outside reviewed venv')
    package_paths=[Path(p).resolve() for p in sys.path if p and any(x in Path(p).parts for x in ('site-packages','dist-packages'))]
    if any(not p.is_relative_to(expected) for p in package_paths):
        raise ValueError('external site-packages path in isolated worker')
    return {'prefix_matches_expected':True,'base_prefix_distinct':True,'user_site_enabled':False,
            'external_package_paths':False}


def owner_metadata_file(dist,name,version):
    """Select the locked distribution's own metadata, excluding vendored copies.

    importlib.metadata.files exposes every path in its RECORD, including nested
    vendored dist-info directories. Only the unique top-level owner is eligible;
    no fallback to a vendor's matching name, version or hash is permitted.
    """
    candidates=[]
    for entry in dist.files or []:
        path=Path(entry)
        if path.name!='METADATA' or not path.parent.name.endswith('.dist-info'):
            continue
        if path.is_absolute() or len(path.parts)!=2 or '..' in path.parts:
            continue
        stem=path.parent.name[:-len('.dist-info')]
        if '-' not in stem:
            continue
        owner,owner_version=stem.rsplit('-',1)
        if re.sub(r'[-_.]+','-',owner).lower()==name and owner_version==version:
            candidates.append(entry)
    if len(candidates)!=1:
        raise ValueError('installed owning METADATA file missing/ambiguous: '+name+
                         '; matching top-level files='+str(len(candidates)))
    return candidates[0]


def verify_installed(cfg,expected_prefix,import_runtime=False):
    import importlib
    import importlib.metadata as metadata
    import platform
    import sys
    report=isolated_prefix(expected_prefix)
    report['target']=runtime_target()
    lock=locked_files(cfg)
    expected={r['name']:r for r in lock['packages']}
    from pip._vendor.packaging.utils import canonicalize_name
    actual={}
    for dist in metadata.distributions():
        name=canonicalize_name(dist.metadata['Name'])
        if name in actual:raise ValueError('duplicate installed distribution')
        actual[name]=dist
    prefix=Path(expected_prefix).resolve()
    if set(actual)!=set(expected)|{'pip'} or actual['pip'].version!='26.2.2':
        raise ValueError('installed environment contains missing/unlocked distributions')
    if not Path(actual['pip'].locate_file('')).resolve().is_relative_to(prefix):
        raise ValueError('bootstrap pip origin outside isolated venv')
    for name,row in expected.items():
        dist=actual[name]
        if dist.version!=row['version'] or not Path(dist.locate_file('')).resolve().is_relative_to(prefix):
            raise ValueError('installed exact version/origin differs: '+name)
        entry=owner_metadata_file(dist,name,row['version'])
        path=Path(dist.locate_file(entry)).resolve()
        if not path.is_relative_to(prefix):
            raise ValueError('installed owning METADATA origin outside isolated venv: '+name)
        data=path.read_bytes();actual_sha=sha(data)
        if actual_sha!=row['core_metadata_sha256']:
            raise ValueError('installed owning METADATA bytes differ: '+name+
                             '; expected='+row['core_metadata_sha256']+'; actual='+actual_sha+
                             '; bytes='+str(len(data)))
    report.update(python_minor=cfg['runtime']['python_minor'],package_count=len(expected),
        exact_versions=True,all_wheel_metadata_hashes_exact=True,unlocked_packages=False,
        owning_top_level_metadata_only=True,
        environment_lock_sha256=cfg['environment_lock']['sha256'])
    if import_runtime:
        modules={'torch':'torch','torchvision':'torchvision','transformers':'transformers',
            'tokenizers':'tokenizers','safetensors':'safetensors','huggingface-hub':'huggingface_hub',
            'numpy':'numpy','accelerate':'accelerate','pillow':'PIL'}
        loaded={}
        for name,module_name in modules.items():
            module=importlib.import_module(module_name)
            if str(module.__version__).split('+')[0]!=cfg['runtime'][name] or not Path(module.__file__).resolve().is_relative_to(prefix):
                raise ValueError('loaded framework version/origin differs: '+name)
            loaded[name]=str(module.__version__)
        report['loaded_runtime_versions']=loaded
        report['loaded_runtime_origins_inside_venv']=True
    return report


def main():
    import argparse
    import os
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['verify-lock','verify-isolation','verify-installed'])
    p.add_argument('--expected-prefix',type=Path)
    p.add_argument('--import-runtime',action='store_true')
    p.add_argument('--report',type=Path)
    args=p.parse_args()
    cfg=json.loads((ROOT/'docs/josev2b-source-contract.json').read_text())
    if args.phase=='verify-lock':
        lock=locked_files(cfg)
        report={'complete':True,'packages':len(lock['packages']),'lock_sha256':cfg['environment_lock']['sha256']}
    else:
        if args.expected_prefix is None:p.error('explicit expected venv prefix required')
        if args.phase=='verify-isolation':report=isolated_prefix(args.expected_prefix)
        else:report=verify_installed(cfg,args.expected_prefix,args.import_runtime)
    if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True))

if __name__=='__main__':main()
