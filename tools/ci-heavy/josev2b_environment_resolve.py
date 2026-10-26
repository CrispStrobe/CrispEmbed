#!/usr/bin/env python3
"""Bounded metadata-only pip resolution. Never permits a wheel/sdist transfer."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from josev2b_environment import MetadataNetworkGuard


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if sys.version_info[:2]!=(3,13) or importlib.metadata.version('pip')!='25.3':
        p.error('reviewed metadata resolver requires existing Python3.13/pip25.3')
    a.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'docs/josev2b-source-contract.json').read_text())
    requirements={name:version for name,version in cfg['runtime'].items() if name!='python_minor'}
    # Existing base pip only; no tool installation or framework import.
    from pip._vendor.requests.adapters import HTTPAdapter
    from pip._internal.cli.main import main as pip_main
    guard=MetadataNetworkGuard(a.output/'network.json',seconds=80)
    original=HTTPAdapter.send;HTTPAdapter.send=guard.wrap(original)
    for key in list(os.environ):
        if key.startswith('PIP_'):os.environ.pop(key)
    os.environ.update(PIP_CONFIG_FILE=os.devnull,PIP_USER='0',PIP_NO_CACHE_DIR='1',PYTHONNOUSERSITE='1')
    args=['--disable-pip-version-check','--no-cache-dir','install','--dry-run','--ignore-installed',
          '--only-binary=:all:','--index-url','https://pypi.org/simple','--report',str(a.output/'pip-report.json'),
          '--python-version','3.13','--implementation','cp','--abi','cp313','--abi','abi3','--abi','none']
    for platform in [f'manylinux_2_{n}_x86_64' for n in range(28,16,-1)]+['manylinux2014_x86_64','manylinux2010_x86_64','manylinux1_x86_64','linux_x86_64']:
        args+=['--platform',platform]
    args += [name+'=='+version for name,version in requirements.items()]
    try:
        result=pip_main(args)
        if result:raise RuntimeError('metadata-only pip resolver rejected: '+str(result))
    finally:
        HTTPAdapter.send=original;guard.save()

if __name__=='__main__':main()
