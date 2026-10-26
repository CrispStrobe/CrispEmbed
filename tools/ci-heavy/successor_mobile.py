#!/usr/bin/env python3
"""Cross-link the actual mobile archive; never claim device execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]

def run(directory,abi):
 if os.environ.get('GITHUB_ACTIONS')!='true':raise ValueError('Hosted only')
 record=json.loads((directory/'consumer.json').read_text());logs=directory/'mobile-link.log'
 source=ROOT/'tests/successor_capi_consumer.c'
 if abi=='ios-arm64':
  archive=directory/'crispembed-ios-arm64/libcrispembed-static.a'
  sdk=subprocess.check_output(['xcrun','--sdk','iphoneos','--show-sdk-path'],text=True).strip()
  command=['xcrun','--sdk','iphoneos','clang','-arch','arm64','-miphoneos-version-min=15.0','-isysroot',sdk,
   str(source),'-I'+str(ROOT/'src'),str(archive),'-lc++','-framework','Foundation','-framework','Accelerate',
   '-framework','Metal','-framework','MetalKit','-o',str(directory/'device-link-check')]
  symbols=subprocess.check_output(['nm','-g',str(archive)],text=True)
  if '_crispembed_init' not in symbols:raise ValueError('iOS public init symbol absent')
 else:
  ndk=Path(os.environ['ANDROID_NDK_HOME'])/'toolchains/llvm/prebuilt/linux-x86_64/bin'
  target='aarch64-linux-android24-clang' if abi=='android-arm64-v8a' else 'armv7a-linux-androideabi24-clang'
  archive=directory/('crispembed-'+abi)/'libcrispembed.so'
  symbols=subprocess.check_output([str(ndk/'llvm-readelf'),'-h','-d','-Ws',str(archive)],text=True)
  machine='AArch64' if abi=='android-arm64-v8a' else 'ARM'
  if ('Machine:' not in symbols or machine not in symbols or 'crispembed_init' not in symbols):raise ValueError('Android ELF architecture/public symbol absent')
  command=[str(ndk/target),str(source),'-I'+str(ROOT/'src'),str(archive),'-Wl,--no-undefined','-o',str(directory/'device-link-check')]
 (directory/'mobile-symbols.txt').write_text(symbols)
 with logs.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
 record.update(mobile_abi=abi,cross_link_exit_code=result.returncode,cross_link_passed=result.returncode==0,
  device_execution_performed=False,qualification=False,
  limits=['Actual archive cross-link and symbol/architecture checks only; device inference remains unverified.',
   'A link failure is a packaging/dependency rejection, not a waived gate.'])
 (directory/'consumer.json').write_text(json.dumps(record,indent=2)+'\n')
 if result.returncode:raise ValueError('Actual mobile archive cross-link failed; preserve dependency log')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--abi',choices=('ios-arm64','android-arm64-v8a','android-armeabi-v7a'),required=True);a=p.parse_args();run(a.output,a.abi)
