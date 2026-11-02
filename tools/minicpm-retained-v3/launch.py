#!/usr/bin/env python3
"""Bounded hosted supervisor; grace permits canonical receipts and child cleanup."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import atomic,sha,require


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--timeout',type=int,required=True);parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args();require(0<args.timeout<=4*3600 and args.command[0]=='--','Explicit bounded hosted command required')
    log=args.receipt.with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic();timed=False
    with log.open('wb')as stream:
        child=subprocess.Popen(args.command[1:],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed=True
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
    atomic(args.receipt,dict(exit=child.returncode,timed_out=timed,seconds=time.monotonic()-started,log=log.name,sha256=sha(log),
                            per_child_rss_measured=False,limits='No RSS estimate substituted; future unobserved benchmark owns per-child wait4 measurements.'))
    raise SystemExit(child.returncode or(124 if timed else 0))


if __name__=='__main__':main()
