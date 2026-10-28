"""Bounded Linux child execution with direct wait4 resource accounting.

No external time utility or cumulative RUSAGE_CHILDREN peak. The returned RSS
is Linux wait4's per-child ru_maxrss (largest process, not concurrent tree sum).
Only this function reaps its Popen child; poll()/wait() must not consume it first.
"""
from contextlib import ExitStack
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run(args,log,*,cwd=None,env=None,timeout=1200,stdout_path=None):
    if not sys.platform.startswith('linux')or not hasattr(os,'wait4'):
        raise ValueError('Linux per-child wait4 accounting required')
    if not isinstance(timeout,(int,float))or isinstance(timeout,bool)or not math.isfinite(timeout)or not 0<timeout<=24*3600:
        raise ValueError('finite positive bounded timeout required')
    args=[str(v)for v in args]
    if not args:raise ValueError('nonempty child argv required')
    start=time.monotonic();deadline=start+timeout;timed_out=False;reaped=False
    with ExitStack()as files:
        stderr=files.enter_context(Path(log).open('wb'))
        stdout=files.enter_context(Path(stdout_path).open('wb'))if stdout_path is not None else stderr
        child=subprocess.Popen(args,cwd=cwd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)

        def wait(options):
            while True:
                try:return os.wait4(child.pid,options)
                except InterruptedError:continue

        def kill_group():
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass

        try:
            while True:
                pid,status,usage=wait(os.WNOHANG)
                if pid==child.pid:
                    reaped=True;break
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    timed_out=True;kill_group();pid,status,usage=wait(0);reaped=True;break
                time.sleep(min(.05,remaining))
        finally:
            if not reaped:
                # Exceptions and interruptions must not leave a model/compiler
                # process group running. Reap exactly our direct child once.
                kill_group();pid,status,usage=wait(0);reaped=True
            child.returncode=os.waitstatus_to_exitcode(status)
        if pid!=child.pid:raise RuntimeError('wait4 returned another child')
        rss_kib=int(usage.ru_maxrss)
        if rss_kib<0:raise RuntimeError('negative per-child RSS')
        return dict(exit=child.returncode,termination_signal=os.WTERMSIG(status)if os.WIFSIGNALED(status)else None,
                    timed_out=timed_out,seconds=time.monotonic()-start,maximum_rss_kib=rss_kib,
                    maximum_rss_bytes=rss_kib*1024,resource_accounting='linux-wait4-per-child',
                    direct_child_pid=child.pid,reaped=True)
