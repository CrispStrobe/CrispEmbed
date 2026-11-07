"""POSIX whole-worker deadline and subprocess-group cleanup for diagnostics."""
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


class WorkerDeadline(TimeoutError):
    pass


class Deadline:
    def __init__(self, seconds):
        if seconds <= 0:
            raise ValueError('positive deadline required')
        self.seconds = seconds

    def __enter__(self):
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        self.previous_timer = signal.getitimer(signal.ITIMER_REAL)
        if self.previous_timer != (0.0, 0.0):
            raise RuntimeError('refuse to overwrite existing deadline')
        self.started = time.monotonic()
        signal.signal(signal.SIGALRM, self.expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    @staticmethod
    def expired(signum, frame):
        raise WorkerDeadline('whole-worker deadline exceeded')

    def close(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous_handler)

    def __exit__(self, *args):
        self.close()


def command(args, log, timeout, receipt, env=None, cwd=None, on_tick=None):
    started = time.monotonic()
    child = None
    peak_rss = 0
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt.update(status='failed', returncode=None)
    try:
        with Path(log).open('w') as stream:
            child = subprocess.Popen(list(map(str, args)), stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=env, cwd=cwd)
            receipt['pid'] = child.pid
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(args, timeout)
                try:
                    rows = dict(x.split(':', 1) for x in Path(f'/proc/{child.pid}/status').read_text().splitlines())
                    peak_rss = max(peak_rss, int(rows.get('VmHWM', '0').split()[0]) * 1024)
                except (OSError, ValueError):
                    pass
                if on_tick is not None:
                    on_tick()
                try:
                    code = child.wait(timeout=min(1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            receipt['returncode'] = code
            if code:
                raise subprocess.CalledProcessError(code, args)
            receipt['status'] = 'completed'
    except BaseException as error:
        receipt['error_type'], receipt['error'] = type(error).__name__, str(error)
        raise
    finally:
        # Descendants can survive even a normally exited parent. Always clean its
        # session group, and reap our actual child before preserving returncode.
        if child is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    break
                if sig == signal.SIGTERM:
                    time.sleep(.05)
            child.wait(timeout=2)
            receipt['returncode'] = child.returncode
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt.update(elapsed_seconds=time.monotonic() - started,
                       child_user_seconds=usage.ru_utime - before.ru_utime, child_system_seconds=usage.ru_stime - before.ru_stime,
                       peak_rss_bytes_observed=peak_rss,
                       rss_sampling_scope="spawned process VmHWM at up to 1s intervals; short-lived peaks may be missed",
                       cumulative_child_peak_rss_bytes=usage.ru_maxrss * 1024)
