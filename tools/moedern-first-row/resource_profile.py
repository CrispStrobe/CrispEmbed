"""One-row CPU resource profile v2; no numerical/runtime policy is changed."""
import os
from pathlib import Path
try:
 import resource
except ImportError:
 resource=None
import shutil
import stat

GIB=2**30
POLICY=dict(schema='crispembed-moedern-first-row-resources-v2',minimum_available_ram_bytes=14*GIB,
 minimum_free_disk_bytes=10*GIB,minimum_cpus=2,maximum_output_bytes=2*GIB,
 minimum_live_disk_reserve_bytes=2*GIB,maximum_single_file_bytes=2*GIB,
 disk_budget_gib=dict(packages_and_caches=4,source_and_build=2,retained_output=2,free_reserve=2))

def require(ok,message):
 if not ok:raise ValueError(message)

def output_extent(root):
 root=Path(root);total=0;files=0
 if root.exists():
  for path in root.rglob('*'):
   row=path.lstat()
   if stat.S_ISREG(row.st_mode)or stat.S_ISLNK(row.st_mode):total+=row.st_size;files+=1
 return dict(bytes=total,files=files)

def live(root,*,free_disk=None,extent=None):
 root=Path(root);probe=root
 while not probe.exists():probe=probe.parent
 free=shutil.disk_usage(probe).free if free_disk is None else free_disk
 size=output_extent(root)if extent is None else extent
 require(size['bytes']<=POLICY['maximum_output_bytes'],'resource-v2 retained output exceeds2GiB; evidence preserved')
 require(free>=POLICY['minimum_live_disk_reserve_bytes'],'resource-v2 live reserve below2GiB; evidence preserved')
 return dict(output_bytes=size['bytes'],output_files=size['files'],free_disk_bytes=free,
             minimum_live_disk_reserve_bytes=POLICY['minimum_live_disk_reserve_bytes'])

def values(ram,disk,cpus):
 require(ram>=POLICY['minimum_available_ram_bytes']and disk>=POLICY['minimum_free_disk_bytes']and cpus>=POLICY['minimum_cpus'],
         'resource-v2 requires14GiB available RAM/10GiB disk/two CPUs')
 return dict(available_ram_bytes=ram,free_disk_bytes=disk,cpu_count=cpus,policy=POLICY)

def admit(path,output_root=None):
 path=Path(path);probe=path
 while not probe.exists():probe=probe.parent
 ram=int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
 report=values(ram,shutil.disk_usage(probe).free,os.cpu_count()or 0)
 report['live']=live(output_root if output_root is not None else path)
 return report

def single_file_limit():
 # Inherited by compiler/package/model children. Existing aggregate capture
 # bounds plus per-command aggregate checks close the one-row output envelope.
 require(resource is not None,'Linux resource limits required for actual execution')
 old_soft,old_hard=resource.getrlimit(resource.RLIMIT_FSIZE)
 limit=POLICY['maximum_single_file_bytes']
 if old_hard!=resource.RLIM_INFINITY:limit=min(limit,old_hard)
 require(limit>=POLICY['maximum_single_file_bytes'],'existing file-size limit below declared2GiB profile')
 resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
 return dict(maximum_single_file_bytes=limit)
