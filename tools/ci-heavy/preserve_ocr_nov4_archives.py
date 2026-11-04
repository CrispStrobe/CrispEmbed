#!/usr/bin/env python3
"""Closed-list byte preservation and uploaded-archive readback; no model work."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
import extract_ocr_nov4_evidence as archive

ROOT = Path(__file__).resolve().parents[2]
SELECTOR = "tools/ci-heavy/preserve_ocr_nov4_archives.py"
HELPER_SHA256 = "5b52a1b14eb8658c3371fff920fade76c2df1346c0659b2d372322144bab84e6"
REPO_ID = 1210669686
SOURCE = "1bb0ddb1d9ac12ded51eb5bd0edd89be0a8259e6"
BRANCH = "audit/mineru2605-aligned-q4-20261104"
TARGETS = tuple(dict(label=label, run_id=run, artifact_id=artifact_id, bytes=size, sha256=digest,
                     source=SOURCE, branch="fix/ocr-qualification-nov4" if label.endswith("receipt") else BRANCH,
                     conclusion="success") for label, run, artifact_id, size, digest in (
    ("mineru2605-receipt",41871894063,13345908772,858185387,"4a8053631886ae78df4bbc131785d702c08b306b62d19e9b930480f85bfdb403"),
    ("mineru2605-formula",41877117676,13347927913,136053962,"7a052a5a73b47d90431c9fabdcea9af10ada9239a105ecd642d21d98a8d0705e"),
    ("mineru2605-japanese",41877121416,13348396823,249816387,"8919d773dc544e30084d3b3da14519fdd884297548bdf5e91d08d993688a868d"),
    ("mineru2605-arabic",41877125216,13348311566,68835208,"fad1d207c52c1b37e0c6f65c10ac10a701e31b4cec84c0392f85792d42a3ae95"),
    ("mineru2605-table",41877129385,13347724003,386986678,"1efd735469a8956aa6364e6fbd9587acf9f93958a48c39ad6a51393edf1164b5"),
    ("mineru2605-layout-page",41877137082,13350809558,1549053766,"701decb5245cc0bdf5e1993370eec2da95fa66e96e94043ac594c3947989aba4"),
    ("mineru2605-layout-rot90",41877141270,13353650933,2466702003,"4474b74df3b369e2d0f3c4f6f204d88f6de2ac7612605cbaa7f846444e3aa212"),
)) + (dict(label="lighton08-inventory-table",run_id=41810175624,artifact_id=13312356863,bytes=3459070891,
           sha256="7517525e72d29d1e535257458cc11942d3df360a6fb7b1ab0182ce8fff2138da",
           source="2f1bbdb11c3612acf7399e31b33ac715027f3184",branch="main",conclusion="failure"),
    dict(label="mineru2605-receipt-initial-failure",run_id=41810161887,artifact_id=13313715962,bytes=745003580,
         sha256="36fc4d014373c8b116d48afd1105f3e3a55f9869c4bea8e8af2785c8b3f52c16",
         source="2f1bbdb11c3612acf7399e31b33ac715027f3184",branch="main",conclusion="failure"))
GERMAN = dict(label="mineru2605-german", run_id=41877132652, artifact_id=13355531707, bytes=2543809676,
              sha256="3ad000ba9da8bbb7155a6770e231d00d25357ab2d4eba8e374a1f3d867f4fd3f",
              source=SOURCE, branch=BRANCH, conclusion="success")
TARGETS = TARGETS + (GERMAN,)
TOTAL = sum(t["bytes"] for t in TARGETS)
require = archive.require


def select_targets(german_only):
    require(type(german_only) is bool, "batch selector must be boolean")
    return (GERMAN,) if german_only else TARGETS


def validate_batch_args(env, german_only):
    require(env.get("PRESERVE_SELECTED_ARGS", "") == ("--german-only" if german_only else ""),
            "closed preservation batch args differ")


def expiry(item):
    require(item.get("expired") is False, "artifact expired")
    value = item.get("expires_at")
    require(isinstance(value, str) and datetime.fromisoformat(value.replace("Z", "+00:00")) > datetime.now(timezone.utc),
            "artifact expiry differs")


def producer(run, source, branch, run_id, completed=True, conclusion=None):
    expected = dict(id=run_id, head_sha=source, head_branch=branch, event="workflow_dispatch",
                    path=".github/workflows/heavy-cpu.yml", run_attempt=1)
    if completed: expected.update(status="completed", conclusion=conclusion)
    for key, value in expected.items(): require(run.get(key) == value, "producer differs: " + key)
    for key in ("repository", "head_repository"):
        require(run.get(key, {}).get("id") == REPO_ID and run[key].get("full_name") == archive.REPOSITORY,
                "producer repository differs")


def artifact_identity(item, artifact_id, run_id, source, branch, name):
    for key, value in dict(id=artifact_id, name=name,
                          url=archive.API + "/actions/artifacts/" + str(artifact_id),
                          archive_download_url=archive.API + "/actions/artifacts/" + str(artifact_id) + "/zip").items():
        require(item.get(key) == value, "artifact differs: " + key)
    expiry(item)
    for key, value in dict(id=run_id, head_sha=source, head_branch=branch, repository_id=REPO_ID,
                          head_repository_id=REPO_ID).items():
        require(item.get("workflow_run", {}).get(key) == value, "artifact producer differs: " + key)
    require(type(item.get("size_in_bytes")) is int and item["size_in_bytes"] > 0 and
            isinstance(item.get("digest"), str) and re.fullmatch(r"sha256:[0-9a-f]{64}", item["digest"]),
            "artifact size/digest invalid")


def validate_metadata(target, run, item):
    require(target in TARGETS, "target outside frozen list")
    producer(run,target["source"],target["branch"],target["run_id"],conclusion=target["conclusion"])
    artifact_identity(item,target["artifact_id"],target["run_id"],target["source"],target["branch"],"heavy-"+str(target["run_id"]))
    require(item["size_in_bytes"] == target["bytes"] and item["digest"] == "sha256:"+target["sha256"],
            "original archive bytes/digest differ")


def stream_hash(stream, expected_bytes, expected_sha, deadline, output=None):
    count, digest = 0, hashlib.sha256()
    while True:
        require(time.monotonic() <= deadline, "archive deadline exceeded")
        block = stream.read(min(archive.CHUNK, expected_bytes-count+1))
        if not block: break
        count += len(block)
        require(count <= expected_bytes, "archive exceeds expected size")
        digest.update(block)
        if output is not None: output.write(block)
    require(count == expected_bytes and digest.hexdigest() == expected_sha, "archive bytes/hash differ")
    return dict(bytes=count,sha256=digest.hexdigest())


def readback(path, targets=TARGETS):
    expected = {t["label"]+".zip":t for t in targets}
    rows = []
    with zipfile.ZipFile(path) as z:
        archive.safe_members(z)
        infos = z.infolist()
        require(len(infos) == len(expected) and {i.filename for i in infos} == set(expected), "outer member list differs")
        require(sum(i.file_size for i in infos) == sum(t["bytes"] for t in targets), "outer expanded bytes differ")
        for info in infos:
            target = expected[info.filename]
            require(not info.is_dir() and info.file_size == target["bytes"], "inner ZIP size differs")
            with z.open(info) as stream:
                rows.append(dict(label=target["label"], **stream_hash(stream,target["bytes"],target["sha256"],time.monotonic()+1200)))
    return rows


def transfer(target, destination, token):
    with urllib.request.build_opener(archive.NoRedirect).open(archive.archive_location(target,token),timeout=60) as stream:
        # Keep task-owned partial bytes if the stream fails; never label them complete.
        with destination.open("xb") as output:
            return stream_hash(stream,target["bytes"],target["sha256"],time.monotonic()+1200,output)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--verify-artifact",type=int)
    parser.add_argument("--german-only",action="store_true")
    args=parser.parse_args()
    targets=select_targets(args.german_only)
    total=sum(target["bytes"] for target in targets)
    out=Path(os.environ["PRESERVE_OUT"]); review=out/"review"; review.mkdir(parents=True,exist_ok=True)
    captured = None
    if args.verify_artifact and (review/"receipt.json").exists():
        captured = archive.read_json((review/"receipt.json").read_bytes())
    receipt=dict(schema="crispembed.ocr-nov4.archive-preservation.v1", mode="verify" if args.verify_artifact else "capture",
                 completed=False, original_forward_calls=0, native_forward_calls=0, builds=0,
                 model_qualified=False, publication_qualified=False, originals=[])
    if isinstance(captured,dict):
        receipt["capture_receipt"]=captured
        receipt["originals"]=captured.get("originals",[])
    def save():
        temporary=review/"receipt.pending.json"
        temporary.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
        temporary.replace(review/"receipt.json")
    save()
    try:
        receipt["stage"]="dispatch-and-source"
        require(hashlib.sha256(Path(archive.__file__).read_bytes()).hexdigest()==HELPER_SHA256,"archive helper changed")
        env=os.environ
        for key,value in dict(GITHUB_ACTIONS="true",GITHUB_REPOSITORY=archive.REPOSITORY,RUNNER_OS="Linux",
                              PRESERVE_SELECTED_SCRIPT=SELECTOR,PRESERVE_SELECTED_RUNNER="ubuntu-latest",
                              PRESERVE_SELECTED_MODE="portable",PRESERVE_SELECTED_PIP="").items():
            require(env.get(key,"")==value,"closed dispatch differs: "+key)
        validate_batch_args(env,args.german_only)
        source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        require(source==env["GITHUB_SHA"] and not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT),"source dirty/different")
        receipt["source"]=source; receipt["targets"]=targets; receipt["german_only"]=args.german_only; receipt["total_original_zip_bytes"]=total
        if args.verify_artifact:
            require(isinstance(captured,dict) and captured.get("mode")=="capture" and
                    captured.get("completed") is True and captured.get("source")==source and
                    captured.get("targets")==list(targets) and captured.get("german_only") is args.german_only,"completed same-source capture receipt required")
            require(len(captured.get("originals",[]))==len(targets),"capture provenance incomplete")
            for target,row in zip(targets,captured["originals"]):
                require(row.get("target")==target and row.get("complete") is True,"capture row differs")
                validate_metadata(target,row["run"],row["artifact"])
                require(row.get("archive")==dict(bytes=target["bytes"],sha256=target["sha256"]),"capture archive differs")
            receipt["originals"]=captured["originals"]
            receipt["capture_completed"]=True
            receipt.pop("capture_receipt",None)
        save()
        scratch=Path(env["PRESERVE_SCRATCH"]); scratch.mkdir(parents=True,exist_ok=True)
        memory=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        require(memory>=2*2**30,"need 2GiB available RAM")
        require(out.stat().st_dev==scratch.stat().st_dev,"scratch/output must share admitted disk")
        required= (total+2*2**30 if args.verify_artifact else 2*total+2*2**30)
        require(shutil.disk_usage(out).free>=required,"insufficient two-copy preservation storage")
        token=env["GH_TOKEN"]
        if args.verify_artifact:
            receipt['stage']='uploaded-artifact-metadata'; save()
            run_id=int(env["GITHUB_RUN_ID"])
            run=archive.fetch_metadata('/actions/runs/'+str(run_id),token)
            producer(run,source,env["GITHUB_REF_NAME"],run_id,completed=False)
            item=archive.fetch_metadata('/actions/artifacts/'+str(args.verify_artifact),token)
            artifact_identity(item,args.verify_artifact,run_id,source,env["GITHUB_REF_NAME"],"ocr-nov4-original-zips-"+str(run_id))
            require(item["size_in_bytes"]<=total+2**20,"outer ZIP exceeds cap")
            receipt['uploaded_artifact']=item; save()
            target=dict(artifact_id=args.verify_artifact,bytes=item['size_in_bytes'],sha256=item['digest'].removeprefix('sha256:'))
            path=scratch/'uploaded-originals.zip'
            receipt['stage']='outer-archive-transfer'; save()
            receipt['outer_archive']=transfer(target,path,token); save()
            receipt['stage']='inner-archive-readback'; save()
            receipt['readback']=readback(path,targets)
            receipt['complete_readback']=True
        else:
            raw=out/'raw'; require(not raw.exists(),"fresh raw directory required"); raw.mkdir()
            for target in targets:
                receipt['stage']='original-'+target['label']; save()
                run=archive.fetch_metadata('/actions/runs/'+str(target['run_id']),token)
                item=archive.fetch_metadata('/actions/artifacts/'+str(target['artifact_id']),token)
                row=dict(target=target,run=run,artifact=item,complete=False); receipt['originals'].append(row); save()
                validate_metadata(target,run,item)
                row['archive']=transfer(target,raw/(target['label']+'.zip'),token); row['complete']=True; save()
        receipt['stage']='complete'; receipt['completed']=True; save()
        return 0
    except Exception as error:
        # Exception text may contain provider signed URLs or local paths.
        receipt['failure']=dict(type=type(error).__name__,message="Preservation failed; inspect the named validation stage without publishing transport URLs")
        save(); raise RuntimeError("OCR archive preservation failed (receipt retained)") from None


if __name__=='__main__':
    raise SystemExit(main())
