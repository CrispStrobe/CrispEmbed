#!/usr/bin/env python3
"""Consume one sealed desktop producer; compile only small boundary consumers.

The empty result is legacy no-model compatibility, not recognized blank-page
quality. No app build, model download, decoded OCR, GPU or mobile qualification.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PRODUCER_RUN = 43302121233
WORKFLOW_SOURCE = '841179b848f8d3aa15e44dd2d5aea65070c82f71'
RUNTIME_SOURCE = '2d2ab3275022e095f4a8f580e276bb2e3aa3f59c'
TARGETS = [dict(runner='ubuntu-24.04', artifact='linux-x86_64'),
           dict(runner='ubuntu-24.04-arm', artifact='linux-arm64'),
           dict(runner='macos-15', artifact='macos-arm64'),
           dict(runner='windows-2022', artifact='windows-x86_64')]
MARKER = 'boundary-json:'
NATIVE_KEYS = {'reported_failure', 'mixed_failure_not_masked', 'legacy_nonnull_empty', 'null_then_empty_recovery'}
RUST_KEYS = {'reported_failure_as_err', 'mixed_failure_as_err', 'legacy_empty_as_ok'}

NATIVE_SOURCE = r'''
#include "crispembed.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void require(int ok) { if (!ok) { fputs("native boundary control failed\n", stderr); exit(2); } }
static void *pipeline(const crispembed_ocr_stage *stages, int count) {
    void *ctx=crispembed_ocr_pipeline_init_stages(0,NULL,NULL,NULL,NULL,NULL,NULL,stages,count,1);
    require(ctx!=NULL); return ctx;
}
static void failure(void *ctx, const char *path) {
    int count=99; const char *text="stale"; float confidence=1;
    const crispembed_ocr_result *rows=crispembed_ocr_pipeline_run(ctx,path,&count,&text,&confidence);
    require(!rows && count==0 && !text && confidence==0);
    require(strstr(crispembed_ocr_pipeline_last_error(ctx),"initialization failed")!=NULL);
    int length=99;
    require(!crispembed_ocr_pipeline_review_json(ctx,&length) && length==0);
}
int main(int argc, char **argv) {
    require(argc==3); crispembed_set_offline(1);
    crispembed_ocr_stage stages[2]={{0},{0}};
    stages[0].engine=14; stages[0].model_a=argv[2];
    void *ctx=pipeline(stages,1); failure(ctx,argv[1]); crispembed_ocr_pipeline_free(ctx);
    ctx=pipeline(stages,2); failure(ctx,argv[1]); crispembed_ocr_pipeline_free(ctx);
    ctx=pipeline(stages+1,1);
    int count=99; const char *text="stale"; float confidence=1;
    crispembed_ocr_pipeline_run(ctx,NULL,&count,&text,&confidence);
    require(!text && count==0 && confidence==0 && *crispembed_ocr_pipeline_last_error(ctx));
    crispembed_ocr_pipeline_run(ctx,argv[1],&count,&text,&confidence);
    require(text && !*text && count==0 && !*crispembed_ocr_pipeline_last_error(ctx));
    crispembed_ocr_pipeline_free(ctx);
    puts("boundary-json:{\"reported_failure\":true,\"mixed_failure_not_masked\":true,\"legacy_nonnull_empty\":true,\"null_then_empty_recovery\":true}");
    return 0;
}
'''

RUST_SOURCE = r'''
use crispembed::{CrispOcrPipeline, OcrCleanupSpec, OcrStageSpec};
fn stage(engine: i32, model: &str) -> OcrStageSpec {
    OcrStageSpec { source_type: 0, engine, model_a: model.into(), model_b: String::new(),
        cleanup: OcrCleanupSpec { enabled: false, ..Default::default() }, det_prob_threshold: 0.3,
        det_box_threshold: 0.5, det_target_short: 736, vlm_max_tokens: 0, vlm_prompt: String::new(),
        min_chars: 0, min_confidence: 0.0 }
}
fn pipeline(stages: &[OcrStageSpec]) -> CrispOcrPipeline {
    CrispOcrPipeline::from_stages(false,None,None,None,None,None,None,stages,1).unwrap()
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let failed=stage(14,&args[2]); let empty=stage(0,"");
    assert!(pipeline(&[failed.clone()]).run(&args[1]).is_err());
    assert!(pipeline(&[failed,empty.clone()]).run(&args[1]).is_err());
    let result=pipeline(&[empty]).run(&args[1]).unwrap();
    assert!(result.full_text.is_empty() && result.regions.is_empty());
    println!("boundary-json:{{\"reported_failure_as_err\":true,\"mixed_failure_as_err\":true,\"legacy_empty_as_ok\":true}}");
}
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header_binding(packaged, committed, member, windows):
    """Bind raw archive bytes, then permit only the observed Windows Git edit.

    Do not normalize files or their recorded hashes. The entire packaged header
    must be exact, or equal a whole-file LF-to-CRLF transform of a CR-free blob.
    """
    raw_sha = hashlib.sha256(packaged).hexdigest()
    if member.get('kind') != 'file' or member.get('bytes') != len(packaged) or member.get('sha256') != raw_sha:
        raise ValueError('Packaged header differs from its verified raw manifest member')
    mode = 'exact-bytes'
    if packaged != committed:
        if not (windows and b'\r' not in committed and b'\n' in committed
                and packaged == committed.replace(b'\n', b'\r\n')):
            raise ValueError('Packaged header differs from exact runtime source')
        mode = 'windows-git-whole-file-lf-to-crlf'
    return {'transformation': mode, 'packaged_sha256': raw_sha,
            'committed_source_sha256': hashlib.sha256(committed).hexdigest(),
            'raw_manifest_member_sha256': member['sha256']}


def verify_header(package, runtime, output, receipt, windows):
    manifest_zip = output / 'manifest.zip'
    if receipt['manifest_artifact']['digest'] != 'sha256:' + digest(manifest_zip):
        raise ValueError('Verified manifest artifact bytes changed')
    with zipfile.ZipFile(manifest_zip) as archive:
        manifest = json.loads(archive.read('successor-native-manifest.json'))
    archive_names = [row['name'] for row in receipt['archives']]
    if len(archive_names) != 1:
        raise ValueError('One verified archive required for header binding')
    archives = [row for row in manifest['files'] if row['name'] == archive_names[0]]
    if len(archives) != 1:
        raise ValueError('Header archive is not bound by verified manifest')
    members = [row for row in archives[0]['members'] if row['path'] == 'include/crispembed.h']
    if len(members) != 1:
        raise ValueError('Exactly one verified public header member required')
    committed = subprocess.check_output(['git', '-C', str(runtime), 'show', RUNTIME_SOURCE + ':src/crispembed.h'])
    if (runtime / 'src/crispembed.h').read_bytes() != committed:
        raise ValueError('Runtime header checkout differs from its exact committed bytes')
    return header_binding((package / 'include/crispembed.h').read_bytes(), committed, members[0], windows)


def api_run():
    return json.loads(subprocess.check_output(['gh', 'api', f'repos/CrispStrobe/CrispEmbed/actions/runs/{PRODUCER_RUN}'], text=True))


def plan(target, service=None):
    if target not in {'all', *(item['artifact'] for item in TARGETS)}:
        raise ValueError('Unknown desktop target')
    service = api_run() if service is None else service
    if any(service.get(key) != value for key, value in {
        'head_sha': WORKFLOW_SOURCE, 'path': '.github/workflows/release.yml',
        'event': 'workflow_dispatch', 'status': 'completed', 'conclusion': 'success'}.items()):
        raise ValueError('Existing producer identity/status differs')
    return {'include': [item for item in TARGETS if target == 'all' or item['artifact'] == target]}


def write_image(path):
    pixels = bytes([255]) * (32 * 32 * 3)
    header = struct.pack('<2sIHHI', b'BM', 54 + len(pixels), 0, 0, 54)
    info = struct.pack('<IiiHHIIiiII', 40, 32, 32, 1, 24, 0, len(pixels), 0, 0, 0, 0)
    path.write_bytes(header + info + pixels)


def parse_control(stdout, keys):
    lines = [line[len(MARKER):] for line in stdout.splitlines() if line.startswith(MARKER)]
    if len(lines) != 1:
        raise ValueError('Exactly one actual execution report is required')
    report = json.loads(lines[0])
    if set(report) != keys or any(value is not True for value in report.values()):
        raise ValueError('Actual boundary control did not pass its declared checks')
    return report


def rust_project(runtime, directory):
    locked = (runtime / 'Cargo.lock').read_text()
    packages = tomllib.loads(locked)['package']
    binding = next(package for package in packages if package['name'] == 'crispembed')
    directory.mkdir()
    (directory / 'src').mkdir()
    (directory / 'src/main.rs').write_text(RUST_SOURCE, encoding='utf-8')
    manifest = ('[package]\nname="packaged-ocr-boundary"\nversion="0.0.0"\nedition="2021"\n'
                '[dependencies]\ncrispembed={path=' + json.dumps((runtime / 'crispembed').as_posix()) +
                ',version=' + json.dumps('=' + binding['version']) + '}\n')
    (directory / 'Cargo.toml').write_text(manifest)
    (directory / 'Cargo.lock').write_text(locked + '\n[[package]]\nname="packaged-ocr-boundary"\nversion="0.0.0"\ndependencies=["crispembed"]\n')


def execute(command, env, logs, label, timeout=120):
    result = subprocess.run(list(map(str, command)), env=env, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    (logs / (label + '.stdout.log')).write_text(result.stdout, encoding='utf-8')
    (logs / (label + '.stderr.log')).write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{label} failed with exit {result.returncode}; see retained logs')
    return result.stdout


def prebuilt_metadata(target, package):
    outputs = list(target.glob('debug/build/crispembed-sys-*/output'))
    if len(outputs) != 1:
        raise ValueError('One actual CrispEmbed prebuilt link record required')
    dirs = [line.split('=', 1)[1] for line in outputs[0].read_text().splitlines() if line.startswith('cargo:LIB_DIR=')]
    if len(dirs) != 1 or Path(dirs[0]).resolve() != package.resolve():
        raise ValueError('Rust consumer did not select the verified packaged library')
    return outputs[0]


def windows_msvc_linker(env):
    """Use the configured MSVC compiler's linker, never PATH's Unix link.exe."""
    compiler = shutil.which('cl.exe', path=env.get('PATH', ''))
    if compiler is None:
        raise ValueError('Configured MSVC cl.exe is required for the Windows Rust consumer')
    linker = Path(compiler).resolve().with_name('link.exe')
    if not linker.is_file():
        raise ValueError('MSVC link.exe must be adjacent to the configured cl.exe')
    return linker


def consume(target, workflow, runtime, output):
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise ValueError('Hosted execution only')
    plan(target)
    if target == 'all':
        raise ValueError('Each worker consumes one package')
    workflow, runtime, output = (path.resolve() for path in (workflow, runtime, output))
    for source in (ROOT, workflow, runtime):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Output must be outside every source checkout')
    controller = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if controller != os.environ['GITHUB_SHA']:
        raise ValueError('Consumer controller checkout differs from its workflow run')
    controller_blob = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                               controller + ':tools/ci-heavy/desktop_ocr_candidate_consumer.py'])
    if hashlib.sha256(controller_blob).hexdigest() != digest(__file__):
        raise ValueError('Executed consumer controller differs from its committed bytes')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    # PR228's validator treats GITHUB_SHA as the validator checkout identity.
    # Scope this override to that child; retain the true controller ID below.
    validator_env = dict(env, GITHUB_SHA=WORKFLOW_SOURCE)
    command = [sys.executable, workflow / 'tools/ci-heavy/successor_assets.py', '--run', str(PRODUCER_RUN),
               '--kind', 'native', '--artifact', target, '--output', output, '--desktop-candidate',
               '--runtime-source-commit', RUNTIME_SOURCE, '--runtime-source-root', runtime]
    subprocess.run(list(map(str, command)), env=validator_env, check=True, timeout=300)
    receipt = json.loads((output / 'consumer.json').read_text())
    if receipt['source_commit'] != RUNTIME_SOURCE or receipt['workflow_source_commit'] != WORKFLOW_SOURCE:
        raise ValueError('Validated producer/runtime receipt differs')
    packages = [path for path in output.glob('crispembed-*') if path.is_dir()]
    if len(packages) != 1:
        raise ValueError('Exactly one verified desktop package required')
    package = packages[0]
    windows = target == 'windows-x86_64'
    library = package / ('crispembed.dll' if windows else 'libcrispembed.dylib' if target == 'macos-arm64' else 'libcrispembed.so')
    if not library.is_file() or not (package / ('crispembed.lib' if windows else library.name)).is_file():
        raise ValueError('Verified packaged link library is missing; source builds are forbidden')
    header = verify_header(package, runtime, output, receipt, windows)
    logs = output / 'logs'
    logs.mkdir()
    env.update(CRISPEMBED_OFFLINE='1', HF_HUB_OFFLINE='1', CRISPEMBED_SYS_LIB_DIR=str(package),
               CARGO_TARGET_DIR=str(output / 'rust-target'), CARGO_HOME=str(output / 'cargo-home'),
               CARGO_BUILD_JOBS='1', CRISPEMBED_THREADS='1', OMP_NUM_THREADS='1')
    for key in ('PATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
        env[key] = str(package) + os.pathsep + env.get(key, '')
    image = output / 'white.bmp'
    write_image(image)
    missing = output / 'missing-ocr-model.gguf'
    if missing.exists():
        raise ValueError('Missing-model fixture unexpectedly exists')
    source = output / 'native-control.c'
    source.write_text(NATIVE_SOURCE)
    native_binary = output / ('native-control.exe' if windows else 'native-control')
    spec = importlib.util.spec_from_file_location('producer_validator', workflow / 'tools/ci-heavy/successor_assets.py')
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    execute(validator.c_consumer_command(package, source, native_binary, windows), env, logs, 'native-compile')
    native = parse_control(execute([native_binary, image, missing], env, logs, 'native-run'), NATIVE_KEYS)
    rust = output / 'rust'
    rust_project(runtime, rust)
    if windows:
        linker = windows_msvc_linker(env)
        env['CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER'] = str(linker)
        linker_proof = {'selection': 'link.exe adjacent to configured MSVC cl.exe',
                        'linker_sha256': digest(linker)}
        (logs / 'rust-linker.json').write_text(json.dumps(linker_proof, indent=2) + '\n')
    rust_stdout = execute(['cargo', 'run', '--quiet', '--locked', '--manifest-path', rust / 'Cargo.toml', '--', image, missing],
                          env, logs, 'rust-run', timeout=600)
    rust_result = parse_control(rust_stdout, RUST_KEYS)
    metadata = prebuilt_metadata(output / 'rust-target', package)
    (logs / 'rust-prebuilt-link.txt').write_bytes(metadata.read_bytes())
    validator.desktop_source_closure(RUNTIME_SOURCE, WORKFLOW_SOURCE, runtime)
    report = dict(schema='crispembed-desktop-ocr-boundary-v1', producer_run=PRODUCER_RUN,
                  controller_source_commit=controller, workflow_source_commit=WORKFLOW_SOURCE,
                  controller_sha256=digest(__file__),
                  runtime_source_commit=RUNTIME_SOURCE, target=target, native=native, rust=rust_result,
                  library_sha256=digest(library), image_sha256=digest(image),
                  header_binding=header,
                  native_control_sha256=digest(source), rust_control_sha256=digest(rust / 'src/main.rs'),
                  rust_lock_sha256=digest(rust / 'Cargo.lock'), prebuilt_link_metadata_verified=True,
                  app_boundary_executed=False, positive_blank_model_executed=False,
                  decoded_ocr_executed=False, gpu_execution_qualified=False, qualification=False,
                  limits=['Actual model-free native/Rust boundary only; legacy empty is not positively recognized blank output.',
                          'CrispSorter application rebuild/error/export paths and decoded OCR remain unexecuted.',
                          'No runtime assets rebuilt; no model download, mobile, WASM or GPU qualification.'])
    if windows:
        report['rust_linker'] = linker_proof
    (output / 'boundary.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=['plan', 'consume'])
    parser.add_argument('--target', default='linux-x86_64')
    parser.add_argument('--workflow-source', type=Path)
    parser.add_argument('--runtime-source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'plan':
        print('matrix=' + json.dumps(plan(args.target), separators=(',', ':')))
    else:
        if any(value is None for value in (args.workflow_source, args.runtime_source, args.output)):
            parser.error('consume requires separate workflow/runtime checkouts and output')
        consume(args.target, args.workflow_source, args.runtime_source, args.output)


if __name__ == '__main__':
    main()
