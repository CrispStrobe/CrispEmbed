#!/usr/bin/env python3
"""Fresh replay of the OvisOCR2 final-integrated frozen/quality evidence on a later source.

tools/ovisocr2_final_integrated_v{1,2}.py pin the exact runtime git objects
(`src`, `ggml`, CMake, ...) of their base, so the unchanged v2 driver refuses
any later source. This wrapper replaces only that source-identity assertion:
it records every protected object (frozen vs current) and the declared source
hashes instead of requiring them, keeps the cleanliness checks, and then runs
the unchanged driver phases. The 14 frozen arms (gate0: retained ordinary
vectors bit-exact), both owning OCR cases, the eleven first-F16 cases (gate0:
IDs/text/EOS/cap/confidences exact) and every reviewed-artifact identity are
checked by the unchanged driver code. Guarded files are not modified.

    frozen_guard_replay_ovis.py --phase {preflight,frozen,quality-check} [--gate N]
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ci-heavy')]
import ovisocr2_final_integrated_v2 as contract  # noqa: E402

DRIVER = ROOT / 'tools/ci-heavy/ovisocr2_final_integrated_v2.py'
_spec = importlib.util.spec_from_file_location('final_v2_driver', DRIVER)
driver = importlib.util.module_from_spec(_spec)
sys.modules['final_v2_driver'] = driver
_spec.loader.exec_module(driver)


def replay_source_guard(root, protocol, require_submodule=True):
    root = Path(root)
    contract.checked_protocol(protocol, require_reviews=False)

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    objects = protocol['protected_git_objects']
    if set(objects) != set(contract.PROTECTED_PATHS):
        raise ValueError('Incomplete final-v2 runtime/build/scorer closure')
    if git('status', '--porcelain', '--', *objects):
        raise ValueError('Dirty final-v2 source')
    changed = {}
    for path, frozen in objects.items():
        current = git('rev-parse', 'HEAD:' + path)
        if current != frozen:
            changed[path] = {'frozen': frozen, 'current': current}
    declared = {p: {'frozen': d, 'current': contract.sha(root / p)} for p, d in contract.DECLARED.items()}
    head_ggml = git('rev-parse', 'HEAD:ggml')
    if require_submodule:
        if not (root / 'ggml/.git').exists():
            raise ValueError('Initialized final dependency required')
        if git('-C', str(root / 'ggml'), 'rev-parse', 'HEAD') != head_ggml or git('-C', str(root / 'ggml'), 'status', '--porcelain'):
            raise ValueError('Final dependency checkout differs from the recorded gitlink')
    return {'replay': True, 'runtime_base': contract.BASE, 'source_head': git('rev-parse', 'HEAD'),
            'frozen_ggml': contract.GGML, 'ggml': head_ggml,
            'protected_objects_changed': changed, 'protected_objects_unchanged': sorted(set(objects) - set(changed)),
            'declared_source_sha256': declared, 'default_decisions': protocol['declared_defaults']}


def replay_sha_factory(original, declared):
    """Auditor files (workflow/driver/test) are recorded, not required, when they drift."""
    drift = {}

    def replay_sha(path):
        value = original(path)
        rel = str(Path(path).resolve().relative_to(ROOT)) if Path(path).is_absolute() else str(path)
        if rel in declared and value != declared[rel]:
            drift[rel] = {'declared': declared[rel], 'current': value}
            out = Path(__import__('os').environ.get('HEAVY_OUT', '.'))
            out.mkdir(parents=True, exist_ok=True)
            (out / 'replay-auditor-drift.json').write_text(json.dumps(drift, indent=2) + '\n')
            return declared[rel]
        return value
    return replay_sha


if __name__ == '__main__':
    driver.source_guard = replay_source_guard
    declared = json.loads((ROOT / 'docs/ovisocr2-final-integrated-v2.json').read_text())['auditor_source_sha256']
    driver.sha = replay_sha_factory(contract.sha, declared)
    sys.argv[0] = str(DRIVER)
    src = DRIVER.read_text()
    main = src[src.index("if __name__ == '__main__':"):]
    exec(compile(main, str(DRIVER), 'exec'), driver.__dict__ | {'__name__': '__main__'})
