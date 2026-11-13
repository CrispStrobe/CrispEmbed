"""Separate second-image-chunk observer, derived from the proven first capture."""
import importlib.util
from pathlib import Path
import re

BASE_PATH = Path(__file__).resolve().parents[1] / 'book-image-layer0-probe/prepare.py'
TARGET = 'lighton4b-book-image-layer0-second-probe'


def configured():
    spec = importlib.util.spec_from_file_location('first_image_prepare', BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probe = module.PROBE
    for before, after in [('if(part.offset==4){', 'if(part.offset==20){'),
                          ('captured_image || replay_cache.tokens!=4 ||', 'captured_image || replay_cache.tokens!=20 ||')]:
        if probe.count(before) != 1:
            raise ValueError('proven observer anchor drift')
        probe = probe.replace(before, after)
    for name, old, new in [('offset', 4, 20), ('cache_before_tokens', 4, 20), ('cache_after_tokens', 20, 36)]:
        probe, count = re.subn(r'(' + name + r'[^:\n]*:)' + str(old) + ',', r'\g<1>' + str(new) + ',', probe, count=1)
        if count != 1:
            raise ValueError('proven capture metadata anchor drift')
    module.PROBE, module.TARGET = probe, TARGET
    return module


base = configured()
SOURCE, DRIVER_SHA, CMAKE_SHA = base.SOURCE, base.DRIVER_SHA, base.CMAKE_SHA
PROBE, STOP = base.PROBE, base.STOP


def generate(driver):
    return configured().generate(driver)


def prepare(root):
    return configured().prepare(root)
