"""Pure controls for reviewed OvisOCR2 precision gate inheritance."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'ovis_vision_env', ROOT / 'tools/ci-heavy/ovisocr2_vision_env.py')
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class VisionEnvironment(unittest.TestCase):
    def test_all_six_reviewed_presets_and_ambient_override(self):
        for name, enabled in subject.PRESETS.items():
            values = {flag: '1' if flag in enabled else '0' for flag in subject.FLAGS}
            producer = {'selected_vision_preset': {'name': name,
                'enabled_gates': list(enabled), 'all_other_native_gates_unchanged': True,
                'no_production_default_promoted': True}, 'native_environment': values}
            selected = subject.reviewed_selection(producer)
            ambient = {**{flag: '1' if values[flag] == '0' else '0'
                         for flag in subject.FLAGS}, 'CRISPEMBED_QWEN35_OCR': '1'}
            actual = subject.apply_selection(ambient, selected)
            self.assertEqual({flag: actual[flag] for flag in subject.FLAGS}, values)
            self.assertEqual(actual['CRISPEMBED_QWEN35_OCR'], '1')

    def test_missing_conflicting_and_unreviewed_modes_rejected(self):
        with self.assertRaises(ValueError):
            subject.reviewed_selection({})
        with self.assertRaises(ValueError):
            subject.apply_selection({}, None)
        producer = {'selected_vision_preset': {'name': 'baseline', 'enabled_gates': [],
                    'all_other_native_gates_unchanged': True,
                    'no_production_default_promoted': True},
                    'native_environment': {flag: '0' for flag in subject.FLAGS}}
        for flag in subject.FLAGS:
            changed = {'selected_vision_preset': dict(producer['selected_vision_preset']),
                       'native_environment': dict(producer['native_environment'])}
            changed['native_environment'][flag] = '1'
            with self.assertRaises(ValueError):
                subject.reviewed_selection(changed)
            changed['native_environment'][flag] = 'true'
            with self.assertRaises(ValueError):
                subject.reviewed_selection(changed)
        producer['selected_vision_preset']['no_production_default_promoted'] = False
        with self.assertRaises(ValueError):
            subject.reviewed_selection(producer)


if __name__ == '__main__':
    unittest.main()
