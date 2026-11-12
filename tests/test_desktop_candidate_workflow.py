"""Model-free dispatch/publication controls; no build or service requests."""
from pathlib import Path
import re
import os
import subprocess
import unittest

WORKFLOW = (Path(__file__).resolve().parents[1] / '.github/workflows/release.yml').read_text()


def condition(expression, event, desktop, ref='refs/heads/candidate', cpu=False):
    # Evaluate only this workflow's small boolean condition vocabulary.
    values = {
        "startsWith(github.ref, 'refs/tags/')": str(ref.startswith('refs/tags/')),
        "github.event_name == 'workflow_dispatch'": str(event == 'workflow_dispatch'),
        "github.event_name != 'workflow_dispatch'": str(event != 'workflow_dispatch'),
        'inputs.desktop_only': str(desktop), 'inputs.cpu_only': str(cpu),
    }
    for key, value in values.items():
        expression = expression.replace(key, value)
    expression = expression.replace('&&', ' and ').replace('||', ' or ').replace('!', ' not ')
    if re.search(r'[^TrueFalsandort ()]', expression):
        raise AssertionError('Unexpected workflow condition: ' + expression)
    return eval(expression, {'__builtins__': {}}, {})


class DesktopCandidateTests(unittest.TestCase):
    def test_mobile_omission_requires_explicit_desktop_dispatch(self):
        self.assertRegex(WORKFLOW, r'desktop_only:\n        type: boolean\n        default: false')
        for job in ('build-android', 'build-ios'):
            expression = re.search(r'  ' + job + r':\n    if: ([^\n]+)', WORKFLOW)[1]
            for event in ('push', 'workflow_dispatch'):
                for desktop in (False, True):
                    for cpu in (False, True):
                        self.assertEqual(condition(expression, event, desktop, cpu=cpu),
                                         not (event == 'workflow_dispatch' and desktop))

    def test_desktop_dispatch_never_publishes_even_on_tag_ref(self):
        guards = re.findall(r'        if: ([^\n]+)\n        uses: softprops/action-gh-release@v3', WORKFLOW)
        self.assertEqual(len(guards), 7)
        for expression in guards:
            for event in ('push', 'workflow_dispatch'):
                for desktop in (False, True):
                    for ref in ('refs/heads/candidate', 'refs/tags/v0.17.20'):
                        self.assertEqual(condition(expression, event, desktop, ref),
                                         ref.startswith('refs/tags/') and not (event == 'workflow_dispatch' and desktop))
        self.assertIn("always() && !cancelled() && github.event_name == 'push'", WORKFLOW)
        self.assertIn('if: ${{ false }}', WORKFLOW)  # pub.dev remains disabled.

    def test_gpu_option_is_independent_of_desktop_scope(self):
        for job in ('build-linux-x86_64-cuda', 'build-windows-x86_64-cuda', 'build-windows-x86_64-vulkan'):
            expression = re.search(r'  ' + job + r':\n    if: ([^\n]+)', WORKFLOW)[1]
            for desktop in (False, True):
                self.assertTrue(condition(expression, 'push', desktop, cpu=True))
                self.assertTrue(condition(expression, 'workflow_dispatch', desktop, cpu=False))
                self.assertFalse(condition(expression, 'workflow_dispatch', desktop, cpu=True))

    def test_candidate_manifest_retains_scoped_inventory_and_artifacts(self):
        self.assertIn('DESKTOP_ONLY: ${{ inputs.desktop_only }}', WORKFLOW)
        self.assertIn('source_root=runtime-source', WORKFLOW)
        self.assertIn('--kind native --scope "$scope"', WORKFLOW)
        self.assertIn('--source-root "$source_root" --workflow-source-root .', WORKFLOW)
        self.assertIn('name: successor-native-manifest', WORKFLOW)
        self.assertIn('retention-days: 90', WORKFLOW)
        self.assertEqual(re.findall(r'^          - name: ([^\n]+)', WORKFLOW, re.M),
                         ['linux-x86_64', 'linux-arm64', 'macos-arm64', 'windows-x86_64'])

    def test_only_desktop_producers_select_explicit_runtime_source(self):
        selected = 'ref: ${{ github.event_name == \'workflow_dispatch\' && inputs.desktop_only && inputs.runtime_source_commit || github.sha }}'
        self.assertEqual(WORKFLOW.count(selected), 4)
        self.assertEqual(WORKFLOW.count("assert source == os.environ['RUNTIME_SOURCE_COMMIT']"), 4)
        self.assertEqual(WORKFLOW.count("'workflow_source_commit': os.environ['GITHUB_SHA']"), 4)
        for job in ('build-android', 'build-ios'):
            block = WORKFLOW.split('\n  ' + job + ':', 1)[1].split('\n  ', 1)[0]
            self.assertNotIn('runtime_source_commit', block)

    def test_runtime_source_validation_rejects_moving_or_injected_refs(self):
        block = WORKFLOW.split('      - name: Validate immutable desktop runtime source', 1)[1]
        script = block.split('        run: |\n', 1)[1].split('      - uses:', 1)[0]
        script = '\n'.join(line[10:] for line in script.splitlines())
        for value in ('2d2ab3275022e095f4a8f580e276bb2e3aa3f59c', '', 'main', '2d2ab327', 'a'*40+'\nsource=other'):
            result = subprocess.run(['bash', '-c', script], env={**os.environ, 'RUNTIME_SOURCE_COMMIT': value}, capture_output=True)
            self.assertEqual(result.returncode == 0, bool(re.fullmatch('[0-9a-f]{40}', value)))


if __name__ == '__main__':
    unittest.main()
