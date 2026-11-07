import sys
from pathlib import Path
import unittest
import re
sys.path.insert(0,str(Path(__file__).parent))
from worker import control_args,text_schema,ROOT,BUILD_FLAGS,validate_profile
sys.path.insert(0,str(ROOT/'tools'))
from lighton4b_q6_recipe import plan


class IntegrationAdmission(unittest.TestCase):
    def test_all_declared_control_cli_arguments(self):
        build,schema=Path('/build'),Path('/frozen.tsv')
        for target in ('test-lighton4b-upstream-physical','test-lighton4b-hf-tiled-out-recipe'):
            self.assertEqual(control_args(target,build,schema),[build/target,schema])
        for target in ('test-lighton4b-physical-f16-attention','test-lighton4b-physical-f16-unsupported','test-lighton4b-physical-f16-wasm-guard','test-lighton4b-physical-native-ops','test-lighton4b-hf-tiled-out-operator','test-lighton4b-q6-tied-head'):
            self.assertEqual(control_args(target,build,schema),[build/target])
        self.assertEqual(control_args('test-lighton4b-release-fence',build,schema),[build/'test-lighton4b-release-fence',schema.parent/'release-fence-fixture.gguf'])
        with self.assertRaises(RuntimeError):control_args('unknown',build,schema)

    def test_every_actual_cpp_cli_signature_matches_invocation(self):
        specs={
            'test-lighton4b-physical-f16-attention':'test_lighton4b_physical_f16_attention.cpp',
            'test-lighton4b-physical-f16-unsupported':'test_lighton4b_physical_f16_fallback.cpp',
            'test-lighton4b-physical-f16-wasm-guard':'test_lighton4b_physical_f16_fallback.cpp',
            'test-lighton4b-physical-native-ops':'test_lighton4b_physical_native_ops.cpp',
            'test-lighton4b-upstream-physical':'test_lighton4b_upstream_physical.cpp',
            'test-lighton4b-release-fence':'test_lighton4b_release_fence.cpp',
            'test-lighton4b-hf-tiled-out-recipe':'test_lighton4b_hf_tiled_out_recipe.cpp',
            'test-lighton4b-hf-tiled-out-operator':'test_lighton4b_hf_tiled_out_operator.cpp',
            'test-lighton4b-q6-tied-head':'test_lighton4b_q6_tied_head.cpp'}
        for target,filename in specs.items():
            code=(ROOT/'tests'/filename).read_text()
            matches=re.findall(r'if\s*\(\s*argc\s*!=\s*(\d+)\s*\)',code)
            args=control_args(target,Path('/build'),Path('/out/closed426.tsv'))
            with self.subTest(target=target):
                if matches:self.assertEqual(len(args),int(matches[0]))
                else:
                    self.assertNotRegex(code,r'\bargc\b|\bargv\b')
                    self.assertEqual(len(args),1)

    def test_actual_native_and_reference_thread_profiles_are_distinct(self):
        native=[f'{k}:BOOL={v}'for k,v in BUILD_FLAGS.items()]
        reference=[f'{k}:BOOL={"ON"if k=="GGML_OPENMP"else v}'for k,v in BUILD_FLAGS.items()if k!='CRISPEMBED_NATIVE']
        validate_profile(native)
        validate_profile(reference,reference=True)
        with self.assertRaises(RuntimeError):validate_profile(native,reference=True)
        with self.assertRaises(RuntimeError):validate_profile(reference)

    def test_exact_closed426_schema(self):
        p=plan();lines=text_schema(p).splitlines()
        self.assertEqual(len(lines),426)
        self.assertEqual(sum(line.split()[1]=='14'for line in lines),33)
        self.assertTrue(next(line for line in lines if line.startswith('l.embed_tokens.weight\t')).startswith('l.embed_tokens.weight\t14\t2560\t248320\t'))
        p['all_tensors'].pop()  # Vision rows are intentionally outside text policy.
        self.assertEqual(len(text_schema(p).splitlines()),426)
        p['all_tensors'].pop(next(i for i,row in enumerate(p['all_tensors'])if row['name'].startswith('l.')))
        with self.assertRaises(RuntimeError):text_schema(p)


if __name__=='__main__':unittest.main()
