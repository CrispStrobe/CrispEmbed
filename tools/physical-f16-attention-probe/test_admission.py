import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parent))
from worker import control_args,text_schema,ROOT
sys.path.insert(0,str(ROOT/'tools'))
from lighton4b_q6_recipe import plan


class IntegrationAdmission(unittest.TestCase):
    def test_all_declared_control_cli_arguments(self):
        build,schema=Path('/build'),Path('/frozen.tsv')
        for target in ('test-lighton4b-upstream-physical','test-lighton4b-hf-tiled-out-recipe'):
            self.assertEqual(control_args(target,build,schema),[build/target,schema])
        for target in ('test-lighton4b-physical-f16-attention','test-lighton4b-physical-f16-unsupported','test-lighton4b-physical-f16-wasm-guard','test-lighton4b-physical-native-ops','test-lighton4b-release-fence','test-lighton4b-hf-tiled-out-operator','test-lighton4b-q6-tied-head'):
            self.assertEqual(control_args(target,build,schema),[build/target])
        with self.assertRaises(RuntimeError):control_args('unknown',build,schema)

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
