"""Model-free contract/negative controls; never import a framework or checkpoint."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bf16_quant_first',ROOT/'tools/lighton4b_bf16_quant_first_score.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Tools set a shared base.PROTOCOL at import; capture this tool's own path
# so later-collected sibling tools in one pytest session cannot redirect it.
OWN_PROTOCOL=m.PROTOCOL
class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header=json.loads((ROOT/'tests/fixtures/lightonocr3-4b-source/model.safetensors.header.json').read_text())
        cls.plan=m.tensor_plan(cls.header)
    def inventory(self,recipe):
        return {'tensors':{name:{'stored_shape':row['stored_shape'],'type':8 if row['quantized']and(recipe=='q8_0'or name=='l.embed_tokens.weight')else 12 if row['quantized']else 0}for name,row in self.plan.items()}}
    def test_source_census_and_patch_geometry(self):
        self.assertEqual((len(self.plan),sum(r['quantized']for r in self.plan.values())),(723,249))
        self.assertEqual(self.plan['v.patch_embed.proj.weight']['source_shape'],[1024,3,2,16,16])
        self.assertEqual(self.plan['v.patch_embed.proj.weight']['stored_shape'],[1536,1024])
        self.assertTrue(all(r['source_shape'][1]%256==0 for r in self.plan.values()if r['quantized']))
    def test_both_declared_recipes(self):
        for recipe in ('q8_0','q4_k'):m.validate_inventory(self.inventory(recipe),self.plan,recipe)
    def test_missing_and_extra_tensor_reject(self):
        data=self.inventory('q8_0');del data['tensors']['l.norm.weight']
        with self.assertRaises(ValueError):m.validate_inventory(data,self.plan,'q8_0')
        data=self.inventory('q8_0');data['tensors']['lm_head.weight']={'type':8,'stored_shape':[2560,248320]}
        with self.assertRaises(ValueError):m.validate_inventory(data,self.plan,'q8_0')
    def test_wrong_protection_embedding_shape_and_recipe_reject(self):
        for key,field,value in [('v.patch_embed.proj.weight','type',1),('l.embed_tokens.weight','type',12),('l.layers.0.linear_attn.in_proj_qkv.weight','stored_shape',[2560,4096])]:
            data=self.inventory('q4_k');data['tensors'][key][field]=value
            with self.assertRaises(ValueError):m.validate_inventory(data,self.plan,'q4_k')
        with self.assertRaises(ValueError):m.validate_inventory(self.inventory('q8_0'),self.plan,'mixed')
    def test_original_driver_binding_and_single_ast_substitution(self):
        protocol=json.loads(OWN_PROTOCOL.read_text())
        data=subprocess.check_output(['git','show',protocol['original_source']+':tools/lighton4b_bf16_freegen.py'],cwd=ROOT)
        self.assertEqual(hashlib.sha256(data).hexdigest(),protocol['original_driver_sha256'])
        transform=m.ReplaceGenerate();tree=transform.visit(ast.parse(data));ast.fix_missing_locations(tree)
        self.assertEqual(transform.count,1);compile(tree,'frozen-original-driver','exec')
        self.assertEqual(protocol['max_new_tokens'],1);self.assertEqual(protocol['expected_forward_arms'],3)
    def test_wrong_generate_owner_rejects(self):
        with self.assertRaises(ValueError):m.ReplaceGenerate().visit(ast.parse('result=other.generate()'))
if __name__=='__main__':unittest.main()
