"""Tiny stdlib-only controls. No framework, model, tokenizer or compiler."""
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/moedern-first-row'))
import common
import prepare
original=common.module(ROOT/'tools/moedern-first-row/original.py','first_row_original_controls')

class Controls(unittest.TestCase):
    def test_exact_scope_closure_and_limits(self):
        c=common.declaration()
        self.assertEqual(c['limits']['native_encode_attempts'],21)
        self.assertEqual(c['limits']['original_targeted_forward_max'],5)
        for group in ('protected_sha256','owned_sha256'):
            for name,pin in c[group].items():self.assertEqual(common.sha(ROOT/name),pin,name)
        self.assertEqual(c['retained']['small']['sha256'],'9d9303913281f294b799bd90cdcf2ee4e47b46b7c314903db3d99ca20d7723ee')
        self.assertEqual(len(c['retained']['small_members']),9)
        self.assertEqual(c['retained']['small_members']['compare.json']['sha256'],'e45d2ed45ce601a7ae5f959127c753e1d553625e3a7d19194ea672fec02f8cdb')

    def test_zip_all_members_and_bad_identity(self):
        def archive(name='a',data=b'abc',duplicate=False):
            b=io.BytesIO()
            with zipfile.ZipFile(b,'w')as z:
                z.writestr(name,data)
                if duplicate:z.writestr(name,data)
            b.seek(0);return zipfile.ZipFile(b)
        pin={'a':dict(bytes=3,sha256=hashlib.sha256(b'abc').hexdigest())}
        self.assertEqual(prepare.inventory(archive(),pin),1)
        for z,pins in [(archive(data=b'abd'),pin),(archive(),{}),(archive('../a'),{'../a':pin['a']}),(archive(duplicate=True),pin)]:
            with self.assertRaises(ValueError):prepare.inventory(z,pins)

    def test_symlink_rejected(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w')as z:
            i=zipfile.ZipInfo('a');i.create_system=3;i.external_attr=0o120777<<16;z.writestr(i,b'abc')
        b.seek(0)
        with self.assertRaises(ValueError):prepare.inventory(zipfile.ZipFile(b),{'a':dict(bytes=3,sha256=hashlib.sha256(b'abc').hexdigest())})

    def test_storage_redirect_has_no_authorization(self):
        import urllib.error
        class Response(io.BytesIO):pass
        class Opener:
            def open(self,request,timeout):
                self.request=request
                raise urllib.error.HTTPError(request.full_url,302,'redirect',{'Location':'https://storage.example/immutable'},None)
        seen=[]
        def storage(url,timeout):seen.append(url);self.assertIsInstance(url,str);return Response(b'abc')
        with tempfile.TemporaryDirectory()as d,patch.dict(os.environ,{'GH_TOKEN':'authored-test-token'}),patch.object(prepare.urllib.request,'build_opener',return_value=Opener()),patch.object(prepare.urllib.request,'urlopen',side_effect=storage):
            prepare.download(dict(id=123,bytes=3,sha256=hashlib.sha256(b'abc').hexdigest()),Path(d)/'artifact.zip')
        self.assertEqual(seen,['https://storage.example/immutable'])

    def test_bad_credential_does_not_fetch(self):
        for value in ('','a b','a\nb'):
            with patch.dict(os.environ,{'GH_TOKEN':value,'CRISPEMBED_ARTIFACT_TOKEN_FILE':''}):
                with self.assertRaises(ValueError):prepare.credential()

    def test_float_negative_zero_and_dictionary_order(self):
        fake=types.SimpleNamespace(Tensor=type('Tensor',(),{}))
        self.assertNotEqual(original.fingerprint(0.,fake),original.fingerprint(-0.,fake))
        self.assertNotEqual(original.fingerprint({'a':1,'b':2},fake),original.fingerprint({'b':2,'a':1},fake))
        for value in (float('nan'),float('inf'),object(),{3:'not-string'}):
            with self.assertRaises(ValueError):original.fingerprint(value,fake)

    def test_tensor_raw_bits_not_numeric_equality(self):
        class Tensor:
            dtype='torch.float32';device='cpu';shape=(1,)
            def __init__(self,raw):self.raw=raw
            def detach(self):return self
            def clone(self):return self
            def contiguous(self):return self
            def cpu(self):return self
            def numpy(self):return self
            def tobytes(self):return self.raw
            def stride(self):return (1,)
        fake=types.SimpleNamespace(Tensor=Tensor)
        dispatcher=types.ModuleType('torch.utils._python_dispatch');dispatcher._disable_current_modes=nullcontext
        with patch.dict(sys.modules,{'torch.utils._python_dispatch':dispatcher}):
            a=original.fingerprint(Tensor(struct.pack('<f',0.)),fake);b=original.fingerprint(Tensor(struct.pack('<f',-0.)),fake)
        self.assertNotEqual(a['sha256'],b['sha256']);self.assertEqual(a['stride'],[1])

    def test_sdpa_dispatch_metadata_scope_order_and_passthrough(self):
        tree=ast.parse((ROOT/'tools/moedern-first-row/original.py').read_text())
        node=next(n for n in ast.walk(tree)if isinstance(n,ast.ClassDef)and n.name=='ActualOperations')
        context=[];records=[];ns=dict(TorchDispatchMode=object,require=common.require,sdpa_active=context,sdpa_dispatches=records,stack=[])
        exec(compile(ast.Module(body=[node],type_ignores=[]),'owned-actual-operation-mode','exec'),ns)
        class Operator:
            def __init__(self,name):self.name=name;self._schema=name+'(Tensor input) -> Tensor';self.calls=[]
            def __str__(self):return self.name
            def __call__(self,*args,**kwargs):self.calls.append((args,kwargs));return args[0]
        mode=ns['ActualOperations']();payload=object()
        a=Operator('aten.transpose.int');b=Operator('aten._scaled_dot_product_flash_attention_for_cpu.default')
        # Outside the wrapper no unrelated operations enter the metadata.
        self.assertIs(mode.__torch_dispatch__(a,(),(payload,),{}),payload);self.assertEqual(records,[])
        context.append('actual.sdpa.0')
        for op in (a,b):self.assertIs(mode.__torch_dispatch__(op,(),(payload,),{}),payload)
        context.pop()
        self.assertEqual([r['operator']for r in records],[str(a),str(b)])
        self.assertEqual([r['sequence']for r in records],[0,1]);self.assertEqual([r['sdpa']for r in records],['actual.sdpa.0']*2)
        self.assertEqual([r['schema']for r in records],[str(a._schema),str(b._schema)])
        self.assertEqual(len(a.calls),2);self.assertEqual(len(b.calls),1)
        self.assertIs(b.calls[0][0][0],payload)
        source=(ROOT/'tools/moedern-first-row/original.py').read_text()
        self.assertIn('dispatches==repeated_dispatches',source)
        self.assertIn("sdpa_dispatch_metadata_repeat_exact=True",source)
        self.assertIn('len(operators)==8',source)

    def test_pinned_out_of_place_publisher_router_proof(self):
        c=common.declaration();p=c['publisher_router_source']
        self.assertEqual(p['sha256'],'528afffbd33210845ed521b12af5569f9b6d0b0eb3f176beebb3efd83cee6668')
        self.assertEqual((p['first_line'],p['last_line']),(235,248))
        self.assertEqual(p['snippet_sha256'],'c0860b582652a7d084acc5dea84a94173efaa433b71fab9b19666acd48e1c0d8')
        assignment=ast.parse(p['normalization_source']).body[0]
        self.assertIsInstance(assignment,ast.Assign);self.assertIsInstance(assignment.value,ast.BinOp);self.assertIsInstance(assignment.value.op,ast.Div)
        self.assertTrue(p['out_of_place']);self.assertEqual(p['augmented_assignments'],0)
        self.assertEqual(p['exact_softmax_calls'],1);self.assertEqual(p['exact_topk_calls'],1);self.assertEqual(p['exact_division_expressions'],1)
        self.assertEqual(p['dense_work'],307*128*16);self.assertLess(p['dense_work'],p['dense_threshold']);self.assertLess(307,p['chunk_tokens'])

    def test_actual_map_excludes_derived_reference(self):
        path=ROOT/'tools/moedern-native-v2/replay.py'
        tree=ast.parse(path.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='stage_map')
        ns={};exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),ns)
        result=common.actual_mapping(types.SimpleNamespace(stage_map=ns['stage_map']))
        self.assertEqual(len(result),46);self.assertFalse(any('derived.'in v for v in result.values()))
        self.assertFalse(any(k.endswith('selection_margin')for k in result))
        self.assertEqual(result['layer.1.ffn.topk_indices'],'actual.layers.1.ffn.topk.indices')

    def test_closed_barriers_reject_missing_or_false(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);common.atomic(root/'barriers.json',dict(completed=False))
            with self.assertRaises((KeyError,ValueError)):original.barriers(root,{})

    def test_immutable_packet_even_if_prepare_is_rewritten(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);text='authored';case=dict(key=common.KEY,id=common.CID,text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                ids=[2,30001]+[1]*305,scoring_mask=[1]*278+[0]*29,anchors={})
            case['ids'][58]=29084
            for side in ('original','native'):
                case['anchors'][side]={}
                for kind in ('all','selected'):
                    name=side+'-'+kind+'.f32';(root/name).write_bytes(b'1234')
                    case['anchors'][side][kind]=dict(path=name,bytes=4,sha256=common.sha(root/name))
            retained=dict(keys=['d:24726600','d:24928817','d:5849439'],cases=[case])
            with zipfile.ZipFile(root/'raw.zip','w')as z:z.writestr('raw/retained/three-doc-input/input-packet.json',json.dumps(retained))
            (root/'small.zip').write_bytes(b'authored-small-zip')
            cfg={'retained':{name:dict(bytes=(root/(name+'.zip')).stat().st_size,sha256=common.sha(root/(name+'.zip')))for name in ('small','raw')}}
            packet=dict(schema='crispembed-moedern-first-row-input-v1',key=common.KEY,case=case)
            common.atomic(root/'input-packet.json',packet)
            def prepared():common.atomic(root/'prepare.json',dict(completed=True,source=common.git('rev-parse','HEAD').decode().strip(),input_packet_sha256=common.sha(root/'input-packet.json'),protocol_sha256=common.sha(common.PATH)))
            prepared();self.assertEqual(common.authenticated_input(root,cfg)['ids'][58],29084)
            packet['case']['text']='changed';packet['case']['text_sha256']=hashlib.sha256(b'changed').hexdigest()
            common.atomic(root/'input-packet.json',packet);prepared()
            with self.assertRaisesRegex(ValueError,'immutable retained packet'):common.authenticated_input(root,cfg)
            (root/'raw.zip').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'ZIP changed'):common.authenticated_input(root,cfg)

    def test_both_barriers_precede_every_observer(self):
        text=(ROOT/'tools/ci-heavy/moedern_first_row_v1.py').read_text()
        ordinary=text.index("command('original-ordinary'");plain=text.index("command('native-plain'")
        closed=text.index("atomic(raw/'barriers.json'");observed=text.index("command('original-observe'");native=text.index("command('native-observed'")
        self.assertTrue(ordinary<plain<closed<observed<native)
        o=(ROOT/'tools/moedern-first-row/original.py').read_text()
        self.assertLess(o.index('baseline=barriers('),o.index('    import torch'))
        self.assertIn('with _disable_current_modes()',o)
        self.assertIn("('aten._softmax.default','aten.topk.default','aten.div.Tensor')",o)
        self.assertNotIn('probabilities=value.detach().clone().softmax',o)
        self.assertNotIn('torch.equal',o.split('def fingerprint')[1].split('def invalid')[0].replace('never torch.equal','never equality'))

    def test_native_call_sequence_preserved_and_counts(self):
        old=(ROOT/'tests/moedern-three-doc/probe.cpp').read_text();new=(ROOT/'tests/moedern-first-row/probe.cpp').read_text()
        start=new.index('int run(int argc');run=new[start:]
        oldrun=old[old.index('int run(int argc'):]
        oldcalls=re.findall(r'(mt::encode|moedern_text_test::encode_observed)\(',oldrun)
        newcalls=re.findall(r'(counted_encode|counted_observed)\(',run)
        self.assertEqual(oldcalls,[{'counted_encode':'mt::encode','counted_observed':'moedern_text_test::encode_observed'}[v]for v in newcalls])
        self.assertIn('attempts == 12 && full == 8 && pre_math_abort == 1',new)
        self.assertIn('attempts == 9 && full == 6 && pre_math_abort == 0',new)
        self.assertIn('callbacks == 1 && first == "input_ids"',new)
        self.assertIn('cases == 1',new);self.assertIn('pairs == 0',new)

    def test_honest_provider_and_no_old_guard(self):
        text=(ROOT/'tools/moedern-first-row/kernel.py').read_text()
        self.assertIn("os.environ.get('GITHUB_ACTIONS')is None",text)
        self.assertNotIn("os.environ['GITHUB_ACTIONS']=",text)
        for path in (ROOT/'tools/moedern-first-row').glob('*.py'):
            self.assertNotIn('old.verify(',path.read_text())
            ast.parse(path.read_text())
        self.assertNotIn('import torch',text)
        self.assertIn('external_exact_version_service_proof_required=True',text)

if __name__=='__main__':unittest.main()
