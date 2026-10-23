"""Fail before models/builds when source qualification or artifact integrity fails."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import os
from unittest.mock import patch
import tempfile
import unittest
import zipfile

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('producer',root/'tools/ci-heavy/ultrabert_retrieval.py')
producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
SOURCE='0e615ea470b8af25925393abc566bbd42606d0df'


def receipt():
    stage=dict(cos_min=1.,max_abs=.0001,mean_abs=.00001,rms=.00002,native_norm=80.,reference_norm=80.,**{'pass':True})
    names=['empty','document','query','unicode','window-boundary','padded-batch']
    return dict(source_commit=SOURCE,status='completed bounded public F32 pooled-output checks; no independent task-quality qualification',
                f32_model_sha256='89478b9cbbad2fd1121be4031d760f1529c989fedf23b6b19251ba2c87422566',
                context_limit=512,original_source='0573f1847db68914052a98341a22fceeb1358518',
                frozen_report_sha256='a3c16863b1a78ebaebcc2d50e24555b696e691d3309ee17b75075c0051032a36',
                library_sha256='c'*64,driver_sha256='d'*64,
                cases=[dict(case=n,ids=[[0]]*(2 if n=='padded-batch' else 1),failures=0,
                            stages={'cls_raw':stage.copy(),'cls_normalized':stage.copy()}) for n in names],
                surfaces={n:True for n in ['c_api','python','cli','http']},
                native_source_sha256={n:'a'*64 for n in ['src/ultrabert_embed.cpp','src/ultrabert_profile.h','src/crispembed.cpp','tests/test_ultrabert_api.cpp']})


class QualificationControls(unittest.TestCase):
    def test_success_and_pending_wrong_source_workflow_rejected(self):
        good=dict(status='completed',conclusion='success',head_sha=SOURCE,path='.github/workflows/ultrabert-api.yml')
        producer.checked_run(good,SOURCE)
        for k,v in [('status','queued'),('conclusion','failure'),('head_sha','a'*40),('path','other.yml')]:
            bad={**good,k:v}
            with self.subTest(key=k),self.assertRaises(ValueError):producer.checked_run(bad,SOURCE)
    def test_complete_receipt_and_partial_or_unqualified_rejected(self):
        producer.checked_receipt(receipt(),SOURCE)
        changes=[lambda d:d.update(source_commit='b'*40),lambda d:d.update(status='started'),
                 lambda d:d['cases'].pop(),lambda d:d['cases'][0].update(failures=1),
                 lambda d:d['cases'][0]['stages']['cls_raw'].update(max_abs=.01),
                 lambda d:d['cases'][0]['stages']['cls_raw'].update(cos_min=float('nan')),
                 lambda d:d['surfaces'].update(python=False),lambda d:d['native_source_sha256'].pop('src/crispembed.cpp'),
                 lambda d:d.pop('library_sha256'),lambda d:d.update(original_source='bad')]
        for change in changes:
            d=receipt();change(d)
            with self.subTest(change=change),self.assertRaises(ValueError):producer.checked_receipt(d,SOURCE)
    def test_preflight_never_calls_original_or_native_models(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            with patch.dict(os.environ,HEAVY_OUT=str(folder/'out'),HEAVY_SCRATCH=str(folder/'scratch'),GITHUB_SHA=SOURCE), \
                 patch.object(sys,'argv',['producer','--api-run','123','--native-source',SOURCE,'--preflight']), \
                 patch.object(producer,'qualification',return_value=(receipt(),{'source':SOURCE})), \
                 patch.object(producer,'original_records') as original,patch.object(producer,'native_records') as native:
                self.assertEqual(producer.main(),0)
                original.assert_not_called();native.assert_not_called()
            self.assertEqual(json.loads((folder/'out/retrieval-producer.json').read_text())['status'],
                             'preflight passed; no model/build executed')
    def test_reused_original_input_and_source_guards(self):
        data=producer.fixture(root/'tests/regression/ultrabert/retrieval-smoke-v1.json')
        rows=[]
        for item in data['queries']+data['documents']:
            text=(producer.QUERY if item['id'].startswith('q') else '')+item['text']
            rows.append(dict(id=item['id'],raw_cls=[1.]+[0.]*767,normalized_cls=[1.]+[0.]*767,
                             ids=[0,42],attention_mask=[1,1],input_utf8_sha256=producer.hashlib.sha256(text.encode()).hexdigest()))
        sources=producer.profile()['inspected_file_sha256']
        p=dict(model=producer.MODEL,revision=producer.REVISION,source_files=sources,hook_neutral=True,
               loaded_custom_source_sha256=sources['modeling_ultrabert.py'],
               weight_sha256='da8be29564b69e4f90fc8f2683e1f369c3d3575e75f7fe798643e28de02e2b9c',
               wrapper_source=producer.ST_SOURCE,wrapper_sha256=producer.ST_WRAPPER_SHA,pooling_sha256=producer.ST_POOL_SHA,
               wrapper='6.0.1',transformers='5.16.1',torch='2.14.0+cpu')
        record=dict(pooling='cls',query_prompt=producer.QUERY,document_prompt='',fixture_sha256=producer.FIXTURE_SHA256,
                    vectors=rows,provenance=p)
        producer.checked_original(data,record)
        for change in [lambda x:x['provenance'].update(hook_neutral=False),
                       lambda x:x['provenance'].update(pooling_sha256='bad'),
                       lambda x:x['vectors'][0].update(input_utf8_sha256='bad'),
                       lambda x:x['vectors'][0].update(attention_mask=[0,1])]:
            bad=deepcopy(record);change(bad)
            with self.assertRaises(ValueError):producer.checked_original(data,bad)

    def test_native_binding_and_library_cannot_fall_back(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);library=root/'libcrispembed.so';binding=root/'python/crispembed/_binding.py'
            producer.checked_binding_paths(library,binding,library,root)
            for loaded,source in [(root/'other.so',binding),(library,root/'other.py')]:
                with self.assertRaises(ValueError):producer.checked_binding_paths(loaded,source,library,root)

    def test_only_exact_small_receipt_is_extracted(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);archive=folder/'a.zip';output=folder/'receipt.json'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('ultrabert-api.json',json.dumps(receipt()));z.writestr('build.log','not extracted')
            self.assertEqual(producer.extract_receipt(archive,output),receipt())
            self.assertEqual(sorted(p.name for p in folder.iterdir()),['a.zip','receipt.json'])
    def test_traversal_duplicate_and_missing_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            for members in [['../ultrabert-api.json'],['ultrabert-api.json','nested/ultrabert-api.json'],['wrong.json']]:
                archive=folder/'a.zip'
                with zipfile.ZipFile(archive,'w') as z:
                    for member in members:z.writestr(member,'{}')
                with self.subTest(members=members),self.assertRaises(ValueError):producer.extract_receipt(archive,folder/'out.json')


if __name__=='__main__':unittest.main()
