"""Weight-free retained-source and archive contract controls."""
import copy
import io
from pathlib import Path
import sys
import unittest
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ci-heavy'))
import ovisocr2_2048_attribution as audit


class Controls(unittest.TestCase):
    def test_frozen_producer_lineage(self):
        p,source=audit.preflight()
        self.assertEqual(p['source_commit'],audit.HEAD)
        self.assertEqual(source['source_run'],40177774574)
        self.assertEqual(p['case'],'large-boundary')

    def test_both_retained_manifest_schemas(self):
        p,_=audit.preflight()
        for arm in ('f32-vision','f32-decoder-prepared','f32-decoder-native-vision'):
            rows=audit.indexed_rows(p,arm)
            self.assertTrue(rows)
            for stage,row in rows.items():self.assertEqual(row['member'],'native-diff/'+arm+'/'+row['stage']+'.f32')

    def test_bad_row_metadata_rejected(self):
        p={'arms':{'x':{'raw':{'coverage_complete':True,'files':[
           {'stage':'prefill.decoder.norm','shape':[1,2,1024],'bytes':8192,'sha256':'0'*64}]}}}}
        self.assertEqual(len(audit.indexed_rows(p,'x')),1)
        for field,value in [('bytes',4),('shape',[1,0,1024]),('file','../escape')]:
            bad=copy.deepcopy(p);bad['arms']['x']['raw']['files'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):audit.indexed_rows(bad,'x')
        bad=copy.deepcopy(p);bad['arms']['x']['raw']['files']*=2
        with self.assertRaises(ValueError):audit.indexed_rows(bad,'x')
        bad=copy.deepcopy(p);bad['arms']['x']['raw']['coverage_complete']=False
        with self.assertRaises(ValueError):audit.indexed_rows(bad,'x')

    def test_position_branch_precedes_embedding_sum(self):
        p,_=audit.preflight()
        stages=audit.vision_stage_order(p['original_capture']['stages'])
        self.assertLess(stages.index('prefill.vision.position'),stages.index('prefill.vision.embedded'))
        self.assertLess(stages.index('prefill.vision.patch_embed'),stages.index('prefill.vision.embedded'))
        self.assertEqual(len(stages),len(set(stages)))

    def test_archive_paths_checked_before_extraction(self):
        for name in ('../escape','/absolute','bad\\path'):
            buf=io.BytesIO()
            with zipfile.ZipFile(buf,'w') as z:z.writestr(name,b'x')
            buf.seek(0)
            with zipfile.ZipFile(buf) as z,self.assertRaises(ValueError):audit.archive_members(z)


if __name__=='__main__':unittest.main()
