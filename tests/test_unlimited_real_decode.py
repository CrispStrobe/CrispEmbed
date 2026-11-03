"""Small immutable receipt/host guards for the retained-ID decoder proof."""
import importlib.util,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('real_decode',ROOT/'tools/unlimited-ocr2/replay_real_decode.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Controls(unittest.TestCase):
    def test_local_replay_rejected_before_model_or_transfer(self):
        with patch.dict(m.os.environ,{},clear=True),patch.object(m.Path,'is_dir',return_value=False),patch.object(m.urllib.request,'urlopen')as request:
            with self.assertRaisesRegex(ValueError,'hosted'):m.replay('unused','unused')
            request.assert_not_called()
    def test_exact_retained_inventory_and_failed_input_visible(self):
        original,native,paired=m.inputs();self.assertEqual(len(original['runs']),20);self.assertEqual(len(native['runs']),20)
        self.assertEqual(paired['eligible_pairs'],19);self.assertEqual(paired['exact_ids'],11)
        self.assertEqual([r['page']for r in paired['rows']if not r['eligible']],['en-form-economic-journal-67'])
    def test_changed_report_rejected_before_decoder_packages(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);out=root/'tests/results/ocr-real-scans';out.mkdir(parents=True)
            source=ROOT/'tests/results/ocr-real-scans';shutil.copyfile(source/'source-identities.json',out/'source-identities.json');(out/'unlimited-original.json').write_text('{}')
            with patch.object(m,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'producer differs'):m.inputs()
if __name__=='__main__':unittest.main()
