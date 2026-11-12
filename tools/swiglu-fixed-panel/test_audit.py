"""Small independent raw-audit corruption and privacy controls; no model data."""
import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
import audit
class AuditControls(unittest.TestCase):
 def build(self,root,values,duplicate=False):
  p=root/'greedy-raw-rows.zip';b=values.astype('<f4').tobytes();row=dict(position=0,bytes=len(b),argmax=7,sha256=hashlib.sha256(b).hexdigest())
  with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_STORED)as z:
   z.writestr('greedy.0.logits.f32',b)
   if duplicate:z.writestr('extra.logits.f32',b)
  record=dict(rows=[row],archive=dict(file=p.name,bytes=p.stat().st_size,sha256=audit.sha(p)));return record,{p.resolve():audit.sha(p)}
 def test_actual_finite_argmax_and_corrupt_rows(self):
  for mode in ['valid','wrong-argmax','nonfinite','extra-member']:
   with self.subTest(mode=mode),tempfile.TemporaryDirectory()as folder:
    root=Path(folder);x=np.zeros(audit.VOCAB,dtype='<f4');x[7]=1
    if mode=='wrong-argmax':x[8]=2
    if mode=='nonfinite':x[8]=np.nan
    r,known=self.build(root,x,mode=='extra-member')
    if mode=='valid':self.assertEqual(audit.raw_archive(root,r,[7],known)['rows'],1)
    else:
     with self.assertRaises(ValueError):audit.raw_archive(root,r,[7],known)
 def test_closed_paths_and_no_book_text(self):
  with tempfile.TemporaryDirectory()as folder:
   with self.assertRaises(ValueError):audit.owned(Path(folder),'../other')
  got=audit.safe_score(dict(raw_text='PROTECTED BOOK SENTENCE',parsed_text='PROTECTED BOOK SENTENCE',parsed_tables=[['PROTECTED']],parse_error='PROTECTED',complete=False,raw={'cer':2.,'wer':3.},content={'cer':1.,'wer':1.}))
  self.assertNotIn('PROTECTED',json.dumps(got));self.assertFalse(got['complete']);self.assertEqual(got['raw']['cer'],2.)
 def test_independent_edit_distance(self):
  self.assertEqual(audit.edit_count('kitten','sitting'),3);self.assertEqual(audit.edit_count('',[1,2]),2);self.assertEqual(audit.edit_count(['one','two'],['one','three','two']),1)
  score=dict(raw={'cer':1/3,'wer':1.},parsed_text='abd',content={'cer':1/3,'wer':1.});v=audit.independent_edits({'text':'abc'},'abd',score);self.assertEqual(v['raw']['character_edits'],1)
if __name__=='__main__':unittest.main()
