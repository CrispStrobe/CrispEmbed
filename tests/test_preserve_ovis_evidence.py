"""Small corruption tests for byte-preserving remote archive handling."""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ci-heavy'))
import preserve_ovis_evidence as preserve


class Controls(unittest.TestCase):
    def test_every_payload_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'raw.zip'
            with zipfile.ZipFile(path,'w') as z:z.writestr('arm/state.f32',b'abcd')
            expected={'arm/state.f32':{'bytes':4,'sha256':hashlib.sha256(b'abcd').hexdigest()}}
            self.assertEqual(preserve.inventory(path,expected),expected)
            with self.assertRaisesRegex(ValueError,'SHA differs'):
                preserve.inventory(path,{'arm/state.f32':{'bytes':4,'sha256':'0'*64}})
            with self.assertRaisesRegex(ValueError,'size differs'):
                preserve.inventory(path,{'arm/state.f32':{'bytes':8,'sha256':expected['arm/state.f32']['sha256']}})
            with self.assertRaisesRegex(ValueError,'coverage differs'):preserve.inventory(path,{})

    def test_unsafe_archive_refused(self):
        for name in ('../escape','/absolute','directory/','windows\\path'):
            buf=io.BytesIO()
            with zipfile.ZipFile(buf,'w') as z:z.writestr(name,b'x')
            buf.seek(0)
            with zipfile.ZipFile(buf) as z,self.assertRaises(ValueError):preserve.safe_members(z)

    def test_frozen_original_identity(self):
        import json
        p=json.loads(preserve.RECEIPT.read_text())
        self.assertEqual(p['run'],preserve.RUN)
        self.assertEqual(p['source_commit'],preserve.HEAD)
        self.assertEqual(p['raw_diff_artifact']['zip_sha256'],preserve.RAW[3])
        self.assertEqual(p['artifact_zip_sha256'],preserve.SMALL[3])


if __name__=='__main__':unittest.main()
