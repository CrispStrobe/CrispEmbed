"""Model-free controls for actual binding output accounting and numerical gates."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bindings',root/'tools/ci-heavy/ultrabert_bindings.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def records():
    vector=[1.]+[0.]*767
    expected=[dict(id='document-0',ids=[0,42],raw_cls=vector.copy(),normalized_cls=vector.copy())]
    actual=dict(rows=[dict(id='document-0',ids=[0,42],raw_cls=vector.copy(),dense=vector.copy(),token_normalized_cls=vector.copy())],
                contracts={k:True for k in ['repeat','batch','context_prefix','rejection_recovery']})
    return expected,actual


class BindingControls(unittest.TestCase):
    def test_versioned_loader_names_and_fallback_rejection(self):
        import tempfile
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);library=folder/'libcrispembed.so.0.17.13';library.touch()
            link=folder/'libcrispembed.so.0';link.symlink_to(library)
            for soname in ['libcrispembed.so','libcrispembed.so.0','libcrispembed.so.0.17.13']:
                p.checked_loaded_library(f'{soname} => {link} (0x123)',library)
            for log in ['libcrispembed.so.0 => not found', 'libother.so => /tmp/other (0x1)',
                        'libcrispembed.so.bad => /tmp/other (0x1)',
                        'libcrispembed.so.0 => /tmp/other (0x1)',
                        f'libcrispembed.so => {link} (0x1)\nlibcrispembed.so.0 => {link} (0x1)']:
                with self.assertRaises(ValueError):p.checked_loaded_library(log,library)

    def test_complete_outputs_and_magnitudes(self):
        a,b=records();metrics=p.compare(a,b)
        self.assertEqual(len(metrics),3)
        self.assertTrue(all(m['passed'] and m['max_abs']==0 and m['reference_norm']==1 for m in metrics))
    def test_wrong_ids_missing_rows_or_contracts_rejected(self):
        a,b=records()
        for change in [lambda x:x['rows'].clear(),lambda x:x['rows'][0].update(id='query-0'),
                       lambda x:x['rows'][0].update(ids=[0,42,15]),lambda x:x['contracts'].pop('repeat'),
                       lambda x:x['contracts'].update(rejection_recovery=False)]:
            bad=deepcopy(b);change(bad)
            with self.assertRaises(ValueError):p.compare(a,bad)
    def test_nonfinite_missing_or_zero_vectors_rejected(self):
        a,b=records()
        for vector in [[1.], [0.]*768, [float('nan')]+[0.]*767, [True]+[0.]*767]:
            bad=deepcopy(b);bad['rows'][0]['raw_cls']=vector
            with self.assertRaises(ValueError):p.compare(a,bad)
    def test_cosine_alone_does_not_waive_absolute_error(self):
        a,b=records();b['rows'][0]['raw_cls'][0]=1.01
        metric=p.compare(a,b)[0]
        self.assertEqual(metric['cosine'],1.)
        self.assertFalse(metric['passed'])


if __name__=='__main__':unittest.main()
