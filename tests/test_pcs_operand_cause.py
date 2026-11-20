#!/usr/bin/env python3
"""New algebra/axis controls, not a model or a replay of prior PCS suites."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from pcs_operand_cause import split_qk, head_operands, embedding_split


class CapturedOperandCause(unittest.TestCase):
    def test_separates_inputs_cross_term_and_actual_local_intervention(self):
        qr=np.array([[1.,2.]]);kr=np.array([[3.,4.]])
        qn=np.array([[2.,3.]]);kn=np.array([[5.,3.]])
        result=split_qk(qr,kr,qn,kn,np.array([[[11.125]]]),np.array([[[19.25]]]),np.array([[[11.375]]]))
        expected={'actualDifference':8.125,'referenceInputLocalError':.25,
                  'queryPerturbationEffect':7.,'keyPerturbationEffect':0.,'crossEffect':1.,
                  'totalOperandEffect':8.,'inputDependentArithmeticRemainder':-.125,'algebraClosureError':0.}
        for key,value in expected.items():self.assertEqual(float(result[key][0,0,0]),value,key)

    def test_canonical_heads_do_not_mix_token_and_channel_axes(self):
        q=np.array([[1.,2.,10.,20.],[3.,4.,30.,40.]])
        qr,kr=head_operands(q,q,(2,2,2))
        np.testing.assert_array_equal(qr,np.array([[[1,2],[3,4]],[[10,20],[30,40]]]))

    def test_wrong_token_shape_refuses(self):
        with self.assertRaisesRegex(ValueError,'token axes'):
            head_operands(np.ones((3,4)),np.ones((2,4)),(2,2,2))

    def test_wrong_head_width_refuses(self):
        with self.assertRaisesRegex(ValueError,'head axes'):
            head_operands(np.ones((2,5)),np.ones((2,5)),(2,2,2))

    def test_nonfinite_operand_refuses(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            head_operands(np.array([[np.nan,1.]]),np.ones((1,2)),(1,1,1))

    def test_embedding_table_interventions_are_only_pre_normalization(self):
        a={'emb.token':np.array([[1.,2.]],dtype=np.float32),
           'emb.position':np.array([[3.,4.]],dtype=np.float32),'emb.type':np.zeros((1,2),np.float32)}
        b={'emb.token':np.array([[2.,2.]],dtype=np.float32),
           'emb.position':np.array([[5.,4.]],dtype=np.float32),'emb.type':np.zeros((1,2),np.float32)}
        a['emb.sum']=(a['emb.token']+a['emb.type'])+a['emb.position'];b['emb.sum']=(b['emb.token']+b['emb.type'])+b['emb.position']
        result=embedding_split(a,b)
        self.assertEqual(result['observedSumDifference'],3.)
        self.assertEqual(result['tokenTableContribution'],1.)
        self.assertEqual(result['positionTableContribution'],2.)
        self.assertEqual(result['additionRoundingRemainder'],0.)
        self.assertEqual(result['tokenOnlyCounterfactualSum'],5.)
        self.assertEqual(result['positionOnlyCounterfactualSum'],6.)


if __name__=='__main__':unittest.main()
