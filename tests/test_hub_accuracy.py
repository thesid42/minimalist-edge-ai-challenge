import unittest
import numpy as np
from edgeprep.hub_accuracy import metrics

class LabelledAccuracyTests(unittest.TestCase):
    def test_labels_and_class_recall_are_used_not_reference_agreement(self):
        truth=np.arange(10);pred=truth.copy();pred[-1]=0
        result=metrics(pred,truth,[str(i) for i in range(10)])
        self.assertEqual(result['correct'],9)
        self.assertEqual(result['accuracy'],.9)
        self.assertEqual(result['worst_class_accuracy'],0)
        self.assertEqual(result['confusion_matrix'][9][0],1)
        self.assertEqual(result['class_support'],[1]*10)

if __name__=='__main__':unittest.main()
