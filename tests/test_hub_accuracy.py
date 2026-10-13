import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from edgeprep.hub_accuracy import metrics, collect

class LabelledAccuracyTests(unittest.TestCase):
    def test_labels_and_class_recall_are_used_not_reference_agreement(self):
        truth=np.arange(10);pred=truth.copy();pred[-1]=0
        result=metrics(pred,truth,[str(i) for i in range(10)])
        self.assertEqual(result['correct'],9)
        self.assertEqual(result['accuracy'],.9)
        self.assertEqual(result['worst_class_accuracy'],0)
        self.assertEqual(result['confusion_matrix'][9][0],1)
        self.assertEqual(result['class_support'],[1]*10)

    def test_collect_uses_recorded_subset_and_rejects_wrong_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sample=root/'separate_sample';folder=root/'run'
            sample.mkdir();folder.mkdir()
            manifest={'source_sha256':'saved-model','indices_sha256':'sample-order',
                'classes':[str(i) for i in range(10)],'shards':[{'start':0,'count':10}],
                'scope':'training_set_subset','caveat':'diagnostic','calibration_disjoint':True}
            (sample/'manifest.json').write_text(json.dumps(manifest))
            logits=np.eye(10,dtype=np.float32)
            np.save(sample/'reference_logits.npy',logits);np.save(sample/'labels.npy',np.arange(10))
            (folder/'labelled_jobs.json').write_text(json.dumps({'manifest':'separate_sample/manifest.json',
                'indices_sha256':'sample-order','jobs':[{'start':0,'count':10,'id':'cached'}]}))
            (folder/'request.json').write_text(json.dumps({'source_sha256':'saved-model','device':'test-phone'}))
            np.save(folder/'labelled_0000.npy',logits)
            with patch('edgeprep.hub_accuracy.ROOT',root),patch('edgeprep.hub_auth.get_client'):
                self.assertTrue(collect(folder))
                result=json.loads((folder/'dataset_accuracy.json').read_text())
                self.assertEqual(result['correct'],10)
                self.assertEqual(result['source_sha256'],'saved-model')
                self.assertEqual(result['manifest'],'separate_sample/manifest.json')
                (folder/'request.json').write_text(json.dumps({'source_sha256':'other-model','device':'test-phone'}))
                with self.assertRaisesRegex(ValueError,'Reference source mismatch'):collect(folder)

if __name__=='__main__' :unittest.main()
