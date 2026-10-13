import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from edgeprep.challenge import load_model, prune_model
from edgeprep.challenge_data import validate_split, balanced_ids
from edgeprep.export import export_fp32


class ChallengeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_split_rejects_overlap_duplicates_and_invalid_indices(self):
        for split in ({'train':[0,1],'val':[1,2]}, {'train':[0,0],'val':[2]},
                      {'train':[-1],'val':[2]}, {'train':[True],'val':[2]},
                      {'train':[],'val':[1]}):
            with self.assertRaises(ValueError):validate_split(split,3)
        valid={'train':[0,1],'val':[2]}
        self.assertEqual(validate_split(valid,3),valid)

    def test_balancing_keeps_partition_and_has_no_duplicates(self):
        labels=[i//10 for i in range(100)]
        ids=balanced_ids(list(range(100)),labels,30)
        self.assertEqual(len(set(ids)),30)
        self.assertEqual(np.bincount([labels[i] for i in ids]).tolist(),[3]*10)

    def test_real_student_pruning_replays_and_exports_fixed_input(self):
        checkpoint='Models/student_mobilenetv2_distilled.pt'
        original=load_model(checkpoint)
        model=prune_model(load_model(checkpoint),.10,scope="expansion")
        replay=prune_model(load_model(checkpoint),.10,scope="expansion")
        self.assertLess(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in original.parameters()))
        self.assertEqual(list(model.state_dict()),list(replay.state_dict()))
        for key,value in model.state_dict().items():torch.testing.assert_close(value,replay.state_dict()[key])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pruned.onnx'
            export_fp32(model,path)
            options=ort.SessionOptions();options.intra_op_num_threads=1
            session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
            self.assertEqual(session.get_inputs()[0].shape,[1,3,224,224])
            logits=session.run(None,{'images':np.zeros((1,3,224,224),np.float32)})[0]
            self.assertEqual(logits.shape,(1,10));self.assertTrue(np.isfinite(logits).all())


if __name__=='__main__':unittest.main(verbosity=2)
