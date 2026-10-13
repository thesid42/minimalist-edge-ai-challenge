import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
import onnxruntime as ort
import torch
from edgeprep.challenge import restore_recipe
from edgeprep.challenge_data import digest
from edgeprep.export import export_fp32

class ResolutionTests(unittest.TestCase):
    def test_recipe_replays_fixed_input_resize_and_matches_export(self):
        torch.set_num_threads(2)
        source=Path('Models/student_mobilenetv2_distilled.pt').resolve()
        with tempfile.TemporaryDirectory() as tmp:
            recipe=Path(tmp)/'recipe.json'
            recipe.write_text(json.dumps({'source_checkpoint':str(source),'source_sha256':digest(source),
               'pruning_ratio':0,'round_to':8,'pruning_scope':'expansion','internal_resolution':160}))
            model,_=restore_recipe(recipe)
            self.assertEqual(sum(p.numel() for p in model.parameters()),2236682)
            path=Path(tmp)/'model.onnx';export_fp32(model,path,size=224,fixed_batch=True)
            graph=onnx.load(path);self.assertTrue(any(n.op_type=='Resize' for n in graph.graph.node))
            opt=ort.SessionOptions();opt.intra_op_num_threads=2
            session=ort.InferenceSession(str(path),opt,providers=['CPUExecutionProvider'])
            self.assertEqual(session.get_inputs()[0].shape,[1,3,224,224])
            inputs=np.random.default_rng(9).normal(size=(1,3,224,224)).astype(np.float32)
            with torch.no_grad(): expected=model(torch.from_numpy(inputs)).numpy()
            actual=session.run(None,{'images':inputs})[0]
            self.assertEqual(actual.shape,(1,10));np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-4)
if __name__=='__main__':unittest.main()
