import tempfile
from pathlib import Path
import unittest
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
import onnxruntime as ort
from edgeprep.weight_storage import convert

class WeightStorageTests(unittest.TestCase):
    def test_half_storage_keeps_float_io_and_computes_rounded_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);x=helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,2]);y=helper.make_tensor_value_info('y',TensorProto.FLOAT,[1,2])
            weights=np.array([.123456,.25008],np.float32)
            g=helper.make_graph([helper.make_node('Add',['x','weights'],['y'])],'test',[x],[y],[numpy_helper.from_array(weights,'weights')])
            model=helper.make_model(g,opset_imports=[helper.make_opsetid('',13)]);model.ir_version=8;onnx.save(model,p/'source.onnx')
            convert(p/'source.onnx',p/'half.onnx')
            graph=onnx.load(p/'half.onnx');self.assertEqual(graph.graph.initializer[0].data_type,TensorProto.FLOAT16)
            opt=ort.SessionOptions();opt.intra_op_num_threads=1
            session=ort.InferenceSession(str(p/'half.onnx'),opt,providers=['CPUExecutionProvider'])
            inputs=np.ones((1,2),np.float32);actual=session.run(None,{'x':inputs})[0]
            self.assertEqual(actual.dtype,np.float32)
            np.testing.assert_array_equal(actual,inputs+weights.astype(np.float16).astype(np.float32))
if __name__=='__main__':unittest.main()
