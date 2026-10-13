from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                     QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process


class ArrayReader(CalibrationDataReader):
    def __init__(self, arrays, name="images"):
        self.arrays, self.name = list(arrays), name
        self.rewind()

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self):
        self.iterator = iter({self.name: np.ascontiguousarray(x, dtype=np.float32)} for x in self.arrays)


def export_fp32(model, output, size=224, opset=13, fixed_batch=True):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.randn(1, 3, size, size)
    torch.onnx.export(model, example, str(output), opset_version=opset,
                      input_names=["images"], output_names=["logits"],
                      dynamic_axes=None if fixed_batch else {"images": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    graph = onnx.load(str(output))
    outputs = {x.name for x in graph.graph.output}
    retained = [x for x in graph.graph.value_info if x.name not in outputs]
    del graph.graph.value_info[:]
    graph.graph.value_info.extend(retained)
    onnx.save(graph, str(output))
    onnx.checker.check_model(str(output))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(example).numpy()
    actual = session.run(None, {"images": example.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    return float(np.max(np.abs(actual - expected)))


def quantize(fp32, output, arrays, unsigned=False, activation_symmetric=True,
             nodes_to_exclude=None, method="MinMax"):
    preprocessed = Path(output).with_suffix(".pre.onnx")
    quant_pre_process(str(fp32), str(preprocessed), skip_symbolic_shape=True)
    quantize_static(str(preprocessed), str(output), ArrayReader(arrays),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8 if unsigned else QuantType.QInt8,
                    weight_type=QuantType.QUInt8 if unsigned else QuantType.QInt8,
                    per_channel=True, calibrate_method=getattr(CalibrationMethod, method),
                    nodes_to_exclude=nodes_to_exclude or [],
                    op_types_to_quantize=["Conv", "Gemm", "MatMul"],
                    extra_options={"ActivationSymmetric": activation_symmetric and not unsigned,
                                   "WeightSymmetric": not unsigned})
    onnx.checker.check_model(str(output))
    preprocessed.unlink(missing_ok=True)
