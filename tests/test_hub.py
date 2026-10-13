import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import onnx
from onnx import helper, TensorProto
from edgeprep.hub_auth import get_client, save_token
from edgeprep.hub_profile import build_plan, load_entries


class HubTests(unittest.TestCase):
    def test_credential_roundtrip_does_not_log_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"client.ini"
            with patch("builtins.print") as output:
                save_token("unit-test-token", path)
            output.assert_not_called()
            with patch("edgeprep.hub_auth.CONFIG_PATH", path), patch.dict("os.environ", {}, clear=True), patch("edgeprep.hub_auth.dotenv_values", return_value={}), patch("edgeprep.hub_auth.hub.Client") as client:
                get_client()
                self.assertEqual(client.call_args.args[0].api_token, "unit-test-token")

    def test_input_contract_and_unsupported_conversion_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.onnx"
            image = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1,3,224,224])
            graph = helper.make_graph([helper.make_node("Identity", ["images"], ["output"])], "test", [image],
                         [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1,3,224,224])])
            onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("",13)]), path)
            args = argparse.Namespace(model=str(path), threads=4, lite_mp=0, runtime="onnx", compute="cpu", direct_onnx=False,
                  precision="none", calibration=None, device="offline-test", performance="burst", range_scheme="mse_minimizer",calibration_count=512,compiled_job=None,quantized_job=None)
            self.assertIn("--compute_unit cpu", build_plan(args)["inference_options"])
            args.compute="npu"
            inference=build_plan(args)["inference_options"]
            self.assertNotIn("--compute_unit",inference)
            self.assertIn("--onnx_execution_providers qnn",inference)
            args.compute="cpu"
            args.precision="w8a8"
            with self.assertRaises(ValueError):build_plan(args)
            args.calibration=str(path);args.runtime="tflite";args.precision="w8a16"
            with self.assertRaises(ValueError):build_plan(args)

    def test_remote_inputs_reject_nonfinite_or_wrong_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"inputs.npy"
            for arrays in [np.zeros((1,224,224,3),np.float32), np.full((1,3,224,224),np.nan,np.float32)]:
                np.save(path,arrays)
                with self.assertRaises(ValueError):load_entries(path,32)
            np.save(path,np.zeros((2,3,224,224),np.float32))
            entries=load_entries(path,1)
            self.assertEqual(entries["images"][0].shape,(1,3,224,224))
            self.assertEqual(len(entries["images"]),1)


if __name__ == "__main__":unittest.main()
