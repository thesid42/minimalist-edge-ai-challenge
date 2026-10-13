"""Compile, quantize and run the student on AI Hub. --plan is entirely offline."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import qai_hub as hub
from .hub_auth import get_client
from .challenge_data import digest


def build_plan(args):
    if args.threads < 1 or not 0 <= args.lite_mp <= 100:
        raise ValueError("Threads must be positive; Lite-MP must be between 0 and 100")
    if args.runtime == "qnn_dlc" and args.compute != "npu":
        raise ValueError("This helper targets QNN DLC to the NPU only")
    if args.runtime == "tflite" and args.precision == "w8a16":
        raise ValueError("Use ONNX or QNN DLC for w8a16")
    if args.direct_onnx and (args.runtime != "onnx" or args.precision != "none"):
        raise ValueError("Direct ONNX inference cannot compile or perform Hub quantization")
    if args.precision != "none" and not args.calibration:
        raise ValueError("Hub quantization requires real calibration data")
    if args.lite_mp and args.precision == "none":
        raise ValueError("Lite-MP requires Hub quantization")
    graph = onnx.load(args.model)
    onnx.checker.check_model(graph)
    input_shape = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
    if len(graph.graph.input) != 1 or graph.graph.input[0].name != "images" or input_shape != [1,3,224,224]:
        raise ValueError("Expected a single fixed float images input (1,3,224,224)")
    qdq = any(n.op_type in ("QuantizeLinear", "DequantizeLinear") for n in graph.graph.node)
    if qdq and not args.direct_onnx:
        raise ValueError("For Hub compilation use our FP32 ONNX and --precision w8a8/w8a16. Local ORT QDQ is not Qualcomm's supported fake-quant format. --direct-onnx can probe ORT compatibility separately.")
    compile_options = f"--target_runtime {args.runtime} --compute_unit {args.compute}"
    inference_options = f"--compute_unit {args.compute} --max_profiler_iterations 100"
    if args.runtime == "onnx":
        inference_options += f" --onnx_options intra_op_num_threads={args.threads};inter_op_num_threads=1;execution_mode=SEQUENTIAL;enable_memory_pattern=true;enable_cpu_memory_arena=true"
        if args.compute == "npu":
            inference_options = inference_options.replace("--compute_unit npu ", "")
            inference_options += ";qnn_htp_performance_mode=" + args.performance + " --onnx_execution_providers qnn"
    elif args.runtime == "tflite":
        inference_options += f" --tflite_options number_of_threads={args.threads}"
    elif args.compute == "npu":
        inference_options += " --qnn_options context_htp_performance_mode=" + args.performance.upper()
    quantize_options = "--range_scheme " + args.range_scheme
    if args.lite_mp:
        quantize_options += f" --lite_mp percentage={args.lite_mp};override_qtype=int16"
    return {"model": str(Path(args.model).resolve()), "source_sha256": digest(args.model),
            "device": args.device, "runtime": args.runtime, "compute_unit_requested": args.compute,
            "precision": args.precision, "input_specs": {"images": [1,3,224,224]},
            "compile_options": compile_options, "inference_options": inference_options,
            "quantize_options": quantize_options if args.precision != "none" else None,
            "calibration_count_requested": args.calibration_count,
            "calibration_sha256": digest(args.calibration) if args.calibration else None,
            "direct_onnx": args.direct_onnx, "compiled_job": args.compiled_job, "quantized_job": args.quantized_job, "local_qdq": qdq,
            "validation_note": "Separate representative-input numerical smoke check; not full-dataset accuracy",
            "target_reference": "Exploration only unless organizers designate this device; preserve laptop CPU baseline",
            "energy": "Inspect returned fields; never convert latency into energy"}


def load_entries(path, count):
    if count < 1:
        raise ValueError("Input sample count must be positive")
    arrays = np.load(path, mmap_mode="r", allow_pickle=False)
    if arrays.ndim != 4 or arrays.shape[1:] != (3,224,224) or arrays.dtype != np.float32 or not len(arrays):
        raise ValueError("Expected nonempty float32 N x 3 x 224 x 224 arrays")
    try:
        values = arrays[:count]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite input data")
        return {"images": [np.array(x[None], dtype=np.float32, order="C", copy=True) for x in values]}
    finally:
        arrays._mmap.close()


def run(args, client):
    plan = build_plan(args)
    entries = load_entries(args.inputs, args.validation_count) if args.inputs else None
    calibration = load_entries(args.calibration, args.calibration_count) if args.precision != "none" else None
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "request.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    jobs = {"device": args.device, "runtime": args.runtime, "sdk_version": str(hub.__version__)}

    def save_job(name, job):
        jobs[name] = job.job_id
        (output / "jobs.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        print(f"{name}: {job.job_id}", flush=True)

    def completed(job):
        status = job.wait()
        jobs[job.job_id + "_status"] = {"code": status.code, "message": status.message}
        (output / "jobs.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        if not status.success:
            raise RuntimeError(f"AI Hub job {job.job_id} failed: {status.message}")

    def target(job):
        completed(job)
        result = job.get_target_model()
        if result is None:
            raise RuntimeError("Job failed; inspect saved job ID in the AI Hub dashboard")
        return result

    if args.compiled_job:
        if args.precision != "none":
            raise ValueError("Reusing a compile job requires --precision none")
        compile_job = client.get_job(args.compiled_job)
        save_job("compile", compile_job); compiled = target(compile_job)
        compile_job.download_target_model(str(output / ("target." + {"onnx":"onnx","tflite":"tflite","qnn_dlc":"dlc"}[args.runtime])))
    elif args.direct_onnx:
        compiled = args.model
    else:
        source = args.model
        if args.quantized_job:
            previous = client.get_job(args.quantized_job)
            save_job("quantize", previous); source = target(previous)
        elif args.precision != "none":
            optimized = client.submit_compile_job(model=source, device=hub.Device(args.device),
                input_specs={"images": (1,3,224,224)}, options="--target_runtime onnx --compute_unit cpu")
            save_job("prepare_onnx", optimized); source = target(optimized)
            optimized.download_target_model(str(output / "optimized_fp32.onnx"))
            quantized = client.submit_quantize_job(model=source, calibration_data=calibration,
                weights_dtype=hub.QuantizeDtype.INT8,
                activations_dtype=hub.QuantizeDtype.INT16 if args.precision == "w8a16" else hub.QuantizeDtype.INT8,
                options=plan["quantize_options"])
            save_job("quantize", quantized); source = target(quantized)
            quantized.download_target_model(str(output / "quantized_fake.onnx"))
        compile_job = client.submit_compile_job(model=source, device=hub.Device(args.device),
                    input_specs={"images": (1,3,224,224)}, options=plan["compile_options"])
        save_job("compile", compile_job); compiled = target(compile_job)
        extension = {"onnx": "onnx", "qnn_dlc": "dlc", "tflite": "tflite"}[args.runtime]
        compile_job.download_target_model(str(output / ("target." + extension)))
    # Synthetic inputs benchmark runtime only. Real normalized data is supplied in
    # a separate inference job so timing does not vary with validation-set length.
    profile = client.submit_inference_job(model=compiled, device=hub.Device(args.device),
                                  inputs=None, profile=True, options=plan["inference_options"])
    save_job("profile_inference", profile); completed(profile)
    profile.download_profile(str(output / "profile.json"))
    if entries:
        job = client.submit_inference_job(model=compiled, device=hub.Device(args.device),
                     inputs=entries, profile=False, options=plan["inference_options"])
        save_job("validation_inference", job); completed(job)
        raw = job.download_output_data()
        if not raw:
            raise RuntimeError("No output data returned by successful remote job")
        values = {k: np.concatenate(v) for k,v in raw.items()}
        np.savez(output / "remote_outputs.npz", **values)
        if len(values) != 1:
            raise ValueError("Expected one ten-class remote output")
        remote = next(iter(values.values()))
        options = ort.SessionOptions(); options.intra_op_num_threads = 1
        local = ort.InferenceSession(args.model, options, providers=["CPUExecutionProvider"])
        reference = np.concatenate([local.run(None, {"images": x})[0] for x in entries["images"]])
        if remote.shape != reference.shape or not np.isfinite(remote).all():
            raise ValueError("Remote outputs have invalid shape or values")
        result = {"examples": len(remote), "scope": "numerical_smoke_check",
                  "top1_agreement_with_source": float(np.mean(remote.argmax(1) == reference.argmax(1))),
                  "max_absolute_logit_error": float(np.max(np.abs(remote-reference))),
                  "source_sha256": plan["source_sha256"], "inputs_sha256": digest(args.inputs),
                  "full_dataset_accuracy": None}
        (output / "numerical_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved results to {output}. Verify actual per-layer compute units before claiming NPU execution.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--model")
    p.add_argument("--device", help="Exact name from the live catalog")
    p.add_argument("--runtime", choices=["onnx", "tflite", "qnn_dlc"], default="onnx")
    p.add_argument("--compute", choices=["cpu", "npu"], default="cpu")
    p.add_argument("--precision", choices=["none", "w8a8", "w8a16"], default="none")
    p.add_argument("--range-scheme", choices=["auto", "mse_minimizer", "min_max"], default="mse_minimizer")
    p.add_argument("--lite-mp", type=int, default=0, help="Beta: percentage of sensitive layers raised to INT16")
    p.add_argument("--performance", choices=["burst", "balanced", "power_saver"], default="burst")
    p.add_argument("--calibration")
    p.add_argument("--calibration-count", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--inputs", help="Normalized .npy samples for separate numerical validation")
    p.add_argument("--validation-count", type=int, default=32)
    p.add_argument("--output", default="results/hub")
    p.add_argument("--quantized-job", help="Reuse a Hub quantize job before target compilation")
    p.add_argument("--compiled-job", help="Reuse a previous successful compile job instead of uploading again")
    p.add_argument("--direct-onnx", action="store_true", help="Skip compilation and probe an existing ONNX with ORT")
    p.add_argument("--plan", action="store_true", help="Validate and print request without network calls")
    args = p.parse_args()
    if args.list_devices:
        if args.plan: p.error("Device discovery requires network access; omit --plan")
        for device in get_client().get_devices():
            print(device.name, device.os, device.attributes)
        return
    if not args.model or not args.device:
        p.error("--model and --device are required, or use --list-devices")
    if args.plan:
        print(json.dumps(build_plan(args), indent=2)); return
    run(args, get_client())


if __name__ == "__main__":
    main()
