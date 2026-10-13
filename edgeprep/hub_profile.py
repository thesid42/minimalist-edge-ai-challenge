"""Explicit remote compile/profile helper. --plan is offline and submits nothing."""
import argparse
import json
from pathlib import Path
import numpy as np
import qai_hub as hub


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--model")
    p.add_argument("--device", help="Exact device name from the LIVE catalog")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--inputs", help="Optional representative .npy input data for numerical validation")
    p.add_argument("--output", default="results/hub")
    p.add_argument("--plan", action="store_true", help="Print proposed request without contacting AI Hub")
    args = p.parse_args()
    if args.list_devices:
        for device in hub.get_devices():
            print(device.name, device.os, device.attributes)
        return
    if not args.model or not args.device:
        p.error("--model and --device are required, or use --list-devices")
    plan = {"model": str(Path(args.model).resolve()), "device": args.device,
            "input_specs": {"images": [1,3,args.size,args.size]},
            "compile_options": "--target_runtime onnx --compute_unit cpu",
            "inference_options": f"--compute_unit cpu --onnx_options intra_op_num_threads={args.threads};inter_op_num_threads=1;execution_mode=SEQUENTIAL",
            "target_reference": "Use the common organizer-provided profile; local CPU fallback is allowed",
            "energy": "Do not infer Joules from latency; check returned profile fields and ask organizers if unavailable"}
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "request.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    compile_job = hub.submit_compile_job(model=args.model, device=hub.Device(args.device),
                    input_specs={"images": (1,3,args.size,args.size)}, options=plan["compile_options"])
    compile_job.wait()
    target = compile_job.get_target_model()
    if target is None:
        raise RuntimeError("Compilation failed; inspect the AI Hub job dashboard")
    entries = None
    if args.inputs:
        arrays = np.load(args.inputs, allow_pickle=False)
        entries = {"images": [np.ascontiguousarray(x[None], dtype=np.float32) for x in arrays[:32]]}
    # qai-hub 0.56 replaced submit_profile_job with submit_inference_job(profile=True).
    job = hub.submit_inference_job(model=target, device=hub.Device(args.device),
                                  inputs=entries, profile=True, options=plan["inference_options"])
    job.wait()
    job.download_profile(str(output / "profile.json"))
    if entries:
        raw = job.download_output_data()
        np.savez(output / "remote_outputs.npz", **{k: np.concatenate(v) for k,v in raw.items()})
    (output / "jobs.json").write_text(json.dumps({"compile_job": compile_job.job_id,
                    "inference_job": job.job_id, "device": args.device}, indent=2), encoding="utf-8")
    print(f"Saved remote results to {output}; inspect before making latency/energy claims.")


if __name__ == "__main__":
    main()
