"""Isolated ONNX CPU worker: does NOT import PyTorch or hold other models in memory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time
import numpy as np
import onnxruntime as ort
import psutil


def benchmark(model, size, threads=1, runs=100, warmup=20, labels=None,
              inputs=None, energy_seconds=0, energy_gpu_ids=None):
    if threads < 1 or runs < 1 or size < 1:
        raise ValueError("Threads, runs and size must be positive")
    process = psutil.Process()
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    before_load = process.memory_info().rss
    peak = [before_load]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    try:
        start = time.perf_counter()
        session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        load_ms = (time.perf_counter() - start) * 1000
        peak[0] = max(peak[0], process.memory_info().rss)
        name = session.get_inputs()[0].name
        x = np.random.default_rng(42).normal(size=(1, 3, size, size)).astype(np.float32)
        data = None
        if inputs is not None:
            data = np.load(inputs, mmap_mode="r", allow_pickle=False)
            if not len(data) or data.ndim != 4 or data.shape[1:] != (3, size, size):
                raise ValueError("Evaluation input shape mismatch or empty dataset")
            x = np.array(data[:1], dtype=np.float32, copy=True)
        for _ in range(warmup):
            session.run(None, {name: x})
        cpu_before = process.cpu_times()
        wall_before = time.perf_counter()
        times = []
        for _ in range(runs):
            started = time.perf_counter_ns()
            logits = session.run(None, {name: x})[0]
            times.append((time.perf_counter_ns() - started) / 1e6)
        wall = time.perf_counter() - wall_before
        cpu_after = process.cpu_times()
        cpu_seconds = (cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system)
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite model outputs")
        result = {"model": str(Path(model).resolve()), "sha256": hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                  "input_shape": [1, 3, size, size], "threads": threads, "runs": runs,
                  "warmup": warmup, "provider": "CPUExecutionProvider", "load_ms": load_ms,
                  "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
                  "mean_ms": float(np.mean(times)), "file_bytes": Path(model).stat().st_size,
                  "peak_rss_mib": peak[0] / 2**20, "rss_before_session_mib": before_load / 2**20,
                  "session_rss_delta_mib": (peak[0] - before_load) / 2**20,
                  "process_cpu_seconds": cpu_seconds,
                  "process_cpu_percent_one_core_basis": 100 * cpu_seconds / wall,
                  "platform": platform.platform(), "cpu": platform.processor(),
                  "onnxruntime": ort.__version__, "accuracy": None, "per_class_accuracy": None,
                  "timing_input": "first provided preprocessed sample" if inputs else "seeded synthetic input",
                  "energy_j_per_inference": None, "energy_kind": "not_measured"}
        if inputs is not None:
            truth = np.load(labels, allow_pickle=False) if labels else None
            if truth is None or len(truth) != len(data) or not len(data):
                raise ValueError("Nonempty inputs and matching labels are required for accuracy")
            if data.shape[1:] != (3, size, size):
                raise ValueError("Evaluation input shape mismatch")
            guesses = np.array([session.run(None, {name: np.ascontiguousarray(a[None], dtype=np.float32)})[0].argmax(1)[0] for a in data])
            result["accuracy"] = float(np.mean(guesses == truth))
            result["validation_examples"] = len(truth)
            result["per_class_accuracy"] = {str(int(c)): float(np.mean(guesses[truth == c] == c)) for c in np.unique(truth)}
        if energy_seconds:
            from codecarbon import OfflineEmissionsTracker
            output = Path(model).parent / "energy"
            output.mkdir(exist_ok=True)
            # gpu_ids=[] avoids attributing idle GPU power to CPU-only ONNX inference.
            tracker = OfflineEmissionsTracker(country_iso_code="USA", output_dir=str(output),
                        tracking_mode="process", measure_power_secs=1, log_level="warning",
                        gpu_ids=[] if energy_gpu_ids is None else energy_gpu_ids,
                        save_to_api=False, save_to_file=True)
            tracker.start()
            energy_runs = 0
            started = time.perf_counter()
            try:
                while time.perf_counter() - started < energy_seconds:
                    session.run(None, {name: x})
                    energy_runs += 1
            finally:
                tracker.stop()
            record = tracker.final_emissions_data
            energy = record.energy_consumed if record else None
            result.update(energy_kind="codecarbon_estimate_not_device_meter",
                          energy_runs=energy_runs, energy_duration_s=time.perf_counter()-started,
                          energy_j_per_inference=energy*3_600_000/energy_runs if energy is not None else None,
                          energy_gpu_ids=[], energy_scope="CPU and RAM attributed by CodeCarbon process mode",
                          energy_note="Windows AMD CPU may use TDP estimation; inspect startup logs and compare both models on the same machine.")
        peak[0] = max(peak[0], process.memory_info().rss)
        # Peak for the inference session, not subsequent CodeCarbon imports.
        return result
    finally:
        stopped.set()
        watcher.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs")
    parser.add_argument("--labels")
    parser.add_argument("--energy-seconds", type=float, default=0)
    args = parser.parse_args()
    result = benchmark(args.model, args.size, args.threads, args.runs, args.warmup,
                       args.labels, args.inputs, args.energy_seconds)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
