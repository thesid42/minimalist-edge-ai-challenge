"""Repeat isolated ONNX CPU measurements without overwriting Windows results."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="artifacts/mac/baseline/model_fp32.onnx")
    parser.add_argument("--candidate", help="Exact recovered192 FP16-storage ONNX artifact")
    parser.add_argument("--output", default="results/mac/runtime")
    parser.add_argument("--inputs", required=True, help="One preprocessed timing sample (.npy)")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--energy-seconds", type=float, default=30)
    args = parser.parse_args()
    if (platform.system() != "Darwin" or args.repetitions < 1 or args.threads < 1
            or args.runs < 1 or args.warmup < 0 or args.energy_seconds < 0):
        parser.error("Requires macOS and valid positive measurement settings")
    models = {"baseline": Path(args.baseline).resolve()}
    if args.candidate:
        models["candidate"] = Path(args.candidate).resolve()
    for path in [*models.values(), Path(args.inputs), Path(args.labels)]:
        if not path.is_file():
            parser.error(f"Missing measurement input: {path}")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Never silently reuse an older run or mix models/settings across invocations.
    if list(out.glob("*_rep*.json")) or (out / "summary.json").exists():
        parser.error("Output already contains results; choose a fresh --output directory")
    metadata = {
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "os": platform.platform(),
        "power_source": subprocess.check_output(["pmset", "-g", "batt"], text=True).strip(),
        "settings": vars(args),
        "candidate_status": "provided" if args.candidate else "not_available_not_measured",
        "protocol": "Serial fresh processes; rotated scenario order each repetition; CPU only; batch size one",
        "power_caveat": "CodeCarbon CPU-load/TDP and RAM heuristic only, not Apple sensors or a device power meter",
        "accuracy_caveat": "Timing sample correctness is only a smoke test, not model accuracy",
    }
    (out / "machine.json").write_text(json.dumps(metadata, indent=2))
    env = os.environ.copy()
    # Force the non-privileged estimation backend; never invoke sudo/powermetrics.
    env["CODECARBON_FORCE_MODE_CPU_LOAD"] = "true"
    scenarios = [(f"{name}_{mode}", path, mode == "sleep")
                 for name, path in models.items() for mode in ("spin", "sleep")]
    measurements = {name: [] for name, _, _ in scenarios}
    for rep in range(1, args.repetitions + 1):
        offset = (rep - 1) % len(scenarios)
        for name, model, sleep in scenarios[offset:] + scenarios[:offset]:
            label = f"{name}_rep{rep}"
            destination = out / f"{label}.json"
            command = [sys.executable, "-m", "edgeprep.benchmark", "--model", str(model),
                       "--threads", str(args.threads), "--runs", str(args.runs),
                       "--warmup", str(args.warmup), "--inputs", args.inputs,
                       "--labels", args.labels, "--energy-seconds", str(args.energy_seconds),
                       "--energy-output", str(out / label), "--output", str(destination)]
            if sleep:
                command.append("--disable-spinning")
            print(f"Measuring {label}", flush=True)
            with (out / f"{label}.stdout.txt").open("w") as stdout, (out / f"{label}.stderr.txt").open("w") as stderr:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=True)
            result = json.loads(destination.read_text())
            measurements[name].append(result)
            print(f"{label}: median {result['median_ms']:.3f} ms; "
                  f"estimated energy {result['energy_j_per_inference']} J/inference", flush=True)
    records = []
    for name, runs in measurements.items():
        energies = [r["energy_j_per_inference"] for r in runs if r["energy_j_per_inference"] is not None]
        powers = [r["codecarbon_record"]["cpu_power"] + r["codecarbon_record"]["ram_power"]
                  for r in runs if r.get("codecarbon_record")]
        records.append({
            "name": name, "model_sha256": runs[0]["sha256"], "file_bytes": runs[0]["file_bytes"],
            "replicates": len(runs), "median_ms": statistics.median(r["median_ms"] for r in runs),
            "mean_ms": statistics.mean(r["mean_ms"] for r in runs),
            "load_ms": statistics.median(r["load_ms"] for r in runs),
            "median_range_ms": [min(r["median_ms"] for r in runs), max(r["median_ms"] for r in runs)],
            "p95_ms": statistics.median(r["p95_ms"] for r in runs),
            "peak_rss_mib": max(r["peak_rss_mib"] for r in runs),
            "session_rss_delta_mib": max(r["session_rss_delta_mib"] for r in runs),
            "process_cpu_percent_one_core_basis": statistics.median(r["process_cpu_percent_one_core_basis"] for r in runs),
            "estimated_energy_j_per_inference": statistics.mean(energies) if energies else None,
            "estimated_energy_stdev_j_per_inference": statistics.stdev(energies) if len(energies) > 1 else None,
            "estimated_energy_range_j_per_inference": [min(energies), max(energies)] if energies else None,
            "estimated_power_w": statistics.mean(powers) if powers else None,
            "cpu_power_mode": runs[0].get("codecarbon_cpu_mode"),
            "cpu_tdp_w": runs[0].get("codecarbon_cpu_tdp_w"),
            "evidence": [str(out / f"{name}_rep{rep}.json") for rep in range(1, len(runs) + 1)],
        })
    summary = {"machine": metadata, "records": records}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
