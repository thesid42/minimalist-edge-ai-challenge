"""Append the completed Mac report, preserving existing Windows evidence and outputs.

Run after measure_mac.py (or after a Windows-only global results refresh).
The new notebook report cell is checked standalone; old cells are not re-executed.
"""
import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN MAC CPU RESULTS -->"
END = "<!-- END MAC CPU RESULTS -->"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write_json(path, value, indent=2, sort_keys=False):
    (ROOT / path).write_text(json.dumps(value, indent=indent, sort_keys=sort_keys) + "\n", encoding="utf-8")


def insert_block(path, body, anchor=None, marker="MAC CPU RESULTS"):
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    begin, finish = f"<!-- BEGIN {marker} -->", f"<!-- END {marker} -->"
    if begin in text:
        start = text.index(begin)
        end = text.index(finish, start) + len(finish)
        end += len(text[end:]) - len(text[end:].lstrip("\n"))
        text = text[:start] + text[end:]
    block = f"{begin}\n{body}\n{finish}\n\n"
    if anchor:
        if text.count(anchor) != 1:
            raise ValueError(f"Expected one report anchor in {path}")
        text = text.replace(anchor, block + anchor)
    else:
        text = text.rstrip() + "\n\n" + block
    target.write_text(text.rstrip("\n") + "\n", encoding="utf-8")


def record_diagnostic_sample(folder):
    if not (ROOT / folder / "candidate_accuracy.json").exists():
        return
    if (ROOT / folder / "manifest.json").exists():
        manifest = read(folder + "/manifest.json")
    else:
        summary = read(folder + "/summary.json")
        manifest = {"evaluation_count": summary["requested_examples"],
                    "class_support": summary["records"][0]["class_support"],
                    "class_to_idx": summary["class_to_idx"], "seed": 42,
                    "available_examples": summary["available_examples"],
                    "evaluation_indices_sha256": summary["records"][0]["evaluation_indices_sha256"]}
        write_json(folder + "/manifest.json", manifest)
    count = manifest["evaluation_count"]
    full = count == manifest.get("available_examples")
    if len(manifest["class_support"]) != 10 or sum(manifest["class_support"]) != count:
        raise ValueError("Class support does not match the evaluated dataset")
    if not full and (count % 10 or manifest["class_support"] != [count // 10] * 10):
        raise ValueError("Expected a class-balanced diagnostic sample")
    phase = "mac_cpu_full_data" if full else f"mac_cpu_diagnostic{count}"
    records = []
    table = ("| Model | Correct / images | Accuracy | Worst-class recall |\n|---|---:|---:|---:|\n" if count > 100 else
             "| Model | Correct / images | Accuracy |\n|---|---:|---:|\n")
    for name in ("baseline", "candidate"):
        result = read(f"{folder}/{name}_accuracy.json")
        original = read(f"results/mac/{name}_accuracy_subset.json")
        model_hash = result.get("sha256", result.get("model_sha256"))
        examples = result.get("validation_examples", result.get("examples"))
        if model_hash != original["sha256"] or examples != count:
            raise ValueError("Diagnostic model hash or evaluation count mismatch")
        correct = round(result["accuracy"] * count)
        line = f"| {name} | {correct:,} / {count:,} | {result['accuracy']*100:.4f}% |"
        if count > 100:
            line += f" {result['worst_class_accuracy']*100:.2f}% |"
        table += line + "\n"
        records.append({"name": f"mac_{name}_eval{count}", "phase": phase, "status": "completed",
                        "model_sha256": model_hash, "diagnostic_examples": count,
                        "diagnostic_accuracy_pct": result["accuracy"] * 100,
                        "worst_class_recall_pct": result["worst_class_accuracy"] * 100 if count > 100 else None,
                        "scope": "training_set" if full else "training_pool_diagnostic_subset", "provider": result["provider"],
                        "evidence": f"{folder}/{name}_accuracy.json", "manifest": folder + "/manifest.json"})
        if full:
            records[-1]["examples"] = records[-1].pop("diagnostic_examples")
            records[-1]["accuracy_pct"] = records[-1].pop("diagnostic_accuracy_pct")
    selection = ("The first half of the earlier 100-image sample was used. " if count == 50 else
                 "The same seeded image indices were used for both models. ")
    coverage = (f"All {count:,} images were evaluated, preserving the dataset's original unequal class counts. " if full else
                f"{count:,} images, {count//10:,} per class. ")
    caveat = ("This is complete training-set accuracy, not held-out generalization. " if full else
              "This is a diagnostic training-pool subset, not held-out or full-dataset accuracy. ")
    body = (f"### {count:,}-image evaluation\n\n" + coverage + selection +
            "Both models ran on CPU with four threads. Input resolution and model weights are unchanged. " + caveat +
            "Earlier subset scores remain intact.\n\n" + table +
            "\nOnly accuracy was rechecked; power and latency were not remeasured. "
            "Use the separate repeated CPU measurements for latency and energy.")
    marker = "MAC FULL DATA RESULTS" if full else ("MAC HALF SAMPLE RESULTS" if count == 50 else f"MAC {count} SAMPLE RESULTS")
    (ROOT / folder / "README.md").write_text(f"# {count:,}-image Mac evaluation\n\n" + body + "\n", encoding="utf-8")
    for path in ("README.md", "results/mac/README.md", "results/mac/paired_runtime/README.md", "experiments/13_mac_cpu/README.md"):
        insert_block(path, body, marker=marker)
    insert_block("experiments/README.md", f"Step 13 also includes a separate [{count:,}-image CPU {'full-data evaluation' if full else 'diagnostic recheck'}](../{folder}/README.md).",
                 marker=marker)
    registry = read("experiments/runs.json")
    registry["records"] = [r for r in registry["records"] if r["phase"] != phase] + records
    write_json("experiments/runs.json", registry)
    notebook = read("experiments/Experiment_Log.ipynb")
    notebook["cells"] = [c for c in notebook["cells"] if c["metadata"].get("mac_cpu_diagnostic_count") != count
                         and not (count == 50 and c["metadata"].get("mac_cpu_half_sample"))]
    notebook["cells"].append({"cell_type": "markdown", "id": f"mac-cpu-eval{count}",
                              "metadata": {"mac_cpu_diagnostic_count": count}, "source": body})
    write_json("experiments/Experiment_Log.ipynb", notebook, indent=1, sort_keys=True)
    insert_block("experiments/Experiment_Log.html", f"<section><h2>{count:,}-image Mac evaluation</h2><pre>" + html.escape(body) + "</pre></section>",
                 "</main>", marker=marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", default="results/mac/paired_runtime")
    parser.add_argument("--accuracy-dir", help="Paired *_accuracy.json evaluation directory, e.g. results/mac/eval27000")
    args = parser.parse_args()
    runtime = args.runtime_dir.rstrip("/")
    summary_path = runtime + "/summary.json"
    summary = read(summary_path)
    expected = {"baseline_spin", "baseline_sleep", "candidate_spin", "candidate_sleep"}
    if {r["name"] for r in summary["records"]} != expected:
        raise ValueError("Complete paired baseline/candidate results are required")
    if args.accuracy_dir:
        scores = {name: read(f"{args.accuracy_dir}/{name}_accuracy.json") for name in ("baseline", "candidate")}
        manifest = read(args.accuracy_dir + "/manifest.json")
        for score in scores.values():
            score["sha256"] = score["model_sha256"]
            score["validation_examples"] = score["examples"]
    else:
        scores = {name: read(f"results/mac/{name}_accuracy_subset.json") for name in ("baseline", "candidate")}
        manifest = read("results/mac/data/manifest.json")
    full = manifest.get("full_dataset_evaluated", False)
    phase = "mac_cpu_full_runtime" if full else "mac_cpu_runtime"
    marker = "MAC FULL CPU RESULTS" if full else "MAC CPU RESULTS"
    report_tag = "full" if full else "diagnostic"
    current = read("best_models/current.json")
    if scores["candidate"]["sha256"] != current["model_sha256"]:
        raise ValueError("Mac score is not for the current saved candidate")
    records = []
    accuracy_label = "Full-data accuracy" if full else "Diagnostic accuracy"
    lines = [f"| Model / mode | ONNX MiB | {accuracy_label} | Median ms | P95 ms | Peak RSS MiB | Estimated W | Estimated J/image, mean +/- SD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for record in summary["records"]:
        model = record["name"].split("_")[0]
        score = scores[model]
        if record["model_sha256"] != score["sha256"] or score["validation_examples"] != manifest["evaluation_count"]:
            raise ValueError("Accuracy and runtime model hashes/counts do not match")
        if not record["estimated_energy_j_per_inference"] or record["cpu_power_mode"] != "cpu_load":
            raise ValueError("Expected CPU-load energy estimates")
        energy_samples = []
        for evidence in record["evidence"]:
            measurement = json.loads(Path(evidence).read_text())
            if measurement["sha256"] != record["model_sha256"] or measurement["provider"] != "CPUExecutionProvider":
                raise ValueError("Per-run provenance mismatch")
            energy_samples.append(measurement["energy_j_per_inference"])
        accuracy_path = (f"{args.accuracy_dir}/{model}_accuracy.json" if args.accuracy_dir else f"results/mac/{model}_accuracy_subset.json")
        row = dict(record, name=("mac_full_" if full else "mac_") + record["name"], phase=phase, status="completed",
                   label=record["name"], diagnostic_accuracy_pct=score["accuracy"] * 100,
                   diagnostic_examples=score["validation_examples"], scope="training_set" if full else "training_pool_diagnostic_subset",
                   evidence=summary_path, accuracy_evidence=accuracy_path,
                   implementation=["scripts/measure_mac.py", "edgeprep/benchmark.py"], machine=summary["machine"])
        if full:
            row["examples"] = row.pop("diagnostic_examples")
            row["accuracy_pct"] = row.pop("diagnostic_accuracy_pct")
            row["worst_class_recall_pct"] = score["worst_class_accuracy"] * 100
            row["correct_predictions"] = score["correct_predictions"]
            row["evaluation_seconds_including_preprocessing"] = score["evaluation_seconds"]
            row["projected_inference_only_energy_j_for_dataset"] = score["examples"] * record["estimated_energy_j_per_inference"]
        row["run_evidence"] = [str(Path(p).relative_to(ROOT)) for p in record["evidence"]]
        row["estimated_energy_range_j_per_inference"] = [min(energy_samples), max(energy_samples)]
        row["estimated_energy_stdev_j_per_inference"] = statistics.stdev(energy_samples) if len(energy_samples) > 1 else 0.0
        records.append(row)
        lines.append(f"| {record['name']} | {record['file_bytes']/2**20:.3f} | {score['accuracy']*100:.4f}% ({score['validation_examples']:,} images) | "
                     f"{record['median_ms']:.3f} | {record['p95_ms']:.3f} | {record['peak_rss_mib']:.2f} | "
                     f"{record['estimated_power_w']:.2f} | {record['estimated_energy_j_per_inference']:.6f} +/- "
                     f"{row['estimated_energy_stdev_j_per_inference']:.6f} |")
    table = "\n".join(lines)
    settings = summary["machine"]["settings"]
    accuracy_note = (f"Accuracy covers all {manifest['evaluation_count']:,} images with their original class counts: "
                     "training-set accuracy, not held-out generalization. " if full else
                     f"Accuracy is a seeded balanced {manifest['evaluation_count']:,}-image diagnostic subset, "
                     "not full-dataset accuracy or held-out generalization. ")
    description = (f"{summary['machine']['cpu']}, {summary['machine']['os']}, CPUExecutionProvider; {settings['threads']} threads, "
                   f"{settings['repetitions']} serial fresh-process repetitions, batch one, {settings['warmup']} warmups, "
                   f"{settings['runs']:,} timings and separate {settings['energy_seconds']:g}-second energy windows. "
                   "Power source is recorded in machine.json. External input remains (1,3,224,224). "
                   "The exact saved winning ONNX was measured without retraining.\n\n"
                   + accuracy_note + "The timing JSON's one-image accuracy "
                   "is only a smoke test. Power/energy use CodeCarbon CPU-load/TDP plus RAM heuristics "
                   "(35 W CPU assumption, GPU excluded), **not hardware-sensor or whole-device readings**. "
                   "Do not mix these Mac baselines with Windows or remote-device results.")
    description += "\n\nPower source at benchmark start: " + summary["machine"]["power_source"].splitlines()[0] + "."
    if full:
        extra = ["\n\n| Model / mode | Worst-class recall | Load ms | Session RAM delta MiB | CPU use, one-core % | Projected dataset inference energy J |",
                 "|---|---:|---:|---:|---:|---:|"]
        for row in records:
            extra.append(f"| {row['label']} | {row['worst_class_recall_pct']:.2f}% | {row['load_ms']:.3f} | "
                         f"{row['session_rss_delta_mib']:.2f} | {row['process_cpu_percent_one_core_basis']:.1f} | "
                         f"{row['projected_inference_only_energy_j_for_dataset']:.2f} |")
        table += "\n".join(extra)
        description += ("\n\nAccuracy is measured across the entire dataset; runtime uses the same fixed real timing image "
                        "for both models. Timed inference excludes disk reads, preprocessing and model loading. "
                        "Projected dataset energy is estimated J/inference multiplied by the image count, not "
                        "a metered full-dataset run. CPU percentages use one core = 100% (four busy cores can reach 400%).")
        description += ("\n\nAggregation: medians of run median/P95/load times, mean estimated watts and J/inference, "
                        "sample SD for energy, maximum inference-session RSS and RAM delta. RSS excludes later "
                        "CodeCarbon imports. Model load time is reported separately from warmed-up inference latency.")
    if any(r["estimated_energy_range_j_per_inference"][1] > 1.5 * r["estimated_energy_range_j_per_inference"][0] for r in records):
        description += ("\n\nEnergy variability warning: repetitions have substantially different estimated energy. "
                        "All repetitions are retained; the cause is unverified. Means and sample SD are reported, "
                        "but a precise energy-saving percentage is not established. Inspect individual run throughput "
                        "and energy before interpreting the average watts or joules.")
    title = "Mac CPU metrics with full-dataset accuracy" if full else "Matched Mac CPU measurement"
    body = "### " + title + "\n\n" + description + "\n\n" + table
    report = ROOT / runtime / "README.md"
    refresh_command = f".venv/bin/python scripts/record_mac_results.py --runtime-dir {runtime}"
    if args.accuracy_dir:
        refresh_command += f" --accuracy-dir {args.accuracy_dir}"
    report.write_text("# Matched Mac CPU results\n\n" + description + "\n\n" + table +
                      f"\n\nAfter a global Windows result refresh, rerun `{refresh_command}` "
                      "to restore this separate Mac section in the main README, log and notebook. "
                      "The new notebook report cell is verified standalone; prior notebook cells are not re-executed.\n", encoding="utf-8")
    insert_block("README.md", body + f"\n\n[Mac evidence]({summary_path}).", "## Models and data", marker=marker)
    insert_block("results/mac/README.md", body + f"\n\nLatest evidence: `{summary_path}`.", marker=marker)
    stage = ROOT / "experiments/13_mac_cpu"
    stage.mkdir(exist_ok=True)
    detail_name = "details_full_runtime.json" if full else "details.json"
    stage_body = description + "\n\n" + table + f"\n\n[Raw evidence](../../{summary_path}); " + f"[exact settings and hashes]({detail_name}).\n"
    if full:
        insert_block("experiments/13_mac_cpu/README.md", "## " + title + "\n\n" + stage_body, marker=marker)
    else:
        (stage / "README.md").write_text("# 13. Matched Mac CPU measurement\n\n" + stage_body, encoding="utf-8")
    write_json("experiments/13_mac_cpu/" + detail_name, records)
    write_json(runtime + "/combined_metrics.json", {"records": records, "machine": summary["machine"]})
    registry = read("experiments/runs.json")
    registry["records"] = [r for r in registry["records"] if r["phase"] != phase] + records
    registry["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json("experiments/runs.json", registry)
    insert_block("experiments/README.md", "## 13. Matched Mac CPU measurement\n\n"
                 "[Details](13_mac_cpu/README.md): exact saved winner against a freshly measured baseline on Apple M4 Pro. "
                 "Subset scores, full-data accuracy and estimated power retain separate evidence from Windows results.", marker=marker)
    table_path = runtime + "/notebook_table.json"
    write_json(table_path, {"text": table})
    source = ('from pathlib import Path\nimport json\nproject = Path.cwd()\n'
              'if not (project / "edgeprep").exists(): project = project.parent\n'
              f'mac_table = json.loads((project / {table_path!r}).read_text())\n'
              'print(mac_table["text"])')
    output = subprocess.check_output([sys.executable, "-c", source], cwd=ROOT, text=True)
    if output != table + "\n":
        raise ValueError("Notebook report cell does not reproduce the measured table")
    notebook = read("experiments/Experiment_Log.ipynb")
    notebook["cells"] = [c for c in notebook["cells"] if not (
        c["metadata"].get("mac_cpu_report_runtime") == runtime or
        (runtime == "results/mac/paired_runtime" and c["metadata"].get("mac_cpu_report") and
         not c["metadata"].get("mac_cpu_report_runtime")))]
    count = max((c.get("execution_count") or 0 for c in notebook["cells"]), default=0) + 1
    notebook["cells"].extend([
        {"cell_type": "markdown", "id": f"mac-cpu-{report_tag}-description", "metadata": {"mac_cpu_report": True, "mac_cpu_report_runtime": runtime},
         "source": "## 13. " + title + "\n\n" + description},
        {"cell_type": "code", "id": f"mac-cpu-{report_tag}-table", "metadata": {"mac_cpu_report": True, "mac_cpu_report_runtime": runtime, "verification": "Standalone Python execution; prior notebook cells not rerun"},
         "source": source, "execution_count": count,
         "outputs": [{"output_type": "stream", "name": "stdout", "text": output}]},
    ])
    write_json("experiments/Experiment_Log.ipynb", notebook, indent=1, sort_keys=True)
    insert_block("experiments/Experiment_Log.html", '<section><h2>13. Matched Mac CPU measurement</h2><p>' +
                 html.escape(description) + '</p><pre>' + html.escape(table) + '</pre></section>', "</main>", marker=marker)
    write_json(runtime + "/report_verification.json", {
        "status": "passed", "candidate_sha256": current["model_sha256"],
        "all_runtime_and_subset_model_hashes_match": True,
        "new_notebook_report_cell_executed_standalone": True, "full_notebook_reexecuted": False,
        "report_table_sha256": hashlib.sha256(table.encode()).hexdigest(),
    })
    for folder in ("results/mac/eval50", "results/mac/eval15000", "results/mac/eval27000"):
        record_diagnostic_sample(folder)
    print("Recorded Mac report, numbered step 13, registry and verified standalone notebook report cell.")


if __name__ == "__main__":
    main()
