"""Apply identical FP16 weight storage to the original baseline and audit file sizes.

No training or cloud jobs. CPU-only paired inference uses the existing labelled500
inputs; full-data accuracy is collected separately with edgeprep.challenge.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
import onnxruntime as ort
from edgeprep.challenge_data import digest
from edgeprep.weight_storage import convert

OUT = ROOT / "results/storage_comparison"
BASELINE = ROOT / "artifacts/baseline/model_fp32.onnx"
HALF = ROOT / "artifacts/baseline_weight16/model_weight16.onnx"
WINNER = ROOT / "best_models/resize192_head_ce_w16_e391533162d7/model_weight16.onnx"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def audit(name, path, accuracy_path):
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    tensors = [numpy_helper.to_array(i) for i in graph.graph.initializer]
    floats = [a for a in tensors if a.dtype in (np.float32, np.float16)]
    size = path.stat().st_size
    accuracy_file = ROOT / accuracy_path
    score = read(accuracy_path) if accuracy_file.exists() else None
    if score and score["model_sha256"] != digest(path):
        raise ValueError("Accuracy evidence belongs to a different model")
    return {
        "name": name, "label": name, "phase": "matched_weight_storage",
        "status": "completed", "model": path.relative_to(ROOT).as_posix(),
        "model_sha256": digest(path), "file_bytes": size, "size_mib": size / 2**20,
        "stored_float_values": sum(a.size for a in floats),
        "float_initializer_payload_bytes": sum(a.nbytes for a in floats),
        "initializer_dtype_counts": dict(Counter(TensorProto.DataType.Name(i.data_type) for i in graph.graph.initializer)),
        "cast_nodes": sum(n.op_type == "Cast" for n in graph.graph.node),
        "resize_nodes": sum(n.op_type == "Resize" for n in graph.graph.node),
        "accuracy_pct": score["accuracy"] * 100 if score else None,
        "worst_recall_pct": score["worst_class_accuracy"] * 100 if score else None,
        "examples": score["examples"] if score else None,
        "scope": score["evaluation_scope"] if score else None,
        "accuracy_evidence": accuracy_path,
        "evidence": "results/storage_comparison/summary.json",
        "implementation": ["scripts/compare_weight_storage.py", "edgeprep/weight_storage.py"],
    }


def smoke():
    sample_path = "results/hub/resize192_labelled500/manifest.json"
    sample = read(sample_path)
    folder = ROOT / sample_path
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    sessions = [ort.InferenceSession(str(p), options, providers=["CPUExecutionProvider"]) for p in (BASELINE, HALF)]
    predictions = [[], []]
    maximum_error = 0.0
    for shard in sample["shards"]:
        path = folder.parent / shard["file"]
        if digest(path) != shard["sha256"]:
            raise ValueError("Changed diagnostic input shard")
        inputs = np.load(path, mmap_mode="r")
        for x in inputs:
            logits = [s.run(None, {s.get_inputs()[0].name: x[None]})[0] for s in sessions]
            for i, output in enumerate(logits):
                if output.shape != (1, 10) or not np.isfinite(output).all():
                    raise ValueError("Invalid model output")
                predictions[i].append(int(output.argmax()))
            maximum_error = max(maximum_error, float(np.abs(logits[0] - logits[1]).max()))
    a, b = map(np.asarray, predictions)
    return {"examples": len(a), "indices_sha256": sample["indices_sha256"],
            "sample_manifest": sample_path, "provider": "CPUExecutionProvider",
            "baseline_fp32_sha256": digest(BASELINE), "baseline_weight16_sha256": digest(HALF),
            "top1_agreement": float((a == b).mean()), "top1_disagreements": int((a != b).sum()),
            "max_absolute_logit_error": maximum_error,
            "note": "Paired output check, separate from full-data labelled accuracy and latency."}


def write_report(summary):
    rows = summary["records"]
    parts = ["# Matched FP16 storage: baseline versus recovered 192\n",
        "Apply the exact same FP16-initializer / Cast-to-FP32 conversion to the unchanged baseline and compare it with the preserved winner. No retraining, resolution change or pruning is applied to the baseline. All local inference uses ONNX Runtime CPU.\n",
        "| Model | Storage | Exact bytes | MiB | Accuracy on 27,000 images | Worst class recall |\n|---|---|---:|---:|---:|---:|"]
    for r in rows:
        accuracy = f'{r["accuracy_pct"]:.4f}%' if r["accuracy_pct"] is not None else "pending"
        worst = f'{r["worst_recall_pct"]:.2f}%' if r["worst_recall_pct"] is not None else "pending"
        storage = "FP16" if r["cast_nodes"] else "FP32"
        parts.append(f'| {r["name"]} | {storage} | {r["file_bytes"]:,} | {r["size_mib"]:.6f} | {accuracy} | {worst} |')
    parts += [f'\nThe recovered192 FP16-storage file is **{summary["candidate_minus_baseline_weight16_bytes"]:,} bytes larger** than the unchanged baseline stored in FP16. The original baseline alone shrinks **{summary["baseline_storage_reduction_pct"]:.4f}%** under this conversion. Both graphs store **2,219,626 floating-point values** and both PyTorch architectures have **2,236,682 parameters**. Exported values differ from the trainable parameter count because export folds batch normalization and reuses constants. The 192 graph includes additional resize/graph metadata; the measured byte difference is not a smaller weight count.\n',
        "The 8.506-to-4.288 MiB comparison is valid as original-FP32 versus selected-FP16 portable-file storage. It does not isolate a benefit of resolution reduction, classifier recovery or structural compression. At equal FP16 storage, there is no file-size improvement over the original student. Our 192 changes reduce spatial computation and recover accuracy; they remove no channels or parameters.\n",
        "All accuracy numbers above use the same 27,000-image training pool and preprocessing; they are training-set accuracy. FP16 storage is rounded once, then restored to FP32 for computation. Accuracy must be checked rather than assumed identical. No new latency or energy claim is made in this size experiment.\n"]
    check = summary.get("paired_output_check")
    if check:
        parts.append(f'Paired output check on the existing balanced 500-image sample: **{check["top1_agreement"]*100:.1f}% top-1 agreement**, {check["top1_disagreements"]} changed predictions, maximum absolute logit difference {check["max_absolute_logit_error"]:.6g}. This check is separate from labelled full-data accuracy.\n')
    parts += ["The saved candidate still compiles to **8,937,126 bytes / 8.523 MiB** in the tested S24 Hub path because the compiler expands stored weights to FP32. A fresh baseline through the same compilation path would be required for a deployed-size ratio. Neither portable file size nor compiled payload size establishes runtime RAM consumption. [Device evidence](saved192_ai_hub.md).\n",
        "## Reproduce\n",
        "```powershell\n.venv\\Scripts\\python.exe -m edgeprep.weight_storage --source artifacts/baseline/model_fp32.onnx --output artifacts/baseline_weight16\n.venv\\Scripts\\python.exe -m edgeprep.challenge evaluate --model artifacts/baseline_weight16/model_weight16.onnx --root data --threads 4 --output results/storage_comparison/baseline_weight16_accuracy.json\n.venv\\Scripts\\python.exe scripts/compare_weight_storage.py\n```\n",
        "Read [size, graph and hash audit](../results/storage_comparison/summary.json), [full-data baseline FP16 accuracy](../results/storage_comparison/baseline_weight16_accuracy.json), and [paired output check](../results/storage_comparison/paired_output_check.json). Routine intermediate ONNX files remain local under ignored artifacts/. The preserved winner and its hashes remain unchanged.\n"]
    (ROOT / "docs/matched_weight_storage.md").write_text("\n".join(parts), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    HALF.parent.mkdir(parents=True, exist_ok=True)
    metadata = convert(BASELINE, HALF)
    recipe = read("artifacts/baseline/recipe.json")
    recipe["weight_storage"] = metadata
    (HALF.parent / "recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    original = onnx.load(BASELINE)
    half = onnx.load(HALF)
    restored = {i.name: numpy_helper.to_array(i) for i in half.graph.initializer}
    for init in original.graph.initializer:
        np.testing.assert_array_equal(restored[init.name + "__storage_fp16"], numpy_helper.to_array(init).astype(np.float16))
    assert [n.SerializeToString() for n in original.graph.node] == [n.SerializeToString() for n in half.graph.node if n.op_type != "Cast"]
    check = smoke()
    (OUT / "paired_output_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    rows = [audit(*item) for item in [
        ("baseline_fp32", BASELINE, "results/all_data/baseline_accuracy.json"),
        ("baseline_weight16", HALF, "results/storage_comparison/baseline_weight16_accuracy.json"),
        ("recovered192_fp32", ROOT / "artifacts/all_data/resize192_head_ce/model_fp32.onnx", "results/all_data/resize192_head_ce_accuracy.json"),
        ("recovered192_weight16", WINNER, "results/all_data/resize192_head_ce_w16_accuracy.json"),
    ]]
    assert len({r["stored_float_values"] for r in rows}) == 1
    assert all(read(path)["parameters"] == 2236682 for path in ("artifacts/baseline/recipe.json", "artifacts/all_data/resize192_head_ce/recipe.json"))
    if (OUT / "baseline_weight16_accuracy.json").exists():
        assert read("results/all_data/baseline_accuracy.json")["evaluation_indices_sha256"] == read("results/storage_comparison/baseline_weight16_accuracy.json")["evaluation_indices_sha256"]
    summary = {"updated_utc": datetime.now(timezone.utc).isoformat(), "onnx_version": onnx.__version__, "onnxruntime_version": ort.__version__, "conversion": metadata,
               "baseline_rounding_verified": True, "baseline_operations_unchanged": True,
               "baseline_storage_reduction_pct": (1 - rows[1]["file_bytes"] / rows[0]["file_bytes"]) * 100,
               "candidate_minus_baseline_weight16_bytes": rows[3]["file_bytes"] - rows[1]["file_bytes"],
               "structural_parameter_reduction": 0, "paired_output_check": check, "records": rows}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
