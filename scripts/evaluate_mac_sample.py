"""Stream a EuroSAT subset or the complete dataset through both CPU ONNX models."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from edgeprep.challenge import evaluate
from edgeprep.challenge_data import dataset


def evaluation_details(count, total):
    if not 0 < count <= total:
        raise ValueError("Evaluation count must be within the dataset")
    full = count == total
    return {"evaluation_scope": "training_set" if full else "training_pool_diagnostic_subset",
            "full_dataset_evaluated": full,
            "accuracy_caveat": ("Complete training-pool accuracy; not held-out generalization" if full else
                                "Balanced subset of the training pool; not full-dataset or held-out generalization")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=15000)
    parser.add_argument("--data", default="data/EuroSAT_RGB")
    parser.add_argument("--baseline", default="artifacts/mac/baseline/model_fp32.onnx")
    parser.add_argument("--candidate", default="best_models/resize192_head_ce_w16_e391533162d7/model_weight16.onnx")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="results/mac/eval15000")
    args = parser.parse_args()
    if args.count < 1 or args.threads < 1:
        parser.error("Count and CPU threads must be positive")
    data = Path(args.data).resolve()
    models = {name: Path(getattr(args, name)).resolve() for name in ("baseline", "candidate")}
    if not data.is_dir() or any(not p.is_file() for p in models.values()):
        parser.error("Dataset directory and both ONNX files must exist")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("*_accuracy.json")) or (out / "summary.json").exists():
        parser.error("Choose a fresh output directory; existing results are preserved")
    torch.set_num_threads(args.threads)
    # Adapt the local ImageFolder layout without moving/copying the dataset or
    # modifying the shared challenge pipeline. Cleanup removes only this temp link.
    with tempfile.TemporaryDirectory(prefix="edgeprep-mac-eval-") as temporary:
        adapter = Path(temporary) / "eurosat"
        adapter.mkdir()
        (adapter / "2750").symlink_to(data, target_is_directory=True)
        ds = dataset(temporary)
        if args.count > len(ds):
            parser.error(f"Count exceeds the {len(ds)} available images")
        scope = evaluation_details(args.count, len(ds))
        records = []
        for name, model in models.items():
            path = out / f"{name}_accuracy.json"
            print(f"Evaluating {name}: {args.count:,} images on CPU", flush=True)
            evaluate(argparse.Namespace(root=temporary, split=None, limit=args.count,
                                        threads=args.threads, model=str(model), output=str(path)))
            score = json.loads(path.read_text())
            score.update(**scope, seed=42,
                         correct_predictions=sum(score["confusion_matrix"][i][i] for i in range(10)),
                         dataset_root=str(data))
            path.write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
            records.append({k: v for k, v in score.items() if k not in ("evaluation_indices", "confusion_matrix")} | {"name": name})
        if records[0]["evaluation_indices_sha256"] != records[1]["evaluation_indices_sha256"]:
            raise ValueError("Baseline and candidate evaluation samples differ")
        summary = {"requested_examples": args.count, "available_examples": len(ds),
                   "class_to_idx": ds.class_to_idx, "provider": "CPUExecutionProvider", "threads": args.threads,
                   "runtime_measurements": "Accuracy only; no new latency or energy benchmark", "records": records,
                   **scope}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        manifest = {"evaluation_count": args.count, "class_support": records[0]["class_support"],
                    "class_to_idx": ds.class_to_idx, "available_examples": len(ds), "seed": 42,
                    "evaluation_indices_sha256": records[0]["evaluation_indices_sha256"],
                    "selection": "Seeded balanced interleaved selection from all available EuroSAT images",
                    "scope": scope["evaluation_scope"], "full_dataset_evaluated": scope["full_dataset_evaluated"]}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
