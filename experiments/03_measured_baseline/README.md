# 03. Our current unchanged-student baseline

After the organizer clarified all-data training and our own laptop reference, evaluated the unchanged student on all 27,000 images. This is training-set accuracy. Exported FP32 ONNX and measured serial CPU runtime/RAM/estimated energy. No recovery or retraining. The table can display newer matched three-repetition timings; baseline_metrics.json preserves the original 20-second reference measurement.

| label | accuracy_pct | worst_recall_pct | size_mib | median_cpu_4t_ms | estimated_cpu_4t_j | examples |
|---|---|---|---|---|---|---|
| Unchanged FP32 student | 99.163 | 95.84 | 8.5063 | 2.7892 | 0.0549 | 27000 |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **Unchanged FP32 student** (`baseline`): [results/all_data/baseline_accuracy.json](../../results/all_data/baseline_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).

## Supporting evidence

- [baseline_metrics.json](../../baseline_metrics.json)
- [scripts/measure_all_data.ps1](../../scripts/measure_all_data.ps1)
