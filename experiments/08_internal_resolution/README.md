# 08. Lower spatial computation, fixed external input

PDF section 3 permits resolution reduction while keeping input (1,3,224,224). Resize inside ONNX to 192 or 160, bilinear align_corners=False without antialias, then run unchanged supplied weights. Initial variants have no training or parameter removal. A separate 192 classifier-only CE recovery trains 12,810 head parameters for one epoch on all 27,000 images, keeping backbone and BN statistics frozen. Evaluate all 27,000 images; measure exported ONNX and matched energy. THOP omits bilinear resize overhead. An additional candidate stores FP16 initializers with explicit Cast-to-FP32 nodes: half-sized weight storage, FP32 CPU compute, no inferred RAM saving.

| label | accuracy_pct | worst_recall_pct | size_mib | median_cpu_4t_ms | estimated_cpu_4t_j | examples |
|---|---|---|---|---|---|---|
| Internal 192 resize, unchanged weights | 98.0 | 93.8 | 8.511 | 2.1325 | 0.0518 | 27000 |
| Internal 160 resize, unchanged weights | 93.0037 | 78.05 | 8.511 | 1.5622 | 0.0352 | 27000 |
| Internal 192 + full-data classifier recovery | 99.1074 | 97.24 | 8.511 | pending / n/a | pending / n/a | 27000 |
| Recovered 192 + FP16 weight storage / FP32 compute | 99.1111 | 97.24 | 4.2882 | 2.0538 | 0.0453 | 27000 |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **Internal 192 resize, unchanged weights** (`resize192`): [results/all_data/resize192_accuracy.json](../../results/all_data/resize192_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py), [edgeprep/resolution.py](../../edgeprep/resolution.py).
- **Internal 160 resize, unchanged weights** (`resize160`): [results/all_data/resize160_accuracy.json](../../results/all_data/resize160_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py), [edgeprep/resolution.py](../../edgeprep/resolution.py).
- **Internal 192 + full-data classifier recovery** (`resize192_head_ce`): [results/all_data/resize192_head_ce_accuracy.json](../../results/all_data/resize192_head_ce_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py), [edgeprep/resolution.py](../../edgeprep/resolution.py).
- **Recovered 192 + FP16 weight storage / FP32 compute** (`resize192_head_ce_w16`): [results/all_data/resize192_head_ce_w16_accuracy.json](../../results/all_data/resize192_head_ce_w16_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py), [edgeprep/resolution.py](../../edgeprep/resolution.py), [edgeprep/weight_storage.py](../../edgeprep/weight_storage.py).

## Supporting evidence

- [edgeprep/resolution.py](../../edgeprep/resolution.py)
- [docs/next_priority_results.md](../../docs/next_priority_results.md)
