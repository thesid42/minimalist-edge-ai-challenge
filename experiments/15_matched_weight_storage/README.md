# 15. Baseline and recovered192 at identical weight precision

Apply the same FP16 initializer storage / FP32 restore conversion to the unchanged baseline. Compare original224 and recovered192 at both FP32 and FP16 storage, verify every rounded baseline initializer and unchanged baseline operations, then measure baseline FP16 accuracy on all 27000 images. A paired 500-image output check is separate. The baseline also becomes about 4.28 MiB, so the selected model has no storage advantage at equal precision; resolution and recovery remove no parameters. Source-versus-compiled deployment size remains separate.

| label | file_bytes | size_mib | stored_float_values | accuracy_pct | worst_recall_pct | examples |
|---|---|---|---|---|---|---|
| baseline_fp32 | 8919497 | 8.506295 | 2219626 | 99.162963 | 95.84 | 27000 |
| baseline_weight16 | 4491583 | 4.283507 | 2219626 | 99.166667 | 95.88 | 27000 |
| recovered192_fp32 | 8924421 | 8.510991 | 2219626 | 99.107407 | 97.24 | 27000 |
| recovered192_weight16 | 4496556 | 4.28825 | 2219626 | 99.111111 | 97.24 | 27000 |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **baseline_fp32** (`baseline_fp32`): [results/storage_comparison/summary.json](../../results/storage_comparison/summary.json); [scripts/compare_weight_storage.py](../../scripts/compare_weight_storage.py), [edgeprep/weight_storage.py](../../edgeprep/weight_storage.py).
- **baseline_weight16** (`baseline_weight16`): [results/storage_comparison/summary.json](../../results/storage_comparison/summary.json); [scripts/compare_weight_storage.py](../../scripts/compare_weight_storage.py), [edgeprep/weight_storage.py](../../edgeprep/weight_storage.py).
- **recovered192_fp32** (`recovered192_fp32`): [results/storage_comparison/summary.json](../../results/storage_comparison/summary.json); [scripts/compare_weight_storage.py](../../scripts/compare_weight_storage.py), [edgeprep/weight_storage.py](../../edgeprep/weight_storage.py).
- **recovered192_weight16** (`recovered192_weight16`): [results/storage_comparison/summary.json](../../results/storage_comparison/summary.json); [scripts/compare_weight_storage.py](../../scripts/compare_weight_storage.py), [edgeprep/weight_storage.py](../../edgeprep/weight_storage.py).

## Supporting evidence

- [docs/matched_weight_storage.md](../../docs/matched_weight_storage.md)
- [results/storage_comparison/summary.json](../../results/storage_comparison/summary.json)
- [results/storage_comparison/baseline_weight16_accuracy.json](../../results/storage_comparison/baseline_weight16_accuracy.json)
