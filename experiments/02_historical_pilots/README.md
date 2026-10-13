# 02. Historical compression and short recovery

Before the organizer clarification, 13 variants were screened on 500 images from a provisional split. The table follows the experimental progression, not reconstructed timestamps. Full quantization, selective quantization, physical pruning, and short CE/KD recovery are different operations. CE/KD used only 500 training images for one epoch. These rows cannot be mixed with current all-data ratios.

| label | accuracy_pct | worst_recall_pct | size_mib | median_cpu_4t_ms | estimated_cpu_4t_j | examples |
|---|---|---|---|---|---|---|
| Unchanged FP32 student | 99.2 | 96.0 | 8.5063 | 2.1668 | pending / n/a | 500 |
| Full INT8, signed / MinMax 128 | 92.6 | 66.0 | 2.5256 | 5.6622 | pending / n/a | 500 |
| Full INT8, unsigned / MinMax 128 | 91.4 | 58.0 | 2.5414 | 4.3796 | pending / n/a | 500 |
| Full INT8 / MinMax 256 | 91.0 | 66.0 | 2.5257 | 5.2509 | pending / n/a | 500 |
| Full INT8 / Percentile 256 | 96.2 | 80.0 | 2.5255 | 5.7884 | pending / n/a | 500 |
| INT8 with first/last/head protected | 96.0 | 90.0 | 3.7159 | 6.0622 | pending / n/a | 500 |
| Selective INT8 / MinMax 256 | 98.4 | 94.0 | 3.772 | 5.3334 | pending / n/a | 500 |
| Uniform channel pruning 5% | 16.0 | 0.0 | 7.6219 | 1.84 | pending / n/a | 500 |
| Uniform channel pruning 10% | 17.2 | 0.0 | 6.9353 | 1.6972 | pending / n/a | 500 |
| Expansion-only pruning 5% | 83.2 | 46.0 | 8.1381 | 2.633 | pending / n/a | 500 |
| Expansion-only pruning 10% | 51.0 | 0.0 | 7.7957 | 2.7552 | pending / n/a | 500 |
| Expansion 5% + short CE recovery | 95.6 | 76.0 | 8.1381 | 2.1636 | pending / n/a | 500 |
| Expansion 5% + short teacher KD recovery | 96.0 | 82.0 | 8.1381 | 2.0575 | pending / n/a | 500 |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **Unchanged FP32 student** (`baseline`): [results/challenge/baseline_accuracy.json](../../results/challenge/baseline_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **Full INT8, signed / MinMax 128** (`qdq_s8`): [results/challenge/qdq_s8_accuracy.json](../../results/challenge/qdq_s8_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **Full INT8, unsigned / MinMax 128** (`qdq_u8`): [results/challenge/qdq_u8_accuracy.json](../../results/challenge/qdq_u8_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **Full INT8 / MinMax 256** (`qdq_s8_256`): [results/challenge/qdq_s8_256_accuracy.json](../../results/challenge/qdq_s8_256_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **Full INT8 / Percentile 256** (`qdq_percentile`): [results/challenge/qdq_percentile_accuracy.json](../../results/challenge/qdq_percentile_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **INT8 with first/last/head protected** (`qdq_protected`): [results/challenge/qdq_protected_accuracy.json](../../results/challenge/qdq_protected_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **Selective INT8 / MinMax 256** (`qdq_pointwise`): [results/challenge/qdq_pointwise_accuracy.json](../../results/challenge/qdq_pointwise_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **Uniform channel pruning 5%** (`prune05`): [results/challenge/prune05_accuracy.json](../../results/challenge/prune05_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **Uniform channel pruning 10%** (`prune10`): [results/challenge/prune10_accuracy.json](../../results/challenge/prune10_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **Expansion-only pruning 5%** (`expansion05`): [results/challenge/expansion05_accuracy.json](../../results/challenge/expansion05_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **Expansion-only pruning 10%** (`expansion10`): [results/challenge/expansion10_accuracy.json](../../results/challenge/expansion10_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **Expansion 5% + short CE recovery** (`expansion05_ce`): [results/challenge/expansion05_ce_accuracy.json](../../results/challenge/expansion05_ce_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **Expansion 5% + short teacher KD recovery** (`expansion05_kd`): [results/challenge/expansion05_kd_accuracy.json](../../results/challenge/expansion05_kd_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).

## Supporting evidence

- [docs/historical_pilot_analysis.md](../../docs/historical_pilot_analysis.md)
- [results/challenge/summary.json](../../results/challenge/summary.json)
