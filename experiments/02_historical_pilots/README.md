# 02. Historical compression and short recovery

Before the organizer clarification, 13 variants were screened on 500 images from a provisional split. The table follows the experimental progression, not reconstructed timestamps. Full quantization, selective quantization, physical pruning, and short CE/KD recovery are different operations. CE/KD used only 500 training images for one epoch. These rows cannot be mixed with current all-data ratios.

| name | accuracy_pct | worst_recall_pct | size_mib | median_cpu_4t_ms | examples |
|---|---|---|---|---|---|
| baseline | 99.2 | 96.0 | 8.5063 | 2.1668 | 500 |
| qdq_s8 | 92.6 | 66.0 | 2.5256 | 5.6622 | 500 |
| qdq_u8 | 91.4 | 58.0 | 2.5414 | 4.3796 | 500 |
| qdq_s8_256 | 91.0 | 66.0 | 2.5257 | 5.2509 | 500 |
| qdq_percentile | 96.2 | 80.0 | 2.5255 | 5.7884 | 500 |
| qdq_protected | 96.0 | 90.0 | 3.7159 | 6.0622 | 500 |
| qdq_pointwise | 98.4 | 94.0 | 3.772 | 5.3334 | 500 |
| prune05 | 16.0 | 0.0 | 7.6219 | 1.84 | 500 |
| prune10 | 17.2 | 0.0 | 6.9353 | 1.6972 | 500 |
| expansion05 | 83.2 | 46.0 | 8.1381 | 2.633 | 500 |
| expansion10 | 51.0 | 0.0 | 7.7957 | 2.7552 | 500 |
| expansion05_ce | 95.6 | 76.0 | 8.1381 | 2.1636 | 500 |
| expansion05_kd | 96.0 | 82.0 | 8.1381 | 2.0575 | 500 |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **baseline**: [results/challenge/baseline_accuracy.json](../../results/challenge/baseline_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **qdq_s8**: [results/challenge/qdq_s8_accuracy.json](../../results/challenge/qdq_s8_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **qdq_u8**: [results/challenge/qdq_u8_accuracy.json](../../results/challenge/qdq_u8_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **qdq_s8_256**: [results/challenge/qdq_s8_256_accuracy.json](../../results/challenge/qdq_s8_256_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **qdq_percentile**: [results/challenge/qdq_percentile_accuracy.json](../../results/challenge/qdq_percentile_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **qdq_protected**: [results/challenge/qdq_protected_accuracy.json](../../results/challenge/qdq_protected_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **qdq_pointwise**: [results/challenge/qdq_pointwise_accuracy.json](../../results/challenge/qdq_pointwise_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **prune05**: [results/challenge/prune05_accuracy.json](../../results/challenge/prune05_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **prune10**: [results/challenge/prune10_accuracy.json](../../results/challenge/prune10_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **expansion05**: [results/challenge/expansion05_accuracy.json](../../results/challenge/expansion05_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **expansion10**: [results/challenge/expansion10_accuracy.json](../../results/challenge/expansion10_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **expansion05_ce**: [results/challenge/expansion05_ce_accuracy.json](../../results/challenge/expansion05_ce_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **expansion05_kd**: [results/challenge/expansion05_kd_accuracy.json](../../results/challenge/expansion05_kd_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).

## Supporting evidence

- [docs/historical_pilot_analysis.md](../../docs/historical_pilot_analysis.md)
- [results/challenge/summary.json](../../results/challenge/summary.json)
