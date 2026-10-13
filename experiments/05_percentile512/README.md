# 05. Selective INT8 with 512 Percentile samples

Same selective precision pattern, now Percentile calibration and 512 balanced samples. This changes two knobs together; it screens a configuration but does not isolate the benefit of either algorithm or sample count. Full-data evaluation and fresh runtime measurements recorded when present.

| label | accuracy_pct | worst_recall_pct | size_mib | median_cpu_4t_ms | estimated_cpu_4t_j | examples |
|---|---|---|---|---|---|---|
| Selective INT8 / Percentile 512 | 98.8037 | 95.36 | 3.7719 | 5.5206 | 0.1322 | 27000 |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **Selective INT8 / Percentile 512** (`qdq_percentile512`): [results/all_data/qdq_percentile512_accuracy.json](../../results/all_data/qdq_percentile512_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).

## Supporting evidence

- [results/all_data/calibration512/manifest.json](../../results/all_data/calibration512/manifest.json)
- [edgeprep/challenge.py](../../edgeprep/challenge.py)
