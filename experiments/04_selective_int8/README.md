# 04. Selective INT8 with 256 MinMax samples

Quantized pointwise convolutions while preserving first/last layers, classifier and depthwise convolutions in FP32. Same learned parameters; smaller weight storage. Calibrated 256 balanced samples and evaluated all 27,000. Smaller does not mean faster: this graph regressed on our laptop CPU.

| label | accuracy_pct | worst_recall_pct | size_mib | median_cpu_4t_ms | examples |
|---|---|---|---|---|---|
| Selective INT8 / MinMax 256 | 98.4111 | 95.4333 | 3.772 | 5.1842 | 27000 |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **Selective INT8 / MinMax 256** (`qdq_pointwise`): [results/all_data/qdq_pointwise_accuracy.json](../../results/all_data/qdq_pointwise_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/export.py](../../edgeprep/export.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).

## Supporting evidence

- [ANALYSIS.md](../../ANALYSIS.md)
- [edgeprep/export.py](../../edgeprep/export.py)
