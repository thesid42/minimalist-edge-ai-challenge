# 12. Direct comparison of 192 model latencies

Fresh rotating-order CPU timing compares unchanged 224, unchanged internal192, recovered192 FP32 and recovered192 FP16 storage. Five serial processes per model, 100 warmups / 1000 timings / four threads, default spinning. No training in parallel. Timing input is one real image; accuracy evidence remains the separate 27,000-image evaluations. FP16 storage still computes FP32; do not attribute noisy small timing differences to half-precision execution.

| label | median_ms | median_min_ms | median_max_ms | p95_ms | replicates | latency_ratio_to_matched_baseline |
|---|---|---|---|---|---|---|
| baseline | 4.89305 | 2.6362 | 5.363 | 8.211165 | 5 | 1.0 |
| resize192 | 2.22175 | 1.96465 | 3.61175 | 5.581005 | 5 | 0.454062 |
| resize192_head_ce | 2.10755 | 1.86515 | 4.2736 | 4.743265 | 5 | 0.430723 |
| resize192_head_ce_w16 | 2.1788 | 1.8374 | 3.8339 | 4.858795 | 5 | 0.445285 |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **baseline** (`latency_baseline`): [['results/latency/resize192/baseline_rep1.json', 'results/latency/resize192/baseline_rep2.json', 'results/latency/resize192/baseline_rep3.json', 'results/latency/resize192/baseline_rep4.json', 'results/latency/resize192/baseline_rep5.json']](../../['results/latency/resize192/baseline_rep1.json', 'results/latency/resize192/baseline_rep2.json', 'results/latency/resize192/baseline_rep3.json', 'results/latency/resize192/baseline_rep4.json', 'results/latency/resize192/baseline_rep5.json']); [scripts/compare_192_latency.py](../../scripts/compare_192_latency.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **resize192** (`latency_resize192`): [['results/latency/resize192/resize192_rep1.json', 'results/latency/resize192/resize192_rep2.json', 'results/latency/resize192/resize192_rep3.json', 'results/latency/resize192/resize192_rep4.json', 'results/latency/resize192/resize192_rep5.json']](../../['results/latency/resize192/resize192_rep1.json', 'results/latency/resize192/resize192_rep2.json', 'results/latency/resize192/resize192_rep3.json', 'results/latency/resize192/resize192_rep4.json', 'results/latency/resize192/resize192_rep5.json']); [scripts/compare_192_latency.py](../../scripts/compare_192_latency.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **resize192_head_ce** (`latency_resize192_head_ce`): [['results/latency/resize192/resize192_head_ce_rep1.json', 'results/latency/resize192/resize192_head_ce_rep2.json', 'results/latency/resize192/resize192_head_ce_rep3.json', 'results/latency/resize192/resize192_head_ce_rep4.json', 'results/latency/resize192/resize192_head_ce_rep5.json']](../../['results/latency/resize192/resize192_head_ce_rep1.json', 'results/latency/resize192/resize192_head_ce_rep2.json', 'results/latency/resize192/resize192_head_ce_rep3.json', 'results/latency/resize192/resize192_head_ce_rep4.json', 'results/latency/resize192/resize192_head_ce_rep5.json']); [scripts/compare_192_latency.py](../../scripts/compare_192_latency.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **resize192_head_ce_w16** (`latency_resize192_head_ce_w16`): [['results/latency/resize192/resize192_head_ce_w16_rep1.json', 'results/latency/resize192/resize192_head_ce_w16_rep2.json', 'results/latency/resize192/resize192_head_ce_w16_rep3.json', 'results/latency/resize192/resize192_head_ce_w16_rep4.json', 'results/latency/resize192/resize192_head_ce_w16_rep5.json']](../../['results/latency/resize192/resize192_head_ce_w16_rep1.json', 'results/latency/resize192/resize192_head_ce_w16_rep2.json', 'results/latency/resize192/resize192_head_ce_w16_rep3.json', 'results/latency/resize192/resize192_head_ce_w16_rep4.json', 'results/latency/resize192/resize192_head_ce_w16_rep5.json']); [scripts/compare_192_latency.py](../../scripts/compare_192_latency.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).

## Supporting evidence

- [scripts/compare_192_latency.py](../../scripts/compare_192_latency.py)
- [results/latency/resize192/summary.json](../../results/latency/resize192/summary.json)
