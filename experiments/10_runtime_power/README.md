# 10. Recovered model and worker-thread power tradeoff

Compare default versus sleeping ORT workers on unchanged FP32 and recovered-192 graphs. Three fresh serial repetitions, four threads, identical timing inputs and 30-second CodeCarbon windows. These are estimates. Worker options belong to the runtime and are not enforced by a submitted ONNX file. Accuracy evidence is separate full-data model evaluation.

| name | median_ms | energy_j_per_inference | energy_stdev | latency_ratio_to_matched_baseline | estimated_energy_ratio_to_matched_baseline |
|---|---|---|---|---|---|
| baseline_spin | 2.7892 | 0.054927 | 0.002059 | 1.0 | 1.0 |
| baseline_sleep | 3.78285 | 0.064004 | 0.001806 | 1.356249 | 1.165244 |
| resize192_head_ce_w16_spin | 2.05385 | 0.045338 | 0.002045 | 0.736358 | 0.825413 |
| resize192_head_ce_w16_sleep | 3.4335 | 0.055279 | 0.004107 | 1.230998 | 1.006399 |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **baseline_spin** (`baseline_spin`): [results/energy/runtime/summary.json](../../results/energy/runtime/summary.json); [scripts/runtime_energy_sweep.py](../../scripts/runtime_energy_sweep.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **baseline_sleep** (`baseline_sleep`): [results/energy/runtime/summary.json](../../results/energy/runtime/summary.json); [scripts/runtime_energy_sweep.py](../../scripts/runtime_energy_sweep.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **resize192_head_ce_w16_spin** (`resize192_head_ce_w16_spin`): [results/energy/runtime/summary.json](../../results/energy/runtime/summary.json); [scripts/runtime_energy_sweep.py](../../scripts/runtime_energy_sweep.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **resize192_head_ce_w16_sleep** (`resize192_head_ce_w16_sleep`): [results/energy/runtime/summary.json](../../results/energy/runtime/summary.json); [scripts/runtime_energy_sweep.py](../../scripts/runtime_energy_sweep.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).

## Supporting evidence

- [scripts/runtime_energy_sweep.py](../../scripts/runtime_energy_sweep.py)
- [docs/next_priority_results.md](../../docs/next_priority_results.md)
- [docs/energy_measurement.md](../../docs/energy_measurement.md)
