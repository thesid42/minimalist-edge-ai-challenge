# 09. Repeated CPU latency and estimated energy

CodeCarbon is the PDF's requested tool. This Windows 10 AMD laptop uses TDP/load estimation. Serial fresh processes, identical weights, 1/2/4/8 threads, three 30-second repetitions. Raw CPU/RAM power and energy CSVs are retained; GPU excluded. Timing uses one real image, dataset accuracy is separate.

| threads | replicates | median_latency_ms | mean_estimated_j_per_inference | energy_stdev | latency_ratio_to_matched_4t | estimated_energy_ratio_to_matched_4t |
|---|---|---|---|---|---|---|
| 1 | 3 | 7.28175 | 0.097657 | 0.000571 | 2.674066 | 1.814572 |
| 2 | 3 | 3.96235 | 0.064672 | 0.001823 | 1.455088 | 1.20167 |
| 4 | 3 | 2.7231 | 0.053818 | 0.000858 | 1.0 | 1.0 |
| 8 | 3 | 2.85495 | 0.085025 | 0.004111 | 1.048419 | 1.579852 |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **FP32 CPU 1 threads** (`cpu_threads_1`): [results/energy/thread_sweep/summary.json](../../results/energy/thread_sweep/summary.json); [scripts/thread_energy_sweep.py](../../scripts/thread_energy_sweep.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **FP32 CPU 2 threads** (`cpu_threads_2`): [results/energy/thread_sweep/summary.json](../../results/energy/thread_sweep/summary.json); [scripts/thread_energy_sweep.py](../../scripts/thread_energy_sweep.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **FP32 CPU 4 threads** (`cpu_threads_4`): [results/energy/thread_sweep/summary.json](../../results/energy/thread_sweep/summary.json); [scripts/thread_energy_sweep.py](../../scripts/thread_energy_sweep.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).
- **FP32 CPU 8 threads** (`cpu_threads_8`): [results/energy/thread_sweep/summary.json](../../results/energy/thread_sweep/summary.json); [scripts/thread_energy_sweep.py](../../scripts/thread_energy_sweep.py), [edgeprep/benchmark.py](../../edgeprep/benchmark.py).

## Supporting evidence

- [docs/energy_measurement.md](../../docs/energy_measurement.md)
- [results/energy/thread_sweep/summary.json](../../results/energy/thread_sweep/summary.json)
