# 13. Matched Mac CPU measurement

Apple M4 Pro, macOS-27.0.1-arm64-arm-64bit, CPUExecutionProvider; 4 threads, 3 serial fresh-process repetitions, batch one, 100 warmups, 1,000 timings and separate 30-second energy windows. Power source is recorded in machine.json. External input remains (1,3,224,224). The exact saved winning ONNX was measured without retraining.

Accuracy is a seeded balanced 100-image **diagnostic subset**, not the previously recorded 27,000-image training-set accuracy or held-out accuracy. The timing JSON's one-image accuracy is only a smoke test. Power/energy use CodeCarbon CPU-load/TDP plus RAM heuristics (35 W CPU assumption, GPU excluded), **not hardware-sensor or whole-device readings**. Do not mix these Mac baselines with Windows or remote-device results.

Energy variability warning: a baseline repetition slowed substantially during its energy window. All repetitions are retained; the cause is unverified. Means and sample SD are reported, but a precise energy-saving percentage is not established. Inspect individual run throughput and energy before interpreting the average watts or joules.

| Model / mode | ONNX MiB | Diagnostic accuracy | Median ms | P95 ms | Peak RSS MiB | Estimated W | Estimated J/image, mean +/- SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_spin | 8.506 | 99.0% (100 images) | 2.227 | 2.378 | 108.56 | 12.93 | 0.039579 +/- 0.012240 |
| baseline_sleep | 8.506 | 99.0% (100 images) | 2.664 | 2.876 | 102.56 | 9.33 | 0.029173 +/- 0.006575 |
| candidate_spin | 4.288 | 100.0% (100 images) | 1.871 | 1.969 | 101.86 | 14.69 | 0.028200 +/- 0.000244 |
| candidate_sleep | 4.288 | 100.0% (100 images) | 2.204 | 2.380 | 101.34 | 8.96 | 0.020962 +/- 0.000544 |

[Raw evidence](../../results/mac/paired_runtime/summary.json); [exact settings and hashes](details.json).

<!-- BEGIN MAC FULL CPU RESULTS -->
## Mac CPU metrics with full-dataset accuracy

Apple M4 Pro, macOS-27.0.1-arm64-arm-64bit, CPUExecutionProvider; 4 threads, 3 serial fresh-process repetitions, batch one, 100 warmups, 1,000 timings and separate 30-second energy windows. Power source is recorded in machine.json. External input remains (1,3,224,224). The exact saved winning ONNX was measured without retraining.

Accuracy covers all 27,000 images with their original class counts: training-set accuracy, not held-out generalization. The timing JSON's one-image accuracy is only a smoke test. Power/energy use CodeCarbon CPU-load/TDP plus RAM heuristics (35 W CPU assumption, GPU excluded), **not hardware-sensor or whole-device readings**. Do not mix these Mac baselines with Windows or remote-device results.

Power source at benchmark start: Now drawing from 'Battery Power'.

Accuracy is measured across the entire dataset; runtime uses the same fixed real timing image for both models. Timed inference excludes disk reads, preprocessing and model loading. Projected dataset energy is estimated J/inference multiplied by the image count, not a metered full-dataset run. CPU percentages use one core = 100% (four busy cores can reach 400%).

Aggregation: medians of run median/P95/load times, mean estimated watts and J/inference, sample SD for energy, maximum inference-session RSS and RAM delta. RSS excludes later CodeCarbon imports. Model load time is reported separately from warmed-up inference latency.

| Model / mode | ONNX MiB | Full-data accuracy | Median ms | P95 ms | Peak RSS MiB | Estimated W | Estimated J/image, mean +/- SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_spin | 8.506 | 99.1630% (27,000 images) | 2.258 | 2.420 | 109.11 | 14.46 | 0.033765 +/- 0.000646 |
| baseline_sleep | 8.506 | 99.1630% (27,000 images) | 2.604 | 2.797 | 107.77 | 9.31 | 0.024802 +/- 0.000263 |
| candidate_spin | 4.288 | 99.1111% (27,000 images) | 1.867 | 1.962 | 105.81 | 14.52 | 0.027979 +/- 0.000356 |
| candidate_sleep | 4.288 | 99.1111% (27,000 images) | 2.295 | 2.402 | 104.12 | 9.05 | 0.021092 +/- 0.000643 |

| Model / mode | Worst-class recall | Load ms | Session RAM delta MiB | CPU use, one-core % | Projected dataset inference energy J |
|---|---:|---:|---:|---:|---:|
| baseline_spin | 95.84% | 9.396 | 61.22 | 399.5 | 911.66 |
| baseline_sleep | 95.84% | 8.028 | 59.55 | 251.9 | 669.65 |
| candidate_spin | 97.24% | 12.048 | 57.88 | 398.2 | 755.44 |
| candidate_sleep | 97.24% | 11.056 | 56.52 | 243.8 | 569.49 |

[Raw evidence](../../results/mac/full_runtime/summary.json); [exact settings and hashes](details_full_runtime.json).

<!-- END MAC FULL CPU RESULTS -->

<!-- BEGIN MAC HALF SAMPLE RESULTS -->
### 50-image evaluation

50 images, 5 per class. The first half of the earlier 100-image sample was used. Both models ran on CPU with four threads. Input resolution and model weights are unchanged. This is a diagnostic training-pool subset, not held-out or full-dataset accuracy. Earlier subset scores remain intact.

| Model | Correct / images | Accuracy |
|---|---:|---:|
| baseline | 50 / 50 | 100.0000% |
| candidate | 50 / 50 | 100.0000% |

Only accuracy was rechecked; power and latency were not remeasured. Use the separate repeated CPU measurements for latency and energy.
<!-- END MAC HALF SAMPLE RESULTS -->

<!-- BEGIN MAC 15000 SAMPLE RESULTS -->
### 15,000-image evaluation

15,000 images, 1,500 per class. The same seeded image indices were used for both models. Both models ran on CPU with four threads. Input resolution and model weights are unchanged. This is a diagnostic training-pool subset, not held-out or full-dataset accuracy. Earlier subset scores remain intact.

| Model | Correct / images | Accuracy | Worst-class recall |
|---|---:|---:|---:|
| baseline | 14,864 / 15,000 | 99.0933% | 95.80% |
| candidate | 14,855 / 15,000 | 99.0333% | 97.20% |

Only accuracy was rechecked; power and latency were not remeasured. Use the separate repeated CPU measurements for latency and energy.
<!-- END MAC 15000 SAMPLE RESULTS -->

<!-- BEGIN MAC FULL DATA RESULTS -->
### 27,000-image evaluation

All 27,000 images were evaluated, preserving the dataset's original unequal class counts. The same seeded image indices were used for both models. Both models ran on CPU with four threads. Input resolution and model weights are unchanged. This is complete training-set accuracy, not held-out generalization. Earlier subset scores remain intact.

| Model | Correct / images | Accuracy | Worst-class recall |
|---|---:|---:|---:|
| baseline | 26,774 / 27,000 | 99.1630% | 95.84% |
| candidate | 26,760 / 27,000 | 99.1111% | 97.24% |

Only accuracy was rechecked; power and latency were not remeasured. Use the separate repeated CPU measurements for latency and energy.
<!-- END MAC FULL DATA RESULTS -->
