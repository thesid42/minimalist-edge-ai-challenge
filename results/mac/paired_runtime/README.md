# Matched Mac CPU results

Apple M4 Pro, macOS-27.0.1-arm64-arm-64bit, CPUExecutionProvider; 4 threads, 3 serial fresh-process repetitions, batch one, 100 warmups, 1,000 timings and separate 30-second energy windows. Power source is recorded in machine.json. External input remains (1,3,224,224). The exact saved winning ONNX was measured without retraining.

Accuracy is a seeded balanced 100-image **diagnostic subset**, not the previously recorded 27,000-image training-set accuracy or held-out accuracy. The timing JSON's one-image accuracy is only a smoke test. Power/energy use CodeCarbon CPU-load/TDP plus RAM heuristics (35 W CPU assumption, GPU excluded), **not hardware-sensor or whole-device readings**. Do not mix these Mac baselines with Windows or remote-device results.

Energy variability warning: a baseline repetition slowed substantially during its energy window. All repetitions are retained; the cause is unverified. Means and sample SD are reported, but a precise energy-saving percentage is not established. Inspect individual run throughput and energy before interpreting the average watts or joules.

| Model / mode | ONNX MiB | Diagnostic accuracy | Median ms | P95 ms | Peak RSS MiB | Estimated W | Estimated J/image, mean +/- SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_spin | 8.506 | 99.0% (100 images) | 2.227 | 2.378 | 108.56 | 12.93 | 0.039579 +/- 0.012240 |
| baseline_sleep | 8.506 | 99.0% (100 images) | 2.664 | 2.876 | 102.56 | 9.33 | 0.029173 +/- 0.006575 |
| candidate_spin | 4.288 | 100.0% (100 images) | 1.871 | 1.969 | 101.86 | 14.69 | 0.028200 +/- 0.000244 |
| candidate_sleep | 4.288 | 100.0% (100 images) | 2.204 | 2.380 | 101.34 | 8.96 | 0.020962 +/- 0.000544 |

After a global Windows result refresh, rerun `.venv/bin/python scripts/record_mac_results.py` to restore this separate Mac section in the main README, log and notebook. The new notebook report cell is verified standalone; prior notebook cells are not re-executed.

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
