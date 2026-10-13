# Matched Mac CPU results

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

After a global Windows result refresh, rerun `.venv/bin/python scripts/record_mac_results.py --runtime-dir results/mac/full_runtime --accuracy-dir results/mac/eval27000` to restore this separate Mac section in the main README, log and notebook. The new notebook report cell is verified standalone; prior notebook cells are not re-executed.
