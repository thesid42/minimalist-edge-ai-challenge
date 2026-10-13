# 06. Real-device CPU/NPU deployment

Authenticated using the ignored .env key. Retrieved 78 catalogue entries. S24 CPU and NPU exploration, plus RB3 Hub-native W8A8 compilation. First S24 NPU inference failed because compute_unit and an explicit provider were both specified; the retry removes compute_unit from inference options. Remote 32-image agreement is a smoke check. New dataset_accuracy.json files measure labelled accuracy on 1,000 balanced images, separately from all-data laptop accuracy. Preserve failures alongside successful runs.

| name | status | device | precision | performance_mode | hub_estimated_latency_ms | accuracy_pct | worst_recall_pct | labelled_samples | labelled_top1_agreement_pct |
|---|---|---|---|---|---|---|---|---|---|
| s24_cpu_fp32 | completed | Samsung Galaxy S24 (Family) | none | burst | 5.533 | 98.8 | 90.0 | 1000.0 | 100.0 |
| s24_npu_float | failed | Samsung Galaxy S24 (Family) | none | burst | pending / n/a | pending / n/a | pending / n/a | pending / n/a | pending / n/a |
| s24_npu_float_retry | completed | Samsung Galaxy S24 (Family) | none | burst | 0.366 | 98.8 | 90.0 | 1000.0 | 100.0 |
| rb3_npu_w8a8 | failed | Dragonwing RB3 Gen 2 Vision Kit | w8a8 | burst | pending / n/a | pending / n/a | pending / n/a | pending / n/a | pending / n/a |
| rb3_onnx_npu_w8a8 | completed | Dragonwing RB3 Gen 2 Vision Kit | w8a8 | burst | 0.614 | 98.3 | 89.0 | 1000.0 | 99.3 |
| rb3_onnx_npu_w8a16 | completed | Dragonwing RB3 Gen 2 Vision Kit | w8a16 | burst | 1.056 | 98.5 | 88.0 | 1000.0 | 99.5 |
| s24_npu_float_retry_balanced | completed | Samsung Galaxy S24 (Family) | none | balanced | 0.778 | 98.8 | 90.0 | 1000.0 | 100.0 |
| s24_npu_float_retry_power_saver | completed | Samsung Galaxy S24 (Family) | none | power_saver | 1.192 | 98.8 | 90.0 | 1000.0 | 100.0 |
| rb3_onnx_npu_w8a8_balanced | completed | Dragonwing RB3 Gen 2 Vision Kit | w8a8 | balanced | 0.648 | 98.3 | 89.0 | 1000.0 | 99.3 |
| rb3_onnx_npu_w8a8_power_saver | completed | Dragonwing RB3 Gen 2 Vision Kit | w8a8 | power_saver | 0.68 | 98.3 | 89.0 | 1000.0 | 99.3 |

All latency values in this table are Hub estimates. S24 CPU/NPU comparisons use the same source; numerical agreement retains the 32-image smoke check; labelled accuracy columns use the new 1,000-image diagnostic sample. Actual placement, memory ranges, failure reasons and job IDs are in details.json.

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **s24_cpu_fp32** (`s24_cpu_fp32`): [results/hub/s24_cpu_fp32](../../results/hub/s24_cpu_fp32); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py), [edgeprep/hub_accuracy.py](../../edgeprep/hub_accuracy.py), [edgeprep/hub_modes.py](../../edgeprep/hub_modes.py).
- **s24_npu_float** (`s24_npu_float`): [results/hub/s24_npu_float](../../results/hub/s24_npu_float); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py), [edgeprep/hub_accuracy.py](../../edgeprep/hub_accuracy.py), [edgeprep/hub_modes.py](../../edgeprep/hub_modes.py).
- **s24_npu_float_retry** (`s24_npu_float_retry`): [results/hub/s24_npu_float_retry](../../results/hub/s24_npu_float_retry); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py), [edgeprep/hub_accuracy.py](../../edgeprep/hub_accuracy.py), [edgeprep/hub_modes.py](../../edgeprep/hub_modes.py).
- **rb3_npu_w8a8** (`rb3_npu_w8a8`): [results/hub/rb3_npu_w8a8](../../results/hub/rb3_npu_w8a8); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py), [edgeprep/hub_accuracy.py](../../edgeprep/hub_accuracy.py), [edgeprep/hub_modes.py](../../edgeprep/hub_modes.py).
- **rb3_onnx_npu_w8a8** (`rb3_onnx_npu_w8a8`): [results/hub/rb3_onnx_npu_w8a8](../../results/hub/rb3_onnx_npu_w8a8); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py), [edgeprep/hub_accuracy.py](../../edgeprep/hub_accuracy.py), [edgeprep/hub_modes.py](../../edgeprep/hub_modes.py).
- **rb3_onnx_npu_w8a16** (`rb3_onnx_npu_w8a16`): [results/hub/rb3_onnx_npu_w8a16](../../results/hub/rb3_onnx_npu_w8a16); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py), [edgeprep/hub_accuracy.py](../../edgeprep/hub_accuracy.py), [edgeprep/hub_modes.py](../../edgeprep/hub_modes.py).
- **s24_npu_float_retry_balanced** (`s24_npu_float_retry_balanced`): [results/hub/s24_npu_float_retry_balanced](../../results/hub/s24_npu_float_retry_balanced); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py), [edgeprep/hub_accuracy.py](../../edgeprep/hub_accuracy.py), [edgeprep/hub_modes.py](../../edgeprep/hub_modes.py).
- **s24_npu_float_retry_power_saver** (`s24_npu_float_retry_power_saver`): [results/hub/s24_npu_float_retry_power_saver](../../results/hub/s24_npu_float_retry_power_saver); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py), [edgeprep/hub_accuracy.py](../../edgeprep/hub_accuracy.py), [edgeprep/hub_modes.py](../../edgeprep/hub_modes.py).
- **rb3_onnx_npu_w8a8_balanced** (`rb3_onnx_npu_w8a8_balanced`): [results/hub/rb3_onnx_npu_w8a8_balanced](../../results/hub/rb3_onnx_npu_w8a8_balanced); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py), [edgeprep/hub_accuracy.py](../../edgeprep/hub_accuracy.py), [edgeprep/hub_modes.py](../../edgeprep/hub_modes.py).
- **rb3_onnx_npu_w8a8_power_saver** (`rb3_onnx_npu_w8a8_power_saver`): [results/hub/rb3_onnx_npu_w8a8_power_saver](../../results/hub/rb3_onnx_npu_w8a8_power_saver); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py), [edgeprep/hub_accuracy.py](../../edgeprep/hub_accuracy.py), [edgeprep/hub_modes.py](../../edgeprep/hub_modes.py).

## Supporting evidence

- [docs/ai_hub.md](../../docs/ai_hub.md)
- [results/hub/devices.json](../../results/hub/devices.json)
- [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py)
