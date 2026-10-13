# 06. Real-device CPU/NPU deployment

Authenticated using the ignored .env key. Retrieved 78 catalogue entries. S24 CPU and NPU exploration, plus RB3 Hub-native W8A8 compilation. First S24 NPU inference failed because compute_unit and an explicit provider were both specified; the retry removes compute_unit from inference options. Remote 32-image agreement is a smoke check, never full-data accuracy. Preserve failures alongside successful runs.

| name | status | device | precision | hub_estimated_latency_ms | top1_agreement_pct |
|---|---|---|---|---|---|
| s24_cpu_fp32 | completed | Samsung Galaxy S24 (Family) | none | 5.533 | 100.0 |
| s24_npu_float | failed | Samsung Galaxy S24 (Family) | none | pending / n/a | pending / n/a |
| s24_npu_float_retry | completed | Samsung Galaxy S24 (Family) | none | 0.366 | 100.0 |
| rb3_npu_w8a8 | failed | Dragonwing RB3 Gen 2 Vision Kit | w8a8 | pending / n/a | pending / n/a |
| rb3_onnx_npu_w8a8 | pending | Dragonwing RB3 Gen 2 Vision Kit | w8a8 | pending / n/a | pending / n/a |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **s24_cpu_fp32**: [results/hub/s24_cpu_fp32](../../results/hub/s24_cpu_fp32); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py).
- **s24_npu_float**: [results/hub/s24_npu_float](../../results/hub/s24_npu_float); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py).
- **s24_npu_float_retry**: [results/hub/s24_npu_float_retry](../../results/hub/s24_npu_float_retry); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py).
- **rb3_npu_w8a8**: [results/hub/rb3_npu_w8a8](../../results/hub/rb3_npu_w8a8); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py).
- **rb3_onnx_npu_w8a8**: [results/hub/rb3_onnx_npu_w8a8](../../results/hub/rb3_onnx_npu_w8a8); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py).

## Supporting evidence

- [docs/ai_hub.md](../../docs/ai_hub.md)
- [results/hub/devices.json](../../results/hub/devices.json)
- [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py)
