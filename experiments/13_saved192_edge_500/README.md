# 13. Saved 192 FP16-storage model on an edge phone

Test the exact saved winner on Samsung Galaxy S24 (Snapdragon 8 Gen 3), separately on CPU and QNN NPU, without additional quantization. Compile from the saved FP16-storage source; inspect compiler output size and actual operation placement. Accuracy uses 500 balanced real images (50/class), disjoint from 512 calibration images, with a local reference from this exact model. Separate microbenchmark profiles give steady inference latency, excluding upload/preprocessing. This is a training-pool diagnostic sample, not 27000-image or held-out accuracy. Jobs can route to separate units of the same named phone.

| name | status | device | precision | performance_mode | hub_estimated_latency_ms | accuracy_pct | worst_recall_pct | labelled_samples | labelled_top1_agreement_pct |
|---|---|---|---|---|---|---|---|---|---|
| resize192_s24_cpu | completed | Samsung Galaxy S24 | none | burst | 4.183 | 99.4 | 96.0 | 500.0 | 100.0 |
| resize192_s24_npu | pending | Samsung Galaxy S24 | none | burst | 0.435 | pending / n/a | pending / n/a | pending / n/a | pending / n/a |

Labelled sample count is 500. Precision 'none' means no extra Hub quantization; the source already has FP16 weight storage with FP32 restores. Read the source/compiled hashes and deployment asset sizes before claiming precision or compression on the device.

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **resize192_s24_cpu** (`resize192_s24_cpu`): [results/hub/resize192_s24_cpu](../../results/hub/resize192_s24_cpu); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py), [edgeprep/hub_accuracy.py](../../edgeprep/hub_accuracy.py), [edgeprep/hub_modes.py](../../edgeprep/hub_modes.py).
- **resize192_s24_npu** (`resize192_s24_npu`): [results/hub/resize192_s24_npu](../../results/hub/resize192_s24_npu); [edgeprep/hub_profile.py](../../edgeprep/hub_profile.py), [edgeprep/hub_auth.py](../../edgeprep/hub_auth.py), [edgeprep/hub_accuracy.py](../../edgeprep/hub_accuracy.py), [edgeprep/hub_modes.py](../../edgeprep/hub_modes.py).

## Supporting evidence

- [docs/saved192_ai_hub.md](../../docs/saved192_ai_hub.md)
- [results/hub/resize192_labelled500/manifest.json](../../results/hub/resize192_labelled500/manifest.json)
- [results/hub/resize192_device.json](../../results/hub/resize192_device.json)
