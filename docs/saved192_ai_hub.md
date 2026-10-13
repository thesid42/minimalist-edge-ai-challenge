# Saved 192 model: 500-image edge-device test

Source: `best_models/resize192_head_ce_w16_e391533162d7/model_weight16.onnx`, SHA-256 `e391533162d7443b8cb74af7e1b73a3c92055a924499b1db10f3d8d243cadfc0`. This is the preserved recovered192 / FP16-storage winner, not the earlier original224 model.

Device: **Samsung Galaxy S24**, Snapdragon 8 Gen 3, Android 14, selected from the live catalogue. CPU and NPU jobs use the same named phone and exact source, but can be routed to different physical units. CPU uses four ONNX Runtime threads; NPU uses QNN in burst mode. No additional W8A8/W8A16 quantization or retraining.

Accuracy: **500 real images, 50 per class**, deterministic seed 20261013, excluding all 512 earlier calibration images. This remains a training-pool diagnostic sample, not held-out or all-data accuracy. Local reference logits are generated from this exact saved ONNX. Two 250-image remote jobs per runtime are combined only when both succeed.

Latency: separate on-device profile, 100 iterations, batch one, using the Hub-reported steady inference estimate. This excludes upload, image preprocessing and model initialization; it is not the elapsed wall time for the 500-image accuracy job.

| Runtime | Profile latency | Labelled accuracy | Correct / 500 | Worst class recall | Agreement with saved source | Actual operation placement |
|---|---:|---:|---:|---:|---:|---|
| CPU / four threads | 4.183 ms | 99.40% | 497 | 96.0% | 100.0% | {'CPU': 56} |
| QNN NPU / burst | 0.435 ms | 99.40% | 497 | 96.0% | 100.0% | {'NPU': 105} |

## Class recall on the 500-image sample

| Class | CPU | NPU |
|---|---:|---:|
| AnnualCrop | 100.0% | 100.0% |
| Forest | 100.0% | 100.0% |
| HerbaceousVegetation | 100.0% | 100.0% |
| Highway | 100.0% | 100.0% |
| Industrial | 100.0% | 100.0% |
| Pasture | 98.0% | 98.0% |
| PermanentCrop | 96.0% | 96.0% |
| Residential | 100.0% | 100.0% |
| River | 100.0% | 100.0% |
| SeaLake | 100.0% | 100.0% |

Both runtimes make the same three errors as the saved model locally: two PermanentCrop images and one Pasture image. Agreement is reported separately from label-based accuracy.


## What happened to weight storage?

The uploaded portable source is **4,496,556 bytes / 4.288 MiB**. Both compiled ONNX target packages contain **8,937,126 bytes / 8.523 MiB** uncompressed, including external weights. The target graph has FLOAT initializers: the compiler folded the source FP16-to-FP32 casts and expanded its weight storage. The original saved file remains unchanged. ZIP size is transfer compression and is not the model-size denominator.

This confirms a distinction between the smaller portable file and the compiler-produced deployment artifact. The source disk saving is real, but it does not survive this specific Hub compile path. NPU execution is verified by per-operation placement; float precision inside the device kernel is not inferred from file size or the source label.

A controlled local experiment applies the same FP16 storage conversion to the original baseline: it becomes 4.283507 MiB versus 4.288250 MiB for this candidate. Equal-precision storage shows no candidate size advantage. [Matched weight-storage evidence](matched_weight_storage.md).

Older original224 results were measured on the S24 family and a different 1,000-image sample. They are historical context, not a fresh matched baseline for this exact-phone 192 test. Do not calculate model speedup or compare accuracy across those samples.

No energy measurement is claimed. [Qualcomm profile metric definitions](https://workbench.aihub.qualcomm.com/docs/hub/howitworks-inference.html). Hub process-memory ranges are retained in profile.json; they are not total board/NPU memory.

## Evidence and reproduction

- [CPU request, jobs, profile and labelled accuracy](../results/hub/resize192_s24_cpu/)
- [NPU request, jobs, profile and labelled accuracy](../results/hub/resize192_s24_npu/)
- [500-image sample manifest](../results/hub/resize192_labelled500/manifest.json)
- [Live device selection](../results/hub/resize192_device.json)

```powershell
.venv\Scripts\python.exe -m edgeprep.hub_accuracy prepare --count 500 --shard-size 250 --sample-dir results/hub/resize192_labelled500 --model best_models/resize192_head_ce_w16_e391533162d7/model_weight16.onnx
.venv\Scripts\python.exe -m edgeprep.hub_profile --model best_models/resize192_head_ce_w16_e391533162d7/model_weight16.onnx --device "Samsung Galaxy S24" --compute cpu --output results/hub/resize192_s24_cpu
.venv\Scripts\python.exe -m edgeprep.hub_profile --model best_models/resize192_head_ce_w16_e391533162d7/model_weight16.onnx --device "Samsung Galaxy S24" --compute npu --output results/hub/resize192_s24_npu
.venv\Scripts\python.exe -m edgeprep.hub_accuracy submit --folder results/hub/resize192_s24_cpu --folder results/hub/resize192_s24_npu --sample-dir results/hub/resize192_labelled500
.venv\Scripts\python.exe -m edgeprep.hub_accuracy collect --folder results/hub/resize192_s24_cpu --folder results/hub/resize192_s24_npu
```

For a new run, choose fresh profile output folders to preserve previous job history. Prepare and labelled-submit are resumable; collect retrieves existing jobs without resubmitting them. Downloaded targets and input/output arrays remain local and ignored by Git.
