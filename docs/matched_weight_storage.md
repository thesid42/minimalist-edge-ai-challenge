# Matched FP16 storage: baseline versus recovered 192

Apply the exact same FP16-initializer / Cast-to-FP32 conversion to the unchanged baseline and compare it with the preserved winner. No retraining, resolution change or pruning is applied to the baseline. All local inference uses ONNX Runtime CPU.

| Model | Storage | Exact bytes | MiB | Accuracy on 27,000 images | Worst class recall |
|---|---|---:|---:|---:|---:|
| baseline_fp32 | FP32 | 8,919,497 | 8.506295 | 99.1630% | 95.84% |
| baseline_weight16 | FP16 | 4,491,583 | 4.283507 | 99.1667% | 95.88% |
| recovered192_fp32 | FP32 | 8,924,421 | 8.510991 | 99.1074% | 97.24% |
| recovered192_weight16 | FP16 | 4,496,556 | 4.288250 | 99.1111% | 97.24% |

The recovered192 FP16-storage file is **4,973 bytes larger** than the unchanged baseline stored in FP16. The original baseline alone shrinks **49.6431%** under this conversion. Both graphs store **2,219,626 floating-point values** and both PyTorch architectures have **2,236,682 parameters**. Exported values differ from the trainable parameter count because export folds batch normalization and reuses constants. The 192 graph includes additional resize/graph metadata; the measured byte difference is not a smaller weight count.

The 8.506-to-4.288 MiB comparison is valid as original-FP32 versus selected-FP16 portable-file storage. It does not isolate a benefit of resolution reduction, classifier recovery or structural compression. At equal FP16 storage, there is no file-size improvement over the original student. Our 192 changes reduce spatial computation and recover accuracy; they remove no channels or parameters.

All accuracy numbers above use the same 27,000-image training pool and preprocessing; they are training-set accuracy. FP16 storage is rounded once, then restored to FP32 for computation. Accuracy must be checked rather than assumed identical. No new latency or energy claim is made in this size experiment.

Paired output check on the existing balanced 500-image sample: **100.0% top-1 agreement**, 0 changed predictions, maximum absolute logit difference 0.0575295. This check is separate from labelled full-data accuracy.

The saved candidate still compiles to **8,937,126 bytes / 8.523 MiB** in the tested S24 Hub path because the compiler expands stored weights to FP32. A fresh baseline through the same compilation path would be required for a deployed-size ratio. Neither portable file size nor compiled payload size establishes runtime RAM consumption. [Device evidence](saved192_ai_hub.md).

## Reproduce

```powershell
.venv\Scripts\python.exe -m edgeprep.weight_storage --source artifacts/baseline/model_fp32.onnx --output artifacts/baseline_weight16
.venv\Scripts\python.exe -m edgeprep.challenge evaluate --model artifacts/baseline_weight16/model_weight16.onnx --root data --threads 4 --output results/storage_comparison/baseline_weight16_accuracy.json
.venv\Scripts\python.exe scripts/compare_weight_storage.py
```

Read [size, graph and hash audit](../results/storage_comparison/summary.json), [full-data baseline FP16 accuracy](../results/storage_comparison/baseline_weight16_accuracy.json), and [paired output check](../results/storage_comparison/paired_output_check.json). Routine intermediate ONNX files remain local under ignored artifacts/. The preserved winner and its hashes remain unchanged.
