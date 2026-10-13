# Saved winner: resize192_head640_head_ce_fp32

Balanced FP32 structural-compression winner: 640 final feature channels and classifier-only recovery on all 27000 images, 7.703 MiB versus 8.506 MiB original baseline, 99.1481% training-set accuracy and 97.40% worst-class recall. Matched three-repetition four-thread CPU measurements show 1.9791 ms and 0.040847 estimated J per image, 21.8% lower latency and 22.8% lower estimated energy than original224. Runtime is effectively similar to unpruned recovered192 within variability; actual size falls 9.5% and accuracy gains eleven correct images. The smaller block-removal model loses more accuracy and class balance. Weights, checkpoint storage and CPU compute remain FP32; older winners are preserved.

- **Inference:** `model.onnx` contains the evaluated model, including stored weights.
- **Further training:** `recovered_state.pt` contains the full recovered PyTorch state (FP32 CPU tensors, including unchanged backbone and trained classifier).
- **Reproduction:** `recipe.json` includes resolution, preprocessing contract, recovery settings and original checkpoint hash. Restore it with `edgeprep.challenge.restore_recipe`; obtain the original supplied checkpoint under `Models/` first.
- **Evidence:** `metrics.json` records full-data accuracy and the hash-matched CPU benchmark; `manifest.json` records checksums and selection reasoning.

Accuracy: 99.1481% on 27,000 training-pool images; worst class 97.40%. CPU median latency 1.9791 ms, 4 threads. These retain the original measurement protocol, not a new benchmark. Energy is an estimate. Weights, checkpoint storage and CPU compute remain FP32.

For the recovered model, load with:

```python
from edgeprep.challenge import restore_recipe
model, recipe = restore_recipe("best_models/resize192_head640_head_ce_fp32_8655a532a076/recipe.json")
```

Run the saved ONNX with CPUExecutionProvider. Preserve this version when promoting a future winner.

[Two-page submission technical report](../../output/pdf/best192_fp32_technical_report.pdf) / [editable report](../../docs/submission_best192.md).
