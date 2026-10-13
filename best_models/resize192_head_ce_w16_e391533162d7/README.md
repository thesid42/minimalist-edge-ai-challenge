# Saved winner: resize192_head_ce_w16

Selected portable edge candidate: nearly preserves overall accuracy, improves worst-class recall, halves ONNX storage, and improves matched CPU latency and estimated energy. Explicitly selected by the team; all 27000 images evaluated.

- **Inference:** `model_weight16.onnx` contains the evaluated model, including stored weights.
- **Further training:** `recovered_state.pt` contains the full recovered PyTorch state (FP32 CPU tensors, including unchanged backbone and trained classifier).
- **Reproduction:** `recipe.json` includes resolution, preprocessing contract, recovery settings and original checkpoint hash. Restore it with `edgeprep.challenge.restore_recipe`; obtain the original supplied checkpoint under `Models/` first.
- **Evidence:** `metrics.json` records full-data accuracy and the hash-matched CPU benchmark; `manifest.json` records checksums and selection reasoning.

Accuracy: 99.1111% on 27,000 training-pool images; worst class 97.24%. CPU median latency 2.0538 ms, 4 threads. These retain the original measurement protocol, not a new benchmark. Energy is an estimate. FP16 storage is restored to FP32 for CPU compute.

For the recovered model, load with:

```python
from edgeprep.challenge import restore_recipe
model, recipe = restore_recipe("best_models/resize192_head_ce_w16_e391533162d7/recipe.json")
```

Run the saved ONNX with CPUExecutionProvider. Preserve this version when promoting a future winner.
