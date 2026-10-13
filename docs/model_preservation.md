# Saved experiment winners

The team selected `resize192_head_ce_w16` as the current best portable edge candidate. Its inference weights and recovered training weights are saved under `best_models/resize192_head_ce_w16_e391533162d7/` and committed to the public repository. The binary files total about 13.1 MiB.

- `model_weight16.onnx`: exact evaluated 192 model, 4,496,556 bytes. Contains FP16 initializers and FP32 CPU computation, with fixed `(1,3,224,224)` input.
- `recovered_state.pt`: complete recovered PyTorch state, 9,193,803 bytes. Contains FP32 CPU tensors for further training; not just the final classifier. Backbone and classifier are both included.
- `recipe.json`: portable checkpoint reference, original source hash, internal resize, training settings and precision metadata. The original organizer checkpoint under `Models/` is required for structural recipe replay; obtain it from the README link.
- `metrics.json`: full 27,000-image accuracy and the matching three-repetition CPU benchmark. Timing-file one-image accuracy is excluded to avoid confusion with dataset accuracy.
- `manifest.json`: SHA-256 for both weight files, size and selection reasoning.
- `validation.json`: saved-file verification, CPU inference and checkpoint replay evidence.

`best_models/current.json` identifies the current winner. `best_models/registry.json` keeps previously selected winners. Each winner gets a separate folder keyed by name and model hash; the save helper rejects different weights in an existing snapshot.

## Future experiments

Training already saves `recovered_state.pt` in its experiment output. Once full-data evaluation and CPU benchmarking justify promoting a candidate, save it before cleanup:

```powershell
.venv\Scripts\python.exe -m edgeprep.champions --name resize192_head_ce_w16 --accuracy results/all_data/resize192_head_ce_w16_accuracy.json --benchmark results/energy/runtime/resize192_head_ce_w16_spin_summary.json --reason "Selected balanced edge candidate: strong accuracy, worst-class recall, size, CPU latency and estimated energy."
```

For a future winner, substitute its experiment name and evidence paths. This validates 27,000-image evaluation, CPU provider and matching model/checkpoint hashes, copies both forms of the weights, preserves old winners and updates the current pointer. Refresh the README/log after promotion and push the saved bundle with the source and evidence. Winners remain explicitly selected using the measured tradeoffs; the helper does not silently choose the model with the highest accuracy.

Load this saved recovery for more GPU training by passing its `recipe.json` to `.venv-train-gpu\Scripts\python.exe -m edgeprep.challenge recover`. For inference, load `model_weight16.onnx` with ONNX Runtime CPUExecutionProvider. Future winners can have different precisions; inspect their recipe and manifest.
