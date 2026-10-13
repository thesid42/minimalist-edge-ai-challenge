# GPU training and CPU laptop inference

The team requested GPU training with CPU laptop inference; AI Hub remains free to use CPU/NPU. The two Python environments are isolated:

| Use | Environment | Verified backend |
|---|---|---|
| Training / optional KD teacher | `.venv-train-gpu` | PyTorch 2.8.0+cu128, CUDA 12.8, RTX 3070 Laptop GPU (8 GB) |
| Export, all-data evaluation, latency, energy, notebook, Hub client | `.venv` | PyTorch 2.8.0+cpu; ONNX Runtime CPUExecutionProvider |
| Remote deployment experiments | AI Hub | Target-specific CPU or NPU; actual placement in saved profiles |

Install with `powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_training.ps1`. Matching CUDA 12.8 wheels are from the [official PyTorch 2.8 installation instructions](https://pytorch.org/get-started/previous-versions/). The installed NVIDIA driver is 617.14; no driver change was required. A system CUDA toolkit is not required for these wheels. Exact installed packages are in `requirements-train-gpu.lock.txt`; the lean additional dependency recipe is `requirements-train-gpu.in`. GPU environment is ignored by Git.

Recovery defaults to `--device cuda` and fails with setup instructions if CUDA is absent. There is no automatic CPU fallback. Mixed precision (`--amp`, default) uses FP16 convolution with a gradient scaler; KD probability calculations remain FP32. Use `--no-amp` for FP32 GPU training. Batch norm running statistics remain frozen during recovery. Start with batch 32 for the student and reduce if GPU memory is insufficient, especially with the ResNet50 teacher. `--workers 0` is a safe Windows default; adjust after checking data-loader throughput.

```powershell
.venv-train-gpu\Scripts\python.exe -m edgeprep.challenge recover --recipe artifacts/all_data/resize192/recipe.json --epochs 1 --batch-size 32 --head-only --output artifacts/all_data/resize192_gpu_head
.venv\Scripts\python.exe -m edgeprep.challenge export --recipe artifacts/all_data/resize192_gpu_head/recipe.json --output artifacts/all_data/resize192_gpu_export
.venv\Scripts\python.exe -m edgeprep.challenge evaluate --model artifacts/all_data/resize192_gpu_export/model_fp32.onnx --threads 4 --output results/all_data/resize192_gpu_accuracy.json
```

Omitting train-limit uses all 27,000 images. Checkpoints store CPU tensors and recipe metadata records the actual training device, CUDA/PyTorch version, AMP, worker count, seed and sample count. Existing full-data classifier recovery was completed on CPU before the GPU preference arrived; it is retained as historical evidence. To reproduce that old training command explicitly, add `--device cpu --no-amp` in `.venv`. CUDA/AMP training need not produce bit-identical weights to CPU training.

## Real pipeline verification

Run `.venv\Scripts\python.exe scripts/verify_gpu_training.py` after setup. It performs full-backbone GPU recovery on 100 balanced real images, five optimizer steps at batch 20, LR 1e-5; then uses the CPU environment for fixed-input ONNX export and 100-image inference. Results: `results/gpu_training/verification.json` and `smoke_accuracy.json`. CUDA/AMP training completed in 8.18 seconds, CPU ONNX parity max error 3.33e-5 and CPU diagnostic accuracy 99/100. This verifies operation, not quality improvement or all-data speedup. The selected 27,000-image candidate is unchanged.

Both environments pass pip check. Keep all local training stopped during serial CPU latency and energy measurements, even when training uses the GPU, because power, thermals and CPU data loading can interfere.
