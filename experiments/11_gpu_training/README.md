# 11. GPU training environment and CPU inference verification

New laptop training uses RTX 3070 CUDA, while laptop evaluation/inference remains CPU-only. This does not constrain AI Hub CPU/NPU. Separate matching-version environments, fail-fast CUDA requirement, AMP and CPU-portable checkpoints. Real 100-image full-backbone recovery / five GPU optimizer steps, followed by CPU export and CPU inference, verifies the pipeline only. The earlier 27,000-image classifier recovery occurred on CPU before this preference and remains the selected candidate.

| name | status | training_examples | optimizer_steps | training_device | amp | seconds | inference_provider |
|---|---|---|---|---|---|---|---|
| gpu_training_pipeline | completed | 100 | 5 | NVIDIA GeForce RTX 3070 Laptop GPU | True | 8.18067269999301 | CPUExecutionProvider |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **RTX 3070 training / CPU inference verification** (`gpu_training_pipeline`): [results/gpu_training/verification.json](../../results/gpu_training/verification.json); [scripts/setup_gpu_training.ps1](../../scripts/setup_gpu_training.ps1), [scripts/verify_gpu_training.py](../../scripts/verify_gpu_training.py), [edgeprep/challenge.py](../../edgeprep/challenge.py).

## Supporting evidence

- [docs/gpu_training.md](../../docs/gpu_training.md)
- [requirements-train-gpu.lock.txt](../../requirements-train-gpu.lock.txt)
- [results/gpu_training/smoke_accuracy.json](../../results/gpu_training/smoke_accuracy.json)
