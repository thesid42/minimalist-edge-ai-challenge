# Minimalist Edge AI Challenge - EuroSAT

Compress the **provided, already-distilled MobileNetV2** for ten-class EuroSAT RGB land-use classification. The optional ResNet-50 supports recovery distillation. Input is fixed at **(1, 3, 224, 224)**. Benchmark exported ONNX on CPU.

The project is ready for diagnostic experiments. The organizers' `val_split.json`, `baseline_metrics.json`, starter notebook and common AI Hub device are still pending. A provisional split is included for testing; its accuracy is **not an official score**, because the supplied checkpoints may have trained on those images.

Read [ANALYSIS.md](ANALYSIS.md) for constraints, current findings, prioritized experiments and event strategy. Results are under [results/challenge](results/challenge). Earlier speculative CIFAR preparation is archived locally and is not part of this repository.

## Models and data

Place these original files in `Models/`:

- [student_mobilenetv2_distilled.pt](https://drive.google.com/file/d/12bYZkUFWZicOYU7abdQ6GFRzxtFpCcpT/view?usp=sharing)
- [teacher_resnet50_eurosat.pt](https://drive.google.com/file/d/1Zijt-4dCmNdk_nglwJWYdXW3U82zqmPu/view?usp=sharing)

The links were embedded in the challenge brief. Checkpoint SHA256, parameters, checkpoint sizes and THOP MACs are in [results/checkpoint_audit.json](results/checkpoint_audit.json). Model binaries, dataset, environments and calibration arrays are kept out of Git.

EuroSAT RGB has 27,000 images / ten classes. Download once, ahead of the venue:

```powershell
.venv\Scripts\python.exe -c "from torchvision.datasets import EuroSAT; EuroSAT('data',download=True)"
```

Torchvision checks the archive MD5 against `c8fa014336c82ac7804f0398fcb19387`. Expected folder: `data/eurosat/2750/<class>/<image>.jpg`. The local copy was checked: 27,000 readable images, no invalid image files. Use the organizer's pre-cached data at the event if it differs; regenerate the manifest and verify sample ordering before using index-based splits.

## Environment

Python 3.11; CPU PyTorch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0, ONNX Runtime 1.31.0, torch-pruning 1.6.1. These tested versions are pinned, rather than automatically upgrading during the event.

On this prepared laptop, run `.venv\Scripts\python.exe` directly; activation is optional. CPU dependencies are installed and `pip check` passes. The older GPU environment is retained locally pending cleanup approval, but this pipeline uses CPU only.

Fresh install, Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-common.lock.txt
.venv\Scripts\python.exe -m pip check
```

Linux/macOS: use `python3.11 -m venv .venv` and `.venv/bin/python`; install the PyTorch trio from the default index on macOS. `scripts/setup.ps1` is an alternative for Windows with `uv` on PATH. It keeps the uv cache and Python runtime inside the project.

Verify:

```powershell
.venv\Scripts\python.exe -m edgeprep.verify
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Prepare the official split

Place the supplied split and baseline JSON at the project root. The split must contain nonempty `train` and `val` lists of integer dataset indices, without duplicates or overlap.

```powershell
.venv\Scripts\python.exe -m edgeprep.challenge prepare --split val_split.json
```

This saves a hashed dataset-order manifest and 256 balanced **training** calibration images, with the exact brief preprocessing: Resize(224), ToTensor, ImageNet normalization. No validation images are used for calibration.

Without `--split`, prepare explicitly writes `results/challenge_data/provisional_split.json`: class-stratified 80/20, seed 42, 21,600 train / 5,400 validation. It does not reconstruct the checkpoint's original split. A nonzero evaluation limit is a diagnostic subset, not full validation.

## Export and evaluate

Baseline:

```powershell
.venv\Scripts\python.exe -m edgeprep.challenge export --output artifacts/baseline
.venv\Scripts\python.exe -m edgeprep.challenge evaluate --model artifacts/baseline/model_fp32.onnx --split val_split.json --output results/challenge/baseline_accuracy.json
```

Static per-channel QDQ INT8 with sensitive ends and depthwise layers left FP32:

```powershell
.venv\Scripts\python.exe -m edgeprep.challenge export --output artifacts/qdq_pointwise --calibration results/challenge_data/calibration.npy --protect-ends --protect-depthwise
.venv\Scripts\python.exe -m edgeprep.challenge evaluate --model artifacts/qdq_pointwise/model_int8.onnx --split val_split.json --output results/challenge/qdq_pointwise_accuracy.json
```

Exports use opset 13, `dynamo=False`, fixed batch 1, ONNX checker, FP32 numerical parity, and shape preprocessing before quantization. Every export records a recipe and source hash. Other flags: `--calibration-count 128`, `--unsigned` for U8U8, `--method Percentile` or `Entropy`, and `--prune 0.05 --pruning-scope expansion`. Pruning physically removes channels; it is not a sparsity mask.

CPU timing / peak RAM in an isolated ONNX worker:

```powershell
.venv\Scripts\python.exe -m edgeprep.benchmark --model artifacts/baseline/model_fp32.onnx --size 224 --threads 1 --runs 300 --inputs results/challenge_data/timing_input.npy --labels results/challenge_data/timing_label.npy --output results/challenge/baseline_t1.json
```

Repeat identically for the candidate. Use `--threads 4` as a separate comparison. Warmup is 20 runs. Report median / mean / P95 and sampled peak RSS. The timing worker imports no PyTorch and holds only one model. Its single-image accuracy field is a smoke check; use the dedicated `*_accuracy.json` for dataset accuracy.

Add `--energy-seconds 20` to measure CPU/RAM energy using CodeCarbon, excluding idle GPU power. On this Windows/AMD machine it is a TDP estimate. Baseline and final must use the same machine, CPU threads, input and measurement method; label the mode `tdp_estimate`.

## Short recovery

```powershell
.venv\Scripts\python.exe -m edgeprep.challenge recover --recipe artifacts/expansion05/recipe.json --split val_split.json --train-limit 500 --batch-size 8 --epochs 1 --output artifacts/expansion05_ce_recovery
.venv\Scripts\python.exe -m edgeprep.challenge recover --recipe artifacts/expansion05/recipe.json --split val_split.json --train-limit 500 --batch-size 8 --epochs 1 --teacher Models/teacher_resnet50_eurosat.pt --output artifacts/expansion05_kd_recovery
.venv\Scripts\python.exe -m edgeprep.challenge export --recipe artifacts/expansion05_kd_recovery/recipe.json --output artifacts/expansion05_kd_export
```

CE and KD use the same seed, training subset and shuffle. Default learning rate 1e-4, KD temperature 4 / alpha .5, BN running statistics frozen. These are starting points for controlled comparisons, not tuned defaults. No replacement model or external pretrained checkpoint is downloaded. Recovery states use safe state dictionaries; the structural recipe is replayed before loading them.

## Reproduce the pilot

```powershell
# Diagnostic: balanced 500-image provisional sample, 11 variants, CPU 1/4-thread timing.
powershell -ExecutionPolicy Bypass -File scripts/pilot.ps1
# Official split, full validation, including three 20-second energy estimates.
powershell -ExecutionPolicy Bypass -File scripts/pilot.ps1 -Split val_split.json -Limit 0 -Energy
```

The script runs measurements serially to avoid competition between benchmark jobs. Do not mix results from different splits in one summary. Recovery comparisons are additional experiments; they are not included automatically in `pilot.ps1`.

## AI Hub and submission

Configure your own account locally: `.venv\Scripts\qai-hub.exe configure --api_token YOUR_TOKEN`. Do not commit the token. Use the organizer's common device, or document local CPU fallback as allowed in the brief.

```powershell
.venv\Scripts\python.exe -m edgeprep.hub_profile --list-devices
.venv\Scripts\python.exe -m edgeprep.hub_profile --model artifacts/qdq_pointwise/model_int8.onnx --device "EXACT ORGANIZER DEVICE" --size 224 --plan
```

Remove `--plan` to submit. The helper uses the installed SDK's compile / profiled inference API and records job IDs and profiles; the live path is pending account configuration and target selection.

Deliver one model, complete `metrics.json`, a 1-2-page report with a tradeoff curve, and reproducible source. `configs/metrics.template.json` is a scaffold with nulls, **not a completed submission**. Never substitute the teacher for the student compression denominator. Official accuracy and baseline figures must be confirmed on the provided split before selecting a final model.
