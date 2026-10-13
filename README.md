# Minimalist Edge AI Challenge - EuroSAT

Compress the supplied, already-distilled MobileNetV2 for ten-class EuroSAT RGB land-use classification. The optional ResNet-50 is a recovery teacher. Input stays **(1, 3, 224, 224)**; benchmark the exported ONNX on CPU.

**Organizer clarification supplied by the team:** there is no official split. Use all **27,000 images** for training, and establish our own baseline by running the unchanged supplied student on our laptop. This supersedes the split instructions and printed baseline placeholders in the PDF. Accuracy on this same dataset is labelled **training-set accuracy**.

Current evidence is under [results/all_data](results/all_data). [baseline_metrics.json](baseline_metrics.json) contains the measured laptop reference once the measurement script completes. The old PDF figures are preserved in [configs/pdf_baseline_reference.json](configs/pdf_baseline_reference.json). Earlier 500-image pilot results are historical under [results/challenge](results/challenge); they must not be mixed with current ratios.

Read [ANALYSIS.md](ANALYSIS.md) for the exploration plan and measured tradeoffs.

## Models and data

Place the original files in `Models/`:

- [student_mobilenetv2_distilled.pt](https://drive.google.com/file/d/12bYZkUFWZicOYU7abdQ6GFRzxtFpCcpT/view?usp=sharing)
- [teacher_resnet50_eurosat.pt](https://drive.google.com/file/d/1Zijt-4dCmNdk_nglwJWYdXW3U82zqmPu/view?usp=sharing)

These are the PDF's model links. Checkpoint hashes and architecture measurements are in [results/checkpoint_audit.json](results/checkpoint_audit.json). Dataset, checkpoints, model exports, environments and calibration arrays are excluded from Git.

EuroSAT RGB is locally verified: 27,000 valid images, ten classes, archive MD5 `c8fa014336c82ac7804f0398fcb19387`. A fresh clone can download it:

```powershell
.venv\Scripts\python.exe -c "from torchvision.datasets import EuroSAT; EuroSAT('data',download=True)"
```

Exact preprocessing: torchvision Resize(224), ToTensor, ImageNet mean [.485,.456,.406] and std [.229,.224,.225]. The manifest hashes sample ordering. Calibration is a balanced representative subset of the whole training pool; it may overlap training-set evaluation under the clarified protocol.

## Setup

Python 3.11 and a tested CPU environment are installed. Fresh Windows install:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-common.lock.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux/macOS: use `python3.11`, `.venv/bin/python`, and the default PyTorch index on macOS. `scripts/setup.ps1` provides an alternative Windows installation with `uv` on PATH. Four functional checks pass. Approved cleanup recovered 8.15 GiB; CPU dependencies and current model/data artifacts remain intact.

## Reproduce the current baseline and comparison

```powershell
powershell -ExecutionPolicy Bypass -File scripts/measure_all_data.ps1
```

This prepares full-data calibration, exports the unchanged student and selective INT8, evaluates both on all 27,000 images, measures CPU 1/4-thread latency and RAM in isolated processes, estimates energy for both with four threads, builds `baseline_metrics.json`, and plots the current tradeoff. Expect several minutes for full-data inference. No baseline weights are retrained.

Baseline size is **FP32 ONNX size**, and final size uses the same format. Original checkpoint size is separately audited. Ratios always use the MobileNetV2 student, never the teacher. CPU threads, input, execution provider and energy method must match between baseline and candidate. On this Windows/AMD laptop, CodeCarbon uses TDP/CPU-load estimation; inspect startup logs on other machines.

Individual commands:

```powershell
.venv\Scripts\python.exe -m edgeprep.challenge prepare
.venv\Scripts\python.exe -m edgeprep.challenge export --output artifacts/all_data/baseline
.venv\Scripts\python.exe -m edgeprep.challenge evaluate --model artifacts/all_data/baseline/model_fp32.onnx --threads 4 --output results/all_data/baseline_accuracy.json
.venv\Scripts\python.exe -m edgeprep.challenge export --output artifacts/all_data/qdq_pointwise --calibration results/all_data/challenge_data/calibration.npy --protect-ends --protect-depthwise
```

Export uses static per-channel QDQ, opset 13 / dynamo=False, fixed batch one, shape preprocessing, ONNX checking and numerical parity. Recipes record original checkpoint hashes and compression settings.

## Recovery on the entire dataset

```powershell
.venv\Scripts\python.exe -m edgeprep.challenge export --prune 0.02 --pruning-scope expansion --output artifacts/all_data/expansion02
.venv\Scripts\python.exe -m edgeprep.challenge recover --recipe artifacts/all_data/expansion02/recipe.json --epochs 1 --batch-size 8 --output artifacts/all_data/expansion02_ce
.venv\Scripts\python.exe -m edgeprep.challenge recover --recipe artifacts/all_data/expansion02/recipe.json --epochs 1 --batch-size 8 --teacher Models/teacher_resnet50_eurosat.pt --output artifacts/all_data/expansion02_kd
```

Omitting `--split` uses all data. Recovery's `--train-limit 0` is now the default and means all 27,000 images. A nonzero limit is a deliberately shorter diagnostic experiment, not full-data training. Evaluation similarly defaults to the whole dataset; `--limit 500` is screening only. Legacy `--split` support remains for reproducing past experiments.

CE/KD use identical sample order, seed 42, LR 1e-4, one epoch, frozen BN running statistics; KD temperature 4 / alpha .5. Cache teacher logits for deterministic training inputs if repeated teacher inference costs too much CPU time. Budget full-data recovery separately from the several-minute inference benchmark.

## Experiment sweep and submission

```powershell
# Quick screening; all images remain available for training.
powershell -ExecutionPolicy Bypass -File scripts/pilot.ps1 -Limit 500
# Full-data scoring for every variant; considerably longer.
powershell -ExecutionPolicy Bypass -File scripts/pilot.ps1 -Limit 0 -Energy
```

Measurements run serially. Current results have explicit evaluation scope and sample counts; do not report subset accuracy as a full-data result. Regenerate the local reference with `python -m edgeprep.local_baseline` after matching baseline accuracy / runtime / energy measurements.

AI Hub is optional if no common target/account is available. Configure your token locally and use `edgeprep.hub_profile --plan` before submitting to the organizer's device. No credentials are committed. The live Hub path is untested.

Deliver one selected model, complete metrics.json, a 1-2-page report with tradeoff curve, and source that recreates it from the checkpoint. `configs/metrics.template.json` is a scaffold, not a completed submission. Report overall training-set accuracy, all class recalls, worst class, parameters, file size, MACs, CPU latency, RAM and estimated/measured energy against our own laptop reference.
