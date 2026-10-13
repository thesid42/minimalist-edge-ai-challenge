# Minimalist Edge AI Challenge - EuroSAT

Compress the supplied, already-distilled MobileNetV2 for ten-class EuroSAT RGB land-use classification. The optional ResNet-50 is a recovery teacher. Input stays **(1, 3, 224, 224)**; benchmark the exported ONNX on CPU.

**Organizer clarification supplied by the team:** there is no official split. Use all **27,000 images** for training, and establish our own baseline by running the unchanged supplied student on our laptop. This supersedes the split instructions and printed baseline placeholders in the PDF. Accuracy on this same dataset is labelled **training-set accuracy**.

Current evidence is under [results/all_data](results/all_data). [baseline_metrics.json](baseline_metrics.json) contains the measured laptop reference once the measurement script completes. The old PDF figures are preserved in [configs/pdf_baseline_reference.json](configs/pdf_baseline_reference.json). Earlier 500-image pilot results are historical under [results/challenge](results/challenge); they must not be mixed with current ratios.

**Start with [the sequential experiment log](experiments/README.md)** for what was tried, how each run differs, and its results. [Experiment_Log.ipynb](experiments/Experiment_Log.ipynb) has readable tables with saved outputs. Read [ANALYSIS.md](ANALYSIS.md) for the exploration plan and measured tradeoffs.

## Latest completed results

Refreshed from saved evidence at 2026-10-13 18:44 UTC. [Full results](docs/next_priority_results.md) and [numbered experiment log](experiments/README.md) are regenerated together.

The strongest measured laptop candidate is **internal 192 + one full-data classifier recovery epoch + FP16 weight storage / FP32 CPU compute**. External input remains `(1,3,224,224)`.

| Matched laptop measurement | Original student | Recovered 192 / weight16 |
|---|---:|---:|
| Accuracy on all 27,000 images | 99.1630% | **99.1111%** |
| Worst class recall | 95.84% | **97.24%** |
| ONNX size | 8.506 MiB | **4.288 MiB** |
| Four-thread median latency, 3 repetitions | 2.789 ms | **2.054 ms** |
| Estimated J/inference, mean of 3 windows | 0.054927 | **0.045338** |

Accuracy decreases 0.0519 percentage points; the candidate is 26.4% faster and uses 17.5% less estimated energy in this matched sequence. CodeCarbon uses CPU TDP/load plus a RAM heuristic on this Windows 10 laptop, with GPU excluded; these are not meter readings. FP16 storage halves file size while ORT folds weights to FP32 for CPU compute; it does not prove a RAM saving. Four threads outperformed 1/2/8 in the measured latency/energy tradeoff. Sleeping workers regressed latency here; the unrecovered 160 model loses too much accuracy (93.00%).

A separate direct latency comparison of all 192 variants is now complete (five repetitions, four CPU threads, default spinning):

| Model | Median latency | Range of run medians |
|---|---:|---:|
| baseline | 4.893 ms | 2.636–5.363 ms |
| resize192 | 2.222 ms | 1.965–3.612 ms |
| resize192_head_ce | 2.108 ms | 1.865–4.274 ms |
| resize192_head_ce_w16 | 2.179 ms | 1.837–3.834 ms |

The range shows substantial Windows timing variability. This separate sequence is not the denominator of the energy comparison above. Small FP32/weight16 differences do not establish compute acceleration.

| AI Hub target / execution | Burst latency | Accuracy on the same 1,000 labelled images |
|---|---:|---:|
| S24 family / CPU FP32 | 5.533 ms | 98.8% |
| S24 family / NPU float | 0.366 ms | 98.8% |
| RB3 Gen 2 Vision Kit / NPU W8A8 | 0.614 ms | 98.3% |
| RB3 Gen 2 Vision Kit / NPU W8A16 | 1.056 ms | 98.5% |

Hub rows use the original 224 model, not the recovered 192 candidate. Actual NPU placement is recorded per operation. These are training-pool diagnostic accuracies (100/class), disjoint from 512 calibration images; they are not all-data scores or held-out results. W8A16 gains two correct predictions over W8A8 on this sample at higher latency; its worst class is 88% vs W8A8's 89%, so it is not a uniform accuracy improvement. Hub power modes expose latency, not measured joules.

New training uses `.venv-train-gpu` / RTX 3070 / CUDA; laptop export, evaluation and inference use `.venv` / CPU. AI Hub can use CPU or NPU. A real 100-image, five-step GPU recovery, CPU export and CPU inference verification passed; it is a pipeline check, not a new competition result. **11 functional tests and dependency checks pass.**

See [GPU setup](docs/gpu_training.md). Refresh the README, log and saved notebook tables after each completed result: `.venv\Scripts\python.exe scripts/refresh_results.py`. This reads evidence only and submits no jobs or training.

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

Linux/macOS: use `python3.11`, `.venv/bin/python`, and the default PyTorch index on macOS. `scripts/setup.ps1` provides an alternative Windows installation with `uv` on PATH. Functional checks cover checkpoint replay, pruning, export, data protocols and deployment helpers. Approved cleanup recovered 8.15 GiB; CPU dependencies and current model/data artifacts remain intact.

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
.venv-train-gpu\Scripts\python.exe -m edgeprep.challenge recover --recipe artifacts/all_data/expansion02/recipe.json --epochs 1 --batch-size 8 --output artifacts/all_data/expansion02_ce
.venv-train-gpu\Scripts\python.exe -m edgeprep.challenge recover --recipe artifacts/all_data/expansion02/recipe.json --epochs 1 --batch-size 8 --teacher Models/teacher_resnet50_eurosat.pt --output artifacts/all_data/expansion02_kd
```

Omitting `--split` uses all data. Recovery's `--train-limit 0` is now the default and means all 27,000 images. A nonzero limit is a deliberately shorter diagnostic experiment, not full-data training. Evaluation similarly defaults to the whole dataset; `--limit 500` is screening only. Legacy `--split` support remains for reproducing past experiments.

CE/KD use identical sample order, seed 42, LR 1e-4, one epoch, frozen BN running statistics; KD temperature 4 / alpha .5. New recovery uses CUDA mixed precision on the laptop GPU; teacher inference during distillation also uses CUDA. Budget full-data recovery separately from the several-minute inference benchmark.

## Experiment sweep and submission

```powershell
# Quick screening; all images remain available for training.
powershell -ExecutionPolicy Bypass -File scripts/pilot.ps1 -Limit 500
# Full-data scoring for every variant; considerably longer.
powershell -ExecutionPolicy Bypass -File scripts/pilot.ps1 -Limit 0 -Energy
```

Measurements run serially. Current results have explicit evaluation scope and sample counts; do not report subset accuracy as a full-data result. Regenerate the local reference with `python -m edgeprep.local_baseline` after matching baseline accuracy / runtime / energy measurements.

AI Hub credentials are read automatically from the ignored project-root `.env` (`QAI_HUB_API_TOKEN=`). Access was verified and the live catalogue saved with 78 entries. Run `.venv\Scripts\python.exe -m edgeprep.hub_auth --check` after editing the key. See [AI Hub device and experiment guide](docs/ai_hub.md) for CPU/NPU, real calibration, reproduction commands and live results.

Deliver one selected model, complete metrics.json, a 1-2-page report with tradeoff curve, and source that recreates it from the checkpoint. `configs/metrics.template.json` is a scaffold, not a completed submission. Report overall training-set accuracy, all class recalls, worst class, parameters, file size, MACs, CPU latency, RAM and estimated/measured energy against our own laptop reference.

## Read the experiments in Jupyter

The summary notebook has saved outputs and can be read on GitHub or through `experiments/Experiment_Log.html`. JupyterLab and ipykernel are installed in the existing CPU environment. Launch it with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/notebook.ps1
```

For a fresh clone, install `requirements-notebook.lock.txt` after CPU setup. The launcher installs a project kernel under .venv, uses local runtime/cache directories, and opens the summary notebook. Its cells only read saved results and plots; executing it does not train models or submit cloud jobs.

Latest work: [device accuracy, latency and energy results](docs/next_priority_results.md), [CodeCarbon measurement guide](docs/energy_measurement.md), and numbered experiment steps 08-12 in the log.

## GPU training, CPU laptop inference

Run `powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_training.ps1` to create the separate `.venv-train-gpu` environment with PyTorch 2.8 / CUDA 12.8. The CPU environment `.venv` stays on CPU PyTorch and CPU ONNX Runtime. Recovery defaults to CUDA and fails if CUDA is unavailable; it never silently switches training to CPU. Use `--no-amp` for FP32 GPU training. Explicit `--device cpu` exists only for reproducing historical CPU training runs.

Export, accuracy evaluation, latency and energy measurement use `.venv` and `CPUExecutionProvider`. Saved recovery checkpoints contain CPU tensors, so they load in the CPU environment. AI Hub experiments may use CPU or NPU according to the device experiment; the laptop inference rule does not constrain cloud devices. See [training setup and verification](docs/gpu_training.md).
