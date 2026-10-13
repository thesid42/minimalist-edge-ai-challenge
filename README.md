# Minimalist Edge AI Challenge - EuroSAT

Compress the supplied, already-distilled MobileNetV2 for ten-class EuroSAT RGB land-use classification. The optional ResNet-50 is a recovery teacher. Input stays **(1, 3, 224, 224)**; benchmark the exported ONNX on CPU.

**Organizer clarification supplied by the team:** there is no official split. Use all **27,000 images** for training, and establish our own baseline by running the unchanged supplied student on our laptop. This supersedes the split instructions and printed baseline placeholders in the PDF. Accuracy on this same dataset is labelled **training-set accuracy**.

Current evidence is under [results/all_data](results/all_data). [baseline_metrics.json](baseline_metrics.json) contains the measured laptop reference once the measurement script completes. The old PDF figures are preserved in [configs/pdf_baseline_reference.json](configs/pdf_baseline_reference.json). Earlier 500-image pilot results are historical under [results/challenge](results/challenge); they must not be mixed with current ratios.

**Start with [the sequential experiment log](experiments/README.md)** for what was tried, how each run differs, and its results. [Experiment_Log.ipynb](experiments/Experiment_Log.ipynb) has readable tables with saved outputs. Read [ANALYSIS.md](ANALYSIS.md) for the exploration plan and measured tradeoffs.

## Latest completed results

Refreshed from saved evidence at 2026-10-13 21:38 UTC. [Full results](docs/next_priority_results.md) and [numbered experiment log](experiments/README.md) are regenerated together.

**Current direction: structural compression of recovered192, keeping FP32 throughout. Every new candidate uses all 27,000 images for evaluation.** [Full FP32 report](docs/fp32_model_reduction.md).

| FP32 model / stage | Parameters | ONNX MiB | Accuracy / 27,000 | Worst class recall |
|---|---:|---:|---:|---:|
| Original student / FP32 | 2,236,682 | 8.506295 | 99.1630% | 95.84% |
| Recovered192 reference / FP32 | 2,236,682 | 8.510991 | 99.1074% | 97.24% |
| Late expansion channels / before recovery | 2,108,202 | 8.023564 | 98.3185% | 96.12% |
| Final 640 channels / before recovery | 2,024,202 | 7.702886 | 98.9963% | 97.40% |
| Residual block 16 removed / before recovery | 1,916,682 | 7.295685 | 96.4296% | 87.80% |
| Final 640 / full-backbone recovery | 2,024,202 | 7.702886 | 98.8815% | 97.65% |
| Remove block 16 / full-backbone recovery | 1,916,682 | 7.295685 | 98.6222% | 95.53% |
| Final 640 / classifier-only recovery | 2,024,202 | 7.702886 | 99.1481% | 97.40% |

New recovery uses GPU FP32 with AMP disabled; export and inference use CPU FP32. The older FP16 comparison below remains historical evidence.

**New matched FP32 laptop measurements:** four CPU threads, three fresh serial repetitions per model.

| FP32 model | Median latency | Range of run medians | Estimated J/image, mean ± SD |
|---|---:|---:|---:|
| original224_fp32 | 2.532 ms | 2.513–2.875 ms | 0.052934 ± 0.001388 |
| parent192_fp32 | 1.966 ms | 1.929–2.941 ms | 0.041391 ± 0.000984 |
| head640_head_ce | 1.979 ms | 1.910–2.125 ms | 0.040847 ± 0.000989 |
| drop16_ce | 1.893 ms | 1.870–2.122 ms | 0.039281 ± 0.000442 |

The 640-channel classifier-recovered model is 21.8% faster and uses 22.8% less estimated energy than original224 in this sequence. Accuracy is evaluated separately on all 27,000 images; latency repeats one real input 1,000 times per run. CodeCarbon estimates CPU energy from TDP/load and RAM from a heuristic; GPU is excluded. Small differences versus unpruned192 remain subject to run variability. [Measurements and full interpretation](docs/fp32_model_reduction.md).

**Historical FP16-storage results follow:** these use separate measurements and an earlier winner.

The measured 192 candidate uses **internal 192 + one full-data classifier recovery epoch + FP16 weight storage / FP32 CPU compute**. External input remains `(1,3,224,224)`.

| Matched laptop measurement | Original student | Recovered 192 / weight16 |
|---|---:|---:|
| Accuracy on all 27,000 images | 99.1630% | **99.1111%** |
| Worst class recall | 95.84% | **97.24%** |
| Portable ONNX size (original FP32 vs candidate FP16 storage) | 8.506 MiB | **4.288 MiB** |
| Four-thread median latency, 3 repetitions | 2.789 ms | **2.054 ms** |
| Estimated J/inference, mean of 3 windows | 0.054927 | **0.045338** |

Accuracy decreases 0.0519 percentage points; the candidate is 26.4% faster and uses 17.5% less estimated energy in this matched sequence. CodeCarbon uses CPU TDP/load plus a RAM heuristic on this Windows 10 laptop, with GPU excluded; these are not meter readings. FP16 storage halves file size while ORT folds weights to FP32 for CPU compute; it does not prove a RAM saving. Four threads outperformed 1/2/8 in the measured latency/energy tradeoff. Sleeping workers regressed latency here; the unrecovered 160 model loses too much accuracy (93.00%).

**Controlled weight-storage comparison:** the baseline gets the same FP16 conversion.

| Model | FP32 ONNX | FP16-storage ONNX |
|---|---:|---:|
| Unchanged original224 | 8.506295 MiB | 4.283507 MiB |
| Recovered192 | 8.510991 MiB | 4.288250 MiB |

At identical FP16 storage, recovered192 is **4,973 bytes larger**. The roughly 50% portable-file reduction comes from weight precision; the baseline gets it too. Resolution reduction lowers spatial computation and recovery restores accuracy, while parameter count stays unchanged. [Full comparison and accuracy check](docs/matched_weight_storage.md).


**Submission report for the selected smaller192 FP32 model:** [two-page PDF](output/pdf/best192_fp32_technical_report.pdf) / [editable text](docs/submission_best192.md). It includes the pipeline tradeoff curve, matched metrics, per-class recall, energy mode and reproduction details.


**Saved winner weights:** [resize192_head640_head_ce_fp32](best_models/resize192_head640_head_ce_fp32_8655a532a076/README.md), with the inference ONNX, full recovered checkpoint, portable recipe, checksums and results. [Current selection](best_models/current.json) / [winner history](best_models/registry.json). Future winners are saved separately; older weights are retained. [Preservation workflow](docs/model_preservation.md).

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

**Saved 192 model on Samsung Galaxy S24 — separate 500-image test:**

| Execution | Hub profile latency | Accuracy / 500 labelled images |
|---|---:|---:|
| CPU / four threads | 4.183 ms | 99.40% (497/500) |
| NPU / burst | 0.435 ms | 99.40% (497/500) |

The exact saved FP16-storage source is uploaded without extra quantization. Hub compilation expands its target payload to 8.523 MiB; the portable source stays 4.288 MiB. The 500-image sample has 50/class and excludes calibration data; accuracy is diagnostic training-pool accuracy. [Full device report and job evidence](docs/saved192_ai_hub.md).


The first Hub table above uses the original 224 model, not the recovered 192 candidate. Actual NPU placement is recorded per operation. These are training-pool diagnostic accuracies (100/class), disjoint from 512 calibration images; they are not all-data scores or held-out results. W8A16 gains two correct predictions over W8A8 on this sample at higher latency; its worst class is 88% vs W8A8's 89%, so it is not a uniform accuracy improvement. Hub power modes expose latency, not measured joules.

New training uses `.venv-train-gpu` / RTX 3070 / CUDA; laptop export, evaluation and inference use `.venv` / CPU. AI Hub can use CPU or NPU. A real 100-image, five-step GPU recovery, CPU export and CPU inference verification passed; it is a pipeline check, not a new competition result. **20 functional tests pass; both environments passed dependency checks.**

See [GPU setup](docs/gpu_training.md). Refresh the README, log and saved notebook tables after each completed result: `.venv\Scripts\python.exe scripts/refresh_results.py`. This reads evidence only and submits no jobs or training.

<!-- BEGIN MAC CPU RESULTS -->
### Matched Mac CPU measurement

Apple M4 Pro, macOS-27.0.1-arm64-arm-64bit, CPUExecutionProvider; 4 threads, 3 serial fresh-process repetitions, batch one, 100 warmups, 1,000 timings and separate 30-second energy windows. Power source is recorded in machine.json. External input remains (1,3,224,224). The exact saved winning ONNX was measured without retraining.

Accuracy is a seeded balanced 100-image **diagnostic subset**, not the previously recorded 27,000-image training-set accuracy or held-out accuracy. The timing JSON's one-image accuracy is only a smoke test. Power/energy use CodeCarbon CPU-load/TDP plus RAM heuristics (35 W CPU assumption, GPU excluded), **not hardware-sensor or whole-device readings**. Do not mix these Mac baselines with Windows or remote-device results.

Energy variability warning: a baseline repetition slowed substantially during its energy window. All repetitions are retained; the cause is unverified. Means and sample SD are reported, but a precise energy-saving percentage is not established. Inspect individual run throughput and energy before interpreting the average watts or joules.

| Model / mode | ONNX MiB | Diagnostic accuracy | Median ms | P95 ms | Peak RSS MiB | Estimated W | Estimated J/image, mean +/- SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_spin | 8.506 | 99.0% (100 images) | 2.227 | 2.378 | 108.56 | 12.93 | 0.039579 +/- 0.012240 |
| baseline_sleep | 8.506 | 99.0% (100 images) | 2.664 | 2.876 | 102.56 | 9.33 | 0.029173 +/- 0.006575 |
| candidate_spin | 4.288 | 100.0% (100 images) | 1.871 | 1.969 | 101.86 | 14.69 | 0.028200 +/- 0.000244 |
| candidate_sleep | 4.288 | 100.0% (100 images) | 2.204 | 2.380 | 101.34 | 8.96 | 0.020962 +/- 0.000544 |

[Mac evidence](results/mac/paired_runtime/summary.json).
<!-- END MAC CPU RESULTS -->

<!-- BEGIN MAC FULL CPU RESULTS -->
### Mac CPU metrics with full-dataset accuracy

Apple M4 Pro, macOS-27.0.1-arm64-arm-64bit, CPUExecutionProvider; 4 threads, 3 serial fresh-process repetitions, batch one, 100 warmups, 1,000 timings and separate 30-second energy windows. Power source is recorded in machine.json. External input remains (1,3,224,224). The exact saved winning ONNX was measured without retraining.

Accuracy covers all 27,000 images with their original class counts: training-set accuracy, not held-out generalization. The timing JSON's one-image accuracy is only a smoke test. Power/energy use CodeCarbon CPU-load/TDP plus RAM heuristics (35 W CPU assumption, GPU excluded), **not hardware-sensor or whole-device readings**. Do not mix these Mac baselines with Windows or remote-device results.

Power source at benchmark start: Now drawing from 'Battery Power'.

Accuracy is measured across the entire dataset; runtime uses the same fixed real timing image for both models. Timed inference excludes disk reads, preprocessing and model loading. Projected dataset energy is estimated J/inference multiplied by the image count, not a metered full-dataset run. CPU percentages use one core = 100% (four busy cores can reach 400%).

Aggregation: medians of run median/P95/load times, mean estimated watts and J/inference, sample SD for energy, maximum inference-session RSS and RAM delta. RSS excludes later CodeCarbon imports. Model load time is reported separately from warmed-up inference latency.

| Model / mode | ONNX MiB | Full-data accuracy | Median ms | P95 ms | Peak RSS MiB | Estimated W | Estimated J/image, mean +/- SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_spin | 8.506 | 99.1630% (27,000 images) | 2.258 | 2.420 | 109.11 | 14.46 | 0.033765 +/- 0.000646 |
| baseline_sleep | 8.506 | 99.1630% (27,000 images) | 2.604 | 2.797 | 107.77 | 9.31 | 0.024802 +/- 0.000263 |
| candidate_spin | 4.288 | 99.1111% (27,000 images) | 1.867 | 1.962 | 105.81 | 14.52 | 0.027979 +/- 0.000356 |
| candidate_sleep | 4.288 | 99.1111% (27,000 images) | 2.295 | 2.402 | 104.12 | 9.05 | 0.021092 +/- 0.000643 |

| Model / mode | Worst-class recall | Load ms | Session RAM delta MiB | CPU use, one-core % | Projected dataset inference energy J |
|---|---:|---:|---:|---:|---:|
| baseline_spin | 95.84% | 9.396 | 61.22 | 399.5 | 911.66 |
| baseline_sleep | 95.84% | 8.028 | 59.55 | 251.9 | 669.65 |
| candidate_spin | 97.24% | 12.048 | 57.88 | 398.2 | 755.44 |
| candidate_sleep | 97.24% | 11.056 | 56.52 | 243.8 | 569.49 |

[Mac evidence](results/mac/full_runtime/summary.json).
<!-- END MAC FULL CPU RESULTS -->
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

Latest work: [device accuracy, latency and energy results](docs/next_priority_results.md), [CodeCarbon measurement guide](docs/energy_measurement.md), and numbered experiment steps 08-14 in the log.

## GPU training, CPU laptop inference

Run `powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_training.ps1` to create the separate `.venv-train-gpu` environment with PyTorch 2.8 / CUDA 12.8. The CPU environment `.venv` stays on CPU PyTorch and CPU ONNX Runtime. Recovery defaults to CUDA and fails if CUDA is unavailable; it never silently switches training to CPU. Use `--no-amp` for FP32 GPU training. Explicit `--device cpu` exists only for reproducing historical CPU training runs.

Export, accuracy evaluation, latency and energy measurement use `.venv` and `CPUExecutionProvider`. Saved recovery checkpoints contain CPU tensors, so they load in the CPU environment. AI Hub experiments may use CPU or NPU according to the device experiment; the laptop inference rule does not constrain cloud devices. See [training setup and verification](docs/gpu_training.md).

<!-- BEGIN MAC HALF SAMPLE RESULTS -->
### 50-image evaluation

50 images, 5 per class. The first half of the earlier 100-image sample was used. Both models ran on CPU with four threads. Input resolution and model weights are unchanged. This is a diagnostic training-pool subset, not held-out or full-dataset accuracy. Earlier subset scores remain intact.

| Model | Correct / images | Accuracy |
|---|---:|---:|
| baseline | 50 / 50 | 100.0000% |
| candidate | 50 / 50 | 100.0000% |

Only accuracy was rechecked; power and latency were not remeasured. Use the separate repeated CPU measurements for latency and energy.
<!-- END MAC HALF SAMPLE RESULTS -->

<!-- BEGIN MAC 15000 SAMPLE RESULTS -->
### 15,000-image evaluation

15,000 images, 1,500 per class. The same seeded image indices were used for both models. Both models ran on CPU with four threads. Input resolution and model weights are unchanged. This is a diagnostic training-pool subset, not held-out or full-dataset accuracy. Earlier subset scores remain intact.

| Model | Correct / images | Accuracy | Worst-class recall |
|---|---:|---:|---:|
| baseline | 14,864 / 15,000 | 99.0933% | 95.80% |
| candidate | 14,855 / 15,000 | 99.0333% | 97.20% |

Only accuracy was rechecked; power and latency were not remeasured. Use the separate repeated CPU measurements for latency and energy.
<!-- END MAC 15000 SAMPLE RESULTS -->

<!-- BEGIN MAC FULL DATA RESULTS -->
### 27,000-image evaluation

All 27,000 images were evaluated, preserving the dataset's original unequal class counts. The same seeded image indices were used for both models. Both models ran on CPU with four threads. Input resolution and model weights are unchanged. This is complete training-set accuracy, not held-out generalization. Earlier subset scores remain intact.

| Model | Correct / images | Accuracy | Worst-class recall |
|---|---:|---:|---:|
| baseline | 26,774 / 27,000 | 99.1630% | 95.84% |
| candidate | 26,760 / 27,000 | 99.1111% | 97.24% |

Only accuracy was rechecked; power and latency were not remeasured. Use the separate repeated CPU measurements for latency and energy.
<!-- END MAC FULL DATA RESULTS -->
