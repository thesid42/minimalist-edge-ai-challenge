# Confirmed challenge and exploration plan

The revised brief fixes the problem: compress an already-distilled torchvision MobileNetV2, trained on EuroSAT RGB satellite images, for ten land-use classes. ResNet-50 is an optional recovery teacher. Input remains 1 x 3 x 224 x 224. The strongest initial direction is **selective static quantization**, followed by cautious expansion-channel pruning with recovery if its measured benefit justifies the accuracy cost.

Small language models do not solve this supplied image-classification task. Replacing the supplied student with an unrelated architecture also weakens checkpoint reproducibility and would require organizer clarification. Work from the provided student.

## What is confirmed and what remains missing

The PDF's only downloadable Drive links point to the two supplied model files. It contains no download for the official split, baseline metrics or starter notebook. It explicitly says these are provided at kickoff. Local fallback testing is useful, but cannot claim official accuracy.

EuroSAT RGB is downloaded and checked: 27,000 images, ten alphabetically ordered classes, no unreadable files, matching torchvision archive checksum. Preprocessing follows the brief exactly: Resize(224), ToTensor, Normalize mean [.485,.456,.406], std [.229,.224,.225]. This is RGB, not the 13-band multispectral dataset. Dataset order is hashed because a numeric split is meaningful only against the same ordered file list.

The diagnostic split is seed-42 class-stratified 80/20, with 21,600 train and 5,400 validation images. The pilot uses the same balanced 500 validation images for every candidate, 50 per class. Calibration uses only the provisional training partition. **The checkpoints' original training membership is unknown; provisional validation may overlap it.** All pilot accuracy is diagnostic and may be optimistic.

Outstanding: official val_split.json, baseline_metrics.json, starter notebook, AI Hub account configuration and common device profile. If organizers explicitly permit a self-created split, retain its manifest and use exactly the same indices for baseline and final; do not silently treat this provisional split as theirs.

## Correct denominator

Measured directly from the supplied files and exact architectures:

| Model | Trainable parameters | Checkpoint MiB | THOP MACs at 224 |
|---|---:|---:|---:|
| MobileNetV2 student | 2,236,682 | 8.772 | 0.3262 G |
| ResNet-50 teacher | 23,528,522 | 90.063 | 4.1317 G |

The brief's baseline table lists [23.53] M parameters, [4.13] G MACs and [22.67] MB, which is internally inconsistent with the provided student. Its parameter/MAC figures match the teacher. The bracketed accuracy and missing worst-class value are also placeholders. **Do not reuse the earlier pre-release 10x compression bands**: the new PDF has a weighted rubric but no numeric bands or exact combined score formula, despite referring to calibrated bands.

Record official JSON separately, reproduce baseline accuracy, and explain any discrepancy before submitting. Compression must be relative to the supplied MobileNetV2. Compare like-for-like ONNX file sizes as an engineering measurement; also report original checkpoint size and use the organizer's prescribed file denominator for official metrics. Quantization changes storage precision, not the number of learned parameters.

## Constraints and scoring implications

Accuracy/reliability 25 points, compression 20, RAM/latency 20, energy 20, reproducibility 10, engineering judgment 5. Mean accuracy alone is insufficient: record all ten recalls, worst-class recall, support and confusion matrix. A small candidate with class collapse is a poor choice even if it wins size.

CPU-only is the preparation assumption; no GPU is provided. Prefer experiments that complete in minutes. Export the actual final model to ONNX and measure that artifact. Use static QDQ with real-image calibration and shape preprocessing, opset 13 / dynamo=False. The fixed external input shape does not establish permission to internally downsample: the brief also says input stays 224, so treat lower internal resolution as a clarification-dependent idea.

Laptop profiling is explicitly allowed if AI Hub is unavailable. The updated brief no longer specifies Arduino UNO Q. Use the common device if organizers name one; do not revive the earlier board/proxy assumption.

Energy is a baseline/final ratio on the same machine. TDP estimation is allowed if labelled. Never rank teams by raw laptop Joules or derive unsupported board energy. Long enough measurement windows, consistent input/thread count and logging of power-estimation mode make the claim reproducible.

## Initial evidence

See results/challenge/summary.csv and summary.json for current consolidated measurements. Each candidate uses the same 500 provisional validation examples. No recovery is applied unless the candidate name explicitly says recovery.

Initial accuracy: FP32 baseline 99.2%; full S8S8 MinMax with 128 calibration images 92.6%; U8U8 at 128 images 91.4%; S8S8 MinMax at 256 images 91.0%; Percentile at 256 images 96.2%; keeping the first/last convolutions and classifier FP32 at 256 images 96.0%; also protecting depthwise layers 98.4%, with worst-class recall 94% versus baseline 96%.

This is a useful controlled comparison: 256-image MinMax versus 256-image Percentile isolates the calibration method; 256-image full MinMax versus end-protected versus end-and-depthwise-protected isolates precision policy. Increasing sample count alone did not help in this run. U8U8 is worth testing on AVX2 but did not recover accuracy here. A local gain does not establish device-toolchain compatibility or official-split accuracy.

Uniform structural pruning, nominal 5% / 10% with channel multiples of eight, dropped accuracy to 16.0% / 17.2%. Dependency handling and export parity pass, but both models collapse semantically. Expansion-only pruning, preserving bottleneck outputs/residual widths and ranking BN-scaled projection-column norms, improves the starting point but still drops to 83.2% / 51.0% without recovery. Nominal channel ratio is not parameter reduction: expansion 5% removes about 4.35% of parameters, while uniform 5% removes about 10.41%. Rounding can make small layers lose a larger fraction than the headline ratio.

The initial results support protecting cheap sensitive layers and targeting high-parameter pointwise convolutions. They do **not** support choosing a pruned model without recovery. Three pipeline tests pass: split misuse rejection, balanced partition sampling, and deterministic dependency-safe replay / fixed-input ONNX export of the actual student. CPU dependency integrity passes. An eight-image teacher-guided recovery smoke test completes and its state replays into a valid ONNX graph; it is not an accuracy-improvement claim.

## Exploration priorities

| Priority | Area | Controlled experiment | What would justify keeping it |
|---|---|---|---|
| 1 | Selective static INT8 | Full quantization vs protect ends vs protect ends + depthwise, same calibration | Near-baseline overall/worst-class recall with smaller storage and measured runtime benefit |
| 1 | Calibration sensitivity | Balanced training samples 64/128/256/512; MinMax vs Percentile vs Entropy at fixed sample count | Improved vulnerable-class recall; stable result on full official validation |
| 2 | Layer sensitivity | Quantize/prune one block at a time; compare margin changes and recall | Identify cheap layers to protect and safe pointwise channels to remove |
| 2 | Expansion-only pruning | 1/2/5% first; round-to 1 vs 8; preserve bottlenecks, stems and head | Physical parameter/MAC reduction that survives short recovery |
| 2 | Recovery cost versus accuracy | Same pruned state, same 500/1000 training images, one epoch: hard-label CE vs CE + teacher KL | Retained worst-class recall and a documented CPU-time budget |
| 3 | BN-stat recalibration | Recompute BN statistics on training-only images without gradients; compare frozen-BN recovery | Cheap recovery of distribution shifts without many optimization steps |
| 3 | Prune -> recover -> quantize | Best recovered small model, recalibrated on representative training images | Gains on both parameter count and disk footprint without compounded accuracy collapse |
| 3 | Teacher-logit caching | Cache ResNet logits for a fixed deterministic training subset, reuse across recovery trials | Less CPU time than repeated teacher inference with equivalent loss/data |
| 4 | Runtime execution | Same graphs with one/four threads; verify graph fusion and hardware target | Consistent latency/RAM/energy improvements without changing classification |
| 4 | QAT or block removal | Only if PTQ/selective precision cannot meet accuracy and time remains | Clear advantage that exports and profiles within the remaining budget |

For cached logits, bind the cache to teacher SHA, training indices, preprocessing and any augmentations. The current recovery has deterministic evaluation preprocessing; augmented images would invalidate a single cached logit per image. Freeze teacher weights, use no_grad, temperature-squared KL, and measure wall time. More teacher compute is a cost to justify, not an automatic requirement.

Avoid spending time on mask-only unstructured sparsity as a size/latency optimization: standard dense storage and kernels retain the zeros. Avoid global aggressive pruning, uncontrolled architecture replacement, dynamic convolution quantization, and accuracy claims from the teacher denominator. FP16 is not a default CPU speedup; test only if the actual runtime/target supports it efficiently. Four-bit weight-only transformer examples do not automatically apply to convolutional MobileNetV2.

## Experiment protocol and selection

Reproduce the official student first. Keep its exact original weights unchanged. Use the organizer split and preprocessing for every accuracy comparison; separate training/calibration data from validation. Log source and output hashes, parameters, MACs, file bytes, confusion matrix, all recalls, sample count, technique settings and CPU training time after every stage.

Timing uses isolated fresh processes: one model, batch one, fixed CPU thread count, 20 warmups, 300 timed runs, median/mean/P95 and 5-ms RSS polling. Accuracy evaluation runs separately, so importing PyTorch for dataset preprocessing does not contaminate the reported ONNX-worker RAM. RSS includes the Python/runtime process footprint; it is not just weights or activation memory. A single real image is sufficient for repeatable timing, not for dataset accuracy.

Treat accuracy, worst-class recall, size, parameters, RAM, latency and energy as separate axes. Plot the Pareto curve. Keep a high-accuracy FP32 fallback until a candidate passes the official full validation and runtime checks. Retain documented failures: the scoring explicitly rewards decisions supported by negative results.

ONNX-size ratio and checkpoint-size ratio can differ due to serialization, folding and quantization metadata. Publish exact bytes and both formats instead of mixing denominators. CPU thread tuning must be applied identically to baseline and final. A parameter reduction need not reduce latency when operator overhead or alignment dominates.

## Competition time budget

1. First 30-45 minutes: obtain official split/baseline, verify dataset order, reproduce student, export FP32 and submit an early working AI Hub job if configured.
2. Next 60-90 minutes: controlled calibration / selective-precision sweep. Full validation for the best two choices; capture per-class failures.
3. Next 90-120 minutes: tiny expansion pruning, short CE/KD or BN recovery; reject class collapse quickly on a diagnostic subset before expensive full validation.
4. Next 60 minutes: quantize the best recovered state and compare against the best unpruned selective-INT8 graph.
5. Reserve at least 90 minutes at the end: same-method baseline/final timing, memory and energy, full validation, final Hub/local profile, Pareto plot, metrics.json, 1-2-page report and clean reproduction run.

With multiple teammates, assign evaluation/reproducibility, quantization, pruning/recovery, and profiling/reporting. Avoid simultaneous CPU-heavy experiments during measurements. No exact scoring optimizer is possible until organizers clarify the final numerical bands or formula; select a defensible Pareto point rather than chase the obsolete pre-release 10x target.

## Submission and remaining work

The brief requires one compressed model (preferably quantized ONNX), complete metrics.json, a 1-2-page technical report with tradeoff curve, and source that rebuilds it from the supplied checkpoint. Current results are preparation, not a final submission. Do not fill missing official fields with fabricated numbers.

AI Hub authentication / common target are pending. The installed SDK helper records requests, job IDs and outputs; its live path is untested. Laptop fallback is explicitly allowed. Configure credentials locally; no secrets are stored in the repository.

Earlier CIFAR code, results and generated practice models are preserved under the local archive. The GPU environment, uv download cache and CIFAR dataset remain on disk because automatic approval review rejected their broad permanent deletion. They are excluded from the public repository and not used by the active challenge pipeline.

## Primary technical references

- [Torchvision EuroSAT RGB documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html)
- [ONNX Runtime quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html): static quantization, preprocessing, QDQ, calibration and selective operator quantization.
- [Torch-Pruning / DepGraph](https://github.com/VainF/Torch-Pruning): physical dependency-aware channel removal.
- [PyTorch knowledge distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)
- [Qualcomm AI Hub SDK documentation](https://workbench.aihub.qualcomm.com/docs/hub/api.html)
