# EuroSAT compression: current plan

The organizers clarified through the team that **no split will be supplied**, training may use **all 27,000 EuroSAT images**, and each team establishes its own baseline by running the supplied MobileNetV2 on its laptop. This supersedes the PDF's split instructions and printed baseline figures. The optional ResNet-50 is a recovery teacher, not the baseline.

The pipeline now follows this protocol by default. `recover --train-limit 0` uses all images and is the default. `evaluate --limit 0` measures all images and is also the default. Calibration draws a representative balanced subset from the full training pool. Accuracy on that pool is labelled **training-set accuracy**; it does not estimate held-out performance.

## Our baseline

Measure the unchanged supplied student before any recovery training or compression. It has **2,236,682 parameters**, approximately **0.32622G THOP MACs**, and **8.5063 MiB FP32 ONNX size**. Its original checkpoint is 8.7723 MiB; size ratios for comparisons use ONNX against ONNX.

Full-data evaluation of the original student gives **99.16296% accuracy** and **95.84% minimum class recall**, for PermanentCrop. The class support sums to 27,000. All candidate accuracy comparisons use exactly the same image indices, preprocessing and class ordering.

`baseline_metrics.json` is generated from full-data accuracy and fresh laptop CPU runtime/RAM/energy measurements. `configs/pdf_baseline_reference.json` preserves the earlier printed reference, with its discrepancy explanation. It has no role in current ratios. Current evidence is under `results/all_data`; the original 500-image pilot is under `results/challenge`.

Preprocessing remains Resize(224), ToTensor and ImageNet normalization. Fixed input is (1, 3, 224, 224). Dataset archive checksum and all 27,000 images have been verified, and sample ordering is hashed. The local CPU environment is tested and pinned. Current code uses no GPU.

## Current full-data comparison

| Metric | Unchanged FP32 student | Selective INT8 |
|---|---:|---:|
| Training-set accuracy, all 27,000 images | 99.16296% | 98.41111% |
| Worst-class recall | 95.84% | 95.43% |
| Learned parameters | 2,236,682 | 2,236,682 |
| Exported ONNX MiB | 8.5063 | 3.7720 |
| CPU mean latency, 4 threads | 2.2676 ms | 5.2400 ms |
| CPU median latency, 4 threads | 2.1036 ms | 5.1842 ms |
| Peak process RSS, 4 threads | 78.73 MiB | 88.11 MiB |
| Estimated energy, 4 threads | 0.04958 J/inference | 0.11916 J/inference |

Selective precision loses 0.75185 percentage points of overall accuracy and 0.40667 points of worst-class recall. Storage is 2.255x smaller, but four-thread median latency is 2.465x higher and estimated energy is 2.403x higher. It remains a storage candidate rather than a winner across efficiency axes. The full-data vulnerable class shifts from PermanentCrop in FP32 to AnnualCrop in selective INT8; that makes crop-class calibration/layer sensitivity a useful next experiment.

Both graphs were measured in separate processes, serially, with identical timing input, provider, warmup/run counts and thread settings. Energy used 20-second four-thread windows; startup logs verify TDP/CPU-load estimation. Baseline mean latency is the primary `latency_ms` field, with median and P95 also retained. No teacher or PDF denominator is used.

## Evidence from the earlier pilot

The historical pilot used a temporary split, 500 evaluation images and short recovery on 500 training images. Those results are screening evidence, not current full-data results or claims of full-data training. The complete record is in [historical_pilot_analysis.md](docs/historical_pilot_analysis.md).

- Full MinMax S8S8 at 256 calibration images reached 91.0% diagnostic accuracy; Percentile at the same sample count reached 96.2%.
- Protecting the ends and depthwise convolutions reached 98.4%, versus 99.2% for the pilot FP32 student. It reduced ONNX size about 2.25x, with unchanged parameter count.
- That selective INT8 graph was about 40% slower at one CPU thread, and slower at four threads too. It did not improve process RAM or estimated energy on this laptop. Keep the FP32 fallback.
- Uniform channel pruning collapsed class performance even at nominal 5%. Expansion-only 5% was better, but fell to 83.2% before recovery.
- One short epoch on 500 training images recovered expansion-pruned accuracy to 95.6% with CE and 96.0% with teacher guidance. Worst-class recalls remained 76% and 82%. The two-image KD gain is not statistically strong evidence of superiority.

Selective INT8 is freshly calibrated using the full training pool and evaluated on all data for the current comparison. At the original protocol update, full-data recovery had not yet run. The later classifier-only full-data recovery is recorded below; full-backbone pruning recovery remains pending. The commands now support it; budget its GPU training time separately from the much shorter full-dataset inference benchmark.

## Experiments to prioritize

| Priority | Area | Controlled experiment | Decision criterion |
|---|---|---|---|
| 1 | Selective precision | Full QDQ vs protected ends vs protected ends + depthwise, same calibration | Retain overall and minimum class recall with useful size/runtime tradeoffs |
| 1 | Calibration sensitivity | 64/128/256/512 balanced samples; fixed-count MinMax vs Percentile vs Entropy | Stable full-data recall for vulnerable classes |
| 2 | Layer sensitivity | Quantize or remove channels from one block at a time | Identify safe pointwise layers and cheap sensitive layers worth protecting |
| 2 | Expansion-channel pruning | Start 1/2/5%, preserve bottleneck outputs/residual dimensions, compare round-to 1 and 8 | Physical parameter/MAC reduction that survives recovery |
| 2 | All-data recovery | Same pruned state, one epoch over all 27,000 images, CE vs teacher KL + CE | Accuracy and worst-class recovery within the available GPU training budget |
| 3 | Teacher-logit caching | Cache teacher outputs for exact deterministic training inputs | Lower repeated recovery cost with matching distillation targets |
| 3 | BN recalibration | Update BN running statistics on the full training pool without gradients | Cheap recovery before longer optimization |
| 3 | Prune -> recover -> quantize | Recalibrate the best recovered student | Gains in parameter count and disk size without compounded accuracy loss |
| 4 | Actual-target execution | Profile the shortlisted graphs on the common AI Hub device if available | Check whether ARM/device kernels improve INT8 beyond laptop behavior |

Cache identity must include teacher hash, preprocessing, image ordering and augmentations. A single cached output per image is valid for deterministic inputs; augmented variants require corresponding targets. Teacher weights stay frozen, outputs use no_grad, and KL is scaled by temperature squared. Compare identical training order and seeds.

Avoid mask-only sparsity as a dense-storage/CPU speed optimization. Avoid aggressive uniform pruning, unsupported dynamic convolution quantization, and substituting unrelated LLMs for the specified image task. PDF section 3 permits lower spatial resolution while preserving the external 224x224 input. An explicit resize inside the exported graph is a valid experiment; accuracy and interpolation must be measured. No current numerical scoring bands justify reviving the old 10x compression target.

## Measurement and selection

For each candidate record training-set accuracy, every class recall, worst class, confusion matrix, parameter count, MACs, exact file bytes, latency, peak RSS and energy. Baseline and candidate must use the same data, CPU thread count, provider and measurement method. Preserve source/output hashes and recipe settings.

Runtime measurements use fresh ONNX-only processes, fixed batch one, 20 warmups, 300 timed runs, mean/median/P95 and 5-ms RSS sampling. Dataset accuracy is computed in a separate process so PyTorch preprocessing imports do not contaminate the ONNX runtime RAM claim. Compare process peak RSS rather than pretending it is weight memory alone.

CodeCarbon uses a TDP/CPU-load estimate on this Windows/AMD laptop. Measure baseline and candidate with identical thread settings and representative input, and at least a 20-second window; record mode and startup logs. These are CPU/RAM estimates, not hardware-meter or target-board energy.

Select a Pareto point across accuracy, minimum recall, size, parameters, RAM, latency and energy. A storage gain with latency/energy regressions is a tradeoff, not an overall efficiency win. Keep an unchanged FP32 fallback until the shortlisted candidate passes complete all-data evaluation and runtime checks.

## Competition budget and deliverables

First reproduce the full-data laptop baseline and export a working fallback. Spend the initial experiment window on calibration/selective precision. Next try tiny expansion pruning and budget one all-data recovery epoch deliberately; new recovery and teacher computation use the RTX 3070 GPU, while all local evaluation/inference remains on CPU. Reserve at least 90 minutes for final full-dataset evaluation, same-method baseline/final runtime and energy, profiling, curve/report generation, and reproduction from the original checkpoint.

Deliver one model, complete metrics.json, a 1-2-page technical report with tradeoff curve, and reproducible source. Include the organizer's all-data clarification in the report so the methodology is explicit. AI Hub remains optional when unavailable; laptop profiling is allowed. No final competition model is selected yet.

The earlier obsolete GPU environment, installation cache and generated CIFAR data/models were removed after explicit approval, recovering 8.15 GiB. Current CPU dependencies, both original checkpoints, EuroSAT and current experiments remain intact. The GitHub repository is public and Anmol-tech's write invitation was sent.

## New experiments and readable log

Use [the numbered experiment log](experiments/README.md) and [the summary notebook](experiments/Experiment_Log.ipynb) as the entry point. They show historical pilots, current all-data measurements and remote trials separately, including failures. Regenerate with `python -m edgeprep.experiment_log` after saved results change.

Selective Percentile calibration with 512 samples achieved **98.80370% training-set accuracy**, **95.36% worst-class recall** (PermanentCrop), and **3.772 MiB** ONNX size. Four-thread median latency was **5.52065 ms**, peak RSS **87.94 MiB**, and TDP/CPU-load estimated energy **0.13219 J/inference**. It improves overall accuracy versus the earlier selective 256-image MinMax run but has slightly lower worst-class recall and remains slower than FP32 on this laptop. Both algorithm and count changed, so this is configuration screening, not evidence that either change alone caused the improvement. Fresh timing/energy measurement happened after full-data evaluation and while remote jobs waited; no concurrent CPU training/evaluation was running. A paired fresh FP32 remeasurement is appropriate before ranking close speed differences.

AI Hub access is now verified using the ignored `.env` file. Galaxy S24 CPU profiling reports 5.533 ms; the corrected ONNX/QNN NPU route reports 0.366 ms and 104/104 profiled operations on NPU. These are Hub latency estimates on remote hardware, separate from laptop timing. The original numerical smoke checks used 32 representative images; the later label-based device accuracy checks use 1,000 balanced images. Read [the AI Hub guide](docs/ai_hub.md) for device catalogue, calibration-format differences, current status and reproducible commands. The first NPU inference rejected conflicting compute/provider flags; both failed jobs and corrected retry are retained. RB3 QNN DLC compiled but profiling failed with MODEL_GRAPH_ERROR; the ONNX/QNN alternative successfully profiles at 0.614 ms, with 104 NPU operations and two CPU I/O conversion operations. Its uncompressed asset is 8.797 MiB and retains floating-point weights, so it is a deployment experiment rather than a storage win. See the log for numerical-check status.

## Next-priority execution (2026-10-13)

Rechecked the challenge PDF: no device is named; sections 5.4-5.5 ask for exported ONNX timing and CodeCarbon energy, section 9 scores same-laptop ratios. Section 3 allows lower resolution if the external input remains (1,3,224,224).

New work is detailed in [next_priority_results.md](docs/next_priority_results.md): labelled 1,000-image S24/RB3 checks; RB3 W8A16; NPU performance modes; repeated laptop thread/energy tests; internal 192/160 resizing; full-data classifier-only recovery; and FP16 weight storage with FP32 CPU compute. Keep source-specific device numbers separate from the new portable CPU candidate.

Unrecovered 192 reached 98.00%, unrecovered 160 reached 93.0037% with worst Pasture recall 78.05%. Reject 160 without further recovery. One 192 classifier-only CE epoch used all 27,000 images, batch32, LR1e-4, frozen backbone and BN; 12,810 head parameters changed, verified by exact state comparison. It took 540.9 seconds and reached 99.1074% overall / 97.24% minimum class recall. The separately validated FP16-storage candidate reaches 99.1111% / 97.24%, is 4.28825 MiB, and retains FP32 CPU computation. This is a 0.05185 percentage-point loss versus the original 99.16296%, with improved minimum recall. It has unchanged learned parameter count (2,236,682); disk savings do not imply smaller inference RAM. Runtime/energy results use new matched repeated references and are recorded in the linked report when complete.

The first larger remote checks: S24 CPU and NPU each 98.8%; RB3 W8A8 98.3%. The subset has 100 examples per class and excludes 512 calibration indices; it remains a diagnostic training-pool subset. Device accuracy is label-based, not numerical agreement. Do not compare this balanced 1,000 score directly against frequency-weighted 27,000-image scores.


## Completed follow-up and training policy

Matched four-thread CPU measurements on the recovered 192 / FP16-storage candidate: 2.05385 ms versus fresh baseline 2.7892 ms (26.4% reduction), and estimated energy 0.0453376 versus 0.0549272 J/inference (17.5% reduction). Three fresh serial repetitions, 100 warmups, 1,000 timings and 30-second energy loops. Energy is CodeCarbon CPU TDP/load plus system-RAM heuristic, not measured hardware energy. Sleeping workers worsened this laptop tradeoff. The new direct five-repetition latency-only comparison reports recovered192 FP32 2.10755 ms and weight16 2.1788 ms, with substantial OS variability retained; it does not establish a precision-related speed gain.

RB3 W8A16 completed: 1.056 ms, 985/1,000 correct, 88% worst-class recall, 104 NPU + 2 CPU I/O operations. W8A8 was 0.614 ms, 983/1,000 correct and 89% worst recall. W8A16 is slower and not a uniform accuracy improvement. S24 and RB3 balanced/power_saver profiles are complete in the log; accuracy comes from the same compiled artifact's burst run. AI Hub profiles here do not provide joules.

The user's later instruction permits GPU training and requires CPU laptop inference; it explicitly allows AI Hub CPU/NPU. `.venv-train-gpu` now contains matching PyTorch 2.8 CUDA 12.8, separate from CPU `.venv`. Real full-backbone CUDA/AMP training on 100 diagnostic images changed 158 tensors, kept BN buffers unchanged and saved finite CPU tensors; CPU export and inference passed. This is pipeline verification, not full-data training or a replacement candidate. New recovery defaults to CUDA and fails if unavailable. Earlier CPU classifier recovery remains valid historical evidence. Full-data pruned-model CE/KD remains a next experiment.

Refresh the README, result report, registry, numbered folders and notebook saved tables using `python scripts/refresh_results.py`; `scripts/refresh_results.ps1 -ExecuteNotebook` additionally executes and exports the readable notebook. Current evidence covers 43 recorded entries across 12 numbered steps. [Latest report](docs/next_priority_results.md) and [GPU setup](docs/gpu_training.md).
