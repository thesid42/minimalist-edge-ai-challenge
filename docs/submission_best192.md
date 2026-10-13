# Smaller 192 FP32 MobileNetV2 - submission technical report

We select a physically smaller MobileNetV2 with internal 192-pixel inference and FP32 weights. It keeps 99.1481% accuracy on all 27,000 EuroSAT images while reducing file size 9.4%, CPU latency 21.8% and estimated energy 22.8% against our unchanged FP32 baseline.

## Model and input contract

Start with the supplied 10-class MobileNetV2 and the saved recovered192 FP32 weights. The parent was adapted by one classifier-only cross-entropy epoch on all 27,000 images. Keep the external FP32 input (1,3,224,224); resize internally to 192x192 with bilinear interpolation (align_corners=False, no antialias). The output stays (1,10), and export uses ONNX opset 13.

## Structural compression and recovery

Physically remove 640 of the final 1,280 feature channels. Rank channels by absolute BN scale times the L2 norm of their classifier-weight column. Torch-Pruning propagates removal through the final convolution, BN and classifier inputs; earlier layers and residual widths are retained. This removes 212,480 parameters; it does not store zero masks or use reduced-precision weights.

Recover the pruned classifier only: 6,410 trainable parameters, one cross-entropy epoch, learning rate 1e-4, batch 32, seed 42, all 27,000 images. Freeze the backbone and BN running statistics. Final recovery ran on the RTX 3070 Laptop GPU in FP32, with AMP and TF32 disabled, and took 37.06 seconds. Export and every reported local inference run use CPU FP32. Only classifier weight and bias changed during recovery.

## Matched results

| Metric | Original 224 FP32 | Recovered 192 FP32 | Selected smaller 192 |
|---|---:|---:|---:|
| Accuracy / 27,000 | 99.1630% | 99.1074% | 99.1481% |
| Worst class recall | 95.84% | 97.24% | 97.40% |
| Parameters | 2,236,682 | 2,236,682 | 2,024,202 |
| ONNX size (MiB) | 8.506 | 8.511 | 7.703 |
| MACs (million) | 326.220 | 239.675 | 232.204 |
| CPU median (ms/image) | 2.5321 | 1.9662 | 1.9791 |
| Sampled peak RSS (MiB) | 86.17 | 79.84 | 86.03 |
| Estimated energy (J/image) | 0.052934 | 0.041391 | 0.040847 |

The final model loses only 0.0148 percentage points versus baseline (four images), while worst-class recall improves 1.56 points. It is 9.5% smaller and gets eleven more images correct than unpruned recovered192. Timing and energy are similar to that parent within run variability; feature pruning does not demonstrate extra CPU acceleration. Sampled peak RSS is essentially unchanged versus baseline, so we claim no RAM reduction. THOP MACs exclude resize overhead.

## Tradeoff curve and selection

Keep classifier-only recovery: pruning starts at 98.9963% accuracy, and recovery restores it to 99.1481% with 97.40% worst-class recall. Discard the separate full-backbone recovery of this same 640-channel architecture: one epoch at learning rate 5e-5 reduced accuracy to 98.8815% (worst class 97.65%). No quantization is applied because this selected submission uses FP32 throughout.

The two-page PDF includes the selected pipeline tradeoff curve, including the discarded full-backbone recovery of the same architecture.

[Two-page PDF](../output/pdf/best192_fp32_technical_report.pdf)

## Measurement protocol

Organizer clarification: train and evaluate on all 27,000 images with our own laptop baseline; these are training-set accuracies. RGB images are resized to 224 and ImageNet-normalized (mean 0.485/0.456/0.406; std 0.229/0.224/0.225). Hardware: Ryzen 7 5800H, 27.86 GiB RAM, Windows 10; ONNX Runtime 1.31.0 CPUExecutionProvider, four threads, batch one. Three serial fresh processes per model use rotating order, 100 warmups and 1,000 timings of one real image. Accuracy separately covers the entire pool.

CodeCarbon 3.3.1 uses CPU TDP/load estimation (45 W) plus a RAM heuristic; GPU is excluded and no training runs concurrently. Three 30-second windows give baseline 0.052934 +/- 0.001388 J/image and final 0.040847 +/- 0.000989 J/image (mean +/- sample SD). These are same-laptop estimates, not physical meter readings or edge-device energy. The selected model has no AI Hub profile yet.

## Per-class recall

| Class | Recall |
|---|---:|
| AnnualCrop | 98.90% |
| Forest | 99.60% |
| HerbaceousVegetation | 99.07% |
| Highway | 99.20% |
| Industrial | 99.52% |
| Pasture | 98.45% |
| PermanentCrop | 97.40% |
| Residential | 99.73% |
| River | 99.28% |
| SeaLake | 99.90% |

## Reproduction

[Saved weights and evidence](../best_models/resize192_head640_head_ce_fp32_8655a532a076/README.md). [Exact implementation and commands](fp32_model_reduction.md).
ONNX SHA-256: `8655a532a07639d47592b634e9596dd36c808ee1632a835fab1f6af8be4612df`.

[Repository](https://github.com/thesid42/minimalist-edge-ai-challenge). Source: challenge PDF sections 2, 3, 5, 6 and 9; the organizer all-data/laptop-baseline clarification governs evaluation.
