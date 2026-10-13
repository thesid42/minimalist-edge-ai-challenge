# 16. Smaller recovered192 architectures with FP32 weights

The team now requires FP32 throughout and all 27000 images for every new comparison. Physically reduce late expansion channels, final feature channels or one residual block. Save exact edits and FP32 state, verify replay and export, measure initial full-data accuracy, recover head640 and drop16 with a full-backbone GPU epoch, then recover the initial head640 with only 6410 classifier parameters trained. GPU AMP remains disabled. Reevaluate all three recovered models on the entire pool on CPU. Matched runtime/estimated energy compares original224, parent192, classifier-recovered head640 and recovered drop16 at FP32, with three fresh serial repetitions each. Older FP16 winners remain preserved.

| label | parameters | size_mib | accuracy_pct | worst_recall_pct | examples | median_cpu_4t_ms | estimated_cpu_4t_j |
|---|---|---|---|---|---|---|---|
| Original student / FP32 | 2236682 | 8.506295 | 99.162963 | 95.84 | 27000 | 2.53215 | 0.052934 |
| Recovered192 reference / FP32 | 2236682 | 8.510991 | 99.107407 | 97.24 | 27000 | 1.96615 | 0.041391 |
| Late expansion channels / before recovery | 2108202 | 8.023564 | 98.318519 | 96.12 | 27000 | pending / n/a | pending / n/a |
| Final 640 channels / before recovery | 2024202 | 7.702886 | 98.996296 | 97.4 | 27000 | pending / n/a | pending / n/a |
| Residual block 16 removed / before recovery | 1916682 | 7.295685 | 96.42963 | 87.8 | 27000 | pending / n/a | pending / n/a |
| Final 640 / full-backbone recovery | 2024202 | 7.702886 | 98.881481 | 97.65 | 27000 | pending / n/a | pending / n/a |
| Remove block 16 / full-backbone recovery | 1916682 | 7.295685 | 98.622222 | 95.533333 | 27000 | 1.8932 | 0.039281 |
| Final 640 / classifier-only recovery | 2024202 | 7.702886 | 99.148148 | 97.4 | 27000 | 1.9791 | 0.040847 |

## Implementation and exact settings

Open [details.json](details.json) for model hashes, pruning/quantization/recovery settings and evidence paths.

- **Original student / FP32** (`original224_fp32`): [results/all_data/baseline_accuracy.json](../../results/all_data/baseline_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/resolution.py](../../edgeprep/resolution.py).
- **Recovered192 reference / FP32** (`parent192_fp32`): [results/all_data/resize192_head_ce_accuracy.json](../../results/all_data/resize192_head_ce_accuracy.json); [edgeprep/challenge.py](../../edgeprep/challenge.py), [edgeprep/resolution.py](../../edgeprep/resolution.py).
- **Late expansion channels / before recovery** (`initial_late_expansion10`): [results/fp32_compression/late_expansion10_accuracy.json](../../results/fp32_compression/late_expansion10_accuracy.json); [scripts/fp32_structural_screen.py](../../scripts/fp32_structural_screen.py), [edgeprep/structural.py](../../edgeprep/structural.py), [edgeprep/challenge.py](../../edgeprep/challenge.py).
- **Final 640 channels / before recovery** (`initial_head640`): [results/fp32_compression/head640_accuracy.json](../../results/fp32_compression/head640_accuracy.json); [scripts/fp32_structural_screen.py](../../scripts/fp32_structural_screen.py), [edgeprep/structural.py](../../edgeprep/structural.py), [edgeprep/challenge.py](../../edgeprep/challenge.py).
- **Residual block 16 removed / before recovery** (`initial_drop16`): [results/fp32_compression/drop16_accuracy.json](../../results/fp32_compression/drop16_accuracy.json); [scripts/fp32_structural_screen.py](../../scripts/fp32_structural_screen.py), [edgeprep/structural.py](../../edgeprep/structural.py), [edgeprep/challenge.py](../../edgeprep/challenge.py).
- **Final 640 / full-backbone recovery** (`head640_ce`): [results/fp32_compression/head640_ce_accuracy.json](../../results/fp32_compression/head640_ce_accuracy.json); [edgeprep/structural.py](../../edgeprep/structural.py), [edgeprep/challenge.py](../../edgeprep/challenge.py), [scripts/fp32_structural_screen.py](../../scripts/fp32_structural_screen.py).
- **Remove block 16 / full-backbone recovery** (`drop16_ce`): [results/fp32_compression/drop16_ce_accuracy.json](../../results/fp32_compression/drop16_ce_accuracy.json); [edgeprep/structural.py](../../edgeprep/structural.py), [edgeprep/challenge.py](../../edgeprep/challenge.py), [scripts/fp32_structural_screen.py](../../scripts/fp32_structural_screen.py).
- **Final 640 / classifier-only recovery** (`head640_head_ce`): [results/fp32_compression/head640_head_ce_accuracy.json](../../results/fp32_compression/head640_head_ce_accuracy.json); [edgeprep/structural.py](../../edgeprep/structural.py), [edgeprep/challenge.py](../../edgeprep/challenge.py), [scripts/fp32_structural_screen.py](../../scripts/fp32_structural_screen.py).

## Supporting evidence

- [docs/fp32_model_reduction.md](../../docs/fp32_model_reduction.md)
- [results/fp32_compression/summary.json](../../results/fp32_compression/summary.json)
- [scripts/fp32_structural_screen.py](../../scripts/fp32_structural_screen.py)
- [edgeprep/structural.py](../../edgeprep/structural.py)
