# 27,000-image Mac evaluation

### 27,000-image evaluation

All 27,000 images were evaluated, preserving the dataset's original unequal class counts. The same seeded image indices were used for both models. Both models ran on CPU with four threads. Input resolution and model weights are unchanged. This is complete training-set accuracy, not held-out generalization. Earlier subset scores remain intact.

| Model | Correct / images | Accuracy | Worst-class recall |
|---|---:|---:|---:|
| baseline | 26,774 / 27,000 | 99.1630% | 95.84% |
| candidate | 26,760 / 27,000 | 99.1111% | 97.24% |

Only accuracy was rechecked; power and latency were not remeasured. Use the separate repeated CPU measurements for latency and energy.
