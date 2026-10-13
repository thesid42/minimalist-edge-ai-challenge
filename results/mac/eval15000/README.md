# 15,000-image Mac evaluation

### 15,000-image evaluation

15,000 images, 1,500 per class. The same seeded image indices were used for both models. Both models ran on CPU with four threads. Input resolution and model weights are unchanged. This is a diagnostic training-pool subset, not held-out or full-dataset accuracy. Earlier subset scores remain intact.

| Model | Correct / images | Accuracy | Worst-class recall |
|---|---:|---:|---:|
| baseline | 14,864 / 15,000 | 99.0933% | 95.80% |
| candidate | 14,855 / 15,000 | 99.0333% | 97.20% |

Only accuracy was rechecked; power and latency were not remeasured. Use the separate repeated CPU measurements for latency and energy.
