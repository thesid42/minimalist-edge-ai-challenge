# 50-image Mac evaluation

### 50-image evaluation

50 images, 5 per class. The first half of the earlier 100-image sample was used. Both models ran on CPU with four threads. Input resolution and model weights are unchanged. This is a diagnostic training-pool subset, not held-out or full-dataset accuracy. Earlier subset scores remain intact.

| Model | Correct / images | Accuracy |
|---|---:|---:|
| baseline | 50 / 50 | 100.0000% |
| candidate | 50 / 50 | 100.0000% |

Only accuracy was rechecked; power and latency were not remeasured. Use the separate repeated CPU measurements for latency and energy.
