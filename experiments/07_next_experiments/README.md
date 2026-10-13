# 07. Next experiments, not yet measured

Same-device W8A8 vs W8A16 is now measured in step 06. The equal-precision baseline storage control is now in step 15; use it in future comparisons and measure its matched latency/energy before attributing gains to a particular knob. Prioritize fixed-count MSE vs MinMax calibration, beta Lite-MP, 1-2% expansion pruning with full-data CE/KD, and BN recalibration. Compare every candidate against a reference on the same device/runtime. Full-data classifier-only recovery of the 192 variant is recorded in step 08 when complete; full-backbone pruned-model CE/KD remains pending.


## Supporting evidence

- [ANALYSIS.md](../../ANALYSIS.md)
- [docs/ai_hub.md](../../docs/ai_hub.md)
