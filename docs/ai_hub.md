# AI Hub deployment and edge experiments

The tested Python 3.11 CPU environment already includes qai-hub 0.56.0. python-dotenv 1.2.4 is now pinned too. No local Qualcomm SDK or GPU environment is needed for hosted jobs. The credential lives in `.env` at the project root, under `QAI_HUB_API_TOKEN=`. The project scripts read it automatically. The stock qai-hub CLI uses its separate home-directory configuration; `.env` and `.qai_hub/` are excluded from Git. `.env.example` is safe to share. Never paste credentials into source or reports.

```powershell
# Verify the key and save the live catalogue, without printing credentials.
.venv\Scripts\python.exe -m edgeprep.hub_auth --check
# Alternative masked prompt, if you prefer a project-local SDK-format credential.
powershell -ExecutionPolicy Bypass -File scripts/configure_ai_hub.ps1
```

Authentication was verified on 2026-10-13: 78 entries, including families and proxies (not 78 distinct physical devices). The exact response is in `results/hub/devices.json`. This proves access to the catalogue; each device/runtime still needs successful compilation and inference.

| Device in our live catalogue | Chipset / platform | Reason to explore |
|---|---|---|
| Samsung Galaxy S23 (Family) | Snapdragon 8 Gen 2 / Android 13 | Older phone comparison |
| Samsung Galaxy S24 (Family) | Snapdragon 8 Gen 3 / Android 14 | First CPU vs NPU comparison |
| Samsung Galaxy S25 (Family) | Snapdragon 8 Elite for Galaxy / Android 15 | Newer phone comparison |
| Dragonwing RB3 Gen 2 Vision Kit | QCS6490 / Qualcomm Linux 1.6 | Embedded vision target, quantized NPU candidate |
| Snapdragon X Elite CRD | Snapdragon X Elite / Windows 11 | ARM laptop deployment |
| Arduino VENTUNO Q | QCS8275 / Ubuntu 24.04 | Actual hosted embedded target worth investigating |
| Dragonwing IQ-9075 EVK | QCS9075 / Qualcomm Linux 1.9 | Another embedded compute target |

These are exploratory targets; the organizers have not designated a common device. Arduino VENTUNO Q is a different device from the earlier UNO Q assumption.

Families route to a member of the named family; use an exact named device for final comparisons if necessary. A proxy measures substitute hardware and must not be presented as the named physical board. [Qualcomm device-selection documentation](https://workbench.aihub.qualcomm.com/docs/hub/devices.html).

## Run our supplied student, not an ImageNet replacement

Keep the original checkpoint, normalization, ten classes and fixed (1,3,224,224) contract. Use our exported `artifacts/baseline/model_fp32.onnx`. `--plan` is offline; remove it to submit jobs.

```powershell
# Matched remote CPU reference.
.venv\Scripts\python.exe -m edgeprep.hub_profile --model artifacts/baseline/model_fp32.onnx --device "Samsung Galaxy S24 (Family)" --inputs results/all_data/calibration512/calibration.npy --output results/hub/s24_cpu_fp32 --plan
# NPU floating-point route; check actual precision/placement in the profile.
.venv\Scripts\python.exe -m edgeprep.hub_profile --model artifacts/baseline/model_fp32.onnx --device "Samsung Galaxy S24 (Family)" --compute npu --inputs results/all_data/calibration512/calibration.npy --output results/hub/s24_npu_float --plan
# Real 512-image calibration for embedded INT8; compilation then inference.
.venv\Scripts\python.exe -m edgeprep.hub_profile --model artifacts/baseline/model_fp32.onnx --device "Dragonwing RB3 Gen 2 Vision Kit" --runtime qnn_dlc --compute npu --precision w8a8 --calibration results/all_data/calibration512/calibration.npy --inputs results/all_data/calibration512/calibration.npy --output results/hub/rb3_npu_w8a8 --plan
```

Generate that calibration file with `python -m edgeprep.challenge prepare --calibration-count 512 --output results/all_data/calibration512`. It uses the full training pool and the same deterministic preprocessing. The helper records hashes and submitted job IDs immediately. `--compiled-job ID` reuses a previously compiled asset; keep its device/runtime consistent. Downloaded deployment binaries remain local and excluded from Git.

The Hub quantization route compiles our FP32 source to optimized ONNX, calibrates using AIMET through `submit_quantize_job`, and compiles the resulting fake-quantized model for the target. Qualcomm documents that fake-quant ONNX retains floating-point weights and is its supported quantized compilation input. Our existing ORT QDQ with integer initializers should instead stay a local candidate; `--direct-onnx` can probe direct ORT execution separately. Use 512 representative inputs as a starting point; Qualcomm recommends roughly 500-1000. [Quantization documentation](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html).

## What to explore next

1. Measure CPU ONNX and NPU ONNX on the same S24 device family. Inspect per-layer compute units and numerical agreement; requesting NPU does not prove every layer used it.
2. Compare Hub W8A8 against W8A16 on the same embedded device. W8A16 spends activation memory to investigate whether crop-class accuracy recovers. Add `--precision w8a16` with ONNX or QNN DLC.
3. Compare MSE-based and MinMax calibration at the same 512 images. Then 512 vs 1000 at a fixed algorithm. Do not attribute a combined algorithm/count change to either factor alone.
4. Try beta Lite-MP with `--lite-mp 5` to raise a few sensitive layers to INT16; compare plain W8A8 first.
5. Compare `--performance burst`, `balanced`, and `power_saver` on the same NPU graph. These are runtime modes, not energy measurements.
6. Use conservative expansion-channel pruning (1-2%) and full-data CE/KD recovery before Hub calibration. Physical channel removal reduces parameters/MACs; quantization alone does not.
7. Investigate batch-normalization recalibration and per-block sensitivity before aggressive pruning. These remain proposed experiments, not measured wins.
8. Consider precompiled QNN context assets for initialization time on a settled chipset; they require deployment/runtime compatibility checks and should stay a separate track from the portable CPU ONNX submission.

The helper supports range selection and beta Lite-MP, and explicitly sets CPU threads and QNN performance mode. [Qualcomm API options](https://workbench.aihub.qualcomm.com/docs/hub/api.html).

QNN context binaries are specific to the target chipset; ONNX with embedded QNN assets is not equivalent to portable CPU ONNX. TensorFlow Lite and QNN DLC are additional deployment candidates, subject to challenge format rules. [Compilation documentation](https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html).

## Evidence rules

Profile with synthetic inputs for stable runtime, then run a separate 32-image numerical check using real normalized EuroSAT inputs. Record source hash, device, actual OS, runtime/version, requested and actual compute units, profile fields, downloaded asset size, and top-1 agreement/logit error. Numerical agreement is a smoke check, not full-data accuracy. [Inference documentation](https://workbench.aihub.qualcomm.com/docs/hub/inference_examples.html).

Keep laptop CPU baseline metrics unchanged. Establish a new reference on each target before computing its improvements. Remote target memory fields and laptop process RSS may have different definitions; compare only like for like. No energy measurement is claimed unless the returned evidence explicitly supplies a documented quantity with units. Cloud profiling is not remote training; recovery remains on our laptop.

## Initial execution findings

S24 CPU and corrected ONNX/QNN NPU profiling completed. CPU reports 5.533 ms; NPU reports 0.366 ms with all 104 profiled operations labelled NPU. The first NPU inference failed because the server rejects combining `--compute_unit` with `--onnx_execution_providers`; the retry uses the explicit QNN provider alone. The helper now checks successful status before downloading results. Raw request, jobs, profiles and numerical checks remain in `results/hub/`.

RB3 W8A8 QNN DLC compiled but profiling failed with `QnnModel_composeGraphsFromDlc: MODEL_GRAPH_ERROR`. Keep this failure in the experiment log; compilation alone is not successful deployment. A second route reuses the saved quantize job through `--quantized-job` and compiles ONNX/QNN instead of recalibrating.

Hub latency values are converted from microseconds; memory ranges are bytes. [Metric definitions](https://workbench.aihub.qualcomm.com/docs/hub/howitworks-inference.html). The notebook/log reads the latest evidence and exposes actual compute counts and whether numerical checking completed.

RB3 ONNX/QNN profiling succeeded at 0.614 ms: 104 NPU operations plus CPU input QuantizeLinear/output DequantizeLinear. The downloaded ONNX directory contains 8,992,256 bytes of external weights and a 231,709-byte graph: **8.797 MiB uncompressed payload**. It retains floating-point weight initializers and is not an on-disk compression win over our 8.506 MiB portable FP32 baseline. Downloaded ZIP size is a transfer size, not the model-size denominator. The failed DLC was smaller but cannot be selected as a working deployment. Numerical validation status is read from saved evidence in the log.
