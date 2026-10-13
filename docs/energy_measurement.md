# Energy measurement

The challenge PDF sections 5.4-5.5 and 9 require exported ONNX timing, joules per inference, the measurement mode, and baseline/final ratios on the same laptop. CodeCarbon 3.3.1 is installed in `.venv`. No extra key is needed for its offline tracker.

On this Ryzen 7 5800H / Windows 10 machine, startup logs select `cpu_load` with a 45 W CPU TDP. CodeCarbon estimates process CPU energy from CPU time/load. RAM is a system-memory heuristic (10 W in these runs), rather than a measured per-process RAM rail. GPU IDs are excluded. These numbers are **TDP estimates**, not actual laptop input energy or remote NPU joules. Windows Energy Meter Interface support requires a compatible Windows 11 machine; no available sensor was detected here. Do not change the OS or install vendor drivers just for this estimate.

Every result now records the CPU measurement mode, TDP, CodeCarbon version, raw CPU/RAM/GPU power, component energy, total kWh, inference count and CSV path. Conversion is `energy_consumed_kWh * 3_600_000 / completed_inferences`. This continuous-loop result represents steady throughput; startup, preprocessing and network upload are excluded. Accuracy fields in these timing files are one-image checks; use dataset accuracy files for accuracy claims.

Run the matched thread sweep (three fresh-process repetitions per setting, 100 warmups, 1,000 timings, 30 seconds of energy inference):

```powershell
.venv\Scripts\python.exe scripts/thread_energy_sweep.py
```

Evidence is in `results/energy/thread_sweep/summary.json`, per-run JSON, startup stderr, and `*/emissions.csv`. Four threads are the matched reference. Keep these new ratios separate from older 20-second runs. Run measurements serially without local training or other inference in parallel; normal OS activity remains a limitation. Repeated runs expose variability, but they do not turn an estimate into a sensor reading.

For a candidate, use the same warmup/run count, threads and energy window as a fresh reference:

```powershell
.venv\Scripts\python.exe -m edgeprep.benchmark --model artifacts/all_data/resize160/model_fp32.onnx --threads 4 --runs 1000 --warmup 100 --inputs results/all_data/challenge_data/timing_input.npy --labels results/all_data/challenge_data/timing_label.npy --energy-seconds 30 --energy-output results/energy/resolution/resize160_rep1 --output results/energy/resolution/resize160_rep1.json
```

For stronger physical evidence, a compatible Linux CPU energy-counter setup can report CPU package joules (still not the whole system), or an external logged power meter can measure laptop/board input energy. For a power trace, integrate watts over seconds and divide by completed inferences; record meter boundary, sampling rate and whether idle power was subtracted. Compare baseline and final with the same physical setup. No external meter is attached in the current evidence.

AI Hub documents latency and memory, not an energy field in the profiles we use. `power_saver` is a runtime configuration, not a joules measurement; latency alone cannot prove an energy saving.

Sources: [CodeCarbon methodology](https://docs.codecarbon.io/latest/explanation/methodology/), [CodeCarbon output](https://docs.codecarbon.io/latest/reference/output/), [Qualcomm profile metric definitions](https://workbench.aihub.qualcomm.com/docs/hub/howitworks-inference.html). Local installed CodeCarbon implementation and saved startup logs establish the actual mode used here.

## ORT worker spinning

`edgeprep.benchmark --disable-spinning` sets both intra/inter worker allow_spinning to zero. This lets idle workers sleep. It can reduce CPU work but may add wakeup latency; measure both. `scripts/runtime_energy_sweep.py` compares default/sleeping workers on identical unchanged and recovered-192 graphs in matched repetitions. This is a runtime configuration improvement: judges must use the same options to reproduce it, and the ONNX file alone cannot enforce session options. [Official ORT threading guidance](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
