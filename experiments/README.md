# Experiment log - start here

Read the numbered folders in order. Each README explains **what changed, implementation, results and interpretation**, with links to the original evidence. `Experiment_Log.ipynb` presents the same tables with saved outputs; no rerunning training is required to read it. `runs.json` is the machine-readable registry.

Quantization reduces numerical precision and weight storage. Channel pruning physically removes parts of the network. Recovery trains a compressed model to restore accuracy. Device compilation prepares it for a particular runtime.

Two accuracy protocols exist: the historical 500-image pilot and the current 27,000-image training-set evaluation. They are separate tables. Remote runs retain the historical 32-image numerical check and now add a labelled 1,000-image balanced diagnostic subset. That subset excludes 512 calibration images but is not held-out generalization or full-dataset accuracy. All model changes still start from the supplied student.

| Step | Experiment | Read |
|---|---|---|
| 01 | Original models and dataset | [Details](01_original_models/README.md) |
| 02 | Historical compression and short recovery | [Details](02_historical_pilots/README.md) |
| 03 | Our current unchanged-student baseline | [Details](03_measured_baseline/README.md) |
| 04 | Selective INT8 with 256 MinMax samples | [Details](04_selective_int8/README.md) |
| 05 | Selective INT8 with 512 Percentile samples | [Details](05_percentile512/README.md) |
| 06 | Real-device CPU/NPU deployment | [Details](06_ai_hub/README.md) |
| 07 | Next experiments, not yet measured | [Details](07_next_experiments/README.md) |
| 08 | Lower spatial computation, fixed external input | [Details](08_internal_resolution/README.md) |
| 09 | Repeated CPU latency and estimated energy | [Details](09_energy_measurement/README.md) |
| 10 | Recovered model and worker-thread power tradeoff | [Details](10_runtime_power/README.md) |
| 11 | GPU training environment and CPU inference verification | [Details](11_gpu_training/README.md) |
| 12 | Direct comparison of 192 model latencies | [Details](12_192_latency/README.md) |

Current CPU results:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>label</th>
      <th>accuracy_pct</th>
      <th>worst_recall_pct</th>
      <th>size_mib</th>
      <th>median_cpu_4t_ms</th>
      <th>estimated_cpu_4t_j</th>
      <th>examples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Unchanged FP32 student</td>
      <td>99.1630</td>
      <td>95.8400</td>
      <td>8.5063</td>
      <td>2.7892</td>
      <td>0.0549</td>
      <td>27000</td>
    </tr>
    <tr>
      <td>Selective INT8 / MinMax 256</td>
      <td>98.4111</td>
      <td>95.4333</td>
      <td>3.7720</td>
      <td>5.1842</td>
      <td>0.1192</td>
      <td>27000</td>
    </tr>
    <tr>
      <td>Selective INT8 / Percentile 512</td>
      <td>98.8037</td>
      <td>95.3600</td>
      <td>3.7719</td>
      <td>5.5206</td>
      <td>0.1322</td>
      <td>27000</td>
    </tr>
    <tr>
      <td>Internal 192 resize, unchanged weights</td>
      <td>98.0000</td>
      <td>93.8000</td>
      <td>8.5110</td>
      <td>2.1325</td>
      <td>0.0518</td>
      <td>27000</td>
    </tr>
    <tr>
      <td>Internal 160 resize, unchanged weights</td>
      <td>93.0037</td>
      <td>78.0500</td>
      <td>8.5110</td>
      <td>1.5622</td>
      <td>0.0352</td>
      <td>27000</td>
    </tr>
    <tr>
      <td>Internal 192 + full-data classifier recovery</td>
      <td>99.1074</td>
      <td>97.2400</td>
      <td>8.5110</td>
      <td>pending / n/a</td>
      <td>pending / n/a</td>
      <td>27000</td>
    </tr>
    <tr>
      <td>Recovered 192 + FP16 weight storage / FP32 compute</td>
      <td>99.1111</td>
      <td>97.2400</td>
      <td>4.2882</td>
      <td>2.0538</td>
      <td>0.0453</td>
      <td>27000</td>
    </tr>
  </tbody>
</table>

To refresh this index after more experiments (reads results only):

```powershell
.venv\Scripts\python.exe -m edgeprep.experiment_log
```

Open the notebook interactively:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/notebook.ps1
```

JupyterLab and the project kernel are installed locally. The notebook reads evidence only; it never reruns training or submits Hub jobs. Timing JSON files also contain a one-image smoke-check field called accuracy; use the separate *_accuracy.json dataset measurements in these tables.
