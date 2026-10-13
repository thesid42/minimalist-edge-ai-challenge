# Experiment log â€” start here

Read the numbered folders in order. Each README explains **what changed, implementation, results and interpretation**, with links to the original evidence. `Experiment_Log.ipynb` presents the same tables with saved outputs; no rerunning training is required to read it. `runs.json` is the machine-readable registry.

Two accuracy protocols exist: the historical 500-image pilot and the current 27,000-image training-set evaluation. They are separate tables. Remote device inference has a 32-image numerical check, not a new full-dataset accuracy score. All model changes still start from the supplied student.

| Step | Experiment | Read |
|---|---|---|
| 01 | Original models and dataset | [Details](01_original_models/README.md) |
| 02 | Historical compression and short recovery | [Details](02_historical_pilots/README.md) |
| 03 | Our current unchanged-student baseline | [Details](03_measured_baseline/README.md) |
| 04 | Selective INT8 with 256 MinMax samples | [Details](04_selective_int8/README.md) |
| 05 | Selective INT8 with 512 Percentile samples | [Details](05_percentile512/README.md) |
| 06 | Real-device CPU/NPU deployment | [Details](06_ai_hub/README.md) |
| 07 | Next experiments, not yet measured | [Details](07_next_experiments/README.md) |

Current CPU results:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>accuracy_pct</th>
      <th>worst_recall_pct</th>
      <th>size_mib</th>
      <th>median_cpu_4t_ms</th>
      <th>examples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>baseline</td>
      <td>99.1630</td>
      <td>95.8400</td>
      <td>8.5063</td>
      <td>2.1036</td>
      <td>27000</td>
    </tr>
    <tr>
      <td>qdq_pointwise</td>
      <td>98.4111</td>
      <td>95.4333</td>
      <td>3.7720</td>
      <td>5.1842</td>
      <td>27000</td>
    </tr>
    <tr>
      <td>qdq_percentile512</td>
      <td>98.8037</td>
      <td>95.3600</td>
      <td>3.7719</td>
      <td>5.5206</td>
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
