param([switch]$ExecuteNotebook)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$python = '.venv\Scripts\python.exe'
& $python scripts/refresh_results.py
if ($LASTEXITCODE -ne 0) { throw 'Result refresh failed.' }
$env:JUPYTER_PATH = Join-Path $PWD '.venv\share\jupyter'
$env:IPYTHONDIR = Join-Path $PWD '.cache\ipython'
$env:JUPYTER_RUNTIME_DIR = Join-Path $PWD '.cache\jupyter-runtime'
if ($ExecuteNotebook) {
    & $python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=edgeprep experiments/Experiment_Log.ipynb
    if ($LASTEXITCODE -ne 0) { throw 'Notebook execution failed.' }
}
& $python -m nbconvert --to html experiments/Experiment_Log.ipynb
if ($LASTEXITCODE -ne 0) { throw 'Notebook HTML export failed.' }
