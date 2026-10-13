$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$projectPython = Join-Path $PWD '.venv\Scripts\python.exe'
$env:JUPYTER_RUNTIME_DIR = Join-Path $PWD '.cache\jupyter-runtime'
$env:JUPYTER_PATH = Join-Path $PWD '.venv\share\jupyter'
$env:IPYTHONDIR = Join-Path $PWD '.cache\ipython'
& $projectPython -m ipykernel install --prefix '.venv' --name edgeprep --display-name 'Edge challenge (.venv)'
if ($LASTEXITCODE -ne 0) { throw 'Install requirements-notebook.lock.txt in .venv first.' }
& $projectPython -m edgeprep.experiment_log
if ($LASTEXITCODE -ne 0) { throw 'Experiment-log refresh failed.' }
& $projectPython -m jupyterlab 'experiments/Experiment_Log.ipynb'
