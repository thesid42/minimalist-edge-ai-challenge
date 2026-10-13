$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:UV_CACHE_DIR = Join-Path $PWD '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.tools\python'
function Invoke-Checked {
    param([string[]]$Arguments)
    & uv @Arguments
    if ($LASTEXITCODE -ne 0) { throw "uv failed: $Arguments" }
}
Invoke-Checked -Arguments @('python', 'install', '3.11')
$environmentName = '.venv'
$index = 'https://download.pytorch.org/whl/cpu'
if (-not (Test-Path "$environmentName\Scripts\python.exe")) {
    Invoke-Checked -Arguments @('venv', $environmentName, '--python', '3.11', '--seed')
}
$pythonPath = "$environmentName\Scripts\python.exe"
Invoke-Checked -Arguments @('pip', 'install', '--python', $pythonPath, 'torch==2.8.0', 'torchvision==0.23.0', 'torchaudio==2.8.0', '--index-url', $index)
$commonFile = if (Test-Path 'requirements-common.lock.txt') { 'requirements-common.lock.txt' } else { 'requirements-common.in' }
Invoke-Checked -Arguments @('pip', 'install', '--python', $pythonPath, '-r', $commonFile)
& $pythonPath -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& $pythonPath -m edgeprep.verify
if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed' }
