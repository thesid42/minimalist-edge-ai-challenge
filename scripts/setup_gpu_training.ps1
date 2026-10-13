$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$uvPath = if ($uvCommand) { $uvCommand.Source } else { Join-Path $env:LOCALAPPDATA 'hermes\bin\uv.exe' }
if (-not (Test-Path -LiteralPath $uvPath)) { throw 'Install uv or add it to PATH first.' }
$gpuPython = '.venv-train-gpu\Scripts\python.exe'
function Invoke-UvChecked {
    param([string[]]$Arguments)
    & $uvPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'GPU environment installation failed.' }
}
if (-not (Test-Path -LiteralPath $gpuPython)) {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run scripts/setup.ps1 for CPU setup first.' }
    Invoke-UvChecked @('venv', '.venv-train-gpu', '--python', '.venv\Scripts\python.exe', '--seed')
}
Invoke-UvChecked @('pip', 'install', '--no-cache', '--python', $gpuPython, 'torch==2.8.0', 'torchvision==0.23.0', '--index-url', 'https://download.pytorch.org/whl/cu128')
Invoke-UvChecked @('pip', 'install', '--no-cache', '--python', $gpuPython, '-r', 'requirements-train-gpu.in')
& $gpuPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'GPU dependency check failed.' }
& $gpuPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0)); print('CPU inference environment remains .venv')"
if ($LASTEXITCODE -ne 0) { throw 'GPU CUDA check failed.' }
