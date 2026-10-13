$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
& '.\.venv\Scripts\python.exe' -m edgeprep.hub_auth
if ($LASTEXITCODE -ne 0) { throw 'AI Hub setup failed. Run this script in an interactive terminal.' }
