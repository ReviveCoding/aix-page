$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root 'src'
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }
& $python (Join-Path $PSScriptRoot 'environment.py')
