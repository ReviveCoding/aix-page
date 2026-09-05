param([switch]$SkipInstall)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Missing canonical venv: $python" }
if (-not $SkipInstall) {
  & $python -m pip install --index-url https://download.pytorch.org/whl/cu130 'torch==2.11.0+cu130'
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  & $python -m pip install 'xgboost==3.4.1'
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$env:PYTHONPATH = Join-Path $root 'src'
& $python (Join-Path $PSScriptRoot 'qualify_gpu.py')
exit $LASTEXITCODE
