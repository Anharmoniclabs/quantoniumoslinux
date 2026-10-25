$ErrorActionPreference = 'Stop'
$workspace = $PSScriptRoot
$env:QUANTONIUM_HOME = $workspace
$python = (Get-Command python.exe -ErrorAction Stop).Source
$env:QUANTONIUM_PYTHON = $python
$native = Join-Path $workspace 'runtime\desktop-shell\src-tauri\target\release\quantoniumos-desktop.exe'
if (-not (Test-Path -LiteralPath $native)) {
    throw "QuantoniumOS desktop app has not been built yet. Run Build-Desktop.ps1 first. Expected: $native"
}
Start-Process -FilePath $native -WorkingDirectory (Join-Path $workspace 'runtime')
