$ErrorActionPreference = 'Stop'
$workspace = $PSScriptRoot
$env:QUANTONIUM_HOME = $workspace
# Load MSVC (VS 2022 Build Tools) and the rustup toolchain, which are installed but not on PATH.
$vsDev = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1'
$env:PATH = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer;$env:PATH"
if (Test-Path -LiteralPath $vsDev) { & $vsDev -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null }
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
foreach ($tool in @('node.exe', 'npm.cmd', 'cargo.exe', 'cl.exe')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Missing build prerequisite: $tool. See https://v2.tauri.app/start/prerequisites/"
    }
}
Push-Location (Join-Path $workspace 'runtime\desktop-shell')
# npm/cargo write progress to stderr; Windows PowerShell 5.1 would treat that as fatal under 'Stop'.
$ErrorActionPreference = 'Continue'
try {
    & npm.cmd ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw 'npm dependency installation failed.' }
    & npm.cmd run tauri build -- --no-bundle
    if ($LASTEXITCODE -ne 0) { throw 'Native Tauri build failed.' }
    Write-Host 'Native desktop build completed. Launch-QuantoniumOS.ps1 will now use it.'
} finally { Pop-Location }
