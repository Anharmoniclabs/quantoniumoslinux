$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot
if (Test-Path "C:\\msys64\\mingw64\\bin") {
  $mingwBin = "C:\\msys64\\mingw64\\bin"
  if (-not (($env:PATH -split ';') -contains $mingwBin)) {
    $env:PATH = "$mingwBin;$env:PATH"
  }
}

$py = if (Test-Path ".\\.venv\\Scripts\\python.exe") { ".\\.venv\\Scripts\\python.exe" } else { "python" }

New-Item -ItemType Directory -Force -Path ".\\logs" | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = ".\\logs\\chatbox_${ts}_stdout.log"
$stderr = ".\\logs\\chatbox_${ts}_stderr.log"

$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:RFTMW_REQUIRE_NATIVE = "1"
if (-not $env:QSHLL_NATIVE_SERVER) { $env:QSHLL_NATIVE_SERVER = "1" }
if (-not $env:QSHLL_NATIVE_MAX_SEQ) { $env:QSHLL_NATIVE_MAX_SEQ = "1024" }
if (-not $env:QSHLL_NATIVE_COMPACT_PROMPT) { $env:QSHLL_NATIVE_COMPACT_PROMPT = "1" }

if (-not $env:QSHLL_HISTORY_MESSAGES) { $env:QSHLL_HISTORY_MESSAGES = "4" }
if (-not $env:QSHLL_MAX_TOKENS) { $env:QSHLL_MAX_TOKENS = "80" }
if (-not $env:QSHLL_TEMPERATURE) { $env:QSHLL_TEMPERATURE = "0.2" }

if (-not $env:RFTMWPK_PATH) {
  $finetuned = Join-Path $repoRoot "qwen35-opus-distil_finetuned.rftmwpk"
  $base35b = Join-Path $repoRoot "qwen35-opus-distil.rftmwpk"
  if (Test-Path $finetuned) {
    $env:RFTMWPK_PATH = (Resolve-Path $finetuned).Path
  } elseif (Test-Path $base35b) {
    $env:RFTMWPK_PATH = (Resolve-Path $base35b).Path
  }
}

$scriptArgs = @($args)
if ($env:RFTMWPK_PATH -and -not ($scriptArgs -contains "--pack")) {
  $scriptArgs = @("--pack", $env:RFTMWPK_PATH) + $scriptArgs
}

$ErrorActionPreference = 'Continue'
& $py -X faulthandler "src\\apps\\qshll_chatbox.py" @scriptArgs 1> $stdout 2> $stderr
$ErrorActionPreference = 'Stop'
$code = $LASTEXITCODE
if ($code -ne 0) {
  Write-Host "Chatbox exited with code $code."
  Write-Host "See logs: $stderr"
  pause
}

