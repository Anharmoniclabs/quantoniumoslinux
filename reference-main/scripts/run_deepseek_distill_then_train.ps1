param(
    [Parameter(Mandatory=$true)]
    [int]$DistillPid,

    [string]$Model = "qwen35-opus-distil.rftmwpk",
    [string]$Data = "ai/training/datasets/distill_deepseek_r1_70b.jsonl",
    [string]$Teacher = "deepseek-r1:70b",
    [int]$MaxSamples = 25
)

$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot\..")

New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = "logs\deepseek_r1_70b_train_after_distill.log"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $log -Value $line
    Write-Host $line
}

Log "Waiting for distillation PID $DistillPid"
try {
    Wait-Process -Id $DistillPid
} catch {
    Log "Wait-Process failed or process already exited: $($_.Exception.Message)"
}

Log "Stopping Ollama teacher $Teacher to free RAM"
try {
    ollama stop $Teacher | Out-Null
} catch {
    Log "ollama stop failed: $($_.Exception.Message)"
}

if (-not (Test-Path $Data)) {
    Log "Training data not found: $Data"
    exit 1
}

$count = (Get-Content $Data | Measure-Object -Line).Lines
Log "Training data lines: $count"
if ($count -le 0) {
    Log "No samples generated; refusing to train"
    exit 1
}
if ($count -lt $MaxSamples) {
    Log "Only $count/$MaxSamples samples generated; refusing partial train"
    exit 1
}

Log "Starting student training on DeepSeek teacher data"
& C:\Python313\python.exe src/apps/train_rftmw_spectral.py `
    --model $Model `
    --data $Data `
    --epochs 1 `
    --lr 1e-4 `
    --max_samples $MaxSamples `
    --log_interval 1 *>> $log

Log "Training command exited with code $LASTEXITCODE"
exit $LASTEXITCODE
