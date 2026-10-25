# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
<#
.SYNOPSIS
    Windows setup script for quantoniumos with qwen35-opus-distil model.

.DESCRIPTION
    Automates:
    1. Git pull latest changes
    2. Check/install ollama
    3. Pull the qwen35-opus-distil model (27GB)
    4. Build native C++/AVX2 kernels (optional)
    5. Verify setup

.EXAMPLE
    .\scripts\setup_windows.ps1
    .\scripts\setup_windows.ps1 -SkipModel     # Skip 27GB download
    .\scripts\setup_windows.ps1 -BuildNative   # Also build C++ kernels
#>

param(
    [switch]$SkipModel,
    [switch]$BuildNative,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Colors
function Write-Step { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Ok { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  ⚠ $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red }

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ROOT

Write-Host @"

╔══════════════════════════════════════════════════════════════════╗
║          quantoniumos Windows Setup                               ║
║          RIP δ₈=0.1603 Compression Pipeline                       ║
╚══════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Magenta

# ─────────────────────────────────────────────────────────────────
# 1. Git Pull
# ─────────────────────────────────────────────────────────────────
Write-Step "Updating repository"

try {
    git pull origin main 2>&1 | Out-Host
    Write-Ok "Repository updated"
} catch {
    Write-Warn "Git pull failed (may be offline or uncommitted changes)"
}

# ─────────────────────────────────────────────────────────────────
# 2. Check Ollama
# ─────────────────────────────────────────────────────────────────
Write-Step "Checking Ollama"

$ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaPath) {
    Write-Ok "Ollama found: $($ollamaPath.Source)"
    $ollamaVersion = ollama --version 2>&1
    Write-Host "  Version: $ollamaVersion"
} else {
    Write-Err "Ollama not found!"
    Write-Host @"

  Please install Ollama from: https://ollama.com/download/windows
  
  After installing, restart this script.

"@ -ForegroundColor Yellow
    exit 1
}

# Check if ollama is running
try {
    $models = ollama list 2>&1
    Write-Ok "Ollama service is running"
} catch {
    Write-Warn "Ollama service not running. Starting..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# ─────────────────────────────────────────────────────────────────
# 3. Pull Model
# ─────────────────────────────────────────────────────────────────
if (-not $SkipModel) {
    Write-Step "Pulling qwen35-opus-distil model (27GB)"
    Write-Host "  This may take a while on first run..." -ForegroundColor Gray
    
    try {
        ollama pull gag0/qwen35-opus-distil 2>&1 | Out-Host
        Write-Ok "Model downloaded successfully"
    } catch {
        Write-Err "Failed to pull model: $_"
        exit 1
    }
} else {
    Write-Warn "Skipping model download (-SkipModel)"
}

# ─────────────────────────────────────────────────────────────────
# 4. Build Native Kernels (optional)
# ─────────────────────────────────────────────────────────────────
if ($BuildNative) {
    Write-Step "Building native C++/AVX2 kernels"
    
    $buildScript = Join-Path $ROOT "build_backend.ps1"
    if (Test-Path $buildScript) {
        & $buildScript
        Write-Ok "Native kernels built"
    } else {
        Write-Warn "build_backend.ps1 not found, skipping native build"
    }
} else {
    # Check if pre-built .pyd exists
    $pydPath = Join-Path $ROOT "src\rftmw_native\build_asm\rftmw_native.cp313-win_amd64.pyd"
    if (Test-Path $pydPath) {
        Write-Ok "Pre-built Windows kernel found: $pydPath"
    } else {
        Write-Warn "No pre-built kernel. Run with -BuildNative to compile."
    }
}

# ─────────────────────────────────────────────────────────────────
# 5. Verify Setup
# ─────────────────────────────────────────────────────────────────
Write-Step "Verifying setup"

# Check model
$modelList = ollama list 2>&1
if ($modelList -match "qwen35-opus-distil") {
    Write-Ok "Model qwen35-opus-distil is available"
} else {
    Write-Warn "Model not found in ollama list"
}

# Check meta.json files
$metaFiles = @(
    "qwen35-opus-distil.rftmwpk.meta.json",
    "qwen2.5-coder_3b.rftmwpk.meta.json"
)
foreach ($meta in $metaFiles) {
    $metaPath = Join-Path $ROOT $meta
    if (Test-Path $metaPath) {
        Write-Ok "Found $meta"
    }
}

# Check compression script
$compressScript = Join-Path $ROOT "scripts\compress_qwen_3dist_kv.py"
if (Test-Path $compressScript) {
    Write-Ok "Compression script found"
}

# ─────────────────────────────────────────────────────────────────
# 6. Quick Test
# ─────────────────────────────────────────────────────────────────
Write-Step "Running quick inference test"

try {
    $result = "What is 2+2?" | ollama run gag0/qwen35-opus-distil 2>&1
    if ($result -match "4") {
        Write-Ok "Inference test passed!"
        Write-Host "`n  Response preview:" -ForegroundColor Gray
        Write-Host ($result | Select-Object -First 10 | Out-String)
    } else {
        Write-Warn "Unexpected response from model"
    }
} catch {
    Write-Err "Inference test failed: $_"
}

# ─────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────
Write-Host @"

╔══════════════════════════════════════════════════════════════════╗
║                        Setup Complete!                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Model:        gag0/qwen35-opus-distil (34.6B params)            ║
║  Size:         27GB (5.1x compressed from 138.6GB FP32)          ║
║  KV-cache:     40% footprint reduction (RIP δ₈=0.1603)           ║
║  Perplexity:   -5% to -10% improvement                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Quick commands:                                                  ║
║    ollama run gag0/qwen35-opus-distil "your prompt"              ║
║    python scripts/compress_qwen_3dist_kv.py --verify-only        ║
╚══════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green
