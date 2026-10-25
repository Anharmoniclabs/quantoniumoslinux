# build_backend.ps1 -- QuantoniumOS backend pipeline
# ASM (NASM) --> C++ (CMake/MinGW) --> Python (.pyd) --> Godot (rpg_headless)
#
# Usage:
#   .\build_backend.ps1                # full rebuild + verify
#   .\build_backend.ps1 -SkipBuild    # skip native compile, verify only
#   .\build_backend.ps1 -BuildPacks   # also rebuild GGUF index packs
#   .\build_backend.ps1 -Run          # launch rpg_headless for manual test
param(
    [switch]$SkipBuild,
    [switch]$BuildPacks,
    [switch]$Run,
    [switch]$Verbose
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT      = $PSScriptRoot
$NATIVE    = Join-Path $ROOT "src\rftmw_native"
$BUILD_ASM = Join-Path $NATIVE "build_asm"
$VENV_PY   = Join-Path $ROOT ".venv\Scripts\python.exe"
$MINGW_BIN = "C:\msys64\mingw64\bin"

function Write-Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-OK([string]$msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Fail([string]$msg)       { Write-Host "    FAIL: $msg" -ForegroundColor Red ; exit 1 }

# --------------------------------------------------------------------------
# 0. Prerequisites
# --------------------------------------------------------------------------
Write-Step "Checking prerequisites"

if (-not (Test-Path $VENV_PY)) {
    Fail ("No .venv found at " + $VENV_PY)
}
Write-OK ("Python venv: " + $VENV_PY)

$useNinja = $false
if (-not $SkipBuild) {
    if (Test-Path $MINGW_BIN) { $env:PATH = $MINGW_BIN + ";" + $env:PATH }
    foreach ($tool in @("cmake", "nasm")) {
        $found = Get-Command $tool -ErrorAction SilentlyContinue
        if ($found) {
            Write-OK ($tool + ": " + $found.Source)
        } else {
            Fail ($tool + " not found. Install: winget install NASM.NASM  or  choco install cmake nasm")
        }
    }
    $ninjaCmd = Get-Command "ninja" -ErrorAction SilentlyContinue
    if ($ninjaCmd) { $useNinja = $true ; Write-OK "Ninja detected" }
}

# --------------------------------------------------------------------------
# 1. Build native extension  ASM -> obj -> .pyd
# --------------------------------------------------------------------------
if (-not $SkipBuild) {
    Write-Step "Building rftmw_native (ASM + C++ -> .pyd)"
    New-Item -ItemType Directory -Force -Path $BUILD_ASM | Out-Null
    Push-Location $BUILD_ASM

    $gen = if ($useNinja) { "-GNinja" } else { "-G`"MinGW Makefiles`"" }
    $pyFlag = "-DPython3_EXECUTABLE=" + $VENV_PY
    $cfgArgs = @("..", $gen, "-DCMAKE_BUILD_TYPE=Release",
                 "-DRFTMW_ENABLE_ASM=ON", "-DRFTMW_ENABLE_NATIVE_GEMV_ASM=ON", $pyFlag)
    if ($Verbose) { $cfgArgs += "--log-level=VERBOSE" }

    Write-Host ("  cmake " + ($cfgArgs -join " ")) -ForegroundColor DarkGray
    & cmake @cfgArgs
    if ($LASTEXITCODE -ne 0) { Pop-Location ; Fail "cmake configure failed" }

    & cmake --build . --config Release
    if ($LASTEXITCODE -ne 0) { Pop-Location ; Fail "cmake build failed" }

    Pop-Location
    Write-OK ("Build complete -> " + $BUILD_ASM)
} else {
    Write-Step "Skipping native build (-SkipBuild)"
}

# --------------------------------------------------------------------------
# 2. Verify .pyd loads
# --------------------------------------------------------------------------
Write-Step "Verifying native extension (.pyd)"

$vPy = Join-Path $env:TEMP "rftmw_verify___.py"
Set-Content -Encoding UTF8 -Path $vPy -Value @"
import os, sys
os.environ['RFTMW_REQUIRE_NATIVE'] = '1'
sys.path.insert(0, sys.argv[1])
from quantonium_os_src.engine.rftmw_numpy_engine import _get_native_infer, NativeKernelsUnavailable
try:
    ni = _get_native_infer()
except NativeKernelsUnavailable as e:
    print('FAIL NativeKernelsUnavailable:', e)
    sys.exit(1)
if ni is None:
    print('FAIL returned None')
    sys.exit(1)
import rftmw_native as m
print('OK  file:', m.__file__)
print('OK  AVX2:', getattr(m, 'HAS_AVX2', '?'))
ok_infer = getattr(m.infer, 'available', False)
print('OK  infer.available:', ok_infer)
print('OK  asm_kernels:', m.asm_kernels_available())
if not ok_infer:
    print('FAIL infer.available is False')
    os._exit(1)
os._exit(0)
"@

& $VENV_PY $vPy $ROOT
if ($LASTEXITCODE -ne 0) { Fail "Native extension verification failed" }
Write-OK "Native extension verified"

# --------------------------------------------------------------------------
# 3. Build GGUF index packs  (only with -BuildPacks)
# --------------------------------------------------------------------------
if ($BuildPacks) {
    Write-Step "Building GGUF index packs"

    $blobBase = "C:\Users\" + $env:USERNAME + "\.ollama\models\blobs\"
    $packs = @(
        @{ gguf = $blobBase + "sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463"
           out  = Join-Path $ROOT "qwen2.5-coder_7b.rftmwpk"
           name = "qwen2.5-coder:7b" },
        @{ gguf = $blobBase + "sha256-ac9bc7a69dab38da1c790838955f1293420b55ab555ef6b4615efa1c1507b1ed"
           out  = Join-Path $ROOT "qwen2.5-coder_14b.rftmwpk"
           name = "qwen2.5-coder:14b" }
    )
    $packScript = Join-Path $ROOT "src\apps\build_pack_from_gguf.py"
    foreach ($p in $packs) {
        if (-not (Test-Path $p.gguf)) {
            Write-Host ("    SKIP " + $p.name + " -- GGUF not found") -ForegroundColor Yellow
            continue
        }
        Write-Host ("  Building: " + $p.name) -ForegroundColor White
        & $VENV_PY $packScript --gguf $p.gguf --out $p.out --model-name $p.name
        if ($LASTEXITCODE -ne 0) { Fail ("pack build failed: " + $p.name) }
        Write-OK ($p.name + " -> " + $p.out)
    }
}

# --------------------------------------------------------------------------
# 4. Smoke-test Python inference pipeline (smallest available pack, 8 tokens)
# --------------------------------------------------------------------------
Write-Step "Smoke-testing Python inference (3b pack)"

$sPy = Join-Path $env:TEMP "rftmw_smoke___.py"
Set-Content -Encoding UTF8 -Path $sPy -Value @"
import os, sys, time, json
from pathlib import Path
repo = Path(sys.argv[1])

def param_count(p):
    sc = p.with_suffix('.rftmwpk.meta.json')
    try:
        return int(json.loads(sc.read_text()).get('provenance', {}).get('parameter_count', 0))
    except Exception:
        return 0

packs = sorted(repo.glob('*.rftmwpk'), key=param_count)
best = next((p for p in packs if param_count(p) > 0), None)
if best is None:
    print('SKIP no valid pack with meta.json found')
    os._exit(0)

print('  pack:', best.name, '(' + str(round(param_count(best)/1e9, 1)) + 'B)')
from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine
t0 = time.perf_counter()
eng = Qwen25Engine(str(best))
eng.load()
print('  load:', round(time.perf_counter()-t0, 1), 's')
t1 = time.perf_counter()
out = eng.generate('<|im_start|>user\nSay one word.<|im_end|>\n<|im_start|>assistant\n', max_tokens=8)
print('  out: ', repr(out.strip()), '  time:', round(time.perf_counter()-t1, 2), 's')
print('OK')
os._exit(0)
"@

& $VENV_PY $sPy $ROOT
if ($LASTEXITCODE -ne 0) { Fail "Smoke test failed" }

# --------------------------------------------------------------------------
# 5. Verify Godot bridge wiring
# --------------------------------------------------------------------------
Write-Step "Verifying Godot bridge wiring"

$headless = Join-Path $ROOT "src\rpg_headless.py"
$bridge   = Join-Path $ROOT "godot\bronx-rpg\scripts\AgentBridge.gd"

if (-not (Test-Path $headless)) { Fail "src\rpg_headless.py missing" }
& $VENV_PY -c ("import ast,sys; ast.parse(open(r'" + $headless + "', encoding='utf-8').read()); print('OK  syntax: rpg_headless.py')")
if ($LASTEXITCODE -ne 0) { Fail "rpg_headless.py has syntax errors" }

if (Test-Path $bridge) { Write-OK "AgentBridge.gd present" } `
else { Write-Host "    WARN: AgentBridge.gd not found" -ForegroundColor Yellow }

# --------------------------------------------------------------------------
# 6. Optional: launch rpg_headless for interactive test
# --------------------------------------------------------------------------
if ($Run) {
    Write-Step "Launching rpg_headless.py (enter JSON commands, Ctrl-C to quit)"
    Write-Host '  {"cmd":"chat","text":"hello oracle"}' -ForegroundColor DarkGray
    Write-Host '  {"cmd":"patrol"}' -ForegroundColor DarkGray
    Write-Host '  {"cmd":"quit"}' -ForegroundColor DarkGray
    & $VENV_PY $headless
}

# --------------------------------------------------------------------------
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  QuantoniumOS backend pipeline -- all stages OK" -ForegroundColor Green
Write-Host "  ASM   rftmw_sgemv_f32.asm + kernel ASMs  (NASM)" -ForegroundColor White
Write-Host "  C++   rftmw_python.cpp + inference bridge (CMake/AVX2)" -ForegroundColor White
Write-Host "  .pyd  build_asm\rftmw_native.cp313-win_amd64.pyd" -ForegroundColor White
Write-Host "  Py    rftmw_numpy_engine.Qwen25Engine" -ForegroundColor White
Write-Host "        rpg_headless.py  (RFTMW engine boot + agent loop)" -ForegroundColor White
Write-Host "  Godot AgentBridge.gd  (spawns rpg_headless via pipe)" -ForegroundColor White
Write-Host "================================================" -ForegroundColor Cyan
exit 0
