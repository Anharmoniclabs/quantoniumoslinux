# Builds the RFTMW native stack (NASM kernels -> C kernels -> C++ engine -> pybind11 module + native CLIs)
# from runtime\src\rftmw_native with MSVC, and installs rftmw_native.pyd next to the Python engines.
$ErrorActionPreference = 'Stop'
$workspace = $PSScriptRoot
$runtime   = Join-Path $workspace 'runtime'
$src       = Join-Path $runtime 'src\rftmw_native'
$build     = Join-Path $src 'build'   # the Python engines import rftmw_native from src\rftmw_native\build
$python    = Join-Path $runtime '.venv\Scripts\python.exe'

$vsRoot = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
$env:PATH = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer;$env:PATH"
& "$vsRoot\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
# clang (GNU driver, MSVC ABI) compiles the GCC-style C99 kernels and flags (-O3 -march=native, VLAs, __attribute__) as written.
$env:PATH = "C:\Program Files\LLVM\bin;C:\Program Files\NASM;$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$env:PATH"
foreach ($tool in @('clang.exe', 'nasm.exe', 'cmake.exe', 'ninja.exe')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "Missing native build prerequisite: $tool" }
}
if (-not (Test-Path -LiteralPath $python)) { throw "Python venv missing: $python" }

$ErrorActionPreference = 'Continue'
Remove-Item -LiteralPath (Join-Path $build 'CMakeCache.txt'), (Join-Path $build 'CMakeFiles') -Recurse -Force -ErrorAction SilentlyContinue
& cmake.exe -S $src -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ `
    "-DCMAKE_C_FLAGS=-DNOMINMAX" "-DCMAKE_CXX_FLAGS=-DNOMINMAX" `
    "-DPython3_EXECUTABLE=$python" -DRFTMW_ENABLE_ASM=ON -DRFTMW_ENABLE_NATIVE_GEMV_ASM=ON -DENABLE_LTO=OFF
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
& cmake.exe --build $build --parallel
if ($LASTEXITCODE -ne 0) { throw 'Native build failed.' }

Get-ChildItem $build -Include '*.exe', '*.pyd' -Recurse -Depth 0 | Select-Object Name, Length
& $python -c "import sys; sys.path.insert(0, r'$build'); import rftmw_native as r; print('rftmw_native loaded:', r.__file__)"
