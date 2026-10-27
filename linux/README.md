# QuantoniumOS for Linux

QuantoniumOS as a full Linux distribution: Arch Linux underneath, an Omarchy-style
Hyprland desktop on top, and the **RFTMW middleware** sitting between them as a
session service that owns model memory and governs every app the desktop launches.

```
┌───────────────────────────────────────────────────────────────────────────┐
│ QuantoniumOS shell (Tauri)  ·  terminal  ·  browser  ·  any app           │
│        launched through qos-run  ──►  rftmwctl  ──►  unix socket          │
├───────────────────────────────────────────────────────────────────────────┤
│ rftmwd — RFTMW middleware (systemd user service, C++ / NASM)              │
│   memory broker   shared mmap of model packs (GGUF / .rftmwpk), page-cache│
│                   residency (mincore), prefetch / pin / evict, LRU budget │
│   pressure        kernel PSI trigger → evict idle packs before thrashing  │
│   workloads       each app = systemd scope in rftmw.slice with            │
│                   MemoryHigh / MemoryMax / MemorySwapMax / CPUWeight      │
│   transforms      RFTMWEngine (canonical Φ-RFT) + AVX2/FMA ASM kernels    │
├───────────────────────────────────────────────────────────────────────────┤
│ Hyprland · waybar (live RFTMW module) · greetd autologin · PipeWire       │
├───────────────────────────────────────────────────────────────────────────┤
│ systemd · zram (zstd) · THP=madvise · PSI · sysctl memory policy          │
├───────────────────────────────────────────────────────────────────────────┤
│ Linux kernel (Arch)                                                       │
└───────────────────────────────────────────────────────────────────────────┘
```

## What "runs on RFTMW" means here

The Linux kernel still schedules processes and manages pages, and no user-space
middleware can replace that. RFTMW sits in the position where the memory bottleneck
of a local-AI desktop actually lives, between the apps and the kernel:

- **Model weights are mapped once.** `PACK.OPEN` maps a pack `MAP_SHARED`, so the
  shell, the native CLIs and Ollama share one page-cache copy instead of each holding
  its own heap copy. `PACK.PREFETCH` warms it, `PACK.PIN` locks it (`mlock`), and
  `PACK.EVICT` pushes it out (`MADV_PAGEOUT` + `POSIX_FADV_DONTNEED`).
- **Memory has a budget.** `--pack-budget-mb` caps resident pack bytes, evicting
  least-recently-used packs first.
- **Pressure is handled before it becomes a freeze.** rftmwd arms a kernel PSI
  trigger (150 ms of stall in a 2 s window). When it fires, idle packs are evicted.
  The `rftmw.slice` ceiling (`MemoryHigh=85%`) throttles apps collectively, while
  the compositor and rftmwd stay outside that slice and remain responsive.
- **Every app is a governed workload.** `qos-run` (used by every Hyprland keybind)
  starts programs as `qos-<name>.scope` with the caller's environment and working
  directory. Each one gets its own memory and CPU limits and its own OOM boundary:
  a runaway app is killed inside its own scope and nothing else is affected.
- **The broker is crash-safe.** Workloads live in their own scopes. If rftmwd
  restarts, the apps keep running and are re-adopted.

## Layout

| Path | What it is |
|---|---|
| `rftmwd/` | Daemon (`rftmwd.cpp`) and client (`rftmwctl.c`). Built from `runtime/src/rftmw_native` headers and NASM kernels. |
| `bin/` | `qos-run` (launch as workload), `qos-session` (greetd → Hyprland), `qos-desktop` (shell + Ollama as workloads), `qos-status` (waybar). |
| `system/` | Files for `/`: user units (`rftmwd.service`, `rftmw.slice`), `user@.service` drop-in, sysctl, zram, THP, greetd, wayland session. |
| `config/` | Hyprland, hypridle, hyprlock, waybar (Omarchy-style keybinds, RFTMW bar module). |
| `packages.txt` | Arch package set (official repos only). |
| `build.sh` | Builds rftmwd, the Python venv, `rftmw_native` (module + CLIs) and the Tauri shell. |
| `install-files.sh` | Installs a built tree into any root (`/` or archiso's airootfs). |
| `install.sh` | Omarchy-style installer for an existing Arch system. |
| `iso/build-iso.sh` | Bootable live ISO (archiso, derived from `releng`). |
| `tests/` | ASM cross-ABI test, rftmwd end-to-end suite, session-tool tests, Arch-chroot ISO build for non-Arch hosts. |
| `bench/` | Kernel benchmark and output-equivalence check against the original sources. |

The Linux daemon's `RFT.FORWARD` and `RFT.SELFTEST` use an FFTW-backed fast
canonical transform for sizes up to 1024. It applies the raw phi-grid transform
with Bluestein FFTs and the Toeplitz Gram inverse square root with a 64-pole
rational approximation and multi-shift CG. The spectral interval is empirically
validated through N=1024; this route is approximate, not a formal error bound.
For numerical failures at N<=256, the engine falls back to its dense canonical
implementation. Build dependency: Arch package `fftw`.

## Install

**On an existing Arch install** (after `archinstall` with a minimal profile):

```sh
git clone <this repo> ~/QuantoniumOS && cd ~/QuantoniumOS
./linux/install.sh        # packages, build, system files, services
reboot                    # greetd logs you straight into QuantoniumOS
```

**Live ISO** (built on Arch as root, about 15 GB of scratch space):

```sh
sudo ./linux/iso/build-iso.sh            # -> linux/iso/out/quantoniumos-YYYY.MM.DD-x86_64.iso
```

On a non-Arch host such as WSL Ubuntu, `sudo linux/tests/wsl_arch_iso.sh` does the
same thing inside an Arch bootstrap chroot under `/var/tmp/qos-arch`.

The live system boots into the session as `qos` (password `qos`). To install it to
disk: run `archinstall`, boot the new system, then run `install.sh`.

**Hardware baseline:** x86-64 with AVX2 and FMA (Haswell or Zen and newer). ISO
binaries are built with `-march=x86-64-v3`. On CPUs without AVX2/FMA, rftmwd skips
the ASM kernels after a runtime CPUID check, but the `rftmw_native` CLIs require them.

## Using the middleware

```sh
rftmwctl stats                                  # RAM, page cache, zram, PSI, packs, workloads
rftmwctl pack.open ~/models/qwen2.5-coder_3b.rftmwpk
rftmwctl pack.prefetch 1 && rftmwctl pack.pin 1 # keep weights hot
rftmwctl pack.budget 8192                       # cap resident packs at 8 GiB
qos-run --name llm --mem-high 12288 --mem-max 14336 --swap-max 0 -- rftmw_qwen25_native ...
rftmwctl procs                                  # per-workload memory.current / peak / limits
rftmwctl rft.selftest 64                        # canonical RFT round-trip + ASM kernel check
rftmwctl rft.forward 1 0 -1 0                    # fast canonical path (N <= 1024)
systemd-cgls --user-unit rftmw.slice            # the same workloads, seen from systemd
```

Keybinds: `Super+Return` terminal, `Super+Space` launcher, `Super+B` browser,
`Super+Q` QuantoniumOS shell, `Super+M` live RFTMW monitor, `Super+Shift+M` evict
idle packs, `Super+W` close window, `Super+1..6` switch workspace.

## Performance

Measured with `linux/bench/run_bench.sh`, which builds the same benchmark against the
original sources (`bench/baseline/`) and the current ones, then checks that every output
matches (worst relative difference 5.8e-7, i.e. float32 rounding). i5-6600, 4 cores,
WSL2:

| Kernel | Before | After | Why |
|---|---|---|---|
| RFT N=256, per call | 1212 ms | 0.017 ms | basis cached per size (was rebuilt, O(N^3), every call) |
| RFT N=256, first call in a process | 1212 ms | 5 ms | basis stored on disk; prebuilt sizes ship in `/usr/share/quantoniumos/rftmw-basis` |
| RFT N=4096 forward / inverse | 621 / 371 ms | 1.4 / 2.7 ms | phasor recurrence instead of per-entry sin/cos, 16-lane blocks, OpenMP, adjoint computed once |
| ASM dot, 2048 floats in cache | 245 ns | 100 ns | 4 accumulators instead of 1 FMA chain |
| sgemv 896x896 (cache-resident) | 33.6 GB/s | 54.8 GB/s | 4 rows per pass, caller thread joins the pool |
| sgemv 11008x2048 (DRAM) | 26 GB/s | 26 GB/s | already at this machine's memory bandwidth |

Large matrix-vector products, which dominate LLM decoding, are limited by memory
bandwidth, not compute. Beyond this point they only get faster by moving fewer bytes
(quantised weights), not by better kernels.

OS-level settings for the same workload:
- **Ollama** (`/etc/quantoniumos/ollama.env`): flash attention and a q8_0 KV cache
  (half the KV memory of f16), one resident model, 30-minute keep-alive.
- **Memory:** zram (zstd) with `swappiness=180`, `page-cluster=0`, early kswapd
  watermarks, THP on madvise, and MGLRU `min_ttl_ms=1000`, so memory exhaustion ends
  in an OOM kill of the largest workload rather than a frozen desktop.
- **Disk:** 2 MiB readahead on block devices, so model files load in larger reads.
- **Build:** `-O3 -march=x86-64-v3`, LTO, and OpenMP for the large-N RFT path and the
  Q4_K/Q6_K mmap matmuls. Those matmuls were written with OpenMP pragmas, but the build
  never enabled OpenMP, so until now they ran on one core.
- `RFTMW_THREADS=N` caps the native thread pool, for example to leave cores for the
  desktop.

Not adopted: sched_ext schedulers (scx_lavd, scx_bpfland). The desktop-oriented ones
schedule by their own interactivity heuristics and don't reliably honor the cgroup CPU
weights rftmwd uses to rank workloads. The kernel's default EEVDF scheduler does honor
them, which keeps the shell ahead of Ollama.

## Verified so far

These were run on real hardware (WSL2 kernel 6.18, systemd 259):
- **NASM kernels on both ABIs** (`tests/test_asm_abi.c`): `rftmw_sgemv_f32.asm` was
  Windows-x64-only and `rftmw_spectral_dot.asm` was System-V-only. Each now has an
  ABI shim and matches the scalar reference on win64 and elf64.
- **rftmwd** (`tests/test_rftmwd.sh`, 19 checks): RFT round-trip error 2.7e-14; ASM dot
  product matches; prefetch/evict really move 128 MiB in and out of the page cache;
  the budget holds; a 300 MiB allocation is OOM-killed inside its 64 MiB scope while
  the daemon survives; after `SIGKILL` the daemon is restarted and re-adopts running
  workloads; spawned apps inherit the caller's environment and cwd.
- **Session tools** (`tests/test_session_tools.sh`): `qos-run` limits, its fallback
  when the broker is down, and `qos-status` output.
- All scripts pass `shellcheck -S warning`.

Run the suites with `linux/tests/run_wsl_tests.sh lint asm build rftmwd session`.
