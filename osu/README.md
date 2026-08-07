# OSU Micro-Benchmarks — Point-to-Point (GPU and CPU buffers)

The [OSU Micro-Benchmarks (OMB)](https://mvapich.cse.ohio-state.edu/benchmarks/) are the standard tool for measuring MPI communication performance.  This directory builds the OMB point-to-point tests and runs them two ways: with **GPU (device) buffers** (`run.sh`), the same path taken by the CG solver in `CG-Tutorial/CG-GPU/`, and with **CPU (host) buffers** (`run_host.sh`), a pure host-memory MPI baseline with no GPU involvement at all.

| Test | What it measures |
|---|---|
| `osu_latency D D` / `H H` | Round-trip latency for a single message between two GPU (or host) buffers |
| `osu_bw D D` / `H H` | Unidirectional bandwidth from one GPU (or host) buffer to another |

The `D D` flag tells OMB to allocate both the send and receive buffers in GPU device memory, exercising the GPU-Aware MPI path end to end.  `H H` allocates both buffers in plain host memory with `malloc()`, exercising ordinary CPU-to-CPU MPI — RCCL (GPU-only collectives) and rocSHMEM (GPU symmetric-heap PGAS model) have no equivalent host-buffer mode, so this is the only host-memory communication test in the suite.

---

## File layout

```
osu/
├── build.sh                    ← downloads OMB source tarball, patches for ROCm, builds
├── run.sh                      ← runs osu_latency + osu_bw with GPU buffers; SPX (INTRA_XCD+INTER_SOCKET) or CPX (INTER_XCD)
├── run_host.sh                 ← runs osu_latency + osu_bw with CPU (host) buffers, 3 affinity modes
├── sbatch_test.sh              ← SLURM job (SPX): build once if needed, then run the INTRA_XCD + INTER_SOCKET sweep
├── sbatch_cpx_test.sh          ← SLURM job (CPX): build once if needed, then run the INTER_XCD sweep
├── sbatch_host_test.sh         ← SLURM job: build once if needed, then run run_host.sh
└── ../set_affinity_mi300a.sh   ← shared per-rank CPU/GPU affinity wrapper
```

After a successful build, `build.sh` creates:

```
osu/
└── build/
    └── libexec/osu-micro-benchmarks/mpi/pt2pt/
        ├── osu_latency
        └── osu_bw
```

---

## Environment Setup

```bash
module load rocm openmpi
```

ROCm provides HIP and the GPU runtime.  The OpenMPI build must include UCX with ROCm support (`-mca pml ucx`) so that `MPI_Send`/`MPI_Recv` can operate on GPU pointers.

---

## How to build

```bash
bash build.sh
```

The script downloads the OSU source tarball from the official MVAPICH site (no official git repo exists for OMB), applies a one-line patch (`hipDeviceReset` → `hipDeviceSynchronize` to avoid a teardown race on ROCm), then configures and builds with `--enable-rocm`.  Build takes about 2 minutes.

---

## How to run

```bash
bash run.sh                # INTRA_XCD (2 CPU sub-modes) + INTER_SOCKET, SPX mode (default)
CPX_MODE=1 bash run.sh     # INTER_XCD, requires a CPX allocation
```

Or submit as SLURM jobs (each builds first if needed):

```bash
sbatch sbatch_test.sh       # SPX: INTRA_XCD + INTER_SOCKET
sbatch sbatch_cpx_test.sh   # CPX: INTER_XCD
```

`run.sh` sweeps three affinity modes automatically, printed **using GPU-level (XCD) nomenclature**, not the CPU-based `_CCD` naming — see the "Naming" section below for why. The output for each mode is prefixed with a header line, for example:

```
======================================================
  Affinity: INTRA_XCD  (CPU sub-mode: INTRA_CCD)
======================================================
--- OSU Latency (GPU buffers) ---
...
--- OSU Bandwidth (GPU buffers, up to 16 MiB) ---
...

======================================================
  Affinity: INTRA_XCD  (CPU sub-mode: INTER_CCD)
======================================================
...

======================================================
  Affinity: INTER_SOCKET  (CPU sub-mode: INTER_SOCKET)
======================================================
...

======================================================
  CPX Affinity: INTER_XCD
======================================================
...
```

### Naming: INTRA_XCD / INTER_XCD / INTER_SOCKET

A physical MI300A package contains 6 XCDs, grouped in pairs under 3 CCDs; a node has 4 such packages.  `set_affinity_mi300a.sh`'s `INTRA_CCD`/`INTER_CCD` env vars only ever change **CPU** pinning — both route both ranks to **GPU 0** (the same XCD), so at the GPU level they are both `INTRA_XCD`.  Getting a genuine `INTER_XCD` data point (crossing XCDs while staying on the *same* package) requires CPX-mode `ROCR_VISIBLE_DEVICES` partitioning, which is not possible in SPX mode at all — this is why `run.sh` needs a `CPX_MODE` switch, mirroring `../rccl-tests/run.sh`.

| GPU-level category | CPU sub-mode | CPU layout | GPU layout |
|---|---|---|---|
| `INTRA_XCD` | `INTRA_CCD` | Both ranks on the same 8-core CCD (shared L3) | Both ranks → GPU 0 |
| `INTRA_XCD` | `INTER_CCD` | Ranks on different CCDs of the same package (separate L3 domains) | Both ranks → GPU 0 |
| `INTER_XCD` *(CPX only)* | — (no CPU pinning) | `ROCR_VISIBLE_DEVICES=0,2` | Two XCDs, **same** package (different CCDs, same die) |
| `INTER_SOCKET` | `INTER_SOCKET` | Ranks on different MI300A packages (different NUMA nodes) | Rank 0 → GPU 0, Rank 1 → GPU 1 (different packages) |

### Affinity sweep

Each benchmark pair is run to reveal how the MI300A memory hierarchy affects GPU-Aware MPI performance:

| Mode | Measured small-msg latency (1 B, SDMA) | Measured large-msg BW (4 MiB, SDMA) |
|---|---|---|
| `INTRA_XCD` (`INTRA_CCD` sub-mode) | **~0.31 µs** | ~430 GB/s |
| `INTRA_XCD` (`INTER_CCD` sub-mode) | ~1.0 µs | ~530 GB/s |
| `INTER_XCD` (CPX) | ~1.23 µs | ~92 GB/s |
| `INTER_SOCKET` | ~2.0 µs | ~53 GB/s |

Key observations from measured results (`INTRA_XCD`/`INTER_SOCKET` on `ppac-pl1-s24-26`; `INTER_XCD` on `ppac-pl1-s25-40`, CPX):

- **Small-message latency** for the two `INTRA_XCD` sub-modes brackets `INTER_XCD` same-CCD (~0.31 µs) < cross-CCD-same-GPU (~1.0 µs), while the genuinely cross-package `INTER_SOCKET` case is highest (~2.0 µs).  `INTER_XCD` (~1.23 µs, crossing XCDs but staying on one package) lands **between** the same-GPU and cross-package cases, exactly as expected for an intermediate hop on the fabric.

- **Large-message bandwidth** at 4 MiB (SDMA) shows the same-GPU `INTRA_XCD` cases far ahead (~430–530 GB/s, since both ranks share one XCD's on-die fabric) with `INTER_XCD` (~92 GB/s) and `INTER_SOCKET` (~53 GB/s) both well below — `INTER_XCD` outperforms `INTER_SOCKET` because the intra-package XCD-to-XCD Infinity Fabric links are wider than the inter-package ones (see `../rccl-tests/README.md`'s architecture note for the same effect measured with RCCL).  With the BLIT copy engine the gap widens further — see the reference output below, where `INTER_XCD` peaks at ~342 GB/s (64 MiB) vs `INTER_SOCKET`'s ~91 GB/s peak (1 GiB).

- **Rendezvous-threshold latency floor** (~512 B–4 KiB, engine- and mode-dependent) is visible in all modes: this is the UCX eager-to-rendezvous protocol transition point.  Bandwidth drops around this size and recovers as message size grows beyond the rendezvous threshold.

> **SLURM note**: `run.sh` passes `--bind-to none` to `mpirun` so that
> `set_affinity_mi300a.sh` can use `taskset` to pin each rank to the intended core
> (SPX mode only — CPX mode applies no CPU pinning; see the naming note above).
> When running manually, use `--exclusive` (or `--cpus-per-task` ≥ 25) in the
> `salloc`/`srun` command so all cores 0–95 are in the job's cgroup.

### Reference output (MI300A, ppac-pl1-s24-26 for INTRA_XCD/INTER_SOCKET, ppac-pl1-s25-40 CPX for INTER_XCD)

```
============================================================
  Affinity mode: INTRA_XCD  (CPU sub-mode: INTRA_CCD)
============================================================
--- osu_latency (D D) ---
# OSU MPI-ROCM Latency Test v7.5
# Size       Avg Latency(us)
1                       0.34
4                       0.31
64                      0.35
256                     0.36
512                     8.43   ← UCX protocol switch (eager → rendezvous)
4096                    8.55
65536                   9.72
1048576                25.29
4194304                26.89

--- osu_bw (D D, up to 4 MiB) ---
# OSU MPI-ROCM Bandwidth Test v7.5
# Size      Bandwidth (MB/s)
256                  1394.81
512                    85.16  ← rendezvous transition dip
4096                  694.49
65536                9441.79
1048576           127015.46
4194304           432602.71

============================================================
  Affinity mode: INTRA_XCD  (CPU sub-mode: INTER_CCD)
============================================================
--- osu_latency (D D) ---
# Size       Avg Latency(us)
1                       1.04
4                       0.99
64                      1.17
256                     1.31
512                     8.09
4096                    8.19
65536                   8.91
1048576                24.03
4194304                26.01

--- osu_bw (D D, up to 4 MiB) ---
# Size      Bandwidth (MB/s)
256                   349.93
512                    79.03
4096                  636.26
65536                8682.84
1048576            163755.00
4194304            529137.15

============================================================
  CPX Affinity: INTER_XCD  (SDMA, HSA_ENABLE_SDMA=1)
============================================================
--- osu_latency (D D) ---
# OSU MPI-ROCM Latency Test v7.5
# Size       Avg Latency(us)
1                       1.23
4                       1.17
64                      1.35
256                     1.49
512                     7.05   ← UCX protocol switch (eager → rendezvous)
4096                   14.05
65536                  14.56
1048576                23.99
4194304                51.34

--- osu_bw (D D, up to 1 GiB) ---
# OSU MPI-ROCM Bandwidth Test v7.5
# Size      Bandwidth (MB/s)
256                    319.07
512                     83.02  ← rendezvous transition dip
4096                   445.29
65536                 6803.08
1048576              56228.45
4194304              91924.06
67108864            114585.81
1073741824          116655.15

============================================================
  CPX Affinity: INTER_XCD  (BLIT, HSA_ENABLE_SDMA=0)
============================================================
--- osu_latency (D D) ---
# Size       Avg Latency(us)
1                       1.01
4                       0.97
64                      1.27
256                     1.30
512                    10.59
4096                    9.99
65536                  10.38
1048576                13.29
4194304                23.03

--- osu_bw (D D, up to 1 GiB) ---
# Size      Bandwidth (MB/s)
256                    320.77
512                     52.89
4096                   900.61
65536                13283.99
1048576             135460.95
4194304             228071.43
67108864            342468.96  ← peak (BLIT outruns SDMA on this path)
1073741824          310078.83

============================================================
  Affinity mode: INTER_SOCKET
============================================================
--- osu_latency (D D) ---
# Size       Avg Latency(us)
1                       1.98
4                       2.00
64                      2.59
256                     2.47
512                     7.41
4096                    7.53
65536                   8.50
1048576                25.56
4194304                79.64

--- osu_bw (D D, up to 4 MiB) ---
# Size      Bandwidth (MB/s)
256                   172.38
512                    77.24
4096                  622.54
65536                8529.62
1048576             42263.48
4194304             53307.89
```

The results reveal four distinct performance tiers of the MI300A interconnect:
same-XCD/same-CCD IPC (`INTRA_XCD`, `INTRA_CCD` sub-mode: ~0.31 µs / ~430 GB/s),
same-XCD/cross-CCD IPC (`INTRA_XCD`, `INTER_CCD` sub-mode: ~1.0 µs / ~530 GB/s),
cross-XCD same-package (`INTER_XCD`: ~1.23 µs / ~92 GB/s SDMA, up to ~342 GB/s BLIT),
and cross-package xGMI (`INTER_SOCKET`: ~2.0 µs / ~53 GB/s).

---

## CPU (host) buffer tests (`run_host.sh`)

```bash
bash run_host.sh
```

Or submit as a SLURM job (builds first if needed):

```bash
sbatch sbatch_host_test.sh
```

`run_host.sh` runs the same `osu_latency`/`osu_bw` pair as `run.sh`, but with `H H` (host) buffers instead of `D D`, and without the SDMA/BLIT copy-engine axis — that toggle only controls which hardware engine performs a *GPU-side* memory copy, so it has no effect when neither buffer is on the GPU.

### Why these labels need no caveat here

Everywhere else in this repo, `INTRA_CCD`/`INTER_CCD`/`INTER_SOCKET` describe **CPU** placement, and a separate note is needed to say what that does or doesn't imply about GPU placement (e.g. rocSHMEM's `run.sh`, where all three modes secretly use the same GPU pair; or RCCL/rocSHMEM's `INTER_XCD` vs `INTER_SOCKET` distinction, which needs CPX-mode GPU partitioning to even express). `run_host.sh` has no such caveat: since
neither buffer ever touches a GPU, the CPU core pair selected by `set_affinity_mi300a.sh` **is** the entire communication path — same-CCD (shared L3), cross-CCD (separate L3, same package), or cross-socket (separate MI300A package, over the CPU-side Infinity Fabric link) — with nothing left ambiguous.

| Mode | CPU layout | Data path |
|---|---|---|
| `INTRA_CCD` | Both ranks on the same 8-core CCD (shared L3) | Same-L3 memory copy |
| `INTER_CCD` | Ranks on different CCDs of the same package (separate L3 domains) | Cross-L3, intra-package cache-coherent copy |
| `INTER_SOCKET` | Ranks on different MI300A packages (different NUMA nodes) | Cross-package transfer over Infinity Fabric |

Result files: `osu_latency_H_<mode>.dat`, `osu_bw_H_<mode>.dat` for each of the three modes.

> **Expectation vs GPU buffers:** host-memory bandwidth is bound by CPU memory
> controllers and cache-coherency traffic rather than the GPU interconnect, so
> absolute numbers will differ from the `D D` results above — but the same ordering
> (`INTRA_CCD` ≤ `INTER_CCD` < `INTER_SOCKET` for latency) is expected, since it comes
> from the same CPU/NUMA hierarchy.  Run `run_host.sh` to collect real numbers for
> your node; no numbers are recorded here yet.

---

## Key concepts

### GPU-Aware MPI (`D D` mode)

When OMB is invoked with `D D`, both the send and receive buffers are allocated with `hipMalloc`.  The MPI library (via UCX) detects that the pointer is in device memory and routes the transfer over the GPU interconnect without staging through host memory. 

Without GPU-Aware MPI (`H H` mode) every send would require:
1. GPU → host copy (`hipMemcpy D→H`)
2. MPI send on host pointer
3. MPI receive on host pointer
4. Host → GPU copy (`hipMemcpy H→D`)

`D D` eliminates steps 1 and 4, which is what `CG-GPU/` does when passing GPU pointers to `MPI_Isend`/`MPI_Irecv`.

### Why two ranks, not four

Latency and bandwidth are point-to-point properties between a *pair* of endpoints. `run.sh` uses `-n 2` so that one rank sends and the other receives.  More ranks would interleave multiple transfers and obscure the per-pair numbers.

---

## Requirements

- ROCm ≥ 7.0 (`hipcc`, HIP runtime)
- GPU-Aware OpenMPI built with UCX + ROCm support
- Internet access from the compute node (first build downloads the tarball from mvapich.cse.ohio-state.edu)
