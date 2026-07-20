# OSU Micro-Benchmarks — GPU Point-to-Point

The [OSU Micro-Benchmarks (OMB)](https://mvapich.cse.ohio-state.edu/benchmarks/) are
the standard tool for measuring MPI communication performance.  This directory builds
and runs the two most important point-to-point tests using **GPU (device) buffers** so
that data never touches the CPU — the same path taken by the CG solver in `CG-Tutorial/CG-GPU/`.

| Test | What it measures |
|---|---|
| `osu_latency D D` | Round-trip latency for a single message between two GPU buffers |
| `osu_bw D D` | Unidirectional bandwidth from one GPU buffer to another |

The `D D` flag tells OMB to allocate both the send and receive buffers in GPU device
memory, exercising the GPU-Aware MPI path end to end.

---

## File layout

```
osu/
├── build.sh                    ← downloads OMB source tarball, patches for ROCm, builds
├── run.sh                      ← runs osu_latency + osu_bw with GPU buffers, 3 affinity modes
├── sbatch_test.sh              ← SLURM job: build once if needed, then run
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

ROCm provides HIP and the GPU runtime.  The OpenMPI build must include UCX with ROCm
support (`-mca pml ucx`) so that `MPI_Send`/`MPI_Recv` can operate on GPU pointers.

---

## How to build

```bash
bash build.sh
```

The script downloads the OSU source tarball from the official MVAPICH site
(no official git repo exists for OMB), applies a one-line patch (`hipDeviceReset` →
`hipDeviceSynchronize` to avoid a teardown race on ROCm), then configures and builds
with `--enable-rocm`.  Build takes about 2 minutes.

---

## How to run

```bash
bash run.sh
```

Or submit as a SLURM job (builds first if needed):

```bash
sbatch sbatch_test.sh
```

`run.sh` sweeps all three affinity modes automatically.  The output for each mode is
prefixed with a header line, for example:

```
======================================================
  Affinity: INTRA_CCD
======================================================
--- OSU Latency (GPU buffers) ---
...
--- OSU Bandwidth (GPU buffers, up to 16 MiB) ---
...

======================================================
  Affinity: INTER_CCD
======================================================
...

======================================================
  Affinity: INTER_SOCKET
======================================================
...
```

### Affinity sweep

Each benchmark pair is run three times with different process placement to reveal how
the MI300A memory hierarchy affects GPU-Aware MPI performance:

| Mode | CPU layout | GPU layout | Measured small-msg latency | Measured large-msg BW (4 MiB) |
|---|---|---|---|---|
| `INTRA_CCD` | Both ranks on the same 8-core CCD (shared L3) | Both ranks → GPU 0 | **~0.31 µs** | ~430 GB/s |
| `INTER_CCD` | Ranks on different CCDs of GPU 0 (separate L3 domains) | Both ranks → GPU 0 | ~1.0 µs | ~530 GB/s |
| `INTER_SOCKET` | Ranks on different GPU dies (different NUMA nodes) | Rank 0 → GPU 0, Rank 1 → GPU 1 | ~2.0 µs | ~53 GB/s |

Key observations from measured results on `ppac-pl1-s24-26`:

- **Small-message latency** shows the expected hierarchy: INTRA_CCD (~0.31 µs) <
  INTER_CCD (~1.0 µs) < INTER_SOCKET (~2.0 µs).  Crossing an L3 boundary triples
  small-message overhead; crossing a GPU die boundary doubles it again.

- **Large-message bandwidth** shows the opposite ordering: INTER_CCD (~530 GB/s) >
  INTRA_CCD (~430 GB/s) >> INTER_SOCKET (~53 GB/s).  Both same-GPU modes saturate the
  intra-die memory fabric, with INTER_CCD slightly ahead because the two ranks have
  independent L3 caches rather than competing for the same one.  INTER_SOCKET is limited
  by the xGMI die-to-die link (~53 GB/s).

- **~8 µs floor at 512 bytes** visible in all three modes: this is the UCX
  eager-to-rendezvous protocol transition point.  Bandwidth drops at 512 B and recovers
  as message size grows beyond the rendezvous threshold.

> **SLURM note**: `run.sh` passes `--bind-to none` to `mpirun` so that
> `set_affinity_mi300a.sh` can use `taskset` to pin each rank to the intended core.
> When running manually, use `--exclusive` (or `--cpus-per-task` ≥ 25) in the
> `salloc`/`srun` command so all cores 0–95 are in the job's cgroup.

### Reference output (MI300A, ppac-pl1-s24-26)

```
============================================================
  Affinity mode: INTRA_CCD
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
  Affinity mode: INTER_CCD
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

The results reveal the three distinct performance tiers of the MI300A interconnect:
same-CCD IPC (~0.31 µs / ~430 GB/s), cross-CCD IPC (~1.0 µs / ~530 GB/s), and
cross-die xGMI (~2.0 µs / ~53 GB/s).

---

## Key concepts

### GPU-Aware MPI (`D D` mode)

When OMB is invoked with `D D`, both the send and receive buffers are allocated with
`hipMalloc`.  The MPI library (via UCX) detects that the pointer is in device memory
and routes the transfer over the GPU interconnect without staging through host memory.

Without GPU-Aware MPI (`H H` mode) every send would require:
1. GPU → host copy (`hipMemcpy D→H`)
2. MPI send on host pointer
3. MPI receive on host pointer
4. Host → GPU copy (`hipMemcpy H→D`)

`D D` eliminates steps 1 and 4, which is what `CG-GPU/` does when passing GPU
pointers to `MPI_Isend`/`MPI_Irecv`.

### Why two ranks, not four

Latency and bandwidth are point-to-point properties between a *pair* of endpoints.
`run.sh` uses `-n 2` so that one rank sends and the other receives.  More ranks would
interleave multiple transfers and obscure the per-pair numbers.

---

## Requirements

- ROCm ≥ 7.0 (`hipcc`, HIP runtime)
- GPU-Aware OpenMPI built with UCX + ROCm support
- Internet access from the compute node (first build downloads the tarball from
  mvapich.cse.ohio-state.edu)
