# RCCL Tests — Collective Communication Benchmarks

[RCCL](https://github.com/ROCm/rccl) (ROCm Collective Communications Library) is AMD's GPU-optimized implementation of the NCCL API.  It provides collective operations (all-reduce, broadcast, send/recv, etc.) that exploit Infinity Fabric / xGMI links between GPU dies for transfers that bypass the CPU entirely.

This directory builds and runs two tests:

| Test | What it measures |
|---|---|
| `sendrecv_perf` | Bidirectional point-to-point bandwidth and latency (2 GPUs) |
| `all_reduce_perf` | All-reduce throughput across all available GPUs |

Together these answer: *how fast can RCCL move data between GPUs*, complementing the MPI-based measurements in `osu/`.

---

## File layout

```
rccl-tests/
├── build.sh                     ← sparse-checkouts rccl-tests from ROCm/rocm-systems, patches, builds
├── run.sh                       ← runs sendrecv_perf (INTER_SOCKET only) + all_reduce_perf
├── sbatch_test.sh               ← SLURM job: build once if needed, then run
├── sbatch_test_system_module.sh ← SLURM job: load system rccl-tests module and run
└── ../set_affinity_mi300a.sh    ← shared per-rank CPU/GPU affinity wrapper
```

After a successful build, binaries live at:

```
rccl-tests/
└── rocm-systems/
    └── projects/rccl-tests/
        └── build/
            ├── sendrecv_perf
            └── all_reduce_perf
```

---

## Environment Setup

```bash
module load rocm openmpi
```

RCCL is bundled with ROCm (`$ROCM_PATH/lib/librccl.so`).  OpenMPI is required to launch multi-rank jobs.

---

## How to build

```bash
bash build.sh
```

The script does a sparse git checkout of only `projects/rccl-tests` from the `ROCm/rocm-systems` monorepo (avoids cloning the full multi-GB repository), detects the local GPU architecture, applies a backward-compatibility patch for the `NCCL_CTA_POLICY_ZERO` symbol missing in ROCm 7.2.4, and builds with `GPU_TARGETS` set to the detected arch (`gfx942` on MI300A).  Build takes about 1 minute.

---

## How to run

```bash
bash run.sh
```

Or submit as a SLURM job (builds first if needed):

```bash
sbatch sbatch_test.sh
```

To run against the system-installed `rccl-tests` module instead of local build artifacts:

```bash
sbatch sbatch_test_system_module.sh
```

`run.sh` runs `sendrecv_perf` under the `INTER_SOCKET` affinity mode, then runs
`all_reduce_perf` once across all GPUs:

```
======================================================
  Affinity: INTER_SOCKET — Send/Recv (1B–4M, 2 GPUs)
======================================================
...

======================================================
  All-Reduce (4M–128M, 4 GPUs)
======================================================
...
```

### Affinity sweep and device partition mode

`sendrecv_perf` is a point-to-point test that requires **2 ranks each with a distinct GPU**.  This constrains which affinity modes are valid depending on the MI300A device partition mode the node is running:

#### SPX mode (default — 4 GPUs per node)

| Affinity mode | SPX support | Notes |
|---|---|---|
| `INTRA_XCD` | **Not supported** | Both ranks map to the same `ROCR_VISIBLE_DEVICES=0`; RCCL finds only 1 GPU and aborts |
| `INTER_XCD` | **Not supported** | Same reason |
| `INTER_SOCKET` | **Supported** | Rank 0 → GPU 0, Rank 1 → GPU 1 (different physical dies/packages) |

#### CPX mode — partition `PPAC_MI300A_CPX` (24 GPUs per node)

In CPX mode the physical MI300A presents **24 virtual GPU devices** — 6 per die (2 per CCD), each with 38 compute units (1/6 of a full XCD).  SLURM remaps the allocated GPU set to local indices 0…N-1 via `ROCR_VISIBLE_DEVICES`.

GPU index layout within a single-node CPX allocation:

```
Die 1 (NUMA 1, cores 24-47):
  Indices 0,1 → CCD 0  (cores 24-31)
  Indices 2,3 → CCD 1  (cores 32-39)
  Indices 4,5 → CCD 2  (cores 40-47)
Die 2 (NUMA 2, cores 48-71)  — only reachable with ≥ 7 GPUs allocated:
  Index  6    → CCD 0  (cores 48-55)
  ...
```

| Affinity mode | GPU selection | Min GPUs to request | Collected |
|---|---|---|---|
| `INTRA_CCD` | `ROCR_VISIBLE_DEVICES=0,1` | 2 | **No — see note** |
| `INTER_XCD` | `ROCR_VISIBLE_DEVICES=0,2` | 3 | Yes |
| `INTER_SOCKET` | `ROCR_VISIBLE_DEVICES=0,6` | 7 | **No** — cross-die bandwidth already measured in SPX mode (`sendrecv_INTER_SOCKET.dat`) |

> **Naming: INTER_XCD vs INTER_SOCKET vs INTER_CCD.**
> A physical MI300A package has 6 XCDs, grouped in pairs under 3 CCDs;
> a node has 4 such packages.  `INTER_XCD` crosses two XCDs **within the same
> package** (indices 0 and 2 — different CCDs, same die); `INTER_SOCKET`
> crosses two XCDs in **different packages** (indices 0 and 6 — different
> dies).  This mode is named `INTER_XCD`, not `INTER_CCD`, because in CPX mode
> no CPU pinning is applied at all (see below) — the mode name describes the
> GPU-level relationship (same package, different XCD), not a CPU-cache
> boundary.
>
> **Why INTRA_CCD is not collected:**
> In CPX mode, GPU indices 0 and 1 are two 38-CU virtual partitions of the
> *same physical CCD* on the same die.  They share the same HBM controller
> region; an RCCL transfer between them does not traverse the Infinity Fabric
> at all and measures intra-HBM copy bandwidth rather than interconnect
> throughput.  This number is not useful for characterizing inter-GPU
> communication and is therefore excluded from the benchmark suite.
> `INTER_XCD` (same package, different CCDs) and `INTER_SOCKET` (different
> packages) are the two meaningful data points.

Run the CPX affinity sweep (INTER_XCD only — INTER_SOCKET is redundant with SPX) with:

```bash
salloc --gres=gpu:7 -p PPAC_MI300A_CPX --exclusive -t 00:30:00
CPX_MODE=1 bash run.sh
```

> **MI300A architecture note:**
> Measured results show a meaningful bandwidth difference between modes:
> INTER_XCD (~113 GB/s peak, same package) outperforms INTER_SOCKET (~84 GB/s
> peak, cross-package) by ~35%.  The Infinity Fabric links that connect
> separate packages on MI300A are narrower than the intra-package fabric
> between XCDs, so cross-package transfers saturate at a lower bandwidth
> ceiling.  Both modes share the same unified HBM address space; the
> difference is purely in how much fabric bandwidth is available on the
> data path between the two virtual GPU partitions.

`all_reduce_perf` always uses all available GPUs on the node and is unaffected by this limitation.

### Reference output (MI300A, SPX — 4 GPUs, INTER_SOCKET):

```
======================================================
  Affinity: INTER_SOCKET — Send/Recv (1B–4M, 2 GPUs)
======================================================
#  Rank  0 ... device  0 [0000:01:00] AMD Instinct MI300A
#  Rank  1 ... device  1 [0001:01:00] AMD Instinct MI300A
#       size         count    type    redop    root     time   algbw   busbw ...
           4             1  float     sum       -1     7.07    0.00    0.00
          64            16  float     sum       -1     7.12    0.01    0.01
        1024           256  float     sum       -1     7.74    0.13    0.13
        4096          1024  float     sum       -1     9.07    0.45    0.45
       65536         16384  float     sum       -1    10.89    6.02    6.02
      262144         65536  float     sum       -1    14.50   18.08   18.08
     1048576        262144  float     sum       -1    21.25   49.35   49.35
     4194304       1048576  float     sum       -1    52.89   79.31   79.31
# Avg bus bandwidth    : 11.69 GB/s
```

### Reference output (MI300A, CPX — `ppac-pl1-s25-40`, INTER_XCD and INTER_SOCKET, Jul 8 2026):

> INTRA_CCD is not collected — see the note in the affinity table above.

```
======================================================
  CPX Affinity: INTER_XCD — Send/Recv (1B–1G)
======================================================
#  Rank  0 Group  0 on ppac-pl1-s25-40 device  0 [0001:01:00] AMD Instinct MI300A
#  Rank  1 Group  0 on ppac-pl1-s25-40 device  1 [0001:01:00] AMD Instinct MI300A
#       size         count    type   redop      time   algbw   busbw  #wrong
           4             1   float     sum     12.75    0.00    0.00       0
        1024           256   float     sum     13.40    0.08    0.08       0
        4096          1024   float     sum     13.22    0.31    0.31       0
     1048576        262144   float     sum     23.88   43.92   43.92       0
     4194304       1048576   float     sum     67.48   62.16   62.16       0
    67108864      16777216   float     sum    627.25  106.99  106.99       0
  1073741824     268435456   float     sum   9464.64  113.45  113.45       0
# Avg bus bandwidth    : 33.98 GB/s

======================================================
  CPX Affinity: INTER_SOCKET — Send/Recv (1B–1G)
======================================================
#  Rank  0 Group  0 on ppac-pl1-s25-40 device  0 [0001:01:00] AMD Instinct MI300A
#  Rank  1 Group  0 on ppac-pl1-s25-40 device  1 [0002:01:00] AMD Instinct MI300A
#       size         count    type   redop      time   algbw   busbw  #wrong
           4             1   float     sum      9.79    0.00    0.00       0
        1024           256   float     sum      8.99    0.11    0.11       0
        4096          1024   float     sum      9.63    0.43    0.43       0
     1048576        262144   float     sum     28.06   37.36   37.36       0
     4194304       1048576   float     sum     82.37   50.92   50.92       0
    67108864      16777216   float     sum    843.20   79.59   79.59       0
  1073741824     268435456   float     sum  12838.20   83.64   83.64       0
# Avg bus bandwidth    : 26.22 GB/s
```

**Key finding:** INTER_XCD peaks at **~113 GB/s** (both ranks on the same package, `[0001:01:00]`) while INTER_SOCKET peaks at **~84 GB/s** (ranks on separate packages, `[0001:01:00]` vs `[0002:01:00]`).  The ~35% bandwidth advantage for same-package transfers reflects that Infinity Fabric links between packages on MI300A are narrower than the intra-package fabric between XCDs.  See the MI300A architecture note above for context.

---

## Key concepts

### Algorithm bandwidth vs bus bandwidth

RCCL reports two bandwidth columns:

- **`algbw`** — total data size ÷ time; what the application sees.
- **`busbw`** — normalized for the number of GPUs in the collective; measures how efficiently the interconnect is used.  For an all-reduce across *N* GPUs, `busbw = algbw × 2(N−1)/N`.

When comparing across GPU counts, `busbw` is the fair metric — it removes the inherent algorithmic scaling factor.

### RCCL vs GPU-Aware MPI

Both RCCL and GPU-Aware MPI (as in `osu/`) can send GPU buffers without host copies. The difference is in the path:

- **GPU-Aware MPI** routes through the UCX transport layer, which may use host memory internally depending on the provider.
- **RCCL** uses ROCm IPC handles and Infinity Fabric directly, bypassing the MPI stack entirely.  On MI300A this gives RCCL a significant bandwidth advantage for large collectives.

This is why `CG-GPU/` exposes an `rccl` communication variant alongside the standard GPU-Aware MPI variants.

### Sparse checkout

`build.sh` uses `git sparse-checkout` so only the `projects/rccl-tests` subtree is fetched from the multi-GB `ROCm/rocm-systems` monorepo:

```bash
git clone --filter=blob:none --no-checkout https://github.com/ROCm/rocm-systems.git
git sparse-checkout set projects/rccl-tests
```

This reduces the initial clone from several GB to a few MB.

---

## Requirements

- ROCm ≥ 7.0 (includes `librccl.so`)
- GPU-Aware OpenMPI (for launching multi-rank tests with `mpirun`)
- Internet access from the compute node (first build clones from GitHub)
