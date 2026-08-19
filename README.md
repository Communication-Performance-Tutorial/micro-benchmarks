# GPU Communication Micro-Benchmarks

This directory contains build and run scripts for three industry-standard benchmark suites that measure **point-to-point and collective communication latency and bandwidth** between AMD GPUs.

These benchmarks complement the application-level measurements in `CG-Tutorial/` by isolating communication performance from computation.  Together they answer: *how fast can data actually move between GPUs on this node?*

---

## Benchmark suites

| Suite | What it measures | Communication layer |
|---|---|---|
| [`osu/`](osu/) | Point-to-point latency & bandwidth (GPU buffer → GPU buffer, and CPU host buffer → host buffer) | GPU-Aware MPI / plain MPI (UCX) |
| [`rccl-tests/`](rccl-tests/) | Send/recv latency & bandwidth; all-reduce throughput | RCCL (ROCm Collective Communications Library) |
| [`rocshmem/`](rocshmem/) | One-sided put/get latency & bandwidth (intra-node) | rocSHMEM IPC backend |

[`plotting-scripts/`](plotting-scripts/) contains Python utilities for visualizing the `.dat` output from all three suites above — see its [README](plotting-scripts/README.md) for details.

---

## Directory layout

```
micro-benchmarks/
├── README.md               ← this file
├── set_affinity_mi300a.sh  ← per-rank CPU/GPU affinity wrapper (used by all run.sh scripts)
├── osu/
│   ├── build.sh                ← downloads OSU source tarball, patches for ROCm, builds
│   ├── run.sh                  ← pt2pt latency + bandwidth with GPU (D D) buffers; SPX (INTRA_XCD+INTER_SOCKET) or CPX (INTER_GCD)
│   ├── run_host.sh             ← pt2pt latency + bandwidth with CPU host (H H) buffers, 3 affinity modes
│   ├── sbatch_test.sh          ← SLURM job (SPX): build once, then run the INTRA_XCD + INTER_SOCKET sweep
│   ├── sbatch_cpx_test.sh      ← SLURM job (CPX): build once, then run the INTER_XCD sweep
│   └── sbatch_host_test.sh     ← SLURM job: build once, then run run_host.sh
├── rccl-tests/
│   ├── build.sh            ← sparse-checkouts rccl-tests from ROCm/rocm-systems, builds
│   ├── run.sh              ← sendrecv latency/bandwidth + all-reduce throughput, 3 affinity modes
│   └── sbatch_test.sh      ← SLURM job: build once, then run
├── rocshmem/
│   ├── build.sh                      ← sparse-checkouts rocshmem from ROCm/rocm-systems, builds (IPC backend)
│   ├── run.sh                        ← ping-pong latency + flood put/get bandwidth, 3 affinity modes
│   └── sbatch_test.sh                ← SLURM job: build once, then run run.sh
└── plotting-scripts/                 ← Python plotting utilities for all three suites' results/ output
    ├── common_style.py                   ← shared styling + .dat file parsers
    ├── osu_ccd_plot.py, osu_xcd_plot.py, osu_host_plot.py
    ├── rccl_sendrecv_plot.py, rccl_allreduce_plot.py
    └── rocshmem_plot.py
```

Each sub-directory is self-contained: `build.sh` fetches its own sources, and `sbatch_test.sh` runs both steps as a single SLURM job.

---

## Quick start

Each benchmark can be submitted independently.  `sbatch_test.sh` automatically skips the build on subsequent runs if the binary is already present.

```bash
sbatch osu/sbatch_test.sh
sbatch rccl-tests/sbatch_test.sh
sbatch rocshmem/sbatch_test.sh    # first run takes ~20 min to build
```

Output is written to `<suite>/test_<jobid>.log`.

```bash
# Check status
squeue -u $USER

# View results as they stream in
tail -f test_<jobid>.log
```

---

## Process affinity sweep

Each `run.sh` script runs its benchmarks **three times** by default (SPX mode), once per affinity mode, by passing `set_affinity_mi300a.sh` to `mpirun` as a per-rank launcher.  The three SPX modes expose different levels of the MI300A memory hierarchy:

| Mode | CPU layout | GPU layout | What it isolates |
|---|---|---|---|
| `INTRA_CCD` / `INTRA_XCD` | Both ranks share one 8-core CCD (same L3 cache) | Both ranks → same GPU die | Baseline: intra-GPU IPC with maximally cache-close host threads |
| `INTER_CCD` / `INTER_XCD` | Ranks on different CCDs of the same GPU die (different L3 domains) | Both ranks → same GPU die | Cost of an L3 boundary without crossing a GPU die boundary |
| `INTER_SOCKET` | Ranks on different GPU dies (different NUMA nodes) | Rank 0 → GPU 0, Rank 1 → GPU 1 | Full inter-die transfer over xGMI/Infinity Fabric |

### MI300A topology recap

Each MI300A is an APU package with 24 CPU cores (3 CCDs) and one GPU (6 GCDs/XCDs, 228 CUs total, presented as a single unified device in SPX mode).  A node has **4 separate MI300A packages**, each connected to every other package directly via Infinity Fabric (xGMI) — a full mesh, not a hierarchy within one package.  The relevant topology for these benchmarks is:

```
Node (4 × MI300A packages, all-to-all Infinity Fabric mesh)
├── Package 0  (GPU 0)  ←  cores  0-23
│   ├── CCD 0: cores  0- 7   (shared L3)
│   ├── CCD 1: cores  8-15   (shared L3)
│   └── CCD 2: cores 16-23   (shared L3)
├── Package 1  (GPU 1)  ←  cores 24-47
│   └── ...
├── Package 2  (GPU 2)  ←  cores 48-71
│   └── ...
└── Package 3  (GPU 3)  ←  cores 72-95
    └── ...
```

`INTRA_CCD` and `INTER_CCD` both keep both ranks on **GPU 0**, so the GPU data path is identical — only the CPU cache topology changes. `INTER_SOCKET` is the "real" two-GPU case where data must cross the Infinity Fabric **package-to-package** link — genuinely different from crossing GCDs *within* one package (only possible via CPX-mode partitioning; see `rccl-tests/README.md` and `rocshmem/README.md` for CPX-mode `INTER_XCD` tests that exercise that intra-package path).

### Measured results (OSU latency + bandwidth, ppac-pl1-s24-26)

| Mode | Small-msg latency (1 B) | Large-msg bandwidth (4 MiB) |
|---|---|---|
| `INTRA_CCD` | **0.31 µs** | ~430 GB/s |
| `INTER_CCD` | ~1.0 µs | ~530 GB/s |
| `INTER_SOCKET` | ~2.0 µs | ~53 GB/s |

Same-GPU transfers (INTRA_CCD, INTER_CCD) are 10× faster in bandwidth than cross-die transfers (INTER_SOCKET).  INTER_CCD beats INTRA_CCD in bandwidth because the two ranks have independent L3 caches and don't compete for the same cache resources.  INTER_SOCKET is limited by the xGMI die-to-die fabric (53 GB/s peak).

### Using the affinity wrapper directly

```bash
# Both ranks on the same CCD and same GPU:
INTRA_CCD=1   mpirun -n 2 --bind-to none bash set_affinity_mi300a.sh <program> [args]

# Ranks on different CCDs but same GPU:
INTER_CCD=1   mpirun -n 2 --bind-to none bash set_affinity_mi300a.sh <program> [args]

# Ranks on different GPU dies:
INTER_SOCKET=1 mpirun -n 2 --bind-to none bash set_affinity_mi300a.sh <program> [args]
```

`--bind-to none` is required so that `mpirun` does not pre-bind ranks to CPU subsets before the wrapper can run `taskset`.  When allocating interactively with `salloc`, use `--exclusive` (or `--cpus-per-task` ≥ 25) so all physical cores 0–95 are inside the job's cgroup and accessible to `taskset`.

The topology constants default to the MI300A values (`NUM_CPUS=96`, `NUM_GPUS=4`, `CORES_PER_CCD=8`) and can be overridden via environment variables for other nodes.

---

## Hardware and software requirements

- AMD Instinct MI300A node (partition `PPAC_MI300A_SPX`, node `ppac-pl1-s24-30`)
- ROCm ≥ 7.2 and GPU-Aware OpenMPI (loaded via `module load rocm openmpi`)
- Internet access from compute nodes (all three suites fetch their sources on first build)

The example SLURM scripts request `--gres=gpu:4` and `--nodelist=ppac-pl1-s24-30`.  Update these lines if running on a different node.
