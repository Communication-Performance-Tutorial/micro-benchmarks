# rocSHMEM — One-Sided GPU Communication Benchmarks

[rocSHMEM](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocshmem) is
AMD's implementation of the OpenSHMEM PGAS (Partitioned Global Address Space) model
for GPUs.  Instead of explicit send/receive pairs, rocSHMEM lets a GPU kernel *put*
data directly into another GPU's memory or *get* data from it — the remote GPU never
needs to call a matching receive.

This directory benchmarks three key operations using the **IPC backend**, which
transfers data via ROCm IPC handles and delivers sub-microsecond latency for
intra-node communication.

| Test | What it measures |
|---|---|
| Ping-Pong (test ID 12) | One-sided put round-trip latency: 1 WG × 1 thread |
| WG Put (test ID 26) | Blocking put bandwidth, 1 WG × 64 threads, quiet included in timing |
| WG Get (test ID 24) | Blocking get bandwidth, 1 WG × 64 threads, quiet included in timing |

Tests 26 and 24 use a single work-group (single context) to give a **single-stream
point-to-point** measurement directly comparable to OSU's `bw` test and RCCL's
`sendrecv_perf`.  Both include `rocshmem_ctx_quiet` in the timing window, so the
timer stops only after all in-flight data is committed — matching OSU's completion
semantics.

---

## File layout

```
rocshmem/
├── build.sh        ← sparse-checkouts rocshmem from ROCm/rocm-systems, builds (IPC backend)
├── run.sh          ← runs ping-pong latency + flood put + WG get tests, 3 affinity modes
└── sbatch_test.sh  ← SLURM job: build once if needed, then run
```

After a successful build, the installed binaries live at:

```
~/rocshmem/share/rocshmem/
├── rocshmem_functional_tests   ← used by run.sh
└── rocshmem_unit_tests
```

Override the install prefix by setting `INSTALL_PREFIX` before running `build.sh`, or
set `ROCSHMEM_INSTALL` at runtime to point `run.sh` at an alternate installation.

---

## Environment Setup

```bash
module load rocm openmpi
```

rocSHMEM uses MPI for process startup and bootstrap; OpenMPI's UCX transport handles
the initial rendezvous.  All data transfers after initialization go through the IPC
backend (ROCm IPC handles), not MPI.

---

## How to build

```bash
bash build.sh
```

The script does a sparse git checkout of only `projects/rocshmem` from the
`ROCm/rocm-systems` monorepo, then builds with the `ipc_single` build config
(`USE_RO=OFF`, `USE_GDA=OFF`, `USE_IPC=ON`).

**First build takes approximately 20 minutes** — rocSHMEM compiles device bitcode for
every supported GPU architecture (`gfx90a`, `gfx942`, `gfx1100`, etc.).

> **Build caveat — Python test dependency:**
> The upstream `ipc_single` config enables `-DBUILD_PYTHON_TESTS=ON`, which requires
> the `scikit-build` Python package.  `build.sh` overrides this with
> `-DBUILD_PYTHON_TESTS=OFF` to avoid the dependency.  The functional and unit test
> binaries used by `run.sh` are unaffected.

---

## How to run

```bash
bash run.sh
```

Or submit as a SLURM job (builds first if needed):

```bash
sbatch sbatch_test.sh
```

`run.sh` sweeps all three affinity modes, running all three tests (ping-pong, flood
put, WG get) for each mode:

```
======================================================
  Affinity: INTRA_CCD  (rank0→core0, rank1→core1)
======================================================
--- Ping-Pong Latency (put, 1 WG x 1 thread, 2 PEs) ---
...
--- Put Bandwidth (1 WG x 64 threads, 1B–4M, 2 PEs) ---
...
--- Get Bandwidth (1 WG x 64 threads, 1B–4M, 2 PEs) ---
...

======================================================
  Affinity: INTER_CCD  (rank0→core0, rank1→core8)
======================================================
...

======================================================
  Affinity: INTER_SOCKET  (rank0→core0, rank1→core24)
======================================================
...
```

Results are saved to `results/pingpong_latency_<MODE>.dat`,
`results/put_bw_<MODE>.dat`, and `results/get_bw_<MODE>.dat`.

### Affinity sweep

| Mode | CPU layout | GPU layout | Expected characteristic |
|---|---|---|---|
| `INTRA_CCD` | Both PEs on the same 8-core CCD (shared L3), cores 0,1 | Rank 0 → GPU 0, Rank 1 → GPU 1 | Lowest CPU-launch overhead; host threads share L3 |
| `INTER_CCD` | PEs on different CCDs of GPU die 0, cores 0,8 | Rank 0 → GPU 0, Rank 1 → GPU 1 | IPC data path unchanged; only CPU L3 locality differs |
| `INTER_SOCKET` | PEs on different GPU dies (NUMA 0 vs NUMA 1), cores 0,24 | Rank 0 → GPU 0, Rank 1 → GPU 1 | IPC handles cross the xGMI die-to-die link |

> **GPU assignment caveat:**
> The rocSHMEM test driver selects GPUs internally via
> `hipSetDevice(OMPI_COMM_WORLD_LOCAL_RANK)`.  This means rank 0 always uses GPU 0
> and rank 1 always uses GPU 1 in **all three affinity modes** — it is not possible
> to assign both ranks to the same GPU device using this test binary.  As a result,
> INTRA_CCD and INTER_CCD differ only in **CPU pinning**, not in the GPU-to-GPU IPC
> path (which is always GPU 0 ↔ GPU 1).
>
> **Do not set `ROCR_VISIBLE_DEVICES`** when running these tests.  Restricting
> visibility to a single device causes `hipSetDevice(1)` in rank 1 to fail with
> `invalid device ordinal`.

### MI300A architecture finding

On MI300A all three modes show nearly identical performance.  rocSHMEM's IPC backend
transfers data directly between GPU HBM pools via Infinity Fabric — a path that does
not change based on which CPU CCD the host thread is pinned to.  The only notable
effect of CPU affinity is in the GPU kernel *launch* overhead, which is minor.

| Mode | Ping-Pong latency | Put BW (4 MiB) | Get BW (4 MiB) |
|---|---|---|---|
| INTRA_CCD | 0.78 µs | 11.82 GB/s | 13.08 GB/s |
| INTER_CCD | 0.76 µs | 11.79 GB/s | 13.09 GB/s |
| INTER_SOCKET | 0.79 µs | 11.76 GB/s | 13.09 GB/s |

All values within measurement noise of each other.  The single-stream put/get
bandwidth (~12–13 GB/s at 4 MiB) is lower than RCCL's ~79 GB/s because rocSHMEM
uses a single work-group issuing sequential puts with a quiet, whereas RCCL's
`sendrecv_perf` uses an optimized collective engine that pipelines transfers.

---

## Reference output (MI300A, IPC backend, intra-node, single-stream)

### INTRA_CCD (rank0→core0, rank1→core1)

```
--- Ping-Pong Latency (put, 1 WG x 1 thread, 2 PEs) ---
### Creating Test:	PingPong	B=ipc PE=2 W=1 Z=1 ###
# Volume (B)   Msg Size (B)   # of timed Msgs   Latency (us)   Bandwidth (GB/s)   Msg Rate (Msg/s)
4              4              10                       0.78             0.00          639386.19

--- Put Bandwidth (1 WG x 64 threads, 1B–4M, 2 PEs) ---
### Creating Test:	Blocking WG level Puts	B=ipc PE=2 W=1 Z=64 ###
# Volume (B)   Msg Size (B)   # of timed Msgs   Latency (us)   Bandwidth (GB/s)   Msg Rate (Msg/s)
1              1              10                       0.68             0.00        1479289.94
1024           1024           10                       0.86             1.11        1162790.70
65536          65536          10                       5.54            11.01         180375.18
1048576        1048576        10                      82.63            11.82          12101.85
4194304        4194304        10                     330.58            11.82           3025.02

--- Get Bandwidth (1 WG x 64 threads, 1B–4M, 2 PEs) ---
### Creating Test:	Blocking WG level Gets	B=ipc PE=2 W=1 Z=64 ###
# Volume (B)   Msg Size (B)   # of timed Msgs   Latency (us)   Bandwidth (GB/s)   Msg Rate (Msg/s)
1              1              10                       0.86             0.00        1162790.70
1024           1024           10                       1.04             0.92         961538.46
65536          65536          10                       5.33            11.46         187687.69
1048576        1048576        10                      75.29            12.97          13282.33
4194304        4194304        10                     298.68            13.08           3348.11
```

### INTER_SOCKET (rank0→core0, rank1→core24) — representative; all modes are similar

```
--- Ping-Pong Latency ---
4              4              10                       0.79             0.00          629722.92
--- Put Bandwidth (peak at 4 MiB) ---
4194304        4194304        10                     332.16            11.76           3010.60
--- Get Bandwidth (peak at 4 MiB) ---
4194304        4194304        10                     298.52            13.09           3349.90
```

The `B=ipc` tag in all test headers confirms the IPC backend is active.  Ping-pong
latency is **~0.8 µs** — about 14× lower than the same test run via the RO
(MPI-routed) backend, which measures ~11.6 µs.

---

## Key concepts

### PGAS vs message passing

In MPI, every `MPI_Send` requires a matching `MPI_Recv`.  In PGAS, any PE can issue a
*put* or *get* without coordination on the remote side:

```c
// PE 0 puts 4 bytes into PE 1's symmetric heap — PE 1 does nothing
rocshmem_putmem(dest_on_pe1, src, 4, 1);
rocshmem_quiet();   // ensure the put is visible before continuing
```

This asymmetric model is natural for GPU kernels where one side computes and
immediately pushes results to another GPU's memory.

### IPC vs RO backends

rocSHMEM supports three backends selected at **compile time** (not runtime when
building a single-backend binary):

| Backend | Transport | Best for | Latency |
|---|---|---|---|
| **IPC** (`ipc_single`) | ROCm IPC handles; direct GPU-to-GPU via Infinity Fabric | Intra-node (same host) | ~0.8 µs |
| RO (Reverse Offload) | GPU kernels offload puts/gets to the CPU, which calls MPI | Inter-node (across network) | ~11–12 µs |
| GDA (GPU Direct Async) | RDMA via network NIC (InfiniBand/RoCE) | Inter-node on RDMA fabrics | varies |

This directory uses `ipc_single` because the benchmarks run on a single node.  If you
need cross-node rocSHMEM measurements, rebuild with the `ro_net` config.

> **Why not `ro_ipc`?**  The `ro_ipc` build config compiles *both* RO and IPC but
> always selects RO at runtime — the dynamic backend selection only activates when all
> three backends (GDA+RO+IPC) are compiled together.  For a guaranteed IPC path,
> `ipc_single` is the correct config.

### Work-groups and threads

Both the WGPut and WGGet tests use `W=1` work-group × `Z=64` threads — a single
concurrent transfer stream per PE.  This matches the single sender/receiver model
of OSU's `bw` test and RCCL's `sendrecv_perf`, making the results directly
comparable.  The timer includes the final `rocshmem_ctx_quiet`, so bandwidth reflects
completed (not just initiated) transfers.

### Symmetric heap

rocSHMEM allocates a fixed shared-memory region on each PE called the *symmetric
heap*.  All remotely accessible buffers must be allocated from this heap.
`ROCSHMEM_HEAP_SIZE=6442450944` (6 GiB) sets its size; lower this value if you
encounter allocation failures.

---

## Requirements

- ROCm ≥ 7.0 (HIP runtime, AMD SMI)
- GPU-Aware OpenMPI (for process launch and bootstrap)
- Internet access from the compute node (first build clones from GitHub)
- ~6 GiB GPU memory per PE for the symmetric heap
