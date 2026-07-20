#!/bin/bash
# set_affinity_mi300a.sh — Per-rank CPU and GPU affinity wrapper for 2-rank MPI jobs
#                          on AMD Instinct MI300A (96-core, 4-GPU configuration).
#
# Usage (as mpirun per-rank launcher):
#   INTRA_CCD=1   mpirun -n 2 bash set_affinity_mi300a.sh <program> [args...]
#   INTER_CCD=1   mpirun -n 2 bash set_affinity_mi300a.sh <program> [args...]
#   INTER_SOCKET=1 mpirun -n 2 bash set_affinity_mi300a.sh <program> [args...]
#
# Affinity modes — set exactly one:
#
#   INTRA_CCD=1     Both ranks share the same CCD (same 8-core L3 domain) and the
#                   same GPU die.  Measures the baseline for intra-GPU IPC transfers
#                   where both host threads are maximally cache-close.
#
#   INTER_CCD=1     Ranks are placed on different CCDs of the same GPU die.
#                   Each rank has its own L3 cache, but the GPU is shared.  Isolates
#                   the cost of crossing an L3 boundary vs. crossing a GPU die boundary.
#
#   INTER_SOCKET=1  Ranks are placed on different GPU dies (different NUMA nodes).
#                   Data transfers cross the xGMI/Infinity Fabric inter-die link.
#                   This is the expected "two-GPU" benchmark configuration.
#
# MI300A topology assumed (matches set_cpu_gpu_mi300a.sh defaults):
#
#   96 CPU cores  — 4 NUMA nodes × 24 cores
#   4 GPU devices — one GCD (GPU Compute Die) per NUMA node
#   8 cores per CCD, 3 CCDs per GPU die
#
#   GPU 0  →  cores   0-23   (CCD 0:  0- 7 | CCD 1:  8-15 | CCD 2: 16-23)
#   GPU 1  →  cores  24-47   (CCD 0: 24-31 | CCD 1: 32-39 | CCD 2: 40-47)
#   GPU 2  →  cores  48-71   (CCD 0: 48-55 | CCD 1: 56-63 | CCD 2: 64-71)
#   GPU 3  →  cores  72-95   (CCD 0: 72-79 | CCD 1: 80-87 | CCD 2: 88-95)
#
# Override topology constants with env vars if your node differs:
#   NUM_CPUS, NUM_GPUS, CORES_PER_CCD

export local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-0}

if [ -z "${NUM_CPUS}" ];    then NUM_CPUS=96;    fi
if [ -z "${NUM_GPUS}" ];    then NUM_GPUS=4;     fi
if [ -z "${CORES_PER_CCD}" ]; then CORES_PER_CCD=8; fi

CORES_PER_GPU=$(( NUM_CPUS / NUM_GPUS ))          # 24
CCDS_PER_GPU=$(( CORES_PER_GPU / CORES_PER_CCD )) #  3

# ── Affinity mode selection ────────────────────────────────────────────────────

if [ -n "${INTRA_CCD}" ]; then
    # Both ranks on CCD 0 of GPU 0: cores 0–7.
    # rank 0 → core 0, rank 1 → core 1 (same CCD, same GPU).
    my_gpu=0
    cpu_start=$(( local_rank ))

elif [ -n "${INTER_CCD}" ]; then
    # Ranks on adjacent CCDs of GPU 0: CCD 0 (cores 0–7) and CCD 1 (cores 8–15).
    # rank 0 → core 0, rank 1 → core 8 (different CCDs, same GPU).
    my_gpu=0
    cpu_start=$(( local_rank * CORES_PER_CCD ))

elif [ -n "${INTER_SOCKET}" ]; then
    # Ranks on different GPU dies: rank 0 → GPU 0, rank 1 → GPU 1.
    # Each rank is pinned to CCD 0 of its respective GPU die.
    # On single-GPU nodes this gracefully degrades to INTER_CCD behaviour
    # (different CCDs, same GPU) so no invalid ROCR_VISIBLE_DEVICES is set.
    if [ "$NUM_GPUS" -gt 1 ]; then
        my_gpu=$(( local_rank ))
        cpu_start=$(( local_rank * CORES_PER_GPU ))
    else
        my_gpu=0
        cpu_start=$(( local_rank * CORES_PER_CCD ))
    fi

else
    # No affinity mode set — run without CPU/GPU pinning.
    exec "$@"
fi

export ROCR_VISIBLE_DEVICES=${my_gpu}

# Pin to the target core if it is inside the SLURM-allocated cpuset.
# On shared (non-exclusive) allocations SLURM may restrict the cpuset to a
# small subset of cores; taskset fails with EINVAL if the core is absent.
# Falling back without CPU pinning is safe: ROCR_VISIBLE_DEVICES already
# routes each rank to the correct GPU, and CPU placement does not affect
# GPU-to-GPU bandwidth on MI300A.
if taskset -c ${cpu_start} true 2>/dev/null; then
    exec taskset -c ${cpu_start} "$@"
else
    exec "$@"
fi
