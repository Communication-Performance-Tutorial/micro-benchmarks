#!/bin/bash
# rocSHMEM intra-node latency and bandwidth benchmarks (IPC backend).
#
# All tests use rocshmem_functional_tests, which is the standard CTest binary.
# Test IDs from CTestFunctionalInstall.cmake:
#   -a 26  : WGPut bandwidth     (1 WG, 64 threads — blocking put + quiet, up to 1 GiB)
#   -a 24  : WGGet bandwidth     (1 WG, 64 threads — blocking get + quiet, up to 1 GiB)
#
# Both tests output a Latency (us) column alongside Bandwidth (GB/s), providing
# a full latency-vs-size sweep (1B–1 GiB) in the same file.  This replaces the
# PingPong test (-a 12), which is hardcoded to a single 4B message size in the
# source (apply_args() overrides any -s flag for PingPongTestType) and therefore
# cannot produce a size sweep.
#
# Tests 26 and 24 use W=1 (single work-group / single context) to give a
# single-stream point-to-point measurement directly comparable to OSU's
# bw/bibw tests and RCCL's sendrecv_perf.  Both include rocshmem_ctx_quiet
# in the timing window, matching the completion semantics of OSU's bandwidth
# test (all in-flight messages flushed before the timer stops).
# -batch 1 limits the buffer to 1 × max_msg_size per direction so the
# 6 GiB symmetric heap is not exceeded when max_msg_size = 1 GiB.
#
# Each test is run three times with different CPU affinity layouts to reveal
# how NUMA/CCD topology affects one-sided rocSHMEM performance:
#
#   INTRA_CCD   — both PEs on the same 8-core CCD (cores 0, 1)
#   INTER_CCD   — PEs on different CCDs of the same GPU die (cores 0, 8)
#   INTER_SOCKET— PEs on different GPU dies / NUMA nodes  (cores 0, 24)
#
# GPU assignment is NOT controlled here. The test driver calls
# hipSetDevice(OMPI_COMM_WORLD_LOCAL_RANK) directly, so rank 0 always uses
# GPU 0 and rank 1 always uses GPU 1 in all three modes.
# Setting ROCR_VISIBLE_DEVICES would cause hipSetDevice(1) to fail for rank 1;
# only taskset is used for topology isolation.
#
# Requires: build.sh completed successfully.
# Binary is installed to ~/rocshmem by default; override with ROCSHMEM_INSTALL.
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TESTS="${ROCSHMEM_INSTALL:-$HOME/rocshmem}/bin/rocshmem_functional_tests"
RESULTS="$SCRIPT_DIR/results"

module load rocm openmpi

mkdir -p "$RESULTS"

# CPU core for each rank per affinity mode (MI300A: 8 cores/CCD, 3 CCDs/die, 4 dies).
# These cores improve NUMA locality but are not required for correctness — rocSHMEM
# assigns GPU N to rank N via hipSetDevice regardless of CPU placement.
# On shared (non-exclusive) allocations, taskset falls back gracefully if the
# requested core is outside the SLURM-allocated cpuset.
declare -A CPU0 CPU1
CPU0[INTRA_CCD]=0;    CPU1[INTRA_CCD]=1
CPU0[INTER_CCD]=0;    CPU1[INTER_CCD]=8
CPU0[INTER_SOCKET]=0; CPU1[INTER_SOCKET]=24

# Base mpirun flags as an array to avoid newline-as-separator issues
# --bind-to none: taskset inside the per-rank launcher owns all CPU pinning
MPIRUN_BASE=(mpirun --bind-to none -mca pml ucx -mca osc ucx
    -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384
    -x ROCSHMEM_HEAP_SIZE=6442450944)

for mode in INTRA_CCD INTER_CCD INTER_SOCKET; do
    c0=${CPU0[$mode]}
    c1=${CPU1[$mode]}

    echo "======================================================"
    echo "  Affinity: ${mode}  (rank0→core${c0}, rank1→core${c1})"
    echo "======================================================"

    echo "--- Put Bandwidth + Latency (1 WG x 64 threads, 1B–1G, 2 PEs) ---"
    "${MPIRUN_BASE[@]}" -x ROCSHMEM_MAX_NUM_CONTEXTS=1 -np 2 \
        bash -c "r=\$OMPI_COMM_WORLD_LOCAL_RANK
                 cpu=\$([ \$r -eq 0 ] && echo $c0 || echo $c1)
                 if taskset -c \$cpu true 2>/dev/null; then
                     exec taskset -c \$cpu '${TESTS}' \"\$@\"
                 else
                     exec '${TESTS}' \"\$@\"
                 fi" -- \
        -a 26 -w 1 -z 64 -s 1073741824 -batch 1 -localbuftype heap \
        | tee "${RESULTS}/put_bw_${mode}.dat"

    echo ""
    echo "--- Get Bandwidth (1 WG x 64 threads, 1B–1G, 2 PEs) ---"
    "${MPIRUN_BASE[@]}" -x ROCSHMEM_MAX_NUM_CONTEXTS=1 -np 2 \
        bash -c "r=\$OMPI_COMM_WORLD_LOCAL_RANK
                 cpu=\$([ \$r -eq 0 ] && echo $c0 || echo $c1)
                 if taskset -c \$cpu true 2>/dev/null; then
                     exec taskset -c \$cpu '${TESTS}' \"\$@\"
                 else
                     exec '${TESTS}' \"\$@\"
                 fi" -- \
        -a 24 -w 1 -z 64 -s 1073741824 -batch 1 -localbuftype heap \
        | tee "${RESULTS}/get_bw_${mode}.dat"

    echo ""
done
