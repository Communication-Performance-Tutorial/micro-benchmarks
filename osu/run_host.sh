#!/bin/bash
# OSU point-to-point latency and bandwidth using CPU (host) buffers (H H = host-to-host).
# Requires: build.sh completed successfully.
#
# Unlike run.sh's GPU-buffer (D D) sweep, this test never touches the GPU at all --
# MPI_Send/MPI_Recv operate directly on host memory allocated with malloc(). This
# means the INTRA_CCD / INTER_CCD / INTER_SOCKET affinity modes are unambiguous
# here: the CPU core placement they select IS the entire data path (same-CCD
# shared-L3 copy, cross-CCD cache-coherent copy, or cross-socket transfer over the
# CPU-side Infinity Fabric link) -- no GPU-assignment caveat is needed, in contrast
# to GPU-buffer tests elsewhere in this repo (e.g. rocshmem/run.sh) where the mode
# name does not always describe the actual GPU-level path.
#
# There is no copy-engine (SDMA/BLIT) axis here: that toggle only affects GPU-side
# memory copies. Host buffers are moved by the CPU / UCX's shared-memory (or CMA)
# transport, which set_affinity_mi300a.sh's ROCR_VISIBLE_DEVICES restriction does
# not affect (harmless no-op for a program that never calls into the GPU runtime).
#
# Result files: osu_latency_H_<mode>.dat, osu_bw_H_<mode>.dat  (3 files each)

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BIN="$SCRIPT_DIR/build/libexec/osu-micro-benchmarks/mpi/pt2pt"
AFFINITY="$SCRIPT_DIR/../set_affinity_mi300a.sh"
RESULTS="$SCRIPT_DIR/results"

module load rocm openmpi

mkdir -p "$RESULTS"

# --bind-to none: let set_affinity_mi300a.sh own all CPU pinning via taskset
MPIRUN="mpirun -n 2 -mca pml ucx --bind-to none"

for mode in INTRA_CCD INTER_CCD INTER_SOCKET; do
    echo "======================================================"
    echo "  Affinity: ${mode}  |  Buffers: Host (H H)"
    echo "======================================================"

    echo "--- OSU Latency (host buffers) ---"
    eval "${mode}=1 ${MPIRUN} bash \"${AFFINITY}\" \"${BIN}/osu_latency\" H H" \
        | tee "${RESULTS}/osu_latency_H_${mode}.dat"

    echo ""
    echo "--- OSU Bandwidth (host buffers, up to 1 GiB) ---"
    eval "${mode}=1 ${MPIRUN} bash \"${AFFINITY}\" \"${BIN}/osu_bw\" -m $((1024*1024*1024)) H H" \
        | tee "${RESULTS}/osu_bw_H_${mode}.dat"

    echo ""
done
