#!/bin/bash
# OSU point-to-point latency and bandwidth using GPU buffers (D D = device-to-device).
# Requires: build.sh completed successfully.
#
# Two sweep dimensions:
#
#   Affinity mode (3 levels):
#     INTRA_CCD   — both ranks on the same CCD (same L3, same GPU die)
#     INTER_CCD   — ranks on different CCDs of the same GPU die
#     INTER_SOCKET— ranks on different GPU dies (xGMI/Infinity Fabric path)
#
#   Copy engine (2 levels), controlled by HSA_ENABLE_SDMA:
#     SDMA  (HSA_ENABLE_SDMA=1) — AMD SDMA hardware copy engine (default)
#     BLIT  (HSA_ENABLE_SDMA=0) — blit kernels (compute-shader copies)
#
# Result files: osu_{latency,bw}_{mode}_{SDMA,BLIT}.dat  (6 files each)

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BIN="$SCRIPT_DIR/build/libexec/osu-micro-benchmarks/mpi/pt2pt"
AFFINITY="$SCRIPT_DIR/../set_affinity_mi300a.sh"
RESULTS="$SCRIPT_DIR/results"

module load rocm openmpi

mkdir -p "$RESULTS"

# --bind-to none: let set_affinity_mi300a.sh own all CPU pinning via taskset
MPIRUN="mpirun -n 2 -mca pml ucx --bind-to none"

for mode in INTRA_CCD INTER_CCD INTER_SOCKET; do
    for engine in SDMA BLIT; do
        # HSA_ENABLE_SDMA=1 → SDMA engine (default); =0 → blit kernels
        sdma_val=$([ "$engine" = "SDMA" ] && echo 1 || echo 0)

        echo "======================================================"
        echo "  Affinity: ${mode}  |  Copy engine: ${engine} (HSA_ENABLE_SDMA=${sdma_val})"
        echo "======================================================"

        echo "--- OSU Latency (GPU buffers) ---"
        eval "HSA_ENABLE_SDMA=${sdma_val} ${mode}=1 ${MPIRUN} bash \"${AFFINITY}\" \"${BIN}/osu_latency\" D D" \
            | tee "${RESULTS}/osu_latency_${mode}_${engine}.dat"

        echo ""
        echo "--- OSU Bandwidth (GPU buffers, up to 1 GiB) ---"
        eval "HSA_ENABLE_SDMA=${sdma_val} ${mode}=1 ${MPIRUN} bash \"${AFFINITY}\" \"${BIN}/osu_bw\" -m $((1024*1024*1024)) D D" \
            | tee "${RESULTS}/osu_bw_${mode}_${engine}.dat"

        echo ""
    done
done
