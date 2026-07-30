#!/bin/bash
# OSU point-to-point latency and bandwidth using GPU buffers (D D = device-to-device).
# Requires: build.sh completed successfully.
#
# Two operating modes, selected by the CPX_MODE environment variable (mirroring
# ../rccl-tests/run.sh):
#
#   SPX mode (default, CPX_MODE unset):
#     Sweeps CPU affinity via set_affinity_mi300a.sh across 3 modes:
#       INTRA_CCD   — both ranks on the same CCD (same L3), both ranks → GPU 0
#       INTER_CCD   — ranks on different CCDs of the same die, both ranks → GPU 0
#       INTER_SOCKET— ranks on different GPU dies (different packages)
#     INTRA_CCD and INTER_CCD both route both ranks to GPU 0 — i.e. both are
#     INTRA_GCD at the GPU level (same GCD), differing only in CPU L3 locality.
#     INTER_SOCKET is genuinely cross-package.
#
#   CPX mode (CPX_MODE=1, requires a PPAC_MI300A_CPX allocation):
#     ROCR_VISIBLE_DEVICES=0,2 restricts rank 0 / rank 1 to two different GCD
#     (XCD) partitions on the SAME physical die — genuine INTER_GCD traffic
#     (same package, different GCD), matching ../rccl-tests/run.sh's CPX mapping.
#     No CPU pinning is applied: GPU bandwidth here is governed entirely by
#     ROCR_VISIBLE_DEVICES, and skipping taskset avoids cpuset issues on shared
#     CPX allocations.
#
# ── Naming: INTRA_GCD / INTER_GCD / INTER_SOCKET ────────────────────────────
# A physical MI300A package contains 6 GCDs (XCDs), grouped in pairs under 3
# CCDs; a node has 4 such packages. INTRA_GCD = both ranks on the same GCD.
# INTER_GCD = different GCDs WITHIN the same package (CPX mode only — SPX mode
# cannot express this). INTER_SOCKET = different packages entirely. The
# INTRA_CCD/INTER_CCD env-var names (inherited from set_affinity_mi300a.sh)
# describe CPU pinning only; at the GPU level both are INTRA_GCD, which is why
# both are grouped under that heading in the README rather than being treated
# as a 3-way GPU-topology sweep on their own.
#
#   Copy engine (2 levels), controlled by HSA_ENABLE_SDMA:
#     SDMA  (HSA_ENABLE_SDMA=1) — AMD SDMA hardware copy engine (default)
#     BLIT  (HSA_ENABLE_SDMA=0) — blit kernels (compute-shader copies)
#
# Result files:
#   SPX (default): osu_{latency,bw}_{INTRA_CCD,INTER_CCD,INTER_SOCKET}_{SDMA,BLIT}.dat
#   CPX (CPX_MODE=1): osu_{latency,bw}_INTER_GCD_{SDMA,BLIT}.dat

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BIN="$SCRIPT_DIR/build/libexec/osu-micro-benchmarks/mpi/pt2pt"
AFFINITY="$SCRIPT_DIR/../set_affinity_mi300a.sh"
RESULTS="$SCRIPT_DIR/results"

module load rocm openmpi

mkdir -p "$RESULTS"

# --bind-to none: let set_affinity_mi300a.sh (SPX) own CPU pinning; unused in CPX
MPIRUN="mpirun -n 2 -mca pml ucx --bind-to none"

if [ -n "${CPX_MODE}" ]; then
    # ── CPX mode: genuine INTER_GCD (same package, different GCD) ───────────
    for engine in SDMA BLIT; do
        sdma_val=$([ "$engine" = "SDMA" ] && echo 1 || echo 0)

        echo "======================================================"
        echo "  CPX Affinity: INTER_GCD  |  Copy engine: ${engine} (HSA_ENABLE_SDMA=${sdma_val})"
        echo "======================================================"

        echo "--- OSU Latency (GPU buffers) ---"
        eval "HSA_ENABLE_SDMA=${sdma_val} ROCR_VISIBLE_DEVICES=0,2 ${MPIRUN} \"${BIN}/osu_latency\" D D" \
            | tee "${RESULTS}/osu_latency_INTER_GCD_${engine}.dat"

        echo ""
        echo "--- OSU Bandwidth (GPU buffers, up to 1 GiB) ---"
        eval "HSA_ENABLE_SDMA=${sdma_val} ROCR_VISIBLE_DEVICES=0,2 ${MPIRUN} \"${BIN}/osu_bw\" -m $((1024*1024*1024)) D D" \
            | tee "${RESULTS}/osu_bw_INTER_GCD_${engine}.dat"

        echo ""
    done
else
    # ── SPX mode (default): INTRA_GCD (2 CPU sub-variants) + INTER_SOCKET ───
    for mode in INTRA_CCD INTER_CCD INTER_SOCKET; do
        for engine in SDMA BLIT; do
            # HSA_ENABLE_SDMA=1 → SDMA engine (default); =0 → blit kernels
            sdma_val=$([ "$engine" = "SDMA" ] && echo 1 || echo 0)

            gpu_label="INTRA_GCD"
            [ "$mode" = "INTER_SOCKET" ] && gpu_label="INTER_SOCKET"

            echo "======================================================"
            echo "  Affinity: ${gpu_label}  (CPU sub-mode: ${mode})  |  Copy engine: ${engine} (HSA_ENABLE_SDMA=${sdma_val})"
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

    echo "  NOTE: INTER_GCD requires a CPX-mode allocation."
    echo "        Run with CPX_MODE=1 on a PPAC_MI300A_CPX allocation. See README.md."
    echo ""
fi
