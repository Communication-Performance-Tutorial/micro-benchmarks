#!/bin/bash
# RCCL send/receive latency+bandwidth and all-reduce throughput.
# Requires: build.sh completed successfully.
#
# Two operating modes, selected by the CPX_MODE environment variable:
#
#   SPX mode (default, CPX_MODE unset):
#     Node presents 4 GPUs.  Only INTER_SOCKET is valid for sendrecv_perf —
#     INTRA_CCD and INTER_CCD both route both ranks to GPU 0 which causes RCCL
#     to see only 1 GPU and abort.
#
#   CPX mode (CPX_MODE=1):
#     Node presents 24 GPUs (6 per physical die, 2 per CCD).  All three affinity
#     modes can be tested by selecting GPU pairs with ROCR_VISIBLE_DEVICES:
#       INTRA_CCD    — skipped (same CCD/HBM, no Infinity Fabric path)
#       INTER_GCD    — local indices 0,2  (diff CCD, same die/package)  ← collected
#       INTER_SOCKET — skipped (cross-die already covered by SPX sendrecv_INTER_SOCKET.dat)
#
#     Named INTER_GCD (not INTER_CCD) because in CPX mode this is purely a GPU
#     partition selection (no CPU pinning is applied — see below): indices 0
#     and 2 are two different GCDs (XCD-partitions) on the SAME physical
#     MI300A package, as distinct from INTER_SOCKET which crosses packages.
#
#     CPU pinning is not used in CPX mode: RCCL bandwidth on MI300A depends
#     only on which GPU pair is selected (via ROCR_VISIBLE_DEVICES), not on
#     CPU core placement.  This avoids cpuset issues on mixed allocations.
#
#     Usage (mixed allocation, no --exclusive needed):
#       CPX_MODE=1 bash run.sh
#
#     The job must have at least 7 GPUs pre-allocated (to span two dies):
#       salloc --gres=gpu:7 -p PPAC_MI300A_CPX ...

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD="$SCRIPT_DIR/rocm-systems/projects/rccl-tests/build"
AFFINITY="$SCRIPT_DIR/../set_affinity_mi300a.sh"
RESULTS="$SCRIPT_DIR/results"

module load rocm openmpi

mkdir -p "$RESULTS"

NGPU=$(rocminfo | grep -c "Device Type.*GPU")

# --bind-to none: let affinity wrapper or explicit taskset own CPU pinning
MPIRUN="mpirun -n 2 -mca pml ucx --bind-to none"

if [ -n "${CPX_MODE}" ]; then
    # ── CPX mode ─────────────────────────────────────────────────────────────
    # SLURM remaps allocated GPUs to local indices 0..N-1 via ROCR_VISIBLE_DEVICES.
    # With 7 GPUs requested: indices 0-5 = die 1 (NUMA 1), index 6 = die 2 (NUMA 2).
    # Each die has 3 CCDs; 2 virtual GPUs per CCD, so adjacent even/odd indices
    # share a CCD (0,1 → CCD0; 2,3 → CCD1; 4,5 → CCD2).
    #
    # CPU base for die 1 = core 24; die 2 = core 48; CCD stride = 8 cores.

    declare -A CPX_GPUS
    CPX_GPUS[INTRA_CCD]="0,1"
    CPX_GPUS[INTER_GCD]="0,2"
    CPX_GPUS[INTER_SOCKET]="0,6"

    # INTRA_CCD (indices 0,1) is intentionally skipped: both virtual GPUs share the
    # same physical CCD and HBM controller, so the transfer does not cross the
    # Infinity Fabric.
    #
    # INTER_SOCKET is intentionally skipped: cross-die bandwidth is already
    # measured in SPX mode (sendrecv_INTER_SOCKET.dat) with full GPU devices.
    # CPX mode adds only the INTER_GCD data point (same package, different GCD).
    #
    # CPU pinning is omitted: RCCL bandwidth on MI300A is determined entirely by
    # ROCR_VISIBLE_DEVICES (which GPU pair each rank uses), not by which CPU core
    # the rank runs on.  Skipping taskset avoids cpuset issues on mixed allocations.
    for MODE in INTER_GCD; do
        echo "======================================================"
        echo "  CPX Affinity: ${MODE} — Send/Recv (1B–1G)"
        echo "======================================================"
        ROCR_VISIBLE_DEVICES=${CPX_GPUS[$MODE]} \
        ${MPIRUN} "${BUILD}/sendrecv_perf" -b 1 -e 1G -f 2 -g 1 \
            | tee "${RESULTS}/sendrecv_cpx_${MODE}.dat"
        echo ""
    done

else
    # ── SPX mode (default) ────────────────────────────────────────────────────
    echo "======================================================"
    echo "  Affinity: INTER_SOCKET — Send/Recv (1B–1G, 2 GPUs)"
    echo "======================================================"
    # sendrecv_perf validates that (nGpus × nRanks) GPUs are visible from rank 0,
    # so ROCR_VISIBLE_DEVICES must NOT restrict visibility here.  CPU-only pinning
    # via taskset places rank 0 on die-0 cores and rank 1 on die-1 cores; RCCL
    # naturally assigns rank N → GPU N.
    # CPU pinning (core 0 for rank 0, core 24 for rank 1) improves NUMA locality
    # but is not required for correctness.  On shared allocations SLURM may
    # restrict the cpuset so core 24 is unavailable; fall back without pinning.
    ${MPIRUN} bash -c "
        r=\$OMPI_COMM_WORLD_LOCAL_RANK
        cpu=\$([ \$r -eq 0 ] && echo 0 || echo 24)
        if taskset -c \$cpu true 2>/dev/null; then
            exec taskset -c \$cpu '${BUILD}/sendrecv_perf' \"\$@\"
        else
            exec '${BUILD}/sendrecv_perf' \"\$@\"
        fi
    " -- -b 1 -e 1G -f 2 -g 1 \
        | tee "${RESULTS}/sendrecv_INTER_SOCKET.dat"
    echo ""

    echo "  NOTE: INTRA_CCD and INTER_GCD sendrecv require CPX-mode nodes."
    echo "        Run with CPX_MODE=1 on a PPAC_MI300A_CPX allocation. See README.md."
    echo ""
fi

echo "======================================================"
echo "  All-Reduce (4M–1G, ${NGPU} GPUs)"
echo "======================================================"
"$BUILD/all_reduce_perf" -b 4M -e 1G -f 2 -g "$NGPU" \
    | tee "${RESULTS}/all_reduce.dat"
