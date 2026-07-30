#!/bin/bash
# rocSHMEM work-group (WG) count saturation sweep — WGPut (test 26) and
# WGGet (test 24), IPC backend.
#
# The existing run.sh always launches a single work-group (-w 1) per PE and
# sweeps message size only. This script instead fixes a large message size
# and sweeps the *number of work-groups* (concurrency), from 1 WG up to the
# source GPU's total compute-unit (CU) count, to find how many WGs are
# needed to saturate a given communication path.
#
# Two operating modes, selected by the CPX_MODE environment variable
# (mirroring ../rccl-tests/run.sh):
#
#   SPX mode (default, CPX_MODE unset):
#     rocSHMEM's test driver always does hipSetDevice(OMPI_COMM_WORLD_LOCAL_RANK),
#     so rank 0 -> GPU 0 and rank 1 -> GPU 1. In SPX mode each of those is a
#     full, unpartitioned GPU die -- i.e. a full MI300A package/socket -- so
#     this is genuinely INTER_SOCKET traffic (crossing packages).
#
#   CPX mode (CPX_MODE=1, requires a PPAC_MI300A_CPX allocation):
#     ROCR_VISIBLE_DEVICES=0,2 restricts rank 0 / rank 1 to two different GCD
#     (XCD) partitions on the SAME physical die -- see the naming note below.
#     This is genuinely INTER_GCD traffic (same package, different GCD).
#
# ── Naming: INTER_GCD vs INTER_SOCKET ───────────────────────────────────────
# A physical MI300A package contains 6 GCDs (XCDs), grouped in pairs under
# 3 CCDs; a node has 4 such packages. INTER_GCD crosses two GCDs WITHIN THE
# SAME package (CPX indices 0 and 2 -- different CCDs, same die, matching
# ../rccl-tests/run.sh's CPX mapping). INTER_SOCKET crosses two GCDs in
# DIFFERENT packages (SPX-mode GPU 0 vs GPU 1). These are genuinely different
# hops on the Infinity Fabric, not just different CPU pinning -- an earlier
# version of this script mistakenly treated "INTER_GCD" as just a CPU-pinning
# variant of the same GPU0<->GPU1 (cross-package) path; that was wrong. Only
# CPX mode can produce true INTER_GCD traffic with this test binary, because
# rocSHMEM's hardcoded hipSetDevice(rank) always targets whichever devices
# ROCR_VISIBLE_DEVICES makes visible as indices 0 and 1.
#
# CU_COUNT (the sweep's target and the "% of device" denominator) is queried
# at runtime via rocminfo AFTER ROCR_VISIBLE_DEVICES is set, so it naturally
# reflects whichever GPU is actually in play: ~228 CUs for a full SPX die,
# ~38 CUs for a single CPX (XCD) partition. Falls back to the MI300A defaults
# (228 / 38) if rocminfo is unavailable or unparsable.
#
# Every result row is annotated with "% of device" = num_wgs / CU_COUNT * 100
# -- the fraction of the source GPU's CUs actively issuing puts/gets for that
# row -- appended as extra columns beside the latency/bandwidth values.
#
# ── Memory-budget note ──────────────────────────────────────────────────────
# WorkGroupPrimitiveTester allocates BOTH the local and remote buffers from
# the per-PE symmetric heap, sized max_msg_size * num_wgs (batch=1) each, i.e.
# 2 * max_msg_size * num_wgs bytes of heap per PE. Sweeping all the way to
# 1 GiB (as the fixed-WG tests in run.sh do) would require hundreds of GiB of
# heap at high WG counts, which is not physically possible. Instead, this
# script uses an 8 GiB heap budget (for these tests only) and picks the
# largest power-of-two max message size that keeps
# 2 * max_msg_size * target_WGs within that budget, so the size range is
# fixed across every WG count within a sweep -- only concurrency varies.
#
# ── CPU pinning ──────────────────────────────────────────────────────────
# INTER_SOCKET (SPX) uses the same CPU cores as run.sh's INTER_SOCKET mode
# (0, 24). INTER_GCD (CPX) applies NO CPU pinning, matching
# ../rccl-tests/run.sh's CPX approach: bandwidth is determined entirely by
# ROCR_VISIBLE_DEVICES (which GCD pair is used), and skipping taskset avoids
# cpuset issues on mixed/shared CPX allocations. Do NOT restrict
# ROCR_VISIBLE_DEVICES to fewer than 2 devices -- rocSHMEM's hipSetDevice(1)
# in rank 1 requires at least 2 visible devices.
#
# Requires: build.sh completed successfully.
# Binary is installed to ~/rocshmem by default; override with ROCSHMEM_INSTALL.
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TESTS="${ROCSHMEM_INSTALL:-$HOME/rocshmem}/share/rocshmem/rocshmem_functional_tests"
RESULTS="$SCRIPT_DIR/results"

module load rocm openmpi

mkdir -p "$RESULTS"

# Heap budget for these tests only (existing run.sh tests keep the default
# 6 GiB). 8 GiB is generously safe against MI300A's per-package HBM capacity
# while allowing a useful size sweep at the target WG counts.
HEAP_BUDGET=$((8 * 1024 * 1024 * 1024))

# ── Detect the (currently visible) GPU's compute-unit (CU) count ──────────
# rocminfo prints one block per agent (CPU or GPU); "Compute Unit:" appears
# before "Device Type:" within each GPU agent's block, so we buffer the CU
# count per-agent and only emit it once we confirm the agent is a GPU. This
# respects ROCR_VISIBLE_DEVICES if already exported, so it reports ~228 for
# a full SPX die or ~38 for a single CPX (XCD) partition as appropriate.
detect_cu_count() {
    local fallback=$1 cu
    cu=$(rocminfo 2>/dev/null | awk '
        /^Agent [0-9]+/      { cu = ""; is_gpu = 0 }
        /Compute Unit:/      { cu = $NF }
        /Device Type:.*GPU/  { is_gpu = 1 }
        is_gpu && cu != ""   { print cu; exit }
    ')
    if [[ "$cu" =~ ^[0-9]+$ ]] && (( cu > 0 )); then
        echo "$cu"
    else
        echo "$fallback"
    fi
}

# ── Largest power-of-two <= n ────────────────────────────────────────────
floor_pow2() {
    local n=$1 p=1
    while (( p * 2 <= n )); do p=$(( p * 2 )); done
    echo "$p"
}

# ── WG-count sequence: powers of two up to (and including) target ─────────
wg_sequence() {
    local target=$1 p=1
    local seq=()
    while (( p <= target )); do
        seq+=("$p")
        p=$(( p * 2 ))
    done
    if (( seq[${#seq[@]}-1] != target )); then
        seq+=("$target")
    fi
    printf '%s\n' "${seq[@]}"
}

# ── Annotate each data row with the % of the device (CUs) in use ──────────
annotate_pct() {
    local wgs=$1 cu_total=$2
    awk -v wgs="$wgs" -v cu="$cu_total" '
        BEGIN { pct = sprintf("%.2f", (wgs / cu) * 100) }
        /^# Volume/ { print $0"   WGs   DeviceUtil(%)"; next }
        /^[0-9]/    { printf "%s   %s   %s\n", $0, wgs, pct; next }
        { print }
    '
}

MPIRUN_BASE=(mpirun --bind-to none -mca pml ucx -mca osc ucx
    -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384
    -x ROCSHMEM_HEAP_SIZE=${HEAP_BUDGET})

# ── Run a single (test-id, WG-count) invocation, with optional CPU pinning ─
run_one() {
    local test_id=$1 wgs=$2 max_size=$3 mode=$4 cu_total=$5 c0=$6 c1=$7 out=$8
    local label="Get"
    [ "$test_id" = 26 ] && label="Put"

    echo "--- WG ${label} Bandwidth+Latency: ${wgs} WGs (${mode}) ---"
    if [ -n "$c0" ]; then
        "${MPIRUN_BASE[@]}" -x ROCSHMEM_MAX_NUM_CONTEXTS=1 -np 2 \
            bash -c "r=\$OMPI_COMM_WORLD_LOCAL_RANK
                     cpu=\$([ \$r -eq 0 ] && echo $c0 || echo $c1)
                     if taskset -c \$cpu true 2>/dev/null; then
                         exec taskset -c \$cpu '${TESTS}' \"\$@\"
                     else
                         exec '${TESTS}' \"\$@\"
                     fi" -- \
            -a "${test_id}" -w "${wgs}" -z 64 -s "${max_size}" -batch 1 -localbuftype heap \
            | annotate_pct "${wgs}" "${cu_total}" \
            | tee -a "$out"
    else
        "${MPIRUN_BASE[@]}" -x ROCSHMEM_MAX_NUM_CONTEXTS=1 -np 2 \
            "${TESTS}" \
            -a "${test_id}" -w "${wgs}" -z 64 -s "${max_size}" -batch 1 -localbuftype heap \
            | annotate_pct "${wgs}" "${cu_total}" \
            | tee -a "$out"
    fi
    echo ""
}

# ── Run one WG-saturation sweep ────────────────────────────────────────────
# Args: mode-name, target-WG-count, max-msg-size, cu-total, cpu0 (optional), cpu1 (optional)
run_wg_sweep() {
    local mode=$1 target=$2 max_size=$3 cu_total=$4 c0=$5 c1=$6
    local put_out="${RESULTS}/put_bw_wgsat_${mode}.dat"
    local get_out="${RESULTS}/get_bw_wgsat_${mode}.dat"
    : > "$put_out"
    : > "$get_out"

    echo "======================================================"
    echo "  ${mode} WG-saturation sweep: 1 -> ${target} WGs"
    echo "  (max message size ${max_size} B, CU_COUNT=${cu_total})"
    echo "======================================================"

    for wgs in $(wg_sequence "$target"); do
        run_one 26 "$wgs" "$max_size" "$mode" "$cu_total" "$c0" "$c1" "$put_out"
        run_one 24 "$wgs" "$max_size" "$mode" "$cu_total" "$c0" "$c1" "$get_out"
    done
}

if [ -n "${CPX_MODE}" ]; then
    # ── INTER_GCD sweep (CPX mode): same package, different GCD ────────────
    # Indices 0,2 match ../rccl-tests/run.sh's CPX mapping: different CCDs,
    # same physical die. No CPU pinning (see header note).
    export ROCR_VISIBLE_DEVICES=0,2

    CU_COUNT=$(detect_cu_count 38)   # MI300A fallback: 1 XCD partition = 38 CUs
    echo "Detected CPX GCD partition CU count: ${CU_COUNT}"
    echo ""

    MAX_SIZE=$(floor_pow2 $(( HEAP_BUDGET / (2 * CU_COUNT) )))
    run_wg_sweep INTER_GCD "$CU_COUNT" "$MAX_SIZE" "$CU_COUNT"
else
    # ── INTER_SOCKET sweep (SPX mode, default): different package ──────────
    unset ROCR_VISIBLE_DEVICES

    CU_COUNT=$(detect_cu_count 228)   # MI300A fallback: full die = 6 XCDs x 38 CUs
    echo "Detected SPX GCD (full die) CU count: ${CU_COUNT}"
    echo ""

    MAX_SIZE=$(floor_pow2 $(( HEAP_BUDGET / (2 * CU_COUNT) )))
    run_wg_sweep INTER_SOCKET "$CU_COUNT" "$MAX_SIZE" "$CU_COUNT" 0 24

    echo "  NOTE: INTER_GCD WG-saturation requires a CPX-mode allocation."
    echo "        Run with CPX_MODE=1 on a PPAC_MI300A_CPX allocation. See README.md."
    echo ""
fi
