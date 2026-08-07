#!/bin/bash
#SBATCH --job-name=bench_affinity
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:4
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=02:00:00
#SBATCH --output=bench_affinity_%j.log

# Build anything not yet built, then run each suite's affinity sweep.
# This script runs the SPX-mode benchmarks (partition PPAC_MI300A_SPX).
# For RCCL CPX-mode results (INTER_GCD), submit separately:
#   sbatch rccl-tests/sbatch_cpx_sendrecv.sh
#
# Results land in:
#   osu/results/          osu_latency_{INTRA_CCD,INTER_CCD,INTER_SOCKET}_{SDMA,BLIT}.dat
#                         osu_bw_{INTRA_CCD,INTER_CCD,INTER_SOCKET}_{SDMA,BLIT}.dat
#   rccl-tests/results/   sendrecv_INTER_SOCKET.dat
#                         sendrecv_cpx_INTER_GCD.dat  ← CPX job
#                         all_reduce.dat
#   rocshmem/results/     pingpong_latency_{INTRA_CCD,INTER_CCD,INTER_SOCKET}.dat
#                         put_bw_{INTRA_CCD,INTER_CCD,INTER_SOCKET}.dat
#                         get_bw_{INTRA_CCD,INTER_CCD,INTER_SOCKET}.dat

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null

MB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================"
echo "  Node:  $(hostname)"
echo "  Date:  $(date)"
echo "  Job:   $SLURM_JOB_ID"
echo "======================================================"
echo ""

# ── OSU ───────────────────────────────────────────────────────────────────────
OSU_BIN="$MB/osu/build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency"
if [[ ! -x "$OSU_BIN" ]]; then
    echo "=== Building OSU ==="
    bash "$MB/osu/build.sh"
else
    echo "=== OSU already built ==="
    module load rocm openmpi
fi

echo ""
echo "=== Running OSU affinity sweep ==="
bash "$MB/osu/run.sh"

# ── RCCL-tests ────────────────────────────────────────────────────────────────
RCCL_BIN="$MB/rccl-tests/rocm-systems/projects/rccl-tests/build/sendrecv_perf"
echo ""
if [[ ! -x "$RCCL_BIN" ]]; then
    echo "=== Building RCCL-tests ==="
    bash "$MB/rccl-tests/build.sh"
else
    echo "=== RCCL-tests already built ==="
fi

echo ""
echo "=== Running RCCL-tests affinity sweep ==="
bash "$MB/rccl-tests/run.sh"

# ── rocSHMEM ──────────────────────────────────────────────────────────────────
# run.sh uses ${ROCSHMEM_INSTALL:-$HOME/rocshmem}; check the same install path.
# (The build-tree binary is an intermediate artifact; build.sh installs to ~/rocshmem.)
SHMEM_BIN="${ROCSHMEM_INSTALL:-$HOME/rocshmem}/share/rocshmem/rocshmem_functional_tests"
echo ""
if [[ ! -x "$SHMEM_BIN" ]]; then
    echo "=== Building rocSHMEM (~20 min) ==="
    bash "$MB/rocshmem/build.sh"
else
    echo "=== rocSHMEM already built ==="
fi

echo ""
echo "=== Running rocSHMEM affinity sweep ==="
bash "$MB/rocshmem/run.sh"

echo ""
echo "======================================================"
echo "  Done: $(date)"
echo "  Results written to:"
echo "    $MB/osu/results/"
echo "    $MB/rccl-tests/results/"
echo "    $MB/rocshmem/results/"
echo "======================================================"
