#!/bin/bash
#SBATCH --job-name=rccl_bench_mod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --nodelist=ppac-pl1-s24-30
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=00:45:00
#SBATCH --output=test_system_module_%j.log

set -eo pipefail

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null

module load rocm openmpi

if ! module load rccl-tests 2>/tmp/rccl_tests_module.$$.err; then
    cat /tmp/rccl_tests_module.$$.err
    rm -f /tmp/rccl_tests_module.$$.err
    echo "ERROR: Unable to locate module 'rccl-tests'"
    echo "Try: module spider rccl-tests"
    exit 1
fi
rm -f /tmp/rccl_tests_module.$$.err

if [[ -z "${RCCL_TESTS_PATH:-}" ]]; then
    echo "ERROR: RCCL_TESTS_PATH is not set by the rccl-tests module"
    exit 1
fi

SENDRECV="${RCCL_TESTS_PATH}/bin/sendrecv_perf"
ALLREDUCE="${RCCL_TESTS_PATH}/bin/all_reduce_perf"
[[ -x "$SENDRECV" ]] || { echo "ERROR: Missing $SENDRECV"; exit 1; }
[[ -x "$ALLREDUCE" ]] || { echo "ERROR: Missing $ALLREDUCE"; exit 1; }

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

SEARCH_DIR="$SUBMIT_DIR"
MB_ROOT=""

# Resolve the micro-benchmarks root by walking up from submit dir.
while [[ "$SEARCH_DIR" != "/" ]]; do
    if [[ -f "$SEARCH_DIR/set_affinity_mi300a.sh" && -d "$SEARCH_DIR/rccl-tests" ]]; then
        MB_ROOT="$SEARCH_DIR"
        break
    fi
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done

if [[ -z "$MB_ROOT" ]]; then
    echo "ERROR: Could not resolve micro-benchmarks root from submit dir: $SUBMIT_DIR"
    exit 1
fi

RCCL_DIR="$MB_ROOT/rccl-tests"
RESULTS="$RCCL_DIR/results"
mkdir -p "$RESULTS"

NGPU=$(rocminfo | grep -c "Device Type.*GPU")
MPIRUN="mpirun -n 2 -mca pml ucx --bind-to none"

echo "======================================================"
echo "  Affinity: INTER_SOCKET - Send/Recv (1B-1G, 2 GPUs)"
echo "  Using system rccl-tests module at: $RCCL_TESTS_PATH"
echo "======================================================"

${MPIRUN} bash -c '
    bin="$1"; shift
    r=$OMPI_COMM_WORLD_LOCAL_RANK
    cpu=$([ "$r" -eq 0 ] && echo 0 || echo 24)
    if taskset -c "$cpu" true 2>/dev/null; then
        exec taskset -c "$cpu" "$bin" "$@"
    else
        exec "$bin" "$@"
    fi
' _ "$SENDRECV" -b 1 -e 1G -f 2 -g 1 \
    | tee "$RESULTS/sendrecv_INTER_SOCKET_system_module.dat"

echo ""
echo "======================================================"
echo "  All-Reduce (4M-1G, ${NGPU} GPUs)"
echo "======================================================"

"$ALLREDUCE" -b 4M -e 1G -f 2 -g "$NGPU" \
    | tee "$RESULTS/all_reduce_system_module.dat"
