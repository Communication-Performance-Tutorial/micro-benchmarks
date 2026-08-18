#!/bin/bash
#SBATCH --job-name=rccl_bench
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --nodelist=ppac-pl1-s24-30
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=00:45:00
#SBATCH --output=test_%j.log

set -eo pipefail

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null

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
BINARY="$RCCL_DIR/rocm-systems/projects/rccl-tests/build/all_reduce_perf"

if [[ ! -x "$BINARY" ]]; then
    echo "=== Build ==="
    bash "$RCCL_DIR/build.sh"
else
    echo "=== Build already done, loading modules ==="
    module load rocm openmpi
fi

echo ""
echo "=== Run ==="
bash "$RCCL_DIR/run.sh"
