#!/bin/bash
#SBATCH --job-name=rccl_cpx_sendrecv
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:7
#SBATCH --nodelist=ppac-pl1-s25-40
#SBATCH --partition=PPAC_MI300A_CPX
#SBATCH --time=00:30:00
#SBATCH --output=cpx_sendrecv_%j.log

set -eo pipefail

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null
module load rocm openmpi 2>/dev/null || true

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
BINARY="$RCCL_DIR/rocm-systems/projects/rccl-tests/build/sendrecv_perf"

if [[ ! -x "$BINARY" ]]; then
    echo "=== Build ==="
    bash "$RCCL_DIR/build.sh"
fi

echo "=== CPX sendrecv: INTER_XCD (INTER_SOCKET already covered by SPX) ==="
CPX_MODE=1 bash "$RCCL_DIR/run.sh"
