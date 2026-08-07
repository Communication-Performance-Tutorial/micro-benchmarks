#!/bin/bash
#SBATCH --job-name=rccl_cpx_sendrecv
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:7
#SBATCH --nodelist=ppac-pl1-s25-40
#SBATCH --partition=PPAC_MI300A_CPX
#SBATCH --time=00:30:00
#SBATCH --output=cpx_sendrecv_%j.log

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null
module load rocm openmpi 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/rocm-systems/projects/rccl-tests/build/sendrecv_perf"

if [[ ! -x "$BINARY" ]]; then
    echo "=== Build ==="
    bash "$SCRIPT_DIR/build.sh"
fi

echo "=== CPX sendrecv: INTER_XCD (INTER_SOCKET already covered by SPX) ==="
CPX_MODE=1 bash "$SCRIPT_DIR/run.sh"
