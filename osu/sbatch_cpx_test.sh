#!/bin/bash
#SBATCH --job-name=osu_cpx_bench
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:3
#SBATCH --nodelist=ppac-pl1-s25-40
#SBATCH --partition=PPAC_MI300A_CPX
#SBATCH --time=00:30:00
#SBATCH --output=cpx_test_%j.log

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null
module load rocm openmpi 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency"

if [[ ! -x "$BINARY" ]]; then
    echo "=== Build ==="
    bash "$SCRIPT_DIR/build.sh"
fi

echo ""
echo "=== Run: CPX affinity sweep (INTER_XCD) ==="
CPX_MODE=1 bash "$SCRIPT_DIR/run.sh"
