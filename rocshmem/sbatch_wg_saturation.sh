#!/bin/bash
#SBATCH --job-name=rocshmem_wgsat
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --nodelist=ppac-pl1-s24-30
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=02:00:00
#SBATCH --output=wgsat_%j.log

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# sbatch copies the submitted script to a spool directory before executing it,
# so BASH_SOURCE[0] resolves there rather than to the real script location.
# Fall back to SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) so
# the sibling scripts (run_wg_saturation.sh, build.sh) can still be found.
if [[ ! -f "$SCRIPT_DIR/run_wg_saturation.sh" && -n "$SLURM_SUBMIT_DIR" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
fi
BINARY="${ROCSHMEM_INSTALL:-$HOME/rocshmem}/share/rocshmem/rocshmem_functional_tests"

if [[ ! -x "$BINARY" ]]; then
    echo "=== Build (first run: ~20 min) ==="
    bash "$SCRIPT_DIR/build.sh"
else
    echo "=== Build already done, loading modules ==="
    module load rocm openmpi
fi

echo ""
echo "=== Run: WG-count saturation sweep (INTER_SOCKET, SPX mode) ==="
echo "    For INTER_XCD (same-package), submit sbatch_wg_saturation_cpx.sh instead."
bash "$SCRIPT_DIR/run_wg_saturation.sh"
