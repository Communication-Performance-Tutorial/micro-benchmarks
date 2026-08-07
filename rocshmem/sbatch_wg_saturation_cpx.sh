#!/bin/bash
#SBATCH --job-name=rocshmem_wgsat_cpx
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:3
#SBATCH --nodelist=ppac-pl1-s25-40
#SBATCH --partition=PPAC_MI300A_CPX
#SBATCH --time=01:00:00
#SBATCH --output=wgsat_cpx_%j.log

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null
module load rocm openmpi 2>/dev/null || true

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
fi

echo "=== Run: WG-count saturation sweep (INTER_XCD, CPX mode) ==="
CPX_MODE=1 bash "$SCRIPT_DIR/run_wg_saturation.sh"
