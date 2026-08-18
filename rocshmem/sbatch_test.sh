#!/bin/bash
#SBATCH --job-name=rocshmem_bench
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --nodelist=ppac-pl1-s24-30
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=01:30:00
#SBATCH --output=test_%j.log

set -eo pipefail

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SEARCH_DIR="$SUBMIT_DIR"
MB_ROOT=""

# Resolve the micro-benchmarks root by walking up from submit dir.
while [[ "$SEARCH_DIR" != "/" ]]; do
    if [[ -f "$SEARCH_DIR/set_affinity_mi300a.sh" && -d "$SEARCH_DIR/rocshmem" ]]; then
        MB_ROOT="$SEARCH_DIR"
        break
    fi
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done

# Fallback: handle submission from a parent workspace directory.
if [[ -z "$MB_ROOT" ]]; then
    for candidate in "$SUBMIT_DIR"/*/micro-benchmarks; do
        if [[ -f "$candidate/set_affinity_mi300a.sh" && -d "$candidate/rocshmem" ]]; then
            MB_ROOT="$candidate"
            break
        fi
    done
fi

if [[ -z "$MB_ROOT" ]]; then
    echo "ERROR: Could not resolve micro-benchmarks root from submit dir: $SUBMIT_DIR"
    exit 1
fi

ROCSHMEM_DIR="$MB_ROOT/rocshmem"
BINARY="${ROCSHMEM_INSTALL:-$HOME/rocshmem}/share/rocshmem/rocshmem_functional_tests"

if [[ ! -x "$BINARY" ]]; then
    echo "=== Build (first run: ~20 min) ==="
    bash "$ROCSHMEM_DIR/build.sh"
else
    echo "=== Build already done, loading modules ==="
    module load rocm openmpi
fi

echo ""
echo "=== Run ==="
bash "$ROCSHMEM_DIR/run.sh"
