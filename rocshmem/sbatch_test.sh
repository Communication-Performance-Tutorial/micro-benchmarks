#!/bin/bash
#SBATCH --job-name=rocshmem_bench
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --nodelist=ppac-pl1-s24-30
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=01:30:00
#SBATCH --output=/shared/prerelease/home/amd_int/slockhar/CommTutorial/micro-benchmarks/rocshmem/test_%j.log

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null

SCRIPT_DIR=/shared/prerelease/home/amd_int/slockhar/CommTutorial/micro-benchmarks/rocshmem
BINARY="${ROCSHMEM_INSTALL:-$HOME/rocshmem}/share/rocshmem/rocshmem_functional_tests"

if [[ ! -x "$BINARY" ]]; then
    echo "=== Build (first run: ~20 min) ==="
    bash "$SCRIPT_DIR/build.sh"
else
    echo "=== Build already done, loading modules ==="
    module load rocm openmpi
fi

echo ""
echo "=== Run ==="
bash "$SCRIPT_DIR/run.sh"
