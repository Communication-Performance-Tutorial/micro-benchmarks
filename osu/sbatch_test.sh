#!/bin/bash
#SBATCH --job-name=osu_bench
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --nodelist=ppac-pl1-s24-30
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=00:30:00
#SBATCH --output=/shared/prerelease/home/amd_int/slockhar/CommTutorial/micro-benchmarks/osu/test_%j.log

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null

SCRIPT_DIR=/shared/prerelease/home/amd_int/slockhar/CommTutorial/micro-benchmarks/osu
BINARY="$SCRIPT_DIR/build/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency"

if [[ ! -x "$BINARY" ]]; then
    echo "=== Build ==="
    bash "$SCRIPT_DIR/build.sh"
else
    echo "=== Build already done, loading modules ==="
    module load rocm openmpi
fi

echo ""
echo "=== Run ==="
bash "$SCRIPT_DIR/run.sh"
