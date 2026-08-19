#!/bin/bash
#SBATCH --job-name=osu_bench
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --nodelist=ppac-pl1-s24-30
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=00:30:00
#SBATCH --output=test_%j.log

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f "$SUBMIT_DIR/build.sh" && -f "$SUBMIT_DIR/run.sh" ]]; then
    SCRIPT_DIR="$SUBMIT_DIR"
elif [[ -f "$SUBMIT_DIR/osu/build.sh" && -f "$SUBMIT_DIR/osu/run.sh" ]]; then
    SCRIPT_DIR="$SUBMIT_DIR/osu"
else
    SCRIPT_DIR=""
    for candidate in "$SUBMIT_DIR"/*/micro-benchmarks/osu; do
        if [[ -f "$candidate/build.sh" && -f "$candidate/run.sh" ]]; then
            SCRIPT_DIR="$candidate"
            break
        fi
    done
    if [[ -z "$SCRIPT_DIR" ]]; then
        echo "ERROR: Could not resolve osu directory from submit dir: $SUBMIT_DIR"
        exit 1
    fi
fi

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
