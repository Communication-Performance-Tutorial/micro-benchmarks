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
fi

echo ""
echo "=== Run: CPX affinity sweep (INTER_XCD) ==="
CPX_MODE=1 bash "$SCRIPT_DIR/run.sh"
