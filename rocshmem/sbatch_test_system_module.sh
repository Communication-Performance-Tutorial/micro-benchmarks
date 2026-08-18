#!/bin/bash
#SBATCH --job-name=rocshmem_mod
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --nodelist=ppac-pl1-s24-30
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=01:30:00
#SBATCH --output=test_system_module_%j.log

set -eo pipefail

source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null

module load rocm openmpi

if ! module load rocshmem 2>/tmp/rocshmem_module.$$.err; then
    cat /tmp/rocshmem_module.$$.err
    rm -f /tmp/rocshmem_module.$$.err
    echo "ERROR: Unable to locate module 'rocshmem'"
    echo "Try: module spider rocshmem"
    exit 1
fi
rm -f /tmp/rocshmem_module.$$.err

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

# Prefer the module's canonical install layout, but fall back to PATH/bin.
if [[ -n "${ROCSHMEM_PATH:-}" && -x "$ROCSHMEM_PATH/share/rocshmem/rocshmem_functional_tests" ]]; then
    TESTS_BIN="$ROCSHMEM_PATH/share/rocshmem/rocshmem_functional_tests"
elif [[ -n "${ROCSHMEM_PATH:-}" && -x "$ROCSHMEM_PATH/bin/rocshmem_functional_tests" ]]; then
    TESTS_BIN="$ROCSHMEM_PATH/bin/rocshmem_functional_tests"
else
    TESTS_BIN="$(command -v rocshmem_functional_tests || true)"
fi

if [[ -z "${TESTS_BIN:-}" || ! -x "$TESTS_BIN" ]]; then
    echo "ERROR: rocshmem_functional_tests not found in the loaded rocshmem module."
    echo "Checked ROCSHMEM_PATH and PATH after 'module load rocshmem'."
    exit 1
fi

echo "Using rocSHMEM module path: ${ROCSHMEM_PATH:-unknown}"
echo "Using test binary: $TESTS_BIN"

# run.sh expects ${ROCSHMEM_INSTALL}/share/rocshmem/rocshmem_functional_tests.
# Create a local shim install prefix that points to the module-provided binary.
SHIM_INSTALL="$ROCSHMEM_DIR/.rocshmem_module_install"
mkdir -p "$SHIM_INSTALL/share/rocshmem"
ln -sf "$TESTS_BIN" "$SHIM_INSTALL/share/rocshmem/rocshmem_functional_tests"

echo ""
echo "=== Run (system rocshmem module) ==="
ROCSHMEM_INSTALL="$SHIM_INSTALL" bash "$ROCSHMEM_DIR/run.sh"
