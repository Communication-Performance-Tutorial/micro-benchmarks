#!/bin/bash
# Build rocSHMEM with the IPC (intra-node) backend.
#
# Uses ipc_single build config: USE_RO=OFF, USE_GDA=OFF, USE_IPC=ON.
# IPC transfers directly between GPU memories on the same node via ROCm IPC
# handles, giving ~1 us latency vs ~12 us for the RO (MPI-routed) backend.
#
# Source is fetched via a sparse checkout of ROCm/rocm-systems so only the
# rocshmem subdirectory is downloaded.
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO="$SCRIPT_DIR/rocm-systems"

module load rocm openmpi

if [[ ! -d "$REPO" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/ROCm/rocm-systems.git "$REPO"
    cd "$REPO"
    git sparse-checkout init --cone
    git sparse-checkout set projects/rocshmem
    git checkout develop
fi

mkdir -p "$REPO/projects/rocshmem/build"
cd "$REPO/projects/rocshmem/build"
# Pass -DBUILD_PYTHON_TESTS=OFF to override the default ON in ipc_single —
# the Python test harness requires scikit-build which may not be installed.
# The install prefix defaults to ~/rocshmem; override with INSTALL_PREFIX env var.
../scripts/build_configs/ipc_single -DBUILD_PYTHON_TESTS=OFF
