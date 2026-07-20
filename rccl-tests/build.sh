#!/bin/bash
# Build RCCL tests with MPI support.
# Uses a sparse checkout of ROCm/rocm-systems to fetch only projects/rccl-tests.
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO="$SCRIPT_DIR/rocm-systems"

module load rocm openmpi

if [[ ! -d "$REPO" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/ROCm/rocm-systems.git "$REPO"
    cd "$REPO"
    git sparse-checkout init --cone
    git sparse-checkout set projects/rccl-tests
    git checkout develop
fi

GPU_ARCH=$(rocminfo | grep -m1 -oE 'gfx[0-9a-f]+')
echo "Building for GPU arch: $GPU_ARCH"

# NCCL_CTA_POLICY_ZERO was added after ROCm 7.2.4; patch common.cu if needed
COMMON_CU="$REPO/projects/rccl-tests/src/common.cu"
if ! grep -q "NCCL_CTA_POLICY_ZERO" "$ROCM_PATH/include/rccl/rccl.h" 2>/dev/null && \
   ! grep -q "compat-patch" "$COMMON_CU"; then
    sed -i '1s|^|/* compat-patch: NCCL_CTA_POLICY_ZERO missing in this RCCL */\n#ifndef NCCL_CTA_POLICY_ZERO\n#define NCCL_CTA_POLICY_ZERO 2\n#endif\n|' "$COMMON_CU"
fi

cd "$REPO/projects/rccl-tests"
make clean
GPU_TARGETS="$GPU_ARCH" make MPI=1 MPI_HOME="$MPI_PATH" HIP_HOME="$ROCM_PATH"
