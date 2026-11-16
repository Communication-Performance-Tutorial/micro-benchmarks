#!/bin/bash
# Build RCCL tests with MPI support.
# Uses a sparse checkout of ROCm/rocm-systems to fetch only projects/rccl-tests.
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO="$SCRIPT_DIR/rocm-systems"

# PrgEnv-amd-openmpi brings in rocm/7.14.0 and the matching GPU-aware OpenMPI.
# It conflicts with the login default PrgEnv-cray, so it has to be swapped in;
# a plain 'module load rocm openmpi' picks up rocm-afar and leaves MPI_PATH unset.
PRGENV=PrgEnv-amd-openmpi/openmpi-5.0.10-ofi-7.14.0
# Compute-node login shells don't add this to MODULEPATH; only uan1 does.
if [[ ":$MODULEPATH:" != *":/shareddata/modules/base:"* ]]; then
    module use --append /shareddata/modules/base
fi
if [[ ":$LOADEDMODULES:" != *":$PRGENV:"* ]]; then
    if [[ ":$LOADEDMODULES:" == *":PrgEnv-cray/"* ]]; then
        module swap PrgEnv-cray "$PRGENV"
    else
        module load "$PRGENV"
    fi
fi
#module load craype-accel-amd-gfx942

if [[ ! -f "$MPI_PATH/include/mpi.h" ]]; then
    echo "error: mpi.h not found under MPI_PATH='$MPI_PATH' after loading $PRGENV" >&2
    module list
    exit 1
fi
echo "Using ROCM_PATH=$ROCM_PATH MPI_PATH=$MPI_PATH"

if [[ ! -d "$REPO" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/ROCm/rocm-systems.git "$REPO"
    cd "$REPO"
    git sparse-checkout init --cone
    git sparse-checkout set projects/rccl-tests
fi

COMMON_CU="$REPO/projects/rccl-tests/src/common.cu"
COMMON_H="$REPO/projects/rccl-tests/src/common.h"

cd "$REPO"
## Drop compat patches from a previous run so they don't block the checkout.
git checkout therock-7.14
#git checkout -- "$COMMON_CU" "$COMMON_H" 2>/dev/null || true
#if ! git cat-file -e "$RCCL_TESTS_REF^{commit}" 2>/dev/null; then
#    git fetch origin
#fi
#git checkout --detach "$RCCL_TESTS_REF"

GPU_ARCH=$(rocminfo | grep -m1 -oE 'gfx[0-9a-f]+')
echo "Building for GPU arch: $GPU_ARCH"

# NCCL_CTA_POLICY_ZERO was added after ROCm 7.2.4; patch common.cu if needed
if ! grep -q "NCCL_CTA_POLICY_ZERO" "$ROCM_PATH/include/rccl/rccl.h" 2>/dev/null && \
   ! grep -q "compat-patch" "$COMMON_CU"; then
    sed -i '1s|^|/* compat-patch: NCCL_CTA_POLICY_ZERO missing in this RCCL */\n#ifndef NCCL_CTA_POLICY_ZERO\n#define NCCL_CTA_POLICY_ZERO 2\n#endif\n|' "$COMMON_CU"
fi

# The rocSHMEM GDA / Anvil SDMA GIN types are missing from RCCL builds without
# ENABLE_ROCSHMEM_GIN; common.h only prints their values in an error message.
# Values match ncclGinType_t in upstream projects/rccl/src/include/nccl_device/core.h.
if ! grep -rq "NCCL_GIN_TYPE_ROCSHMEM_GDA" "$ROCM_PATH/include/nccl_device" 2>/dev/null && \
   ! grep -q "compat-patch" "$COMMON_H"; then
    sed -i '1s|^|/* compat-patch: rocSHMEM GIN types missing in this RCCL */\n#ifndef NCCL_GIN_TYPE_ROCSHMEM_GDA\n#define NCCL_GIN_TYPE_ROCSHMEM_GDA 6\n#endif\n#ifndef NCCL_GIN_TYPE_ANVIL_SDMA\n#define NCCL_GIN_TYPE_ANVIL_SDMA 7\n#endif\n|' "$COMMON_H"
fi

cd "$REPO/projects/rccl-tests"
make clean
GPU_TARGETS="$GPU_ARCH" make MPI=1 MPI_HOME="$MPI_PATH" HIP_HOME="$ROCM_PATH"
