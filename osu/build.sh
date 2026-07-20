#!/bin/bash
# Build OSU Micro-Benchmarks with ROCm / GPU-aware MPI support.
# OMB has no official git repo (MVAPICH only distributes tarballs), so the
# source is fetched directly from mvapich.cse.ohio-state.edu on first build.
# This keeps the script self-contained and independent of any other
# directory on the system.
set -e

OSU_VERSION="7.5"
OSU_URL="https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tar.gz"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SRC="$SCRIPT_DIR/src"

module load rocm openmpi

if [[ ! -d "$SRC" ]]; then
    TARBALL="$SCRIPT_DIR/osu-micro-benchmarks-${OSU_VERSION}.tar.gz"
    curl -fsSL "$OSU_URL" -o "$TARBALL"
    tar -xzf "$TARBALL" -C "$SCRIPT_DIR"
    mv "$SCRIPT_DIR/osu-micro-benchmarks-${OSU_VERSION}" "$SRC"
    rm -f "$TARBALL"
fi

cd "$SRC"
# hipDeviceReset() after MPI_Finalize() triggers a teardown race on ROCm;
# hipDeviceSynchronize() is the safe replacement.
sed -i 's/hipDeviceReset/hipDeviceSynchronize/g' c/util/osu_util_mpi.c

./configure --prefix="$SCRIPT_DIR/build" \
    CC=$(which mpicc) CXX=$(which mpicxx) \
    CPPFLAGS=-D__HIP_PLATFORM_AMD__=1 \
    --enable-rocm --with-rocm="$ROCM_PATH"

make -j$(nproc) && make install
