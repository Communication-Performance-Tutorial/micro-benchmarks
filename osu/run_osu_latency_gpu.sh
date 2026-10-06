#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BIN="$SCRIPT_DIR/build/libexec/osu-micro-benchmarks/mpi/pt2pt"
OSU_LATENCY=${BIN}/osu_latency

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <core1> <core2> [osu_latency args...]" >&2
    exit 1
fi

core1=$1
core2=$2
shift 2

CORES_PER_GPU=24
gpu1=${GPU1:-$(( core1 / CORES_PER_GPU ))}
gpu2=${GPU2:-$(( core2 / CORES_PER_GPU ))}


for c in "$core1" "$core2"; do
    if ! [[ $c =~ ^[0-9]+$ ]]; then
        echo "Error: '$c' is not a valid core id" >&2
        exit 1
    fi
    if ! taskset -c "$c" true 2>/dev/null; then
        echo "Error: cannot bind to core $c (not in your allocation?)" >&2
        exit 1
    fi
done

echo "# osu_latency: rank 0 -> core $core1 and gpu $gpu1, rank 1 -> core $core2 and gpu $gpu2"

mpirun -mca pml ucx --bind-to none \
    -np 1 env ROCR_VISIBLE_DEVICES="$gpu1" taskset -c "$core1" "$OSU_LATENCY" "$@" D D : \
    -np 1 env ROCR_VISIBLE_DEVICES="$gpu2" taskset -c "$core2" "$OSU_LATENCY" "$@" D D

