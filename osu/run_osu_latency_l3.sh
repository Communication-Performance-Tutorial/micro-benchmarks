#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BIN="$SCRIPT_DIR/build/libexec/osu-micro-benchmarks/mpi/pt2pt"
OSU_LATENCY=${BIN}/osu_latency

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <num_msgs> [osu_latency args...]" >&2
    exit 1
fi

num_msgs=$1
shift 1

if ! [[ $num_msgs =~ ^[1-4]$ ]]; then
    echo "Error: num_msgs must be between 1 and 4, got '$num_msgs'" >&2
    exit 1
fi

for (( k = 0; k < num_msgs; k++ )); do
    c1=$(( k ))
    c2=$(( k + 4 ))
    mpirun -mca pml ucx --bind-to none \
        -np 1 taskset -c "$c1" "$OSU_LATENCY" "$@" : \
        -np 1 taskset -c "$c2" "$OSU_LATENCY" "$@" &	
done



