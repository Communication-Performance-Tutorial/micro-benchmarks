#!/bin/bash
#SBATCH --job-name=cpx_topo_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:7
#SBATCH --exclusive
#SBATCH --nodelist=ppac-pl1-s25-40
#SBATCH --partition=PPAC_MI300A_CPX
#SBATCH --time=00:15:00
#SBATCH --output=/shared/prerelease/home/amd_int/slockhar/CommTutorial/micro-benchmarks/rccl-tests/cpx_topo_%j.log

# Topology probe for ppac-pl1-s25-40 (CPX mode, 24 GPUs / 192 CPUs).
# Goal: determine the correct CPU core → die/CCD mapping so that
# run.sh CPX_CPU0/CPX_CPU1 assignments can be updated.

echo "================================================================"
echo "  Node: $(hostname)   Job: $SLURM_JOB_ID"
echo "  Allocated CPUs : $SLURM_JOB_CPUS_PER_NODE"
echo "  Allocated GPUs : $SLURM_GPUS_ON_NODE"
echo "================================================================"
echo ""

echo "----------------------------------------------------------------"
echo "  SLURM cpuset visible to this job"
echo "----------------------------------------------------------------"
cat /proc/self/status | grep -E "^(Cpus_allowed|Mems_allowed)"
echo ""

echo "----------------------------------------------------------------"
echo "  lscpu"
echo "----------------------------------------------------------------"
lscpu
echo ""

echo "----------------------------------------------------------------"
echo "  numactl --hardware"
echo "----------------------------------------------------------------"
numactl --hardware
echo ""

echo "----------------------------------------------------------------"
echo "  CPU → NUMA / core topology (lscpu --extended)"
echo "----------------------------------------------------------------"
lscpu --extended=CPU,SOCKET,NODE,CORE
echo ""

echo "----------------------------------------------------------------"
echo "  rocminfo (GPU agent summary)"
echo "----------------------------------------------------------------"
source /etc/profile 2>/dev/null
source ~/.bashrc 2>/dev/null
module load rocm 2>/dev/null || true
rocminfo | grep -A 4 "Agent [0-9]"
echo ""

echo "================================================================"
echo "  Done.  Use the NUMA / core layout above to set CPX_CPU0/CPX_CPU1"
echo "  in rccl-tests/run.sh."
echo "================================================================"
