#!/bin/bash
# PBS job script used for the experiments; adapt the queue (-q) and resources (-l) to your cluster.
# Synthetic change point detection benchmark (CPU only).
# Submit: qsub experiments/pbs_cpd.sh
#PBS -N rcaf_cpd
#PBS -q cpuq
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=24:00:00
#PBS -j oe
cd "${PBS_O_WORKDIR:-$PWD}"   # repository root: submit the job from there
export OMP_NUM_THREADS=1
echo "host=$(hostname) ncpus=$NCPUS start=$(date)"
/usr/bin/time -v python3 -W ignore experiments/run_cpd.py
echo "end=$(date)"
