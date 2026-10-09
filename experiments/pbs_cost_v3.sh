#!/bin/bash
# PBS job script used for the experiments; adapt the queue (-q) and resources (-l) to your cluster.
# Single-thread CPU cost of the v3 components.
#PBS -q cpuq
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=04:00:00
#PBS -j oe
cd "${PBS_O_WORKDIR:-$PWD}"   # repository root: submit the job from there
export OMP_NUM_THREADS=1 RCAF_GPUS=4 CUDA_VISIBLE_DEVICES=
for attempt in 1 2 3; do
  python3 -W ignore experiments/measure_cost_v3.py RWD2 && break
  sleep 60
done
