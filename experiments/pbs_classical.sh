#!/bin/bash
# PBS job script used for the experiments; adapt the queue (-q) and resources (-l) to your cluster.
# Classical baselines (seasonal naive, SARIMA-RT, XGBoost) on CPU.
# Submit: qsub experiments/pbs_classical.sh
#PBS -N rcaf_classical
#PBS -q cpuq
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=24:00:00
#PBS -j oe
cd "${PBS_O_WORKDIR:-$PWD}"   # repository root: submit the job from there
export OMP_NUM_THREADS=2 NCPUS=2
echo "start=$(date)"
python3 -W ignore experiments/run_classical.py
echo "end=$(date)"
