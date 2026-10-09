#!/bin/bash
# PBS job script used for the experiments; adapt the queue (-q) and resources (-l) to your cluster.
# Semi-synthetic CPD benchmark for one change type (KIND = L, A or LA). CPU only, single process.
# Submit: qsub -v KIND=L -N semi_L experiments/pbs_semi.sh
#PBS -q cpuq
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=24:00:00
#PBS -j oe
cd "${PBS_O_WORKDIR:-$PWD}"   # repository root: submit the job from there
export OMP_NUM_THREADS=1
echo "kind=$KIND start=$(date)"
python3 -W ignore experiments/cpd_semisynth.py $KIND
echo "end=$(date)"
