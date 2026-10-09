#!/bin/bash
# PBS job script used for the experiments; adapt the queue (-q) and resources (-l) to your cluster.
# Seasonal naive and SARIMA-RT over the validation + test spans (CPU, single process).
# Submit: qsub -v RCAF_DS=RWD1+RWD2 -N clas_v3 experiments/pbs_classical_v3.sh
#PBS -q cpuq
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=24:00:00
#PBS -j oe
cd "${PBS_O_WORKDIR:-$PWD}"   # repository root: submit the job from there
export OMP_NUM_THREADS=1 RCAF_GPUS=4
echo "ds=$RCAF_DS start=$(date)"
# processes can be killed by the host's out-of-memory daemon; every stage resumes from completed outputs
for attempt in 1 2 3 4 5 6; do
  python3 -W ignore experiments/run_classical_v3.py && break
  echo "attempt $attempt failed (rc=$?) at $(date)"; sleep 120
done
echo "end=$(date)"
