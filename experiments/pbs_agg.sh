#!/bin/bash
# PBS job script used for the experiments; adapt the queue (-q) and resources (-l) to your cluster.
# Online aggregation + conformal intervals (CPU, single process) for the datasets in AGG_DS (space-separated).
# Submit: qsub -v AGG_DS="RWD1 RWD2" -N agg_dev experiments/pbs_agg.sh
#PBS -q cpuq
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=24:00:00
#PBS -j oe
cd "${PBS_O_WORKDIR:-$PWD}"   # repository root: submit the job from there
export OMP_NUM_THREADS=2
echo "ds=$AGG_DS start=$(date)"
# processes can be killed by the host's out-of-memory daemon; every stage resumes from completed outputs
for attempt in 1 2 3 4 5 6; do
  python3 -W ignore experiments/aggregate.py $AGG_DS && break
  echo "attempt $attempt failed (rc=$?) at $(date)"; sleep 120
done
echo "end=$(date)"
