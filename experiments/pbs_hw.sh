#!/bin/bash
# PBS job script used for the experiments; adapt the queue (-q) and resources (-l) to your cluster.
# Post-hoc highway ablation (run_ablation_hw.py) for the datasets in RCAF_DS on one GPU (MIG) slice.
# Submit: qsub -v RCAF_DS=DE+FR -N hw_DE_FR experiments/pbs_hw.sh
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe
cd "${PBS_O_WORKDIR:-$PWD}"   # repository root: submit the job from there
# activate your Python environment here if needed, e.g.: source venv/bin/activate
export RCAF_GPUS=${RCAF_GPUS:-auto} RCAF_WPG=${RCAF_WPG:-6} OMP_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES
echo "host=$(hostname) ds=$RCAF_DS gpu=$RCAF_GPUS start=$(date)"
# processes can be killed by the host's out-of-memory daemon; every stage resumes from completed outputs
for attempt in 1 2 3 4 5 6; do
  python3 -W ignore experiments/run_ablation_hw.py all && break
  echo "attempt $attempt failed (rc=$?) at $(date)"; sleep 120
done
echo "end=$(date)"
