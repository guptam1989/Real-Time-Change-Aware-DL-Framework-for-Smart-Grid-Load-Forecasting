#!/bin/bash
# PBS job script used for the experiments; adapt the queue (-q) and resources (-l) to your cluster.
# Forecasting pipeline for one dataset on one GPU.
# Submit: qsub -v RCAF_DS=RWD1 -N rcaf_RWD1 experiments/pbs_forecast.sh
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=48:00:00
#PBS -j oe
cd "${PBS_O_WORKDIR:-$PWD}"   # repository root: submit the job from there
# activate your Python environment here if needed, e.g.: source venv/bin/activate
export RCAF_GPUS=${RCAF_GPUS:-auto} RCAF_WPG=${RCAF_WPG:-5} OMP_NUM_THREADS=1
# A MIG device UUID can be passed in RCAF_GPUS (qsub -v RCAF_GPUS=MIG-...)
unset CUDA_VISIBLE_DEVICES
echo "host=$(hostname) pbs_gpu=$CUDA_VISIBLE_DEVICES ds=$RCAF_DS start=$(date)"
python3 -W ignore experiments/run_forecast.py ${STAGE:-all}
echo "end=$(date)"
