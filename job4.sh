#!/bin/bash
#SBATCH --job-name=sp_750
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH -o sp_750.out
#SBATCH -e sp_750.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node8
#SBATCH --partition=debug

echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load openmpi/4.1
module load gcc/9.3
module load cuda/11
ulimit -s unlimited
python3 model_sai_optim_s_p.py  n