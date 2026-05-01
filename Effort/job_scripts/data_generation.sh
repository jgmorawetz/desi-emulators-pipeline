#!/bin/bash
#SBATCH --account=desi
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --job-name=data_gen
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=300G


module load julia

export JULIA_DEPOT_PATH=/global/homes/j/jgmorawe/.julia
export JULIA_PROJECT=/global/homes/j/jgmorawe/desi-emulators-pipeline
export JULIA_WORKER_TIMEOUT=600
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

julia --project=$JULIA_PROJECT -e 'using Pkg; Pkg.instantiate();'

export JULIA_TOTAL_TASKS=$(
  scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | grep -m1 '^NumTasks=' | cut -d= -f2
)

srun -N1 -n1 --mpi=none --cpu-bind=none --overlap julia --project=$JULIA_PROJECT Effort/job_scripts/data_generation.jl
