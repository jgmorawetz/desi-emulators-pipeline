#!/bin/bash
#SBATCH --account=desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --job-name=data_generation
#SBATCH --time=08:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G


module load julia

export JULIA_DEPOT_PATH=/global/homes/j/jgmorawe/.julia
export JULIA_PROJECT=/global/homes/j/jgmorawe/emulators_pipeline
export JULIA_WORKER_TIMEOUT=120
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

julia --project=$JULIA_PROJECT -e 'using Pkg; Pkg.instantiate();'

export JULIA_TOTAL_TASKS=$(
  scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | grep -m1 '^NumTasks=' | cut -d= -f2
)

srun -N1 -n1 --mpi=none --cpu-bind=none --overlap julia --project=$JULIA_PROJECT data_generation.jl
