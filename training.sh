#!/bin/bash
#SBATCH --account=desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --job-name=training
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --array=0-8

sleep $((SLURM_ARRAY_TASK_ID * 30))

module load julia

PARAMS=(
    "11 0"
    "11 2"
    "11 4"
    "loop 0"
    "loop 2" 
    "loop 4"
    "ct 0"
    "ct 2"
    "ct 4"
)

PARAM_SET=(${PARAMS[$SLURM_ARRAY_TASK_ID]})
component="${PARAM_SET[0]}"
multipole="${PARAM_SET[1]}"

home_dir="/global/homes/j/jgmorawe/emulators_pipeline"
path_input="${home_dir}/effort_velocileptors_rept_mnuw0wacdm_50000"
path_output="${home_dir}/trained_effort_velocileptors_rept_mnuw0wacdm_50000"

julia "${home_dir}/training.jl" --component="$component" --multipole="$multipole" --path_input="$path_input" --path_output="$path_output"