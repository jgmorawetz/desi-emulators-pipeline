#!/bin/bash
#SBATCH --account=desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --job-name=training_Mapse
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --array=0-1

sleep $((SLURM_ARRAY_TASK_ID * 30))

module load julia

PARAMS=(
    "cb"
    "mm"
)

spectrum="${PARAMS[$SLURM_ARRAY_TASK_ID]}"

home_dir="/global/homes/j/jgmorawe/desi-emulators-pipeline"
scratch_dir="/pscratch/sd/j/jgmorawe"
path_input="${scratch_dir}/mapse_class_mnuw0wacdm_100000"
path_output="${home_dir}/trained_mapse_class_mnuw0wacdm_100000"
nn_setup_path="${home_dir}/Mapse/supporting_files/nn_setup.json"
n_epoch=2000
n_run=20
batchsize=512
var_ratio=0.999999

julia -t $SLURM_CPUS_PER_TASK  "${home_dir}/Mapse/codes/training.jl" --spectrum="$spectrum" --path_input="$path_input" --path_output="$path_output" --var_ratio=$var_ratio --nn_setup_path="$nn_setup_path" --n_epoch=$n_epoch --n_run=$n_run --batchsize=$batchsize