#!/bin/bash
#SBATCH --account=desi
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --job-name=training_capse_camb
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3

sleep $((SLURM_ARRAY_TASK_ID * 30))

module load julia

PARAMS=(
    "TT"
    "TE"
    "EE"
    "PP"
)

PARAM_SET=(${PARAMS[$SLURM_ARRAY_TASK_ID]})
spectrum="${PARAM_SET}"

home_dir="/global/homes/j/jgmorawe/desi-emulators-pipeline"
scratch_dir="/pscratch/sd/j/jgmorawe"
dataset_directory="${scratch_dir}/capse_camb_mnuw0wacdm_100000"
save_directory="${home_dir}/trained_capse_camb_mnuw0wacdm_100000/${spectrum}"
network_architecture_path="${home_dir}/nn_setup_64x5.json"
lmax=9500
n_nodes=512
n_epoch=2000
n_run=30
batchsize=1024
learning_rates="1e-4,3e-5,1e-5,3e-6,1e-6,3e-7"


# PP is only rescaled by As (no tau factor), so it has its own postprocessing
postprocessing="postprocessing"
if [ "$spectrum" == "PP" ]; then
    postprocessing="postprocessing_PP"
fi
cp "${home_dir}/class_velocileptors/supporting_files/Capse/${postprocessing}.py" "${save_directory}/postprocessing.py"
cp "${home_dir}/class_velocileptors/supporting_files/Capse/${postprocessing}.jl" "${save_directory}/postprocessing.jl"


julia -t $SLURM_CPUS_PER_TASK camb/training_capse.jl --spectrum="$spectrum" --n_nodes=$n_nodes --lmax=$lmax --dataset_directory="$dataset_directory" --save_directory="$save_directory" --network_architecture_path="$network_architecture_path"  --batchsize=$batchsize --n_epoch=$n_epoch --n_run=$n_run --learning_rates="$learning_rates"