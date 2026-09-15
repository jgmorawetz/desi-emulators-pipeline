using Pkg
Pkg.activate(".")
using DataFrames
using Effort
using ArgParse
using DelimitedFiles
using Mapse
using Random
using LinearAlgebra
using Statistics
include("training_engine.jl")



# Argument parsing 
config = ArgParseSettings()
@add_arg_table config begin
    "--multipole"
    help = "Specify the multipole to be trained. One of 0, 2, 4."
    arg_type = String
    required = true
    "--component"
    help = "Specify the component to be trained. One of '11', 'loop', 'ct'."
    arg_type = String
    required = true
    "--dataset_directory"
    help = "Specify the folder path where the data samples are stored."
    arg_type = String
    required = true
    "--save_directory"
    help = "Specify the directory where the emulator results will be stored (must be created in advance along with subfolders)."
    arg_type = String
    required = true
    "--network_architecture_path"
    help = "Specify the file path where the neural network architecture is specified."
    arg_type = String
    required = true
    "--batchsize"
    help = "Specify the batch size for computing the gradients."
    arg_type = Int 
    required = true
    "--n_epoch"
    help = "Specify the number of epochs per run (for each run results are updated in file)."
    arg_type = Int 
    required = true
    "--n_run"
    help = "Specify the number of runs for each learning rate."
    arg_type = Int 
    required = true 
    "--learning_rates"
    help = "Specify (comma separated) the different learning rates to iterate through."
    arg_type = String
    required = true
end
parsed_args = parse_args(config)
multipole = parsed_args["multipole"]
component = parsed_args["component"]
dataset_directory = parsed_args["dataset_directory"]
save_directory = parsed_args["save_directory"]
network_architecture_path = parsed_args["network_architecture_path"]
batchsize = parsed_args["batchsize"]
n_epoch = parsed_args["n_epoch"]
n_run = parsed_args["n_run"]
learning_rates = parsed_args["learning_rates"]
learning_rates = parse.(Float64, split(learning_rates, ","))



# Defines the necessary preprocessing functions (undoes primordial amplitude and growth factor)
function D_ODE(z, H0, ombh2, omch2, Mnu, w0, wa)
    cosmology = Effort.w0waCDMCosmology(ln10Aₛ=3.044, nₛ=0.9649, h=H0/100, ωb=ombh2, ωc=omch2, mν=Mnu, w0=w0, wa=wa)
    return Effort.D_z(z, cosmology)
end

preprocess(z, As, H0, ombh2, omch2, Mnu, w0, wa) = As * D_ODE(z, H0, ombh2, omch2, Mnu, w0, wa)^2


# Reads the contents of the data samples (and extracts number of samples and input/output dimensions for each)
subdirs = readdir(dataset_directory)
n_samples = length(subdirs)
example_param_dict = JSON.parsefile(joinpath(dataset_directory, subdirs[1], "effort_dict.json"))
n_input = length(example_param_dict)
if haskey(example_param_dict, "w0") && haskey(example_param_dict, "wa") && ! haskey(example_param_dict, "Mnu")
    model = "w0waCDM"
elseif haskey(example_param_dict, "w0") && haskey(example_param_dict, "wa") && haskey(example_param_dict, "Mnu")
    model = "Mnuw0waCDM"
end
nk = length(npzread(joinpath(dataset_directory, subdirs[1], "kv.npy")))
nk_factor = component == "11" ? 3 : component == "loop" ? 9 : component == "ct" ? 4 : error("Wrong component!")
n_output = nk * nk_factor

# Constructs matrices to store the input and output features
input_params = Matrix{Float64}(undef, n_samples, n_input)
targets = Matrix{Float64}(undef, n_samples, n_output)


# Goes through the data samples and adds input/target features and then at end constructs dataframe
for (idx, sub) in enumerate(subdirs)
    if idx % 50000 == 0
        println("Loaded up to $(idx) samples.")
    end
    sub_path = joinpath(dataset_directory, sub)
    param_dict = JSON.parsefile(joinpath(sub_path, "effort_dict.json"))
    # Determines whether w0waCDM or Mnuw0waCDM model
    if model == "w0waCDM"
        z, ln10As, ns, H0, ombh2, omch2, w0, wa = (param_dict[param] for param in ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "w0", "wa"])
        Mnu = 0.06
        input_params[idx, :] = [z, ln10As, ns, H0, ombh2, omch2, w0, wa]
    elseif model == "Mnuw0waCDM"
        z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa = (param_dict[param] for param in ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mnu", "w0", "wa"])
        input_params[idx, :] = [z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa]
    end
    factor = preprocess(z, exp(ln10As)*1e-10, H0, ombh2, omch2, Mnu, w0, wa)
    # Reshapes the velocileptors table into the relevant emulator components and applies necessary rescaling
    target_file = npzread(joinpath(sub_path, "pk_$(multipole).npy"))
    if component == "11"
        targets[idx, :] = vec(Array(target_file)[:, 1:3]) ./ factor
    elseif component == "loop"
        targets[idx, :] = vec(Array(target_file)[:, 4:12]) ./ factor^2
    elseif component == "ct"
        targets[idx, :] = vec(Array(target_file)[:, 13:16]) ./ factor 
    end
end
if model == "w0waCDM"
    df = DataFrame(z=input_params[:, 1], ln10As=input_params[:, 2], ns=input_params[:, 3], 
                   H0=input_params[:, 4], ombh2=input_params[:, 5], omch2=input_params[:, 6],
                   w0=input_params[:, 7], wa=input_params[:, 8],
                   observable=[targets[i, :] for i in 1:n_samples])
elseif model == "Mnuw0waCDM"
    df = DataFrame(z=input_params[:, 1], ln10As=input_params[:, 2], ns=input_params[:, 3], 
                   H0=input_params[:, 4], ombh2=input_params[:, 5], omch2=input_params[:, 6],
                   Mnu=input_params[:, 7], w0=input_params[:, 8], wa=input_params[:, 9],
                   observable=[targets[i, :] for i in 1:n_samples])
end

# Now runs the training itself
run_training(df, save_directory, network_architecture_path, learning_rates, n_run, n_epoch, batchsize)