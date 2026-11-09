using Pkg
Pkg.activate(".")
using DataFrames
using PyCall
using Capse
using ArgParse
using DelimitedFiles
using Random
using LinearAlgebra
using Statistics
include(joinpath(@__DIR__, "..", "training_engine.jl"))



# Argument parsing 
config = ArgParseSettings()
@add_arg_table config begin
    "--spectrum"
    help = "Specify the spectrum to be trained. Either TT, EE, TE, PP."
    arg_type = String 
    required = true
    "--n_nodes"
    help = "Specify the size of the output grid for l (number of desired Lobatto nodes)."
    arg_type = Int 
    required = true
    "--lmax"
    help = "Specify the maximum l for training (can differ from data generation)."
    arg_type = Int 
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
spectrum_kind = parsed_args["spectrum"]
n_nodes = parsed_args["n_nodes"]
lmax = parsed_args["lmax"]
dataset_directory = parsed_args["dataset_directory"]
save_directory = parsed_args["save_directory"]
network_architecture_path = parsed_args["network_architecture_path"]
batchsize = parsed_args["batchsize"]
n_epoch = parsed_args["n_epoch"]
n_run = parsed_args["n_run"]
learning_rates = parsed_args["learning_rates"]
learning_rates = parse.(Float64, split(learning_rates, ","))



# Defines the necessary preprocessing and helper functions
preprocess(ln10As, tau) = exp(ln10As) * 1e-10 * exp(-2 * tau)
preprocess_PP(ln10As) = exp(ln10As) * 1e-10

cubic_spline = pyimport("scipy.interpolate").CubicSpline

function interpolate_dense_chunk(dense, ell_dense, nodes)
    """Interpolates the output onto new input grid (the nodes)."""
    spline = cubic_spline(ell_dense, dense; axis=1)
    return vec(spline(nodes))
end

function lobatto_nodes(node_count; lower=2, upper=lmax)
    """Computes the lobatto nodes in terms of the number of nodes and lower/upper bounds."""
    theta = range(0, π; length=node_count)
    nodes = 0.5 .* (lower + upper) .- 0.5 .* (upper - lower) .* cos.(theta)
    nodes[1] = lower
    nodes[end] = upper
    return nodes
end

function hybrid_lowell_nodes(lobatto_node_count; dense_lower=2, dense_upper=20, upper=lmax)
    """For EE spectrum specifically, uses denser grid for low l and then lobatto grid for above this."""
    low_nodes = collect(dense_lower:dense_upper-1)
    high_nodes = lobatto_nodes(lobatto_node_count; lower=Float64(dense_upper), upper)
    nodes = vcat(low_nodes, high_nodes)
    return nodes
end


# Specifies the nodes to use for interpolation
if spectrum_kind == "EE"
    nodes = hybrid_lowell_nodes(n_nodes, dense_lower=2, dense_upper=20, upper=lmax)
else
    nodes = lobatto_nodes(n_nodes, lower=2, upper=lmax)
end


# Reads the contents of the data samples (and extracts number of samples and input/output dimensions for each)
subdirs = readdir(dataset_directory)
n_samples = length(subdirs)
example_param_dict = JSON.parsefile(joinpath(dataset_directory, subdirs[1], "capse_dict.json"))
# Input order: the standard parameters present in the samples, then any extra ones (sorted)
standard_params = ["ln10As", "ns", "H0", "ombh2", "omch2", "tau", "Mnu", "w0", "wa"]
param_names = vcat(filter(param -> haskey(example_param_dict, param), standard_params),
                   sort(filter(param -> !(param in standard_params), collect(keys(example_param_dict)))))
if param_names[1] != "ln10As" || param_names[6] != "tau"
    error("The postprocessing assumes ln10As is the first input and tau the sixth, got $(param_names)")
end
n_input = length(param_names)
nl = length(npzread(joinpath(dataset_directory, subdirs[1], "TT.npy")))
n_output = length(nodes)

# Constructs matrices to store the input and output features
input_params = Matrix{Float64}(undef, n_samples, n_input)
targets = Matrix{Float64}(undef, n_samples, n_output)


# Goes through the data samples and adds input/target features and then at end constructs dataframe
for (idx, sub) in enumerate(subdirs)
    if idx % 50000 == 0
        println("Loaded up to $(idx) samples.")
    end
    sub_path = joinpath(dataset_directory, sub)
    param_dict = JSON.parsefile(joinpath(sub_path, "capse_dict.json"))
    input_params[idx, :] = [param_dict[param] for param in param_names]
    ln10As, tau = param_dict["ln10As"], param_dict["tau"]
    if spectrum_kind == "PP"
        factor = preprocess_PP(ln10As) # applies different preprocessing depending on component
    else
        factor = preprocess(ln10As, tau)
    end
    ells = Array(2:lmax)
    target_file = npzread(joinpath(sub_path, "$(spectrum_kind).npy"))[1:lmax-1]
    targets[idx, :] = interpolate_dense_chunk(vec(Array(target_file)) ./ factor, ells, nodes)
end
df = DataFrame(input_params, param_names)
df.observable = [targets[i, :] for i in 1:n_samples]

# Saves the l grid of the outputs (the nodes), which the emulator needs to interpolate back onto all l
npzwrite(joinpath(save_directory, "l.npy"), nodes)

# Now runs the training itself
run_training(df, save_directory, network_architecture_path, learning_rates, n_run, n_epoch, batchsize)