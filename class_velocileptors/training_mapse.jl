using Pkg
Pkg.activate(".")
using DataFrames
using Effort
using Mapse
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
    help = "Specify the spectrum to be trained. Either 'cb' or 'mm'."
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
    "--var_ratio"
    help = "Specify the variance ratio of PCA components to keep."
    arg_type = Float64 
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
dataset_directory = parsed_args["dataset_directory"]
save_directory = parsed_args["save_directory"]
var_ratio = parsed_args["var_ratio"]
network_architecture_path = parsed_args["network_architecture_path"]
batchsize = parsed_args["batchsize"]
n_epoch = parsed_args["n_epoch"]
n_run = parsed_args["n_run"]
learning_rates = parsed_args["learning_rates"]
learning_rates = parse.(Float64, split(learning_rates, ","))



# Reads the contents of the data samples (and extracts number of samples and input/output dimensions for each)
subdirs = readdir(dataset_directory)
n_samples = length(subdirs)
example_param_dict = JSON.parsefile(joinpath(dataset_directory, subdirs[1], "mapse_dict.json"))
# Input order: the standard parameters present in the samples, then any extra ones (sorted); As, ns not to be included here
standard_params = ["z", "H0", "ombh2", "omch2", "Mnu", "w0", "wa"]
param_names = vcat(filter(param -> haskey(example_param_dict, param), standard_params),
                   sort(filter(param -> !(param in vcat(standard_params, ["ln10As", "ns"])), collect(keys(example_param_dict)))))
n_input = length(param_names)
k_grid = npzread(joinpath(dataset_directory, subdirs[1], "kv.npy"))
nk = length(k_grid)

# Constructs matrices to store the input and output features
input_params = Matrix{Float64}(undef, n_samples, n_input)
targets = Matrix{Float64}(undef, n_samples, nk)


# Goes through the data samples and adds input/target features and then at end constructs dataframe
for (idx, sub) in enumerate(subdirs)
    if idx % 50000 == 0
        println("Loaded up to $(idx) samples.")
    end
    sub_path = joinpath(dataset_directory, sub)
    param_dict = JSON.parsefile(joinpath(sub_path, "mapse_dict.json"))
    input_params[idx, :] = [param_dict[param] for param in param_names]
    z, ln10As, ns, H0, ombh2, omch2 = (param_dict[param] for param in ["z", "ln10As", "ns", "H0", "ombh2", "omch2"])
    # Fixed values if not varied, as in the data generation
    Mnu = get(param_dict, "Mnu", 0.06)
    w0 = get(param_dict, "w0", -1.0)
    wa = get(param_dict, "wa", 0.0)
    # Load true spectrum (Pcb or Pmm)
    spec_file = spectrum_kind == "cb" ? "pk_cb.npy" : "pk_mm.npy"
    P_true = npzread(joinpath(sub_path, spec_file))
    # Compute growth, primordial Pk, and neutrino correction factors (and applies rescaling to get transfer function effectively)
    h = H0 / 100
    Ωcb0 = (ombh2 + omch2) / h^2
    D = Effort.D_z(z, Ωcb0, h; mν=Mnu, w0=w0, wa=wa)
    As = exp(ln10As) * 1e-10
    P_prim = Mapse.primordial_Pk(As, ns, k_grid)
    log10_k = log10.(k_grid)
    ων = Mnu / 93.14
    Δω = omch2 + ων - ombh2
    ωm = ombh2 + omch2 + ων
    DIFF = exp.(0.4971733969600907 .+ (-24.849067935704547 .- log.((((((0.731102574104348 .^ log10_k) .+ Δω) ./ 0.17522861267519874) .^ log10_k) .+ ((63.65597287231169 .^ (log10_k .+ 0.0472474783701488)) .* ((0.9899093975978591 .^ (log10_k ./ (cos.(log10_k ./ ((1.1964213875807956 ^ -2.3661897652294015) ./ cos.(log10_k ./ -1.8173117588773222))) ./ 0.20037856443385513))) ./ (Δω ^ 0.7767030041348179)))) .+ (0.14823981687164764 * ωm))))
    targets[idx, :] = P_true ./ (D^2 .* P_prim .* DIFF .^ 2)
end

# Does PCA compression (and saves to file for later usage to undo it)
pca_mean = vec(mean(targets, dims=1))
centered_targets = targets .- pca_mean'
U, S, V = svd(centered_targets)
total_var = sum(S .^ 2)
cum_var = cumsum(S .^ 2) / total_var
n_pca = findfirst(>=(var_ratio), cum_var)
if isnothing(n_pca)
    n_pca = length(S)
end
pca_basis = V[:, 1:n_pca]
pca_coefs = centered_targets * pca_basis
Mapse.save_pca_metadata(save_directory, pca_mean, pca_basis)

df = DataFrame(input_params, param_names)
df.observable = [pca_coefs[i, :] for i in 1:n_samples]

# Now runs the training itself
run_training(df, save_directory, network_architecture_path, learning_rates, n_run, n_epoch, batchsize)