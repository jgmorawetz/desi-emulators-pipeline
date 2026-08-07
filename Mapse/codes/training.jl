using Pkg; Pkg.activate(".")
using EmulatorsTrainer
using DataFrames
using NPZ
using JSON
using AbstractCosmologicalEmulators
using Mapse
using SimpleChains
using ArgParse
using DelimitedFiles
using Random
using LinearAlgebra
using Statistics


config = ArgParseSettings()
@add_arg_table config begin
    "--spectrum"
    help = "Specify the spectrum to be trained. Either cb or mm."
    arg_type = String
    required = true
    "--path_input"
    help = "Specify the path to the input folder (training data)."
    arg_type = String
    required = true
    "--path_output"
    help = "Specify the path to the output folder (trained emulator)."
    arg_type = String
    required = true
    "--var_ratio"
    help = "Specify the variance ratio of PCA components to keep."
    arg_type = Float64 
    required = true
    "--nn_setup_path"
    help = "Specify the file path to the neural network setup json file."
    arg_type = String
    required = true
    "--n_epoch"
    help = "Specify the number of epochs."
    arg_type = Int
    required = true
    "--n_run"
    help = "Specify the number of runs (per learning rate)."
    arg_type = Int
    required = true
    "--batchsize"
    help = "Specify the batchsize."
    arg_type = Int
    required = true
end
parsed_args = parse_args(config)
const SpectraKind = parsed_args["spectrum"]
const DatasetDirectory = parsed_args["path_input"]
const OutDirectory = parsed_args["path_output"]
const var_ratio = parsed_args["var_ratio"]
nn_setup_path = parsed_args["nn_setup_path"]
n_epoch = parsed_args["n_epoch"]
n_run = parsed_args["n_run"]
batchsize = parsed_args["batchsize"]
@info "Spectrum kind: $SpectraKind"
@info "Dataset directory: $DatasetDirectory"
@info "Output directory: $OutDirectory"
@info "Target variance ratio: $var_ratio"

# Load the data
subdirs = readdir(DatasetDirectory)
n_samples = length(subdirs)
@info "Found $n_samples samples in the dataset directory."
if n_samples == 0
    error("No training data found in $DatasetDirectory.")
end

# Load k_grid from the first folder (and sets size by reading it)
first_sub = joinpath(DatasetDirectory, subdirs[1])
k_grid = npzread(joinpath(first_sub, "k.npy"))
nk = length(k_grid)

# Arrays to store input features and target spectra (need to adjust if number of input parameters changes)
n_input = length(JSON.parsefile(joinpath(DatasetDirectory, subdirs[1], "mapse_dict.json"))) - 2 # subtracts two since As, ns not to be included here
features = Matrix{Float64}(undef, n_input, n_samples)
targets = Matrix{Float64}(undef, nk, n_samples)


# Preprocesses targets by finding transfer function ratio (need to adjust depending on which input parameters are used)
@info "Preprocessing targets by extracting the transfer function ratio..."
for (idx, sub) in enumerate(subdirs)
    sub_path = joinpath(DatasetDirectory, sub)
    dict = JSON.parsefile(joinpath(sub_path, "mapse_dict.json"))
    # Extract parameters
    z = dict["z"]
    ln10As = dict["ln10As"]
    ns = dict["ns"]
    H0 = dict["H0"]
    ombh2 = dict["ombh2"]
    omch2 = dict["omch2"]
    Mnu = dict["Mnu"]
    w0 = dict["w0"]
    wa = dict["wa"]
    features[:, idx] = [z, H0, ombh2, omch2, Mnu, w0, wa]
    # Load true spectrum (Pcb or Pmm)
    spec_file = SpectraKind == "cb" ? "pk_cb.npy" : "pk_mm.npy"
    P_true = npzread(joinpath(sub_path, spec_file))
    # Compute growth, primordial Pk, and neutrino correction factors
    h = H0 / 100
    Ωcb0 = (ombh2 + omch2) / h^2
    D = D_z(z, Ωcb0, h; mν=Mnu, w0=w0, wa=wa)
    As = exp(ln10As) * 1e-10
    P_prim = Mapse.primordial_Pk(As, ns, k_grid)
    log10_k = log10.(k_grid)
    ων = Mnu / 93.14
    Δω = omch2 + ων - ombh2
    ωm = ombh2 + omch2 + ων
    DIFF = exp.(0.4971733969600907 .+ (-24.849067935704547 .- log.((((((0.731102574104348 .^ log10_k) .+ Δω) ./ 0.17522861267519874) .^ log10_k) .+ ((63.65597287231169 .^ (log10_k .+ 0.0472474783701488)) .* ((0.9899093975978591 .^ (log10_k ./ (cos.(log10_k ./ ((1.1964213875807956 ^ -2.3661897652294015) ./ cos.(log10_k ./ -1.8173117588773222))) ./ 0.20037856443385513))) ./ (Δω ^ 0.7767030041348179)))) .+ (0.14823981687164764 * ωm))))
    # Target: T^2 = P_true / (D^2 * P_prim * DIFF^2)
    targets[:, idx] = P_true ./ (D^2 .* P_prim .* DIFF .^ 2)
end


# Performs PCA based on variance ratio (does SVD and then selects number of PCA elements based on variance ratio criteria)
pca_mean = mean(targets, dims=2)[:, 1]
centered_targets = targets .- reshape(pca_mean, :, 1)
U, S, V = svd(centered_targets)
total_var = sum(S .^ 2)
cum_var = cumsum(S .^ 2) / total_var
n_pca = findfirst(>=(var_ratio), cum_var)
if isnothing(n_pca)
    n_pca = length(S)
end
@info "Retained $n_pca PCA components for cumulative variance ratio >= $var_ratio"
pca_basis = U[:, 1:n_pca]
pca_coefs = pca_basis' * centered_targets


# Create folder structure for output and save PCA metadata and kgrid
folder_output = joinpath(OutDirectory, "Pk_lin_$SpectraKind")
mkpath(folder_output)
Mapse.save_pca_metadata(folder_output, pca_mean, pca_basis)
npzwrite(joinpath(folder_output, "k.npy"), k_grid)


# Prepares DataFrame for training (need to adjust if different parameters are used)
df = DataFrame(
    z = features[1, :],
    H0 = features[2, :],
    ombh2 = features[3, :],
    omch2 = features[4, :],
    Mnu = features[5, :],
    w0 = features[6, :],
    wa = features[7, :],
    observable = [pca_coefs[:, i] for i in 1:n_samples])

# List of input parameters corresponding to the dataframe (must change depending on which model is used)
array_pars_in = ["z", "H0", "ombh2", "omch2", "Mnu", "w0", "wa"]
in_array, out_array = EmulatorsTrainer.extract_input_output_df(df)
in_MinMax = EmulatorsTrainer.get_minmax_in(df, array_pars_in)
out_MinMax = EmulatorsTrainer.get_minmax_out(out_array)

# Saves input and output minimums and maximums to file (as they are later used to undo the normalization for the output)
npzwrite(joinpath(folder_output, "inminmax.npy"), in_MinMax)
npzwrite(joinpath(folder_output, "outminmax.npy"), out_MinMax)

# Applies the normalization to the inputs/outputs for emulator purposes
EmulatorsTrainer.maximin_df!(df, in_MinMax, out_MinMax)

# Initializes the neural network architecture (must create setup .json file in advance, sample version found in github)
# and need to adjust path accordingly
NN_dict = JSON.parsefile(nn_setup_path)
NN_dict["n_output_features"] = n_pca
NN_dict["n_input_features"] = n_input
# Saves configured nn_setup.json to output directory
open(joinpath(folder_output, "nn_setup.json"), "w") do io
    JSON.print(io, NN_dict)
end

mlpd = AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict)
X, Y, Xtest, Ytest = EmulatorsTrainer.getdata(df)
p = SimpleChains.init_params(mlpd)
G = SimpleChains.alloc_threaded_grad(mlpd)


# Initializes the losses
mlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(Y))
mlpdtest = SimpleChains.add_loss(mlpd, SquaredLoss(Ytest))

report = p -> begin
    train = mlpdloss(X, p)
    test = mlpdtest(Xtest, p)
    @info "Loss:" train test
end

pippo_loss = mlpdtest(Xtest, p)
println("Initial Test Loss: ", pippo_loss)
lr_list = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

# Iterates through different learning rates and does multiple runs for each, the user may wish
# to modify the number of runs, the number of epochs, the batchsize, etc
for lr in lr_list
    for i in 1:n_run
        @time SimpleChains.train_batched!(G, p, mlpdloss, X, SimpleChains.ADAM(lr), n_epoch; batchsize=batchsize)
        report(p)
        test = mlpdtest(Xtest, p)
        if pippo_loss > test
            npzwrite(joinpath(folder_output, "weights.npy"), p)
            npzwrite(joinpath(folder_output, "best_test_loss.npy"), Array([test, lr, i])) # keeps track of best loss so far in case crash prematurely
            global pippo_loss = test
            @info "New best test loss: $test (Saved weights)"
        end
    end
end