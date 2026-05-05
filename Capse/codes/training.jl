using Pkg; Pkg.activate(".")
using EmulatorsTrainer
using DataFrames
using NPZ
using JSON
using AbstractCosmologicalEmulators
using SimpleChains
using ArgParse
using DelimitedFiles


config = ArgParseSettings()
@add_arg_table config begin
    "--spectrum"
    help = "Specify the spectrum to be trained. Either TT, EE, TE, PP."
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
global SpectraKind = parsed_args["spectrum"]
CℓDirectory = parsed_args["path_input"]
OutDirectory = parsed_args["path_output"]
nn_setup_path = parsed_args["nn_setup_path"]
n_epoch = parsed_args["n_epoch"]
n_run = parsed_args["n_run"]
batchsize = parsed_args["batchsize"]
@info SpectraKind
@info CℓDirectory
@info OutDirectory

# The length of the l grid (automatically extracts by reading in one of the data training samples)
global nl = length(npzread(joinpath(CℓDirectory, readdir(CℓDirectory)[1], "TT.npy")))

# Function to compute rescaling factor (to remove and then reapply after emulator prediction)
preprocess(ln10As, τ) = exp(ln10As) * 1e-10 * exp(-2 * τ)

# Takes in cosmological parameters and Cl result and outputs tuple of results
# If certain parameters are added/removed from model, they must be modified in this function!
function get_observable_tuple(cosmo_pars, Cl)
    ωb = cosmo_pars["ombh2"]
    ωcdm = cosmo_pars["omch2"]
    τ = cosmo_pars["τ"]
    H0 = cosmo_pars["H0"]
    ln10As = cosmo_pars["ln10As"]
    ns = cosmo_pars["ns"]
    Mν = cosmo_pars["Mnu"]
    w0 = cosmo_pars["w0"]
    wa = cosmo_pars["wa"]
    factor = preprocess(ln10As, τ)
    return (ln10As, ns, H0, ωb, ωcdm, τ, Mν, w0, wa, Cl[1:nl] ./ factor)
end

# The number of input features (automatically extracts by reading in one of the data training samples)
n_input_features = length(JSON.parsefile(joinpath(CℓDirectory, readdir(CℓDirectory)[1], "capse_dict.json")))
n_output_features = nl

# Function for adding observable to dataframe
observable_file = "/" * SpectraKind * ".npy"
param_file = "/capse_dict.json"
add_observable!(df, location) = EmulatorsTrainer.add_observable_df!(df, location, param_file, observable_file, get_observable_tuple)

# Initiates empty dataframe and adds results (must change which parameters are included here depending on model)
df = DataFrame(ln10A_s=Float64[], ns=Float64[], H0=Float64[], omega_b=Float64[], omega_cdm=Float64[], τ=Float64[], Mnu=Float64[], w0=Float64[], wa=Float64[], observable=Array[])
@time EmulatorsTrainer.load_df_directory!(df, CℓDirectory, add_observable!)

# List of input parameters corresponding to the dataframe (must change depending on which model is used)
array_pars_in = ["ln10A_s", "ns", "H0", "omega_b", "omega_cdm", "τ", "Mnu", "w0", "wa"]
in_array, out_array = EmulatorsTrainer.extract_input_output_df(df)
in_MinMax = EmulatorsTrainer.get_minmax_in(df, array_pars_in)
out_MinMax = EmulatorsTrainer.get_minmax_out(out_array)

# Saves input and output minimums and maximums to file (as they are later used to undo the normalization for the output)
folder_output = OutDirectory * "/" * string(SpectraKind)
npzwrite(folder_output * "/inminmax.npy", in_MinMax)
npzwrite(folder_output * "/outminmax.npy", out_MinMax)

# Applies the normalization to the inputs/outputs for emulator purposes
EmulatorsTrainer.maximin_df!(df, in_MinMax, out_MinMax)


# Initializes the neural network architecture (must create setup .json file in advance, sample version found in github)
# and need to adjust path accordingly
NN_dict = JSON.parsefile(nn_setup_path)
NN_dict["n_output_features"] = n_output_features
NN_dict["n_input_features"] = n_input_features

mlpd = AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict)
X, Y, Xtest, Ytest = EmulatorsTrainer.getdata(df)
p = SimpleChains.init_params(mlpd)
G = SimpleChains.alloc_threaded_grad(mlpd)


# Saves the l vector and the modified setup .json file
l = Array(2:nl+1)
dest = joinpath(folder_output, "l.npy")
npzwrite(dest, l)
dest = joinpath(folder_output, "nn_setup.json")
json_str = JSON.json(NN_dict)
open(dest, "w") do file
    write(file, json_str)
end


# Adds supporting Capse.jl files (for postprocessing terms) to the trained emulator directory
dest = joinpath(folder_output, "postprocessing.py")
run(`cp Capse/supporting_files/postprocessing.py $dest`)
dest = joinpath(folder_output, "postprocessing.jl")
run(`cp Capse/supporting_files/postprocessing.jl $dest`)


# Initializes the losses
mlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(Y))
mlpdtest = SimpleChains.add_loss(mlpd, SquaredLoss(Ytest))

report = let mtrain = mlpdloss, X = X, Xtest = Xtest, mtest = mlpdtest
    p -> begin
        let train = mlpdloss(X, p), test = mlpdtest(Xtest, p)
            @info "Loss:" train test
        end
    end
end

pippo_loss = mlpdtest(Xtest, p)
println("Initial Loss: ", pippo_loss)
lr_list = [1e-4, 7e-5, 5e-5, 2e-5, 1e-5, 7e-6, 5e-6, 2e-6, 1e-6, 7e-7, 5e-7, 2e-7]

# Iterates through different learning rates and does multiple runs for each, the user may wish
# to modify the number of runs, the number of epochs, the batchsize, etc
for lr in lr_list
    for i in 1:n_run
        @time SimpleChains.train_batched!(G, p, mlpdloss, X, SimpleChains.ADAM(lr), n_epoch; batchsize=batchsize)
        report(p)
        test = mlpdtest(Xtest, p)
        if pippo_loss > test
            npzwrite(folder_output * "/weights.npy", p) # continuously saves the weights to file if they get better
            global pippo_loss = test
            @info "Saving coefficients! Test loss is equal to :" test
        end
    end
end