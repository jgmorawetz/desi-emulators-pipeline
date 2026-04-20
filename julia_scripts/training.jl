using Pkg; Pkg.activate(".")
using EmulatorsTrainer
using DataFrames
using NPZ
using JSON
using AbstractCosmologicalEmulators
using Effort
using SimpleChains
using ArgParse
using DelimitedFiles


config = ArgParseSettings()
@add_arg_table config begin
    "--component"
    help = "Specify the component to be trained. Either 11, loop or ct."
    default = "11"
    "--multipole", "-l"
    help = "Specify the multipole to be trained. Either 0, 2, or 4."
    arg_type = Int
    default = 0
    "--path_input", "-i"
    help = "Specify the path to the input folder (training data)."
    required = true
    "--path_output", "-o"
    help = "Specify the path to the output folder (trained emulator)."
    required = true
end
parsed_args = parse_args(config)
global Componentkind = parsed_args["component"]
ℓ = parsed_args["multipole"]
PℓDirectory = parsed_args["path_input"]
OutDirectory = parsed_args["path_output"]
@info ℓ
@info PℓDirectory
@info OutDirectory
@info Componentkind

# The length of the k grid (this must be changed if a different k grid is used)
global nk = 55

# The output size depends on which component is being computed (based on EFT expansion)
if Componentkind == "11"
    nk_factor = 3
elseif Componentkind == "loop"
    nk_factor = 9
elseif Componentkind == "ct"
    nk_factor = 4
else
    @error "Wrong component!"
end

# Function to take the power spectrum training sample and separate into the components and apply factor rescaling
function reshape_Pk(Pk, factor)
    if Componentkind == "11"
        result = vec(Array(Pk)[:, 1:3]) ./ factor
    elseif Componentkind == "loop"
        result = vec(Array(Pk)[:, 4:12]) ./ factor^2
    elseif Componentkind == "ct"
        result = vec(Array(Pk)[:, 13:16]) ./ factor
    else
        @error "Wrong component!"
    end
    return result
end

# Computes the linear growth factor
function D_ODE(z, ωb, ωcdm, h, Mν, w0, wa)
    cosmology = Effort.w0waCDMCosmology(
        ln10Aₛ=3.0, nₛ=0.96, h=h,
        ωb=ωb, ωc=ωcdm, mν=Mν,
        w0=w0, wa=wa)
    return Effort.D_z(z, cosmology)
end

# Function to compute rescaling factor (to remove and then reapply after emulator prediction)
preprocess(z, As, ωb, ωcdm, h, Mν, w0, wa) = As * D_ODE(z, ωb, ωcdm, h, Mν, w0, wa)^2

# Takes in cosmological parameters and power spectrum result and outputs tuple of results
# If certain parameters are added/removed from model, they must be modified in this function!
function get_observable_tuple(cosmo_pars, Pk)
    z = cosmo_pars["z"]
    ωb = cosmo_pars["ombh2"]
    ωcdm = cosmo_pars["omch2"]
    Mν = cosmo_pars["Mnu"]
    h = cosmo_pars["H0"] / 100
    As = exp(cosmo_pars["ln10As"]) * 1e-10
    w0 = cosmo_pars["w0"]
    wa = cosmo_pars["wa"]
    factor = preprocess(z, As, ωb, ωcdm, h, Mν, w0, wa)
    return (cosmo_pars["z"], cosmo_pars["ln10As"], cosmo_pars["ns"], cosmo_pars["H0"],
            cosmo_pars["ombh2"], cosmo_pars["omch2"], cosmo_pars["Mnu"], cosmo_pars["w0"], cosmo_pars["wa"], 
            reshape_Pk(Pk, factor))
end

# The number of input features (must be changed if a different model is used)
n_input_features = 9
n_output_features = nk * nk_factor

# Function for adding observable to dataframe
observable_file = "/pk_" * string(ℓ) * ".npy"
param_file = "/effort_dict.json"
add_observable!(df, location) = EmulatorsTrainer.add_observable_df!(df, location, param_file, observable_file, get_observable_tuple)

# Initiates empty dataframe and adds results (must change which parameters are included here depending on model)
df = DataFrame(z=Float64[], ln10A_s=Float64[], ns=Float64[], H0=Float64[], omega_b=Float64[], omega_cdm=Float64[], Mnu=Float64[], w0=Float64[], wa=Float64[], observable=Array[])
@info PℓDirectory
@time EmulatorsTrainer.load_df_directory!(df, PℓDirectory, add_observable!)

# List of input parameters corresponding to the dataframe (must change depending on which model is used)
array_pars_in = ["z", "ln10A_s", "ns", "H0", "omega_b", "omega_cdm", "Mnu", "w0", "wa"]
in_array, out_array = EmulatorsTrainer.extract_input_output_df(df)
in_MinMax = EmulatorsTrainer.get_minmax_in(df, array_pars_in)
out_MinMax = EmulatorsTrainer.get_minmax_out(out_array)

# Saves input and output minimums and maximums to file (as they are later used to undo the normalization for the output)
folder_output = OutDirectory * "/" * string(ℓ) * "/" * string(Componentkind)
folder_output_before = OutDirectory * "/" * string(ℓ)
npzwrite(folder_output * "/inminmax.npy", in_MinMax)
npzwrite(folder_output * "/outminmax.npy", out_MinMax)

# Applies the normalization to the inputs/outputs for emulator purposes
EmulatorsTrainer.maximin_df!(df, in_MinMax, out_MinMax)


# Initializes the neural network architecture (must created setup .json file in advance, sample version found in github)
# and need to adjust path accordingly
NN_dict = JSON.parsefile("nn_setup.json")
NN_dict["n_output_features"] = n_output_features
NN_dict["n_input_features"] = n_input_features

mlpd = AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict)
X, Y, Xtest, Ytest = EmulatorsTrainer.getdata(df)
p = SimpleChains.init_params(mlpd)
G = SimpleChains.alloc_threaded_grad(mlpd)


# Saves the k vector and the modified setup .json file
dest = joinpath(folder_output, "k.npy")
run(`cp kv.npy $dest`)
dest = joinpath(folder_output, "nn_setup.json")
json_str = JSON.json(NN_dict)
open(dest, "w") do file
    write(file, json_str)
end


# Adds supporting Effort.jl files (for postprocessing and stochastic terms) to the trained emulator directory
# for each multipole and component
if Componentkind == "loop"
    dest = joinpath(folder_output, "postprocessing.py")
    run(`cp postprocessing_loop.py $dest`)
    dest = joinpath(folder_output, "postprocessing.jl")
    run(`cp postprocessing_loop.jl $dest`)
else
    dest = joinpath(folder_output, "postprocessing.py")
    run(`cp postprocessing.py $dest`)
    dest = joinpath(folder_output, "postprocessing.jl")
    run(`cp postprocessing.jl $dest`)
end
if ℓ == 0
    dest = joinpath(folder_output, "stochmodel.py")
    run(`cp stochmodel_0.py $dest`)
    dest = joinpath(folder_output, "stochmodel.jl")
    run(`cp stochmodel_0.jl $dest`)
elseif ℓ == 2
    dest = joinpath(folder_output, "stochmodel.py")
    run(`cp stochmodel_2.py $dest`)
    dest = joinpath(folder_output, "stochmodel.jl")
    run(`cp stochmodel_2.jl $dest`)
elseif ℓ == 4
    dest = joinpath(folder_output, "stochmodel.py")
    run(`cp stochmodel_4.py $dest`)
    dest = joinpath(folder_output, "stochmodel.jl")
    run(`cp stochmodel_4.jl $dest`)
else
    @error "Unsupported multipole"
end

# Separately adds the supporting Effort.jl files (for stochastic, biascombination, jacobian) to the
# trained emulator directory for each multipole (not for each component separately)
# Technically this duplicates the copying multiple times across 11,loop,ct versions but doesn't
# matter because just copying the same file anyway (makes code cleaner)
dest = joinpath(folder_output_before, "biascombination.py")
run(`cp biascombination.py $dest`)
dest = joinpath(folder_output_before, "biascombination.jl")
run(`cp biascombination.jl $dest`)
dest = joinpath(folder_output_before, "jacbiascombination.py")
run(`cp jacbiascombination.py $dest`)
dest = joinpath(folder_output_before, "jacbiascombination.jl")
run(`cp jacbiascombination.jl $dest`)
if ℓ == 0
    dest = joinpath(folder_output_before, "stochmodel.py")
    run(`cp stochmodel_0.py $dest`)
    dest = joinpath(folder_output_before, "stochmodel.jl")
    run(`cp stochmodel_0.jl $dest`)
elseif ℓ == 2
    dest = joinpath(folder_output_before, "stochmodel.py")
    run(`cp stochmodel_2.py $dest`)
    dest = joinpath(folder_output_before, "stochmodel.jl")
    run(`cp stochmodel_2.jl $dest`)
elseif ℓ == 4
    dest = joinpath(folder_output_before, "stochmodel.py")
    run(`cp stochmodel_4.py $dest`)
    dest = joinpath(folder_output_before, "stochmodel.jl")
    run(`cp stochmodel_4.jl $dest`)
else
    @error "Unsupported multipole"
end



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
# to modify the number of trials, the number of iterations, the batchsize, etc
n_run = 10
n_iter = 2000
batchsize = 256
for lr in lr_list
    for i in 1:n_run
        @time SimpleChains.train_batched!(G, p, mlpdloss, X, SimpleChains.ADAM(lr), n_iter; batchsize=batchsize)
        report(p)
        test = mlpdtest(Xtest, p)
        if pippo_loss > test
            npzwrite(folder_output * "/weights.npy", p) # continuously saves the weights to file if they get better
            global pippo_loss = test
            @info "Saving coefficients! Test loss is equal to :" test
        end
    end
end
