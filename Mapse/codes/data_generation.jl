using Distributed
using NPZ
using SlurmClusterManager
using EmulatorsTrainer
using JSON3
using Random
using LinearAlgebra
using PyCall


ENV["SLURM_NTASKS"] = ENV["JULIA_TOTAL_TASKS"]
mgr = SlurmManager(;launch_timeout = 600.0, srun_post_exit_sleep = 2.0)
addprocs(mgr)


@everywhere begin
    using NPZ, EmulatorsTrainer, JSON3, Random, LinearAlgebra, PyCall
end


@everywhere begin

    # Define the fixed parameters
    const FIXED_LN10AS = 3.044
    const FIXED_NS = 0.965

    # Specify the emulator input parameters and their lower/upper bounds in the desired order (needs to be adjusted by model type)
    # Excludes As and ns (since they are not part of the emulator component of the calculation)
    pars = ["z", "H0", "ombh2", "omch2", "Mnu", "w0", "wa"]
    lb = [0.2, 50, 0.02, 0.08, 0.0, -3, -3]
    ub = [1.6, 90, 0.025, 0.18, 0.5, 1, 2]

    # Specify the desired number of data samples (note: if w0wa extension is considered, some samples will be discarded for w0+wa>0)
    n = 100000

    # Specify the random seed for reproducibility purposes (or set to nothing otherwise)
    seed = nothing
    if seed != nothing
        Random.seed!(seed)
    end

    # Uses latin hypercube sampling, and removes samples with w0+wa>0 since unphysical
    # Must change the indices to match positions of w0,wa in the input vector!
    s = EmulatorsTrainer.create_training_dataset(n, lb, ub)
    w0_ind, wa_ind = 6, 7
    s_cond = [s[w0_ind, i] + s[wa_ind, i] for i in 1:n] 
    s = s[:, s_cond.<0.0]

    # Specify the directory path to store training samples (recommended to change depending on which model/code being used)
    root_dir = "/pscratch/sd/j/jgmorawe/mapse_class_mnuw0wacdm_" * string(n)

    # For importing python modules from Julia
    classy = pyimport("classy")

    # Sets k grid in physical units (not h/Mpc) from 1e-3 to 10
    nk = 2000
    k_grid = exp.(range(log(1e-3), log(10), length=nk))

    # Function which takes in the training sample parameters and saves statistics to file
    function classy_script(CosmoDict, root_path)
        try
            # Creates subfolders to store each training sample in (uses random string for this)
            rand_str = root_path * "/" * randstring(10)
            # Dictionary of the input features to Class (may need to change depending on model)
            z = CosmoDict["z"]
            cosmo_params = Dict(
                "output" => "mPk",
                "P_k_max_1/Mpc" => 15.0, # physical k max
                "z_pk" => string(z),
                "ln10^{10}A_s" => FIXED_LN10AS,
                "n_s" => FIXED_NS,
                "h" => CosmoDict["H0"] / 100,
                "omega_b" => CosmoDict["ombh2"],
                "omega_cdm" => CosmoDict["omch2"],
                "m_ncdm" => CosmoDict["Mnu"],
                "w0_fld" => CosmoDict["w0"],
                "wa_fld" => CosmoDict["wa"],
                "tau_reio" => 0.0568,
                "N_ur" => 2.0308,
                "N_ncdm" => 1,
                "use_ppf" => "yes",
                "fluid_equation_of_state" => "CLP",
                "cs2_fld" => 1,
                "Omega_Lambda" => 0,
                "Omega_scf" => 0)

            # Initializes the Class object and then computes statistics
            cosmo = classy.Class()
            cosmo.set(cosmo_params)
            cosmo.compute()

            # Compute Pcb and Pmm in physical units (Mpc^3)
            Pcb = [cosmo.pk_cb(k, z) for k in k_grid]
            Pmm = [cosmo.pk(k, z) for k in k_grid]

            if any(isnan, Pcb) || any(isnan, Pmm)
                @error "NaN values encountered for parameters: $CosmoDict"
            else
                # Creates directory for the particular training sample and saves relevant files to it
                mkdir(rand_str)
                npzwrite(rand_str * "/pk_cb.npy", Pcb)
                npzwrite(rand_str * "/pk_mm.npy", Pmm)
                npzwrite(rand_str * "/k.npy", k_grid)

                # Store all 9 parameters in mapse_dict.json
                CosmoDict_full = Dict(
                    "z" => z,
                    "ln10As" => FIXED_LN10AS,
                    "ns" => FIXED_NS,
                    "H0" => CosmoDict["H0"],
                    "ombh2" => CosmoDict["ombh2"],
                    "omch2" => CosmoDict["omch2"],
                    "Mnu" => CosmoDict["Mnu"],
                    "w0" => CosmoDict["w0"],
                    "wa" => CosmoDict["wa"])
                open(rand_str * "/mapse_dict.json", "w") do io
                    JSON3.write(io, CosmoDict_full)
                end
            end
        catch e
            println("Something went wrong during calculation!")
            println(CosmoDict)
        end
    end
end

EmulatorsTrainer.compute_dataset(s, pars, root_dir, classy_script, :distributed)