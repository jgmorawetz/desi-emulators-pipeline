using Distributed
using NPZ
using SlurmClusterManager
using EmulatorsTrainer
using JSON3
using Random
using PyCall


ENV["SLURM_NTASKS"] = ENV["JULIA_TOTAL_TASKS"]
mgr = SlurmManager(;launch_timeout = 600.0, srun_post_exit_sleep = 2.0)
addprocs(mgr)


@everywhere begin
    using NPZ, EmulatorsTrainer, JSON3, Random, PyCall
end


@everywhere begin

    # Specify the emulator input parameters and their lower/upper bounds in the desired order (needs to be adjusted by model type)
    pars = ["ln10As", "ns", "H0", "ombh2", "omch2", "τ", "Mnu", "w0", "wa"]
    lb = [2.3, 0.8, 50, 0.02, 0.08, 0.01, 0, -3, -3]
    ub = [3.7, 1.1, 90, 0.025, 0.18, 0.15, 0.5, 1, 2]

    # Specify the desired number of data samples (note: if w0wa extension is considered, some samples will be discarded for w0+wa>0)
    n = 50000

    # Specify the random seed for reproducibility purposes (or set to nothing otherwise)
    seed = 0
    if seed != nothing
        Random.seed!(seed)
    end

    # Uses latin hypercube sampling, and removes samples with w0+wa>0 since unphysical
    # Must change the indices to match positions of w0,wa in the input vector!
    s = EmulatorsTrainer.create_training_dataset(n, lb, ub)
    w0_ind, wa_ind = 8, 9
    s_cond = [s[w0_ind, i] + s[wa_ind, i] for i in 1:n] 
    s = s[:, s_cond.<0.0]

    # Specify the directory path to store training samples (recommended to change depending on which model/code being used)
    root_dir = "/pscratch/sd/j/jgmorawe/capse_class_mnuw0wacdm_" * string(n)

    # Import necessary python modules from Julia (velocileptors_free is modified version of velocileptors to allow for passing k grid directly)
    classy = pyimport("classy")
    np = pyimport("numpy")


    # Function which takes in the training sample parameters and saves statistics to file
    function classy_script(CosmoDict, root_path)
        try
            # Creates subfolders to store each training sample in (uses random string for this)
            rand_str = root_path * "/" * randstring(10)
            # Dictionary of the input features to Class (may need to change depending on model)
            cosmo_params = Dict(
                "output" => "tCl pCl lCl",
                "l_max_scalars" => 10000,
                "lensing" => "yes",
                "h" => CosmoDict["H0"] / 100,
                "omega_b" => CosmoDict["ombh2"],
                "omega_cdm" => CosmoDict["omch2"],
                "ln10^{10}A_s" => CosmoDict["ln10As"],
                "n_s" => CosmoDict["ns"],
                "tau_reio" => CosmoDict["τ"],
                "N_ur" => 2.0308,
                "N_ncdm" => 1,
                "m_ncdm" => CosmoDict["Mnu"],
                "use_ppf" => "yes",
                "w0_fld" => CosmoDict["w0"],
                "wa_fld" => CosmoDict["wa"],
                "fluid_equation_of_state" => "CLP",
                "cs2_fld" => 1.0,
                "Omega_Lambda" => 0.0,
                "Omega_scf" => 0.0,
                "accurate_lensing" => 1,
                "non_linear" => "hmcode")
            # Initializes the Class object and then computes statistics
            cosmo = classy.Class()
            cosmo.set(cosmo_params)
            cosmo.compute()
            cl = cosmo.lensed_cl(lmax=10000)
            ell = np.arange(length(cl["tt"])) # multipole array goes from 0 up to l_max (inclusive)
            factor = ell .* (ell .+ 1) ./ 2 ./ np.pi
            tt = 7.42715e12 .* (factor .* cl["tt"])[3:10000]
            ee = 7.42715e12 .* (factor .* cl["ee"])[3:10000]
            te = 7.42715e12 .* (factor .* cl["te"])[3:10000]
            pp = (ell .* (ell .+ 1) .* ell .* (ell .+ 1) .* cl["pp"] ./ 2 ./ np.pi)[3:10000]
            if any(isnan, tt) || any(isnan, ee) || any(isnan, te) || any(isnan, pp)
                @error "There are nan values!"
            else
                mkdir(rand_str)
                npzwrite(rand_str * "/TT.npy", tt)
                npzwrite(rand_str * "/EE.npy", ee)
                npzwrite(rand_str * "/TE.npy", te)
                npzwrite(rand_str * "/PP.npy", pp)
                open(rand_str * "/capse_dict.json", "w") do io
                    JSON3.write(io, CosmoDict)
                end
            end
        catch e
            println("Something went wrong during calculation!")
            println(CosmoDict)
        end
    end
end

EmulatorsTrainer.compute_dataset(s, pars, root_dir, classy_script, :distributed)
