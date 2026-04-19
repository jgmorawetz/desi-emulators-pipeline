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

    # Specify the emulator input parameters and their lower/upper bounds in the desired order (needs to be adjusted by model type)
    pars = ["z", "ln10As", "ns", "H0", "ombh2", "omch2", "Mnu", "w0", "wa"]
    lb = [0.2, 2.3, 0.8, 50, 0.02, 0.08, 0, -3, -3]
    ub = [1.6, 3.7, 1.1, 90, 0.025, 0.18, 0.5, 1, 2]

    # Specify the desired number of data samples (note: if w0wa extension is considered, some samples will be discarded for w0+wa>0)
    n = 50000

    # Specify the random seed for reproducibility purposes (or set to nothing otherwise)
    seed = nothing
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
    root_dir = "effort_velocileptors_rept_mnuw0wacdm_" * string(n)

    # Import necessary python modules from Julia (velocileptors_free is modified version of velocileptors to allow for passing k grid directly)
    classy = pyimport("classy")
    EPT = pyimport("velocileptors_free.EPT.ept_fullresum_fftw")
    UTILS = pyimport("velocileptors_free.Utils.pnw_dst")
    np = pyimport("numpy")

    # 'kv_target' is the desired output k grid for the emulator (can change if needed)
    konhmin = 1e-4
    konhmax = 10
    nk = 20000
    konh = np.logspace(np.log10(konhmin), np.log10(konhmax), nk)
    kv_target = np.concatenate(([0.0005],
                                np.logspace(np.log10(0.001), np.log10(0.01), 7, endpoint=false), 
                                np.logspace(np.log10(0.01), np.log10(0.03), 4, endpoint=false), 
                                np.arange(0.029, 0.1, 0.006), 
                                np.arange(0.103, 0.195, 0.008), 
                                np.arange(0.2, 0.295, 0.012), 
                                np.arange(0.3, 0.395, 0.016), 
                                np.arange(0.4, 0.51, 0.025)))


    # Function which takes in the training sample parameters and saves statistics to file
    function velocileptors_script(CosmoDict, root_path)
        try
            # Creates subfolders to store each training sample in (uses random string for this)
            rand_str = root_path * "/" * randstring(10)
            # Dictionary of the input features to Class (may need to change depending on model)
            z = CosmoDict["z"]
            cosmo_params = Dict(
                "output" => "mPk",
                "P_k_max_h/Mpc" => 20.0,
                "z_pk" => "0.0,3.",
                "h" => CosmoDict["H0"] / 100,
                "omega_b" => CosmoDict["ombh2"],
                "omega_cdm" => CosmoDict["omch2"],
                "ln10^{10}A_s" => CosmoDict["ln10As"],
                "n_s" => CosmoDict["ns"],
                "tau_reio" => 0.0568,
                "N_ur" => 2.0308,
                "N_ncdm" => 1,
                "m_ncdm" => CosmoDict["Mnu"],
                "use_ppf" => "yes",
                "w0_fld" => CosmoDict["w0"],
                "wa_fld" => CosmoDict["wa"],
                "fluid_equation_of_state" => "CLP",
                "cs2_fld" => 1.0,
                "Omega_Lambda" => 0.0,
                "Omega_scf" => 0.0)
            
            # Initializes the Class object and then computes statistics
            cosmo = classy.Class()
            cosmo.set(cosmo_params)
            cosmo.compute()
            f = cosmo.scale_independent_growth_factor_f(z)
            plin = [cosmo.pk_cb(k * CosmoDict["H0"] / 100, z) * (CosmoDict["H0"] / 100)^3 for k in konh]
            
            knw, Pnw = UTILS.pnw_dst(konh, plin)
            PT = EPT.REPT(knw, plin, pnw=Pnw, kvec=kv_target, beyond_gauss=true, one_loop=true, N=2000, 
                          extrap_min=-6, extrap_max=2, cutoff=100, threads=1)
            # Does NOT apply the AP to the statistic (must apply analytically after calling emulator)
            PT.compute_redshift_space_power_multipoles_tables(f, apar=1.0, aperp=1.0, ngauss=4) 
            
            if any(isnan, PT.p0ktable) || any(isnan, PT.p2ktable) || any(isnan, PT.p4ktable)
                @error "There are nan values!"
            else
                # Creates directory for the particular training sample and saves relevant files to it
                mkdir(rand_str)
                npzwrite(rand_str * "/kv.npy", vec(PT.kv))
                npzwrite(rand_str * "/pk_0.npy", PT.p0ktable)
                npzwrite(rand_str * "/pk_2.npy", PT.p2ktable)
                npzwrite(rand_str * "/pk_4.npy", PT.p4ktable)
                open(rand_str * "/effort_dict.json", "w") do io
                    JSON3.write(io, CosmoDict)
                end
            end
        catch e
            println("Something went wrong during calculation!")
            println(CosmoDict)
        end
    end
end

EmulatorsTrainer.compute_dataset(s, pars, root_dir, velocileptors_script, :distributed)