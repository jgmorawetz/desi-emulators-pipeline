using Distributed
using SlurmClusterManager
using ArgParse
using Random
using EmulatorsTrainer
using PyCall
using NPZ
using JSON3
using LinearAlgebra


# =============================================================================
# Argument parsing 
# =============================================================================

config = ArgParseSettings()
@add_arg_table config begin
    "--emulator_type"
    help = "Specify which emulator type to create: 'effort', 'ace', 'capse', 'mapse'."
    arg_type = String
    required = true
    "--parameters"
    help = "Specify (comma separated) the labels for the parameters of the emulator."
    arg_type = String
    required = true
    "--lower_bounds"
    help = "Specify (comma separated) the lower bounds for each of the parameters."
    arg_type = String
    required = true
    "--upper_bounds"
    help = "Specify (comma separated) the upper bounds for each of the parameters."
    arg_type = String
    required = true
    "--n_samples"
    help = "Specify the raw (before filtering) number of data samples to generate."
    arg_type = Int64
    required = true
    "--random_seed"
    help = "Specify (if desired) the random seed so the data samples can be reproduced exactly."
    arg_type = Int64
    default = nothing
    "--root_dir"
    help = "Specify the directory where the data samples will be stored."
    arg_type = String 
    required = true
    "--k_grid_path"
    help = "Specify (if using Effort) the path to read in the desired k grid."
    arg_type = String
    default = nothing
end
parsed_args = parse_args(config)
emulator_type = parsed_args["emulator_type"]
parameters = parsed_args["parameters"]
lower_bounds = parsed_args["lower_bounds"]
upper_bounds = parsed_args["upper_bounds"]
n_samples = parsed_args["n_samples"]
random_seed = parsed_args["random_seed"]
root_dir = parsed_args["root_dir"]
k_grid_path = parsed_args["k_grid_path"]
parameters = String.(split(parameters, ","))
lower_bounds = parse.(Float64, split(lower_bounds, ","))
upper_bounds = parse.(Float64, split(upper_bounds, ","))



# =============================================================================
# Sets up Slurm workers
# =============================================================================

ENV["SLURM_NTASKS"] = ENV["JULIA_TOTAL_TASKS"]
mgr = SlurmManager(;launch_timeout = 600.0, srun_post_exit_sleep = 2.0)
addprocs(mgr)

@everywhere begin
    using Random, EmulatorsTrainer, PyCall, NPZ, JSON3, LinearAlgebra
end

@everywhere begin
    # Generates the latin hypercube samples
    if random_seed != nothing
        Random.seed!(random_seed)
    end
    samples = create_training_dataset(n_samples, lower_bounds, upper_bounds)

    # Filters out bad unphysical samples with w0+wa>0 (if exist)
    if "w0" in parameters && "wa" in parameters
        w0_ind = findfirst(==("w0"), parameters)
        wa_ind = findfirst(==("wa"), parameters)
        w0wa_cond = (samples[w0_ind, :] .+ samples[wa_ind, :]) .< 0
        samples = samples[:, w0wa_cond]
    end

    # Imports necessary python packages from PyCall
    np = pyimport("numpy")
    classy = pyimport("classy")
    velocileptors = pyimport("velocileptors_free")


    function classy_script(cosmo_dict, root_dir)
        """
        Function to generate the necessary data samples for the given set of cosmological parameters.
        Arguments:
            'cosmo_dict' -> The dictionary of cosmological parameters associated with the particular data sample.
            'emulator_type' -> Either "effort", "ace", "capse", "mapse" (needed to determine which inputs to apply in class and which outputs to save).
        """ 
        try
            rand_str = root_path * "/" * randstring(10) # creates folder path for this particular data sample

            # Extracts the cosmological parameters from the dictionary
            h = cosmo_dict["H0"] / 100
            ombh2 = cosmo_dict["ombh2"]
            omch2 = cosmo_dict["omch2"]
            z = haskey(cosmo_dict, "z") ? cosmo_dict["z"] : nothing
            ln10As = haskey(cosmo_dict, "ln10As") ? cosmo_dict["ln10As"] : 3.044 # fixes ln10As, ns to fiducial values for Mapse when only transfer function emulated
            ns = haskey(cosmo_dict, "ns") ? cosmo_dict["ns"] : 0.9649
            w0 = haskey(cosmo_dict, "w0") ? cosmo_dict["w0"] : -1 # fixes w0, wa = -1, 0 if not varied in model
            wa = haskey(cosmo_dict, "wa") ? cosmo_dict["wa"] : 0
            Mnu_in_dict = haskey(cosmo_dict, "Mnu")
            Mnu = Mnu_in_dict ? cosmo_dict["Mnu"] : 0.06 # sum of neutrino mass
            N_ncdm = Mnu_in_dict ? 3 : 1
            m_ncdm = Mnu_in_dict ? "$(Mnu/3),$(Mnu/3),$(Mnu/3)" : Mnu # three denegerate species or single species
            tau = haskey(cosmo_dict, "tau") ? cosmo_dict["tau"] : nothing

            # Sets the cosmological parameters (and adds additional parameters depending on which observable is measured)
            cosmo_params = Dict("ln10^{10}A_s" => ln10As, "n_s" => ns, "h" => h, "omega_b" => ombh2, "omega_cdm" => omch2, 
                                "w0_fld" => w0, "wa_fld" => wa, "use_ppf" => "yes", "fluid_equation_of_state" => "CLP", "cs2_fld" => 1, "Omega_Lambda" => 0, "Omega_scf" => 0,
                                "Neff" => 3.044, "N_ncdm" => N_ncdm, "m_ncdm" => m_ncdm)
            if emulator_type in ["effort", "ace"]
                cosmo_params["output"] = "mPk"
                cosmo_params["z_pk"] = string(z)
                cosmo_params["P_k_max_h/Mpc"] => 20
            elseif emulator_type == "mapse"
                cosmo_params["output"] = "mPk"
                cosmo_params["z_pk"] = string(z)
                cosmo_params["P_k_max_1/Mpc"] => 15
            elseif emulator_type == "capse"
                cosmo_params["output"] = "tCl pCl lCl"
                cosmo_params["l_max_scalars"] = 10000
                cosmo_params["lensing"] = "yes"
                cosmo_params["tau_reio"] = tau 
                cosmo_params["accurate_lensing"] = 1
                cosmo_params["non_linear"] = "hmcode" 
            end

            # Initializes the cosmology
            cosmo = classy.Class()
            cosmo.set(cosmo_params)
            cosmo.compute()

            # Computes and saves different statistics depending which emulator being used
            if emulator_type == "effort"
                # Computes the EFT loop table given the linear matter power spectrum
                konhmin, konhmax, nk = 1e-4, 10, 20000
                konh = np.logspace(np.log10(konhmin), np.log10(konhmax), nk)
                ktarget = npzread(k_grid_path) # desired emulator grid
                f = cosmo.scale_independent_growth_factor_f(z)
                plin = [cosmo.pk_cb(k * h, z) * h^3 for k in konh]
                knw, Pnw = velocileptors.Utils.pnw_dst.pnw_dst(konh, plin)
                PT = velocileptors.EPT.ept_fullresum_fftw.REPT(knw, plin, pnw=Pnw, kvec=ktarget, beyond_gauss=true, one_loop=true, N=2000, # currently uses velocileptors_free version but need to change if repos get merged!
                                                            extrap_min=-6, extrap_max=2, cutoff=100, threads=1)
                PT.compute_redshift_space_power_multipoles_tables(f, apar=1, aperp=1, ngauss=4) # computes without AP (emulator incorporates analytically)
                if any(isnan, PT.p0ktable) || any(isnan, PT.p2ktable) || any(isnan, PT.p4ktable)
                    @error "There are nan values!"
                else
                    mkdir(rand_str)
                    npzwrite(rand_str * "/kv.npy", vec(PT.kv))
                    npzwrite(rand_str * "/pk_0.npy", PT.p0ktable)
                    npzwrite(rand_str * "/pk_2.npy", PT.p2ktable)
                    npzwrite(rand_str * "/pk_4.npy", PT.p4ktable)
                    open(rand_str * "/effort_dict.json", "w") do io
                        JSON3.write(io, cosmo_dict)
                    end
                end

            elseif emulator_type == "ace"
                # Computes background quantities (sigma8, sigma8(z), rs_drag, H(z), r(z), D(z), f(z))
                # and saves outputs in either ln10As or sigma8 basis depending what inputs are
                sigma8 = cosmo.sigma8
                cosmo_dict["sigma8"] = sigma8
                sigma8_z = cosmo.sigma(8/h, z)
                r_drag = cosmo.rs_drag
                H_z = cosmo.Hubble(z) * 299792.458
                r_z = cosmo.comoving_distance(z)
                D_z = cosmo.scale_independent_growth_factor(z)
                f_z = cosmo.scale_independent_growth_factor_f(z)
                result_ln10As_basis = [sigma8, sigma8_z, r_drag, H_z, r_z, D_z, f_z] # input in ln10As basis
                result_sigma8_basis = [ln10As, sigma8_z, r_drag, H_z, r_z, D_z, f_z] # input in sigma8 basis
                if any(isnan, result_ln10As_basis) || any(isnan, result_sigma8_basis)
                    @error "There are nan values!"
                else
                    mkdir(rand_str)
                    npzwrite(rand_str * "/result_ln10As_basis.npy", result_ln10As_basis)
                    npzwrite(rand_str * "/result_sigma8_basis.npy", result_sigma8_basis)
                    open(rand_str * "/ace_dict.json", "w") do io 
                        JSON3.write(io, cosmo_dict)
                    end
                end

            elseif emulator_type == "capse"
                # Computes the temperature, polarization, lensing angular CMB power spectra quantities
                cl = cosmo.lensed_cl(lmax=10000)
                ell = np.arange(length(cl["tt"]))
                factor = ell .* (ell .+ 1) ./ (2 * np.pi)
                tt = 7.42715e12 .* (factor .* cl["tt"])[3:end] # slices from l=2 to lmax (excludes l=0,1)
                te = 7.42715e12 .* (factor .* cl["te"])[3:end]
                ee = 7.42715e12 .* (factor .* cl["ee"])[3:end]
                pp = (ell .* (ell .+ 1) .* ell .* (ell .+ 1) .* cl["pp"] ./ (2 * np.pi))[3:end]
                if any(isnan, tt) || any(isnan, te) || any(isnan, ee) || any(isnan, pp)
                    @error "There are nan values!"
                else
                    mkdir(rand_str)
                    npzwrite(rand_str * "/TT.npy", tt)
                    npzwrite(rand_str * "/TE.npy", te)
                    npzwrite(rand_str * "/EE.npy", ee)
                    npzwrite(rand_str * "/PP.npy", pp)
                    open(rand_str * "/capse_dict.json", "w") do io
                        JSON3.write(io, cosmo_dict)
                    end
                end
            
            elseif emulator_type == "mapse"
                # Computes the linear matter power spectrum
                nk = 2000
                k_grid = exp.(range(log(1e-3), log(10), length=nk))
                Pcb = [cosmo.pk_cb(k, z) for k in k_grid] # uses physical Mpc (not h units)
                Pmm = [cosmo.pk(k, z) for k in k_grid]
                if any(isnan, Pcb) || any(isnan, Pmm)
                    @error "There are nan values!"
                else
                    mkdir(rand_str)
                    npzwrite(rand_str * "/pk_cb.npy", Pcb)
                    npzwrite(rand_str * "/pk_mm.npy", Pmm)
                    npzwrite(rand_str * "/k.npy", k_grid)
                    cosmo_dict_full = copy(cosmo_dict)
                    cosmo_dict_full["ln10As"] = 3.044 # these are not originally in the dictionary
                    cosmo_dict_full["ns"] = 0.9649
                    open(rand_str * "/mapse_dict.json", "w") do io
                        JSON3.write(io, cosmo_dict_full)
                    end
                end
            end
            cosmo.struct_cleanup()
            cosmo.empty()

        catch e
            println("Something went wrong during calculation!")
            println(cosmo_dict)
        end
    end
end


# =============================================================================
# Runs code
# =============================================================================

EmulatorsTrainer.compute_dataset(samples, parameters, root_dir, classy_script, :distributed)