# emulators_pipeline

This repository contains the scripts for training neural network emulators on the NERSC supercomputer.

## Scripts

#### data_generation.jl
This script generates samples prior to the training process (inputs are cosmological parameters and outputs are the statistics). The existing code is tailored to the mnuw0waCDM extension but it can be generalized to any model of interest. Here are instructions for how to modify the script to accomodate any model:

1. `pars` lists the cosmological parameter labels in the order which will be passed as input to the emulator. `lb` and `ub` denote the lower and upper bounds (for the latin hypercube) associated with the parameters. The boundaries should be wide enough to accomodate the parameter regions of interest but not unnecessarily wide as this degrades emulator accuracy.

2. `n` specifies the number of training samples. This will vary depending on the desired accuracy of the emulator balanced with increased training resources.

3. `seed` specifies the random seed used for generating the latin hypercube sampling. This is merely for reproducibility purposes and can be set to `nothing` if no seed is wanted.

4. The three lines below `s = EmulatorsTrainer.create_training_dataset(n, lb, ub)` filter out unphysical samples (w0+wa>0). Because of this, the actual number of training samples will be less than `n` in practice. Note that the indices 8,9 are currently tailored for mnuw0wacdm so these indices will have to be changed manually depending where w0,wa are found in the input vector (or removed altogether if w0,wa is not part of the model)!

5. `root_dir` is the directory where the training samples will be stored. The directory needs to be changed to match your installed environment. And the title for the emulator folder itself should ideally be labelled with the code/model/number of training samples (not a requirement, but important for preventing overwriting or confusing different emulators).

6. The emulator output k grid `kv_target` was selected to provide sufficient resolution (for interpolation purposes) across different scales but not unnecessarily high as this reduces emulator accuracy. The resolution is chosen to get coarser at higher k to avoid contributing disproportionately to the cost function. However, the choice is completely arbitrary and the user is welcome to modify as desired.

7. Inside the `velocileptors_script` function, the `cosmo_params` dictionary contains the various parameters that need to be specified in Class. Currently, only the parameters (ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa) are allowed to vary freely. The user needs to free up/fix certain parameters depending on which model is desired. E.g. if the neutrino mass is fixed, swap `"m_ncdm" => CosmoDict["Mnu"]` with `"m_ncdm" => 0.06`.

8. NOTE: To avoid issues with interpolation, we have modified the original `velocileptors` repository (to `velocileptors_free` which is installed in the environment) to allow the freedom to pass our own k grid manually (the original code sets logarithmic spacing by default), so the line calling `EPT.REPT` uses slightly different arguments than original velocileptors.

9. Currently, the emulator computes the power spectrum without the AP effect incorporated (sets apar=aperp=1 in the code), so the AP must be applied analytically in your theory code (Effort.jl has the necessary functions to do this). But if desired, the AP can be reincorporated by calculating the AP parameters and applying them to the line `PT.compute_redshift_space_power_multipoles_tables`. 
