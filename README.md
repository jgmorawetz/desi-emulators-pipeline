# desi-emulators-pipeline

This repository contains the files and scripts necessary for training neural network emulators on the NERSC supercomputer.

## Installation 

First, we must create an environment which permits the use of PyCall within Julia. `cd` to your preferred directory and run the following steps:

1. 
```
module load python
conda create -n classy_env python=3.10 -y
conda activate classy_env
pip install numpy scipy cython
pip install classy
pip install git+https://github.com/marcobonici/velocileptors_free.git
```
2. Run ```which python``` and copy/paste the folder path (will be needed later).

3. Open a new terminal and `cd` to the folder where you started, and run the following steps. In the line `ENV["PYTHON"]`, paste the directory path that was copied from step 2.
```
module load julia
julia
using Pkg
Pkg.activate(".")
Pkg.add("PyCall")
ENV["PYTHON"] = <which-python-folder-path>
Pkg.build("PyCall")
exit()
```

4. To install the remainder of the necessary packages, `cp` the `Project.toml` and `Manifest.toml` files to your directory, and run the following commands:
```
julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
exit()
```

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



#### training.jl
This script performs the training itself after the data generation has finished. The existing code is again tailored to the mnuw0waCDM extension but it can be generalized to any model of interest. Here are instructions for how to modify the script to accomodate any model:

1. `nk` at the top of the script is the size of the emulator k grid, which must be adjusted depending on which k grid the user chooses to apply.

2. NOTE: The function `preprocess` applies a rescaling to remove degrees of freedom that are easy to incorporate analytically in a separate step and thus improve emulator accuracy. E.g. for the linear matter power spectrum component, the statistic is divided by $A_s D(z)^2$ and then remultiplied by this after the emulator prediction is made.

3. The function `get_observable_tuple` takes in the cosmology parameters and the full power spectrum result and outputs the tuple of the parameters and the relevant component of the power spectrum. This is tailored currently to mnuw0waCDM, and the user needs to adjust if a different model is used. E.g. if neutrino mass is fixed, `Mν = cosmo_pars["Mnu"]` -> `Mν = 0.06` and remove the entry `cosmo_pars["Mnu"]` from the tuple as it is no longer an emulator parameter. However, the set of parameters that are passed to `preprocess` within this function must remain the same so the existing parameters must be kept (even if they are made to fixed values). In the returned tuple, however, only the free emulator parameters should be listed and they should be listed in the same order as the emulator input.

4. `n_input_features` is the number of input parametes to the emulator. It is currently tailored to mnuw0waCDM (including ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa) and needs to be adjusted by the user if a different model is used.

5. `df` initiates the dataframe which holds the parameters and the outputs. It is currently tailored to mnuw0waCDM and parameters must be removed/added depending on which parameters are allowed to vary by the emulator.

6. `array_pars_in` is the list containing the parameter labels that are passed to `df`. The user must adjust this to ensure they are the same.

7. NOTE: The inputs and outputs are normalized between -1 and 1 (based on the minimum/maximum values of the inputs and outputs) for emulator purposes and then are unnormalized at the end when the emulator prediction is computed. The minimum and maximum values are thus saved to file in advance.

8. `NN_dict` initializes the neural network architecture. It reads in the file `nn_setup.json` (which the user needs to `cp` their directory in advance) which contains the number of hidden layers, the number of neurons in each layer, etc. The user needs to adjust these parameters beforehand depending on what their desired settings are. The current settings are not necessarily better and are simply placeholders.
