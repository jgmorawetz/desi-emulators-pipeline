using EmulatorsTrainer
using JSON
using SimpleChains
using NPZ 
using AbstractCosmologicalEmulators



function run_training(df, folder_output, nn_setup_path, lr_list, n_run, n_epoch, batchsize)
    """
    Takes in the dataframe of input parameters and output statistics along with other details,
    and runs the training, continuously saving the updated weights to file each time the test loss improves.
    
    Arguments:
        'df' -> The dataframe containing the input parameters and outputs (labelled "observable").
        'folder_output' -> The folder path where the training weights and information will be stored.
        'nn_setup_path' -> The file path containing the neural network architecture.
        'lr_list' -> The list of learning rates.
        'n_run' -> The number of repetitions of a given number of epochs for a fixed learning rate.
        'n_epoch' -> The number of epochs for each of the above runs.
        'batchsize' -> The batchsize for calculating gradients.
    """ 

    # extracts parameter labels
    array_pars_in = names(df, Not(:observable))

    # extracts min/max input and output information and saves to file, then performs normalization
    in_array, out_array = EmulatorsTrainer.extract_input_output_df(df)
    in_minmax = EmulatorsTrainer.get_minmax_in(df, array_pars_in)
    out_minmax = EmulatorsTrainer.get_minmax_out(out_array)
    npzwrite(joinpath(folder_output, "inminmax.npy"), in_minmax)
    npzwrite(joinpath(folder_output, "outminmax.npy"), out_minmax)
    EmulatorsTrainer.maximin_df!(df, in_minmax, out_minmax)

    # reads in the file specifying the neural network architecture and saves modified version to file
    NN_dict = JSON.parsefile(nn_setup_path)
    NN_dict["n_input_features"] = length(array_pars_in)
    NN_dict["n_output_features"] = length(df[!, "observable"])
    open(joinpath(folder_output, "nn_setup.json"), "w") do io
        JSON.print(io, NN_dict)
    end

    # initializes the minimization and calculates initial training/test losses
    mlpd = AbstractCosmologicalEmulators._get_nn_simplechains(NN_dict)
    X, Y, Xtest, Ytest = EmulatorsTrainer.getdata(df)
    p = SimpleChains.init_params(mlpd)
    G = SimpleChains.alloc_threaded_grad(mlpd)
    mlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(Y))
    mlpdtest = SimpleChains.add_loss(mlpd, SquaredLoss(Ytest))
    report = p -> begin
        train = mlpdloss(X, p)
        test = mlpdtest(Xtest, p)
        @info "Loss:" train test
    end
    test_loss = mlpdtest(Xtest, p)
    println("Initial Test Loss: ", test_loss)

    # iterates through the different learning rates performs a certain number of separate runs per number of epochs specified 
    # and continually saves progress (if test loss improves) as you go
    for lr in lr_list
        for i in 1:n_run
            @time SimpleChains.train_batched!(G, p, mlpdloss, X, SimpleChains.ADAM(lr), n_epoch; batchsize=batchsize)
            report(p)
            test = mlpdtest(Xtest, p)
            if test_loss > test
                npzwrite(joinpath(folder_output, "weights.npy"), p)
                npzwrite(joinpath(folder_output, "best_test_loss.npy"), Array([test, lr, i])) # keeps track of best test loss so far (and which run it occurred)
                global test_loss = test
                @info "New best test loss: $test (Saved weights)"
            end
        end
    end
end