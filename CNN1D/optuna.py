
import optuna
import pytorch_lightning as pl
import pickle
from mgr.models import NNET1D, NNET2D, LitNet
from mgr.utils_mgr import create_dataloaders
import torch
import torch.nn as nn


# questo lo salvo qua solo per ricordamelo
def load_optuna( file_path = "./trial.pickle"):
    # Specify the path to the pickle file


    # Open the pickle file in read mode
    with open(file_path, "rb") as file:
        # Load the data from the pickle file
        best_optuna = pickle.load(file)
    
    #best_optuna.params["lr"] = 0.01 #changing learning rate
    hyperparameters = best_optuna.params
    
    return hyperparameters
 
def define_model(trial=None,optuna_params=None,in_features=4096):

    # Get from trial the number of n_components
    
    if trial is not None:
        # Optimize number of layers, hidden units and dropout rate
        n_layers = trial.suggest_int('n_layers', 2, 10)
        layers = []
        in_features = in_features
        for i in range(n_layers):
            out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 256)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = trial.suggest_float('dropout_l{}'.format(i), 0.2, 0.5) 
            layers.append(nn.Dropout(p))
            in_features = out_features

        layers.append(nn.Linear(in_features, 8)) # 10 classes to classify
        layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)
    
    elif optuna_params is not None:
        # Optimize number of layers, hidden units and dropout rate
        n_layers = optuna_params['n_layers']
        layers = []
        in_features = 4096
        for i in range(n_layers):
            out_features = optuna_params['n_units_l{}'.format(i)]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            p = optuna_params['dropout_l{}'.format(i)]
            layers.append(nn.Dropout(p))
            in_features = out_features

        layers.append(nn.Linear(in_features, 8))

        return nn.Sequential(*layers)
 
def objective(trial):

    model_net = NNET1D()

    # Set the hyperparameters in the config dictionary
    hyperparameters = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
    }

    # Keys necessary for config optimizer: lr, weight_decay, lr_step, lr_gamma

    # Configure optimizer Adam
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hyperparameters["lr"], weight_decay=0.01)

    trainer = pl.Trainer(max_epochs=5, check_val_every_n_epoch=1, log_every_n_steps=5, deterministic=True)

    trainer.logger.log_hyperparams(hyperparameters)

    config_optimizer = {"lr_step": 3, "lr_gamma": 0.1}
   
    model = LitNet(model_net, optimizer, config_optimizer)

    """
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(PATH_DATA = "data/",
                                                                            transforms=transforms, 
                                                                            net_type=net_type,
                                                                            batch_size=batch_size,
                                                                            num_workers=num_workers,
                                                                            mfcc=mfcc,
                                                                            normalize=normalize)
    """

    train_dataloader, val_dataloader = create_dataloaders(PATH_DATA = "data/",
                                                            net_type="1D",
                                                            batch_size=32,
                                                            mfcc=False,
                                                            normalize=False)
   
    trainer.fit(model, train_dataloader, val_dataloader)
    return trainer.callback_metrics["val_loss"].item()


def HyperTune(study_name="first-study", n_trials=3, timeout=300):

    pruner = optuna.pruners.NopPruner()
    # print(pruner) <optuna.pruners._nop.NopPruner object at 0x7f4c2466ed50>
    # print(type(pruner)) <class 'optuna.pruners._nop.NopPruner'>

    study = optuna.create_study(study_name=study_name, direction="minimize", pruner=pruner, load_if_exists=True) #storage="sqlite:///myfirstoptimizationstudy.db"
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    return trial

if __name__ == "__main__":
    pl.seed_everything(42)
    trial = HyperTune() 

    # Save trial as pickle
    with open('trialv2.pickle', 'wb') as f:
        pickle.dump(trial, f)