import optuna
import pytorch_lightning as pl
import pickle
from mgr.models import NNET1D, NNET2D, LitNet
from mgr.utils_mgr import create_dataloaders
import torch
import torch.nn as nn
import os


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

    
    config_optimizer = {'lr': trial.suggest_float('lr', 1e-5, 1e-1),
              'lr_step': 10,
              'lr_gamma': 0.05,
              'weight_decay': 0.005,
              }


    config_train = {"fast_dev_run":False,
                    'max_epochs': 1,
                    'batch_size': 64,
                    'num_workers': os.cpu_count(),
                    'patience': 20,
                    'net_type':'1D',
                    'mfcc': False,
                    'normalize': False
                    }
    

    # Set the hyperparameters in the config dictionary

    # Keys necessary for config optimizer: lr, weight_decay, lr_step, lr_gamma

    # Define the optimizer as Adam
    optimizer = torch.optim.Adam(model_net.parameters(), lr = config_optimizer['lr'], weight_decay = config_optimizer['weight_decay'])
     
    trainer = pl.Trainer(
        max_epochs=config_train['max_epochs'],
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        deterministic=True,
 
        devices = "auto",
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        fast_dev_run=config_train['fast_dev_run'],
    )

    trainer.logger.log_hyperparams(config_optimizer)
   
    model = LitNet(model_net, optimizer = optimizer, config_optimizer = config_optimizer)


    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(PATH_DATA="../data/", transforms=None, net_type=config_train['net_type'], batch_size = config_train['batch_size'], num_workers = config_train['num_workers'], mfcc = config_train['mfcc'], normalize = config_train['normalize'])

    trainer.fit(model, train_dataloader, val_dataloader)
    return trainer.callback_metrics["val_loss"].item()


def HyperTune(study_name="first-study", n_trials=1, timeout=300):

    pruner = optuna.pruners.BasePruner
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