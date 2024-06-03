from mgr.models import NNET1D, NNET2D, LitNet
from mgr.utils_mgr import main_train
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking
import audiomentations as audio
import os


if __name__ == "__main__":

    # Data augmentation
    train_transform = audio.Compose([   
        # add gaussian noise to the samples
        audio.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        # adds silence
        audio.TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0)
    ])

    
    eval_transorm = None

    # This gets updated by the hyperparameter optimization
    config_optimizer = {'lr': 1e-4,
              'lr_step': 100,
              'lr_gamma': 0,
              'weight_decay': 0.005,
              }

    model_net = NNET1D()

    config_train = {"fast_dev_run":False,
                    'max_epochs': 100,
                    'batch_size': 64,
                    'num_workers': os.cpu_count(),
                    'patience': 20,
                    'net_type':'1D',
                    'mfcc': False,
                    'normalize': False,
                    'schedule': False
                    }
 

    main_train(model_net =model_net,
                train_transforms=train_transform,
                eval_transforms= eval_transorm,
                PATH_DATA="../data/", 
                config_optimizer=config_optimizer,
                config_train=config_train,
                )
