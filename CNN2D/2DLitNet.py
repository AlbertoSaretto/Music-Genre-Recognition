from mgr.models import CONV1D, NNET1D, CONV2D, NNET2D, LitNet, MixNet
from mgr.utils_mgr import main_train, RandomApply, GaussianNoise
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking
import os
from torch.optim import Adam


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score, MulticlassAccuracy
import torcheval.metrics



if __name__ == "__main__":

    train_transform = v2.Compose([
        v2.ToTensor(),
        RandomApply(FrequencyMasking(freq_mask_param=30), prob=0.5),     #Time and Freqeuncy are inverted bacause of the data are transposed
        RandomApply(TimeMasking(time_mask_param=2), prob=0.5),
        RandomApply(GaussianNoise(std = 0.015), prob=0.5),
    ])

    eval_transform = v2.Compose([
        v2.ToTensor(),
    ])
   
    
    config_optimizer = {'lr': 5e-5,
              'lr_step': 100,
              'lr_gamma': 0,
              'weight_decay': 0.005,
              }
    
    config_train = {"fast_dev_run":False,
                    'max_epochs': 100,
                    'batch_size': 64,
                    'num_workers': os.cpu_count(),
                    'patience': 20,
                    'net_type':'2D',
                    'mfcc': True,
                    'normalize': True,
                    'schedule': False
                    }

    main_train(model_net = NNET2D(),
                train_transforms=train_transform,
                eval_transforms= eval_transform,
                PATH_DATA="../data/", 
                config_optimizer=config_optimizer,
                config_train=config_train,
                )
    
