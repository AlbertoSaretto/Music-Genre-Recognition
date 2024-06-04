from mgr.models import NNET1D, NNET2D, LitNet, MixNet, CONV1D, CONV2D
from mgr.utils_mgr import main_train, RandomApply, GaussianNoise
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking
import audiomentations as audio
import os



if __name__ == "__main__":

    train_transform = { '2D':v2.Compose([
                            v2.ToTensor(),
                            RandomApply(FrequencyMasking(freq_mask_param=30), prob=0.5),     #Time and Freqeuncy are inverted bacause of the data are transposed
                            RandomApply(TimeMasking(time_mask_param=2), prob=0.5),
                            RandomApply(GaussianNoise(std = 0.015), prob=0.5),
                            ]),

                        '1D': audio.Compose([   
                            audio.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                            audio.TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0)
                            ])
    }



    eval_transform = {'2D':v2.Compose([v2.ToTensor(),             
                                      ]),
                      '1D': None           
    }

    model_net = MixNet(CONV1D(), CONV2D())
   
    
    config_optimizer = {'lr': 5e-5,
              'lr_step': 100,
              'lr_gamma': 0,
              'weight_decay': 0,
              }
    
    config_train = {"fast_dev_run":False,
                    'max_epochs': 100,
                    'batch_size': 32,
                    'num_workers': 4,
                    'patience': 20,
                    'net_type':'Mix',
                    'mfcc': True,
                    'normalize': True,
                    'schedule': False
                    }
    
    main_train(model_net = model_net,
                train_transforms=train_transform,
                eval_transforms= eval_transform,
                PATH_DATA="../data/", 
                config_optimizer=config_optimizer,
                config_train=config_train,
                )
    