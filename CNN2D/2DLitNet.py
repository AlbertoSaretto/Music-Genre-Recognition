from mgr.models import NNET1D, LitNet
from mgr.utils_mgr import main_train, RandomApply
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking
import os

#Housing anywhere case monaco 48 ore per vedere e dire ok sei tutelato, tipo booking

#OPTUNA RESULTS:
# weight decay: 0.000572


#Class to apply random transformations to the data
class RandomApply:
    def __init__(self, transform, prob):
        self.transform = transform
        self.prob = prob

    def __call__(self, x):
        if torch.rand(1) < self.prob:
            return self.transform(x)
        return x
    
class GaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        #return tensor + absolute value of noise
        return tensor + torch.abs(torch.randn(tensor.size()) * self.std + self.mean)




# Start by removing stuff that requires an experiment, like Dropout or BatchNorm
class NNET2D(nn.Module):
        
    def __init__(self, initialisation="xavier"):
        super(NNET2D, self).__init__()
        
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256,kernel_size=(4,20)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout2d(.1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(2,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout2d(.1)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(1,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout2d(.1)
        )
                

        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 8),
            nn.Softmax(dim=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Initialize only self.classifer weights
        # We need the weights of the trained CNNs
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
        
        
    def forward(self,x):
        c1 = self.c1(x) 
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        x = c1 + c3
        max_pool = F.max_pool2d(x, kernel_size=(125,1))
        avg_pool = F.avg_pool2d(x, kernel_size=(125,1))
        x = torch.cat([max_pool,avg_pool],dim=1)
        x = self.fc(x.view(x.size(0), -1)) # Reshape x to fit in linear layers. Equivalent to F.Flatten
        return x  


if __name__ == "__main__":

    train_transform = v2.Compose([
        v2.ToTensor(),
        RandomApply(FrequencyMasking(freq_mask_param=40), prob=0.5),     #Time and Freqeuncy are inverted bacause of the data are transposed
        RandomApply(TimeMasking(time_mask_param=3), prob=0.5),
        #Add Gaussian noise on spectrogram with v2
        RandomApply(GaussianNoise(std = 0.025), prob=0.5),
    ])

    train_transform2 = v2.Compose([
        v2.ToTensor(),
        FrequencyMasking(freq_mask_param=30),     #Time and Freqeuncy are inverted bacause of the data are transposed
        TimeMasking(time_mask_param=2),
        #Add Gaussian noise on spectrogram with v2
        GaussianNoise(std = 0.02),
    ])


    eval_transform = v2.Compose([
        v2.ToTensor(),
    ])
   
    
    config_optimizer = {'lr': 5e-5,
              'lr_step': 100,
              'lr_gamma': 0,
              'weight_decay': 0.0057,
              }
    
    config_train = {"fast_dev_run":False,
                    'max_epochs': 100,
                    'batch_size': 64,
                    'num_workers': 6,
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
    

##tensorboard --logdir=lightning_logs/ 
# to visualize logs