"""
ME LO SEGNO QUA PERCHé NON SAPREI DOVE ALTRO SEGNARLO

from argparse import ArgumentParser


def main(hparams):
    model = LightningModule()
    trainer = Trainer(accelerator=hparams.accelerator, devices=hparams.devices)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)

"""


from mgr.models import NNET1D, NNET2D, LitNet
from mgr.utils_mgr import main_train
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking
import audiomentations as audio
import os

"""
Here you can find the list of the models searched during the hyperparameter optimization.
At the end, the best performing one is NNET1D_BN, which is saved in the MGR function.

We also find that transformations help the model to generalize better, so we add some data augmentation techniques.
"""

# Start by removing stuff that requires an experiment, like Dropout or BatchNorm
class NNET1D_plain(nn.Module):
        
    def __init__(self):
        super(NNET1D_plain, self).__init__()
        
        
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=128, stride=32, padding=64),
         
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=4, stride=4),
          
        )
        

        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=32, stride=2, padding=16),
           
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
          
        )

        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16, stride=2, padding=8),
            
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
         
        )
        
        #Trying to add 4th convolutional block
        self.c4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=8,stride=2, padding=4),
        
            nn.ReLU(inplace = True),
           
        )
        

        self.fc = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace = True),

            nn.Linear(512, 128),
            nn.ReLU(inplace = True),

            nn.Linear(128,128),
            nn.ReLU(inplace = True),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
           
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
        

    def forward(self, x):

        c1 = self.c1(x)
        
        c2 = self.c2(c1)
        
        c3 = self.c3(c2)
        
        c4 = self.c4(c3)


        max_pool = F.max_pool1d(c4, kernel_size=64)
        avg_pool = F.avg_pool1d(c4, kernel_size=64)

        #Concatenate max and average pooling
        x = torch.cat([max_pool, avg_pool], dim = 1) 

        
        # x dimensions are [batch_size, channels, length, width]
        # All dimensions are flattened except batch_size  
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        return x 

# Adding BatchNorm to avoid overfitting
class NNET1D_BN(nn.Module):
        
    def __init__(self):
        super(NNET1D_BN, self).__init__()
        
        
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=128, stride=32, padding=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=4, stride=4),
          
        )
        

        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
          
        )

        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
         
        )
        
        #Trying to add 4th convolutional block
        self.c4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=8,stride=2, padding=4),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
           
        )
        

        self.fc = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace = True),

            nn.Linear(512, 128),
            nn.ReLU(inplace = True),

            nn.Linear(128,128),
            nn.ReLU(inplace = True),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
           
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
        

    def forward(self, x):

        c1 = self.c1(x)
        
        c2 = self.c2(c1)
        
        c3 = self.c3(c2)
        
        c4 = self.c4(c3)


        max_pool = F.max_pool1d(c4, kernel_size=64)
        avg_pool = F.avg_pool1d(c4, kernel_size=64)

        #Concatenate max and average pooling
        x = torch.cat([max_pool, avg_pool], dim = 1) 

        
        # x dimensions are [batch_size, channels, length, width]
        # All dimensions are flattened except batch_size  
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        return x 

# Adding Dropout to avoid overfitting
class NNET1D_BN_DropOut(nn.Module):
        
    def __init__(self):
        super(NNET1D_BN_DropOut, self).__init__()
        
        
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=128, stride=32, padding=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.5)
          
        )
        

        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
          
        )

        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        
        #Trying to add 4th convolutional block
        self.c4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=8,stride=2, padding=4),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5)
        )
        

        self.fc = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(128,128),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
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
        

    def forward(self, x):

        c1 = self.c1(x)
        
        c2 = self.c2(c1)
        
        c3 = self.c3(c2)
        
        c4 = self.c4(c3)


        max_pool = F.max_pool1d(c4, kernel_size=64)
        avg_pool = F.avg_pool1d(c4, kernel_size=64)

        #Concatenate max and average pooling
        x = torch.cat([max_pool, avg_pool], dim = 1) 

        
        # x dimensions are [batch_size, channels, length, width]
        # All dimensions are flattened except batch_size  
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        return x 

# Changing Kernels
class NNET1D_K(nn.Module):
        
    def __init__(self):
        super(NNET1D_K, self).__init__()
        
        
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=512,stride=256),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace = True),
            #nn.MaxPool1d(kernel_size=4, stride=4),
        )
                

        self.c2 = nn.Sequential(
                    nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8,),
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace = True),
                # nn.MaxPool1d(kernel_size=4, stride=4),
                )


        self.c3 = nn.Sequential(
                    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4,),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace = True),
                    #nn.AvgPool1d(kernel_size=8, stride=4),
                )

        self.c4 = nn.Sequential(
                    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace = True),
                    #nn.AvgPool1d(kernel_size=16, stride=8),
                )

        self.fc = nn.Sequential(
            nn.Linear(7680, 1024), 
            nn.ReLU(inplace = True),

            nn.Linear(1024, 512), 
            nn.ReLU(inplace = True),

            nn.Linear(512, 128),
            nn.ReLU(inplace = True),

           # nn.Linear(128,128),
          #  nn.ReLU(inplace = True),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
           
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
        

    def forward(self, x):
        
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        
        max_pool = F.max_pool1d(c4, kernel_size=64)
        avg_pool = F.avg_pool1d(c4, kernel_size=64)

        #Concatenate max and average pooling
        x = torch.cat([max_pool, avg_pool], dim = 1) 

        
        # x dimensions are [batch_size, channels, length, width]
        # All dimensions are flattened except batch_size  
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x 


# 
class NNET1D_Lite(nn.Module):
        
    def __init__(self):
        super(NNET1D_Lite, self).__init__()
        
        
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=128, stride=32, padding=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=4, stride=4),
          
        )
        

        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
          
        )

        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
         
        )
        
        #Trying to add 4th convolutional block
        self.c4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=8,stride=2, padding=4),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
           
        )
        

        self.fc = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(inplace = True),

            nn.Linear(512, 128),
            nn.ReLU(inplace = True),

            nn.Linear(128,128),
            nn.ReLU(inplace = True),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
           
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
        

    def forward(self, x):

        c1 = self.c1(x)
        
        c2 = self.c2(c1)
        
        c3 = self.c3(c2)
        
       # c4 = self.c4(c3)


        max_pool = F.max_pool1d(c3, kernel_size=64)
        avg_pool = F.avg_pool1d(c3, kernel_size=64)

        #Concatenate max and average pooling
        x = torch.cat([max_pool, avg_pool], dim = 1) 

        
        # x dimensions are [batch_size, channels, length, width]
        # All dimensions are flattened except batch_size  
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        return x 


# Full tuning with Optuna
class NNET1D_BN_hyper(nn.Module):
        
    def __init__(self,trial=None,optuna_params=None):
        super(NNET1D_BN_hyper, self,).__init__()
        
        self.trial = trial
        self.optuna_params = optuna_params
        
        if trial is not None:
            channels = [trial.suggest_int(f'channels_{i}', 64, 256) for i in range(4)]
           

        elif optuna_params is not None:
            print("Using optuna params")
            #channels = [optuna_params[f'channels_{i}'] for i in range(4)]
            # Benchmark = [64,128,256,512]
            channels = [256,256,512,1024] #SETTING MANUALLY. 30-03 TRYING HIGHER CHANNELS FOR EACH LAYER
          
        else: 
            print("Using default channels")
            channels = [64,128,256,512]
        
        
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=channels[0], kernel_size=128, stride=32, padding=64),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=4, stride=4),
          
        )
        

        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
          
        )

        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=2, stride=2),
         
        )
        
        
        self.c4 = nn.Sequential(
            nn.Conv1d(in_channels=channels[2], out_channels=channels[3], kernel_size=8,stride=2, padding=4),
            nn.BatchNorm1d(channels[3]),
            nn.ReLU(inplace = True),
           
        )
        

        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        # Initialize only self.classifer weights
        # We need the weights of the trained CNNs
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def _define_fc(self,in_features):
        # Get from trial the number of n_components
       
        if self.trial is not None:
            
            # Optimize number of layers, hidden units and dropout rate
            n_layers = self.trial.suggest_int('n_layers', 4, 10)
            layers = []
            in_features = in_features
            for i in range(n_layers):
                out_features = self.trial.suggest_int('n_units_l{}'.format(i), 4, 256)
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.ReLU())
                p = self.trial.suggest_float('dropout_l{}'.format(i), 0.2, 0.5) 
                layers.append(nn.Dropout(p))
                in_features = out_features

            layers.append(nn.Linear(in_features, 8)) # 10 classes to classify
            layers.append(nn.Softmax(dim=1))

            return nn.Sequential(*layers)
        
        elif self.optuna_params is not None:
            # Optimize number of layers, hidden units and dropout rate
            n_layers = self.optuna_params['n_layers']
            layers = []
            in_features = in_features
            for i in range(n_layers):
                out_features = self.optuna_params['n_units_l{}'.format(i)]
                layers.append(nn.Linear(in_features, out_features))
                layers.append(nn.ReLU())
                p = self.optuna_params['dropout_l{}'.format(i)]
                layers.append(nn.Dropout(p))
                in_features = out_features

            layers.append(nn.Linear(in_features, 8))

            return nn.Sequential(*layers)
        
        
    def dont_optimize_fc(self,in_features):
        
        fc = nn.Sequential(
            nn.Linear(in_features, 512), 
            nn.ReLU(inplace = True),

            nn.Linear(512, 128),
            nn.ReLU(inplace = True),

            nn.Linear(128,128),
            nn.ReLU(inplace = True),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
           
            nn.Linear(64, 8),
            nn.Softmax(dim=1)
        )

        return fc

    def forward(self, x):

        c1 = self.c1(x)
        
        c2 = self.c2(c1)
        
        c3 = self.c3(c2)
        
        c4 = self.c4(c3)


        max_pool = F.max_pool1d(c4, kernel_size=64)
        avg_pool = F.avg_pool1d(c4, kernel_size=64)

        #Concatenate max and average pooling
        x = torch.cat([max_pool, avg_pool], dim = 1) 

        
        # x dimensions are [batch_size, channels, length, width]
        # All dimensions are flattened except batch_size  
        x = torch.flatten(x, start_dim=1)
        self.fc = self.dont_optimize_fc(in_features=x.shape[1])
        
        x = self.fc(x)
        return x 


# questo lo salvo qua solo per ricordamelo
def load_optuna(file_path = "./trial.pickle"):
    import pickle
    # Specify the path to the pickle file
    # Open the pickle file in read mode
    with open(file_path, "rb") as file:
        # Load the data from the pickle file
        best_optuna = pickle.load(file)
    
    #best_optuna.params["lr"] = 0.01 #changing learning rate
    hyperparameters = best_optuna.params
    
    return hyperparameters
 

if __name__ == "__main__":

    # Data augmentation
    import audiomentations as audio

    train_transform = audio.Compose([   
        # add gaussian noise to the samples
        audio.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        # change the speed or duration of the signal without changing the pitch
        #audio.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        # pitch shift the sound up or down without changing the tempo
        #audio.PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
        # adds silence
        audio.TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0)
    ])


    
    eval_transorm = None

    # This gets updated by the hyperparameter optimization
    config_optimizer = {'lr': 1e-4,
              'lr_step': 10,
              'lr_gamma': 0.05,
              'weight_decay': 0.005,
              }

    optuna_hyper = load_optuna("./cnn-hypertune-last-of-today-30-03.pickle")

    config_optimizer.update(optuna_hyper)

   # model_net = NNET1D_BN_hyper(optuna_params=optuna_hyper)

    model_net = NNET1D_BN()

    config_train = {"fast_dev_run":False,
                    'max_epochs': 100,
                    'batch_size': 16,
                    'num_workers': os.cpu_count(),
                    'patience': 20,
                    'net_type':'1D',
                    'mfcc': False,
                    'normalize': False
                    }
 



    main_train(model_net =model_net,
                train_transforms=train_transform,
                eval_transforms= eval_transorm,

                PATH_DATA="../data/", 
                config_optimizer=config_optimizer,
                config_train=config_train,
                )

        
  
    
