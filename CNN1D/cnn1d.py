"""
ME LO SEGNO QUA PERCHé NON SAPRE IDOVE ALTRO SEGNARLO

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

if __name__ == "__main__":

    
   
    """
    ATTENZIONE: NORMALIZZAZIONE???
   
    """

    main_train(max_epochs=100,
                model_net = LitNet(NNET1D()),
                net_type='1D',
                transforms=None,
                PATH_DATA="../data/",
                batch_size=32,
                fast_dev_run=False,
                lr=0.0001
                )
        
  
    
