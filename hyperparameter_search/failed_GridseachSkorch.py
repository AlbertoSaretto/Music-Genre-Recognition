import numpy as np
from sklearn.datasets import make_classification
from torch import nn

from skorch import NeuralNetClassifier
import numpy as np
import torchvision.transforms.v2 as v2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os 
from torch.optim import Adadelta
from tqdm import tqdm
import gc
import optuna
import pytorch_lightning as pl
import time
import pickle
import utils
from utils_mgr import readheavy, get_stft, clip_stft, DataAudio, create_subset

 
import skorch

from sklearn.model_selection import GridSearchCV
from skorch import NeuralNet



def import_and_preprocess_data():
    # Load metadata and features.
    tracks = utils.load('data/fma_metadata/tracks.csv')

    #Check tracks format
    print("track shape",tracks.shape)

    #Select the desired subset among the entire dataset
    sub = 'small'
    raw_subset = tracks[tracks['set', 'subset'] <= sub] 
    
    #Creation of clean subset for the generation of training, test and validation sets
    meta_subset= create_subset(raw_subset)

    # Remove corrupted files
    corrupted = [98565, 98567, 98569, 99134, 108925, 133297]
    meta_subset = meta_subset[~meta_subset['index'].isin(corrupted)]

    #Split between taining, validation and test set according to original FMA split

    train_set = meta_subset[meta_subset["split"] == "training"]
    val_set   = meta_subset[meta_subset["split"] == "validation"]
    test_set  = meta_subset[meta_subset["split"] == "test"]

    # Standard transformations for images
    # Mean and std are computed on one file of the training set
    transforms = v2.Compose([v2.ToTensor(),
        v2.RandomResizedCrop(size=(128,513), antialias=True), 
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[1.0784853], std=[4.0071154]),
        ])

    # Create the datasets and the dataloaders
    train_dataset    = DataAudio(train_set, transform = transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count())

    val_dataset      = DataAudio(val_set, transform = transforms)
    val_dataloader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())

    test_dataset     = DataAudio(test_set, transform = transforms)
    test_dataloader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())


    return train_dataloader, val_dataloader, test_dataloader


class NNET2(nn.Module):
        
    def __init__(self,initialisation="xavier"):
        super(NNET2, self).__init__()
        
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256,kernel_size=(4,513)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(.2)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(2,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(.2)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(1,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(.2)
        )
               

        self.fc = nn.Sequential(
            nn.Linear(256, 300),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(150, 8),
            nn.Softmax(dim=1)
        )
    """
       if self.initialisation == "xavier":
            self.reset_parameters()

        elif self.initialisation == "model_parameters":
            qui voglio assicurarmi che se non ho un modello salvato, allora lo inizializzo con xavier
            altrimenti uso i parametri del modello salvato

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    """
    def forward(self,x):
        
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        x = c1 + c3
        max_pool = F.max_pool2d(x, kernel_size=(125,1))
        avg_pool = F.avg_pool2d(x, kernel_size=(125,1))
        x = max_pool + avg_pool
        x = self.fc(x.view(x.size(0), -1)) # maybe I should use flatten instead of view
        return x 

net = NeuralNet(
    NNET2,
    max_epochs=10,
    criterion=nn.CrossEntropyLoss(),
    optimizer=Adadelta,
    optimizer__rho=0.9,
    optimizer__eps=1e-06,
    optimizer__weight_decay=0,
    lr=0.1,
    train_split=False,
)

# deactivate skorch-internal train-valid split and verbose logging
net.set_params(train_split=False, verbose=0)
params = {
    'lr': np.linspace(0.001, 0.9, 100),
    "optimizer__rho": np.linspace(0.01, 1, 10),
    "optimizer__eps": np.linspace(1e-7, 1e-4, 10),
    "optimizer__weight_decay": [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
}
gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

train_dataloader, val_dataloader, test_dataloader = import_and_preprocess_data()
X, y = next(iter(train_dataloader))

# Convert y to single-label format
y = y.argmax(dim=1)

gs.fit(X, y)
print(gs.best_score_, gs.best_params_)

