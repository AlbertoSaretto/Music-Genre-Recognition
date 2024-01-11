import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import librosa
import librosa.display
import os
import pytorch_lightning as pl
import os 
from torch.optim import Adadelta


import torchvision.transforms.v2 as v2
import torch
from torchvision.transforms import Compose, ToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings

import utils
import utils_mgr
from utils_mgr import getAudio

import pickle


print("let's start")

##tensorboard --logdir=lightning_logs/ 
# to visualize logs

def import_and_preprocess_data(PATH_DATA="/home/diego/fma/data/"):
    # Load metadata and features.
    tracks = utils.load(PATH_DATA + 'fma_metadata/tracks.csv')


    #Select the desired subset among the entire dataset
    sub = 'small'
    raw_subset = tracks[tracks['set', 'subset'] <= sub] 
    
    #Creation of clean subset for the generation of training, test and validation sets
    meta_subset= utils_mgr.create_subset(raw_subset)

    # Remove corrupted files
    corrupted = [98565, 98567, 98569, 99134, 108925, 133297]
    meta_subset = meta_subset[~meta_subset['index'].isin(corrupted)]

    #Split between taining, validation and test set according to original FMA split

    train_set = meta_subset[meta_subset["split"] == "training"]
    val_set   = meta_subset[meta_subset["split"] == "validation"]
    test_set  = meta_subset[meta_subset["split"] == "test"]

    # Standard transformations for images
    # Mean and std are computed on one file of the training set
    transforms = v2.Compose([torch.Tensor])

    # Create the datasets and the dataloaders
    train_dataset    = utils_mgr.DataAudio(train_set, transform = transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count())

    val_dataset      = utils_mgr.DataAudio(val_set, transform = transforms)
    val_dataloader   = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count())

    test_dataset     = utils_mgr.DataAudio(test_set, transform = transforms)
    test_dataloader  = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count())


    return train_dataloader, val_dataloader, test_dataloader


# Remember to add initialisation of weights
class Block1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            Block1d(in_channels=1, out_channels=16, kernel_size=256, stride=64, padding=128),
            nn.MaxPool1d(kernel_size=4, stride=4),
            Block1d(in_channels=16, out_channels=32, kernel_size=32, stride=2, padding=16),
            nn.MaxPool1d(kernel_size=2, stride=2),
            Block1d(in_channels=32, out_channels=64, kernel_size=16, stride=2, padding=8),
            nn.MaxPool1d(kernel_size=2, stride=2),
            Block1d(in_channels=64, out_channels=128, kernel_size=8, stride=2, padding=4)
        )

    def forward(self, x):
        return self.block(x) # [128,1,65]
    

class Baseline1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = ConvBlock1d()
        self.avg_pool = nn.MaxPool1d(kernel_size=64)
        self.max_pool = nn.AvgPool1d(kernel_size=64)
        self.classifier = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64,8)
        )
        
    def forward(self, x):
        x = self.conv_block(x) # [128, 64]
        x_avg = self.avg_pool(x) # [128, 1]
        x_max = self.max_pool(x) # [128, 1]
        x = torch.cat([x_avg, x_max], dim = 1) # [256, 1]
        return self.classifier(x.reshape((x.shape[0], -1)))
    




# Define a LightningModule (nn.Module subclass)
# A LightningModule defines a full system (ie: a GAN, autoencoder, BERT or a simple Image Classifier).
class LitNet(pl.LightningModule):
    
    def __init__(self, config: dict):
       
        super().__init__()
        # super(NNET2, self).__init__() ? 
        
        print('Network initialized')
        
        self.net = Baseline1d()
    def __init__(self):
        super().__init__()
        self.conv_block = ConvBlock1d()
        self.avg_pool = nn.MaxPool1d(kernel_size=64)
        self.max_pool = nn.AvgPool1d(kernel_size=64)
        self.classifier = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64,8)
        )
        
    def forward(self, x):
        x = self.conv_block(x) # [128, 64]
        x_avg = self.avg_pool(x) # [128, 1]
        x_max = self.max_pool(x) # [128, 1]
        x = torch.cat([x_avg, x_max], dim = 1) # [256, 1]
        return self.classifier(x.reshape((x.shape[0], -1)))()
        self.val_loss = []
        self.train_loss = []
        self.best_val = np.inf
        self.config = config
        

    def forward(self,x):
        return self.net(x)

    # Training_step defines the training loop. 
    def training_step(self, batch, batch_idx=None):
        # training_step defines the train loop. It is independent of forward
        x_batch = batch[0]
        label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch) # Diego nota: aggiungere weights in base a distribuzione classi dataset?
        self.train_loss.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx=None):
        # validation_step defines the validation loop. It is independent of forward
        # When the validation_step() is called,
        # the model has been put in eval mode and PyTorch gradients have been disabled. 
        # At the end of validation, the model goes back to training mode and gradients are enabled.
        x_batch = batch[0]
        label_batch = batch[1]

        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch)

        val_acc = np.sum(np.argmax(label_batch.detach().cpu().numpy(), axis=1) == np.argmax(out.detach().cpu().numpy(), axis=1)) / len(label_batch)

            
        #validation_loss = np.append(validation_loss,1 - val_acc)
        #print(f"accuracy: {val_acc*100} %")
        
        # Should I save the model based on loss or on accuracy?7
        """
        LitNet doesnt need to save the model, it is done automatically by pytorch lightning
        if loss.item() < self.best_val:
            print("Saved Model")
            torch.save(self.net.state_dict(), "saved_models/nnet2/model.pt")
            self.best_val = loss.item()
        """


        self.val_loss.append(loss.item())
        self.log("val_loss", loss.item(), prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adadelta(self.net.parameters(), lr=self.config["lr"],rho=self.config["rho"], eps=self.config["eps"], weight_decay=self.config["weight_decay"])
        return optimizer
    


def load_optuna( file_path = "./trial.pickle"):
    # Specify the path to the pickle file


    # Open the pickle file in read mode
    with open(file_path, "rb") as file:
        # Load the data from the pickle file
        best_optuna = pickle.load(file)
    
    #best_optuna.params["lr"] = 0.01 #changing learning rate
    hyperparameters = best_optuna.params
    
    return hyperparameters
 

def main():
    pl.seed_everything(666)
      
    # Define the EarlyStopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        min_delta=0.01,     # Minimum change in the monitored metric
        patience=3,          # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # Mode: 'min' if you want to minimize the monitored quantity (e.g., loss)
    )


    # I think that Trainer automatically takes last checkpoint.
    trainer = pl.Trainer(max_epochs=100, check_val_every_n_epoch=5, log_every_n_steps=1, 
                         deterministic=True,callbacks=[early_stop_callback], ) # profiler="simple" remember to add this and make fun plots
    model = LitNet(hyperparameters)

    """
    # Load model weights from checkpoint
    CKPT_PATH = "./lightning_logs/version_1/checkpoints/epoch=29-step=3570.ckpt"
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    """

    train_dataloader, val_dataloader, test_dataloader = import_and_preprocess_data(PATH_DATA="Data/")
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model=model,dataloaders=test_dataloader,verbose=True)


if __name__ == "__main__":
    main()