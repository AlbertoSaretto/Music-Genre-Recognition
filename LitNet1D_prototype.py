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
class NNET1D(nn.Module):
        
    def __init__(self):
        super(NNET1D, self).__init__()
        
        
        self.c1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=256,kernel_size=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.c2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=4,padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.c3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=4,padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 300),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(150, 8),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        x = c1 + c3

        max_pool = F.max_pool1d(x, kernel_size=125)
        avg_pool = F.avg_pool1d(x, kernel_size=125)
        x = max_pool + avg_pool
        x = torch.flatten(x,start_dim = 1)
        x = self.fc(x)
        return x 



# Define a LightningModule (nn.Module subclass)
# A LightningModule defines a full system (ie: a GAN, autoencoder, BERT or a simple Image Classifier).
class LitNet(pl.LightningModule):
    
    def __init__(self, config: dict):
       
        super().__init__()
        # super(NNET2, self).__init__() ? 
        
        print('Network initialized')
        
        self.net = NNET1D()
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
    

    

def main():
    pl.seed_everything(666)
  
    # Set the hyperparameters in the config dictionary
    # Parameters found with Optuna. Find a way to automatically import this
    
    # Specify the path to the pickle file
    file_path = "./trial.pickle"

    # Open the pickle file in read mode
    with open(file_path, "rb") as file:
        # Load the data from the pickle file
        best_optuna = pickle.load(file)
    
    best_optuna.params["lr"] = 0.01 #changing learning rate
    hyperparameters = best_optuna.params
    

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