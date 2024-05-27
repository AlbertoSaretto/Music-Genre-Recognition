import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import librosa
from tensorflow.keras.utils import to_categorical
import mgr.utils as utils
import librosa
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm
import warnings
import h5py
import os
import gc



#Function to extract audio signal from .mp3 file
def getAudio(idx, sr_i=None, PATH_DATA="data/"):
    #Get the audio file path
    filename = utils.get_audio_path(PATH_DATA+"fma_small/", idx)

    #Load the audio (sr = sampling rate, number of audio carries per second)
    x, sr = librosa.load(filename, sr=sr_i, mono=True)  #sr=None to consider original sampling rate

    return x, sr


#Function to convert labels into array of probabilities
def conv_label(label):
    le = LabelEncoder()
    y = le.fit_transform(label)
    y = to_categorical(y, 8)
    return y


#This function create a dataframe containing index and label for a subset, simplifying its structure and making it suitable 
#for the split into test, validation and validation sets, moreover labels are converted into numpy arrays
def create_subset(raw_sub):
    labels = conv_label(raw_sub['track']['genre_top']) 
    subset = pd.DataFrame(columns = ['index', 'genre_top', 'split', 'labels'])
    
    for j in range(len(raw_sub['track'].index)):
        index = raw_sub['track'].index[j]
        genre = raw_sub['track', 'genre_top'][index]
        split = raw_sub['set', 'split'][index]
        label = labels[j]
        # Create a new row as a Series
        new_row = pd.Series({'index': index, 'genre_top': genre, 'split':split, 'labels':label})
        subset.loc[len(subset)] = new_row
        
    return subset



def import_and_preprocess_data(PATH_DATA="../data/"):
    # Load metadata and features.
    tracks = utils.load(PATH_DATA+'fma_metadata/tracks.csv')


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

    return train_set, val_set, test_set


def create_dataloaders(PATH_DATA="data/",transforms=None,batch_size=64,num_workers=os.cpu_count(),net_type='1D', mfcc=False, normalize=False, train_transforms=None, eval_transforms=None):
  
    from mgr.datasets import DataAudio, DataAudioMix
    from torch.utils.data import DataLoader
    
    train_set, val_set, test_set = import_and_preprocess_data(PATH_DATA)

    if net_type == '1D' or net_type == '2D':
        train_dataset  = DataAudio(train_set, transform = train_transforms, PATH_DATA=PATH_DATA, net_type=net_type, mfcc=mfcc, normalize=normalize)
        val_dataset    = DataAudio(val_set, transform = eval_transforms, PATH_DATA=PATH_DATA, net_type=net_type, mfcc=mfcc, normalize=normalize)
        test_dataset   = DataAudio(test_set, transform = eval_transforms, PATH_DATA=PATH_DATA, net_type=net_type, test = True, mfcc=mfcc, normalize=normalize)

    elif net_type == 'Mix':
        train_dataset  = DataAudioMix(train_set, transform = train_transforms, PATH_DATA=PATH_DATA, mfcc=mfcc, normalize=normalize)
        val_dataset    = DataAudioMix(val_set, transform = eval_transforms, PATH_DATA=PATH_DATA, mfcc=mfcc, normalize=normalize)
        test_dataset   = DataAudioMix(test_set, transform = eval_transforms, PATH_DATA=PATH_DATA, test = True, mfcc=mfcc, normalize=normalize)  



    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader


#Function for main training of the network
def main_train(model_net, 
               config_optimizer= None,
               config_train = None,
               PATH_DATA = "data/",
    
               train_transforms=None,
               eval_transforms=None,):
    
    import pytorch_lightning as pl
    from mgr.models import LitNet
    import torch

    pl.seed_everything(0)

    # Set the device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # release all unoccupied cached memory
        torch.cuda.empty_cache()
        # printo GPU info
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print('{} {} GPU available'.format(str(device_count), str(device_name)))

    # Define the optimizer as Adam
    optimizer = torch.optim.Adam(model_net.parameters(), lr = config_optimizer['lr'], weight_decay = config_optimizer['weight_decay'])
        
    # Define the EarlyStopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        min_delta=0.01,     # Minimum change in the monitored metric
        patience=config_train['patience'],          # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # Mode: 'min' if you want to minimize the monitored quantity (e.g., loss)
    )

    # Set the trainer's device to GPU if available
    trainer = pl.Trainer(
        max_epochs=config_train['max_epochs'],
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        deterministic=True,
        callbacks=[early_stop_callback],
        devices = "auto",
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        fast_dev_run=config_train['fast_dev_run'],
    )

    model = LitNet(model_net, optimizer = optimizer, config_optimizer = config_optimizer, schedule = config_train['schedule'])

    train_dataloader,  val_dataloader, test_dataloader  = create_dataloaders(PATH_DATA=PATH_DATA, train_transforms=train_transforms, eval_transforms=eval_transforms,net_type=config_train['net_type'], batch_size = config_train['batch_size'], num_workers = config_train['num_workers'], mfcc = config_train['mfcc'], normalize = config_train['normalize'])
    

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model=model,dataloaders=test_dataloader,verbose=True)


    return model


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

