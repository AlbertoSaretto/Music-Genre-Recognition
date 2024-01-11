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

from utils_mgr import readheavy, get_stft, clip_stft, DataAudio


"""
Script per processare i dati e salvarli in un dataloader.
Lo script prende le clip, le trasforma in stft e applica data augmentation e normalizzazione con torchvision.transforms.v2
In output ottengo un dataloader che posso usare per addestrare la rete. Il dataloader viene salvato in un file pickle. 
Tieni d'occhio il nome con cui salvi tale file per non sovrascrivere con file precedenti.

Lo script si divide in due parti: 
1) import_and_preprocess_data: legge i dati e crea le clip
2) preprox: applica le trasformazioni e crea il dataloader.

Dato che questo script è fatto di fretta e furia, bisogna modificarlo manualmente all'uopo.
Il mio pc è in grado di elaborare un solo file alla volta.
Ecco quindi che è necessario commentare tutte le reference a "train" o "valid" a seconda che si voglia elaborare il train o il validation set.
(eg se voglio creare il dataloader per il train, commento tutte le reference a "valid" e viceversa)

Ho fatto due funzioni diverse perché pare che in questo modo la memoria venga leggermente alleggerita. 
Potrebbe essere che a volte il kernel muoia lo stesso, ma stando attenti a chiudere/riaprire, spegnere/riaccendere dovrebbe funzionare (16GB di RAM richiesti)
"""


def import_and_preprocess_data(config: dict, test = False,n_train=1):
    
    """
    if test:
        test = readheavy("test",2,"Audio/")
        test = get_stft(test)
        test_clip = clip_stft(test, 128)
        transforms = Compose([ ToTensor(), ])
        test_dataset = DataAudio(data=test_clip,transform=transforms)
        # Qui imposto una batch size arbitraria. Lo faccio perché  temo che la funzione di main mi dia problemi
        test_dataloader = DataLoader(test_dataset, 64, shuffle=False, num_workers=os.cpu_count())
        return test_dataloader
    """
    print("reading train")
    #Convert Audio into stft data
    #train = readheavy("training",1,f"Audio/")
    
    #train =  np.load(f"Audio/training_{n_train}.npy", allow_pickle = True)
    
    print("getting stft")
    #train_stft = get_stft(train)
   
    #del train
    gc.collect()

    valid = readheavy("validation",1,"Audio/")
    valid_stft = get_stft(valid)

    del valid
    gc.collect()

    # take each song and splits it into clips of n_samples 
    # creates 
    print("making clips")
    #train_clip = clip_stft(train_stft, 128)
    print("making clips")
    valid_clip = clip_stft(valid_stft, 128)

    return valid_clip

def preprox(train_clip, config = {"batch_size": 64}):
    
    # Compute mean and std of the dataset
    print("stacking")
    train_stacked = np.stack(train_clip[:,0],axis=0)
    mean_train = np.mean(train_stacked,axis=(0,1,2))
    std_train = np.std(train_stacked,axis=(0,1,2))

    del train_stacked
    gc.collect()    
    
    # Computing mean and std only on the training set.
    # In real life we will not have access to the test set so we train onlt with the training set values

    transforms = v2.Compose([v2.ToTensor(),
    v2.RandomResizedCrop(size=train_clip[:,0][0].shape, antialias=True), 
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[mean_train], std=[std_train]),
    ])

    
    train_dataset = DataAudio(data=train_clip,transform=transforms)
    #valid_dataset = DataAudio(data=valid_clip,transform=transforms)

    del train_clip
    #del valid_clip

    gc.collect()

    #Creation of dataloader classes
    batch_size = config["batch_size"]
    print("train dataloader")
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=os.cpu_count())
    print("valid dataloader")
    #valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=os.cpu_count())
    print("data loaded")

    del train_dataset
    #del valid_dataset
    gc.collect()

    #return train_dataloader, valid_dataloader
    return train_dataloader


def main():
    valid_clip = import_and_preprocess_data(config = {"batch_size": 64}, test = False,n_train=1)
    valid_dataloader = preprox(valid_clip)

    # Save train_dataloader as pickle
    with open('valid_dataloader_2D_1.pkl', 'wb') as f:
        pickle.dump(valid_dataloader, f)
 
   
    
if __name__ == "__main__":
    main()